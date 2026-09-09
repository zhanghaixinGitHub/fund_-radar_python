"""保留三试点未来源值变化；不回填、不更改已有样本和模型。"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql as pg

revision = "20260909_19"
down_revision = "20260909_18"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "cash_source_observation",
        sa.Column(
            "observation_id",
            sa.BigInteger(),
            sa.Identity(),
            primary_key=True,
            comment="观察记录游标，按生成顺序分页，不当提交顺序证明",
        ),
        sa.Column(
            "source_id",
            pg.UUID(),
            sa.ForeignKey("source_registry.source_id"),
            nullable=False,
            comment="实际来源主键，不跨来源拼接",
        ),
        sa.Column(
            "fund_code",
            sa.String(32),
            sa.ForeignKey("fund_share_class.fund_code"),
            nullable=False,
            comment="仅三只现金研究试点",
        ),
        sa.Column("kind", sa.String(16), nullable=False, comment="NAV净值或DIVIDEND现金分红事件"),
        sa.Column("record_key", sa.String(64), nullable=False, comment="净值业务日ISO字符串或稳定的来源分红事件键"),
        sa.Column("event_date", sa.Date(), comment="净值日或分红净值除息/除息/公告日，均缺失时保留空值"),
        sa.Column("operation", sa.String(8), nullable=False, comment="源行新增、更新或删除；删除不是无事件证明"),
        sa.Column(
            "observed_at",
            sa.DateTime(timezone=True),
            nullable=False,
            comment="触发器实际执行的数据库时间，不是首次公告时间或事务提交时间",
        ),
        sa.Column(
            "expires_at",
            sa.DateTime(timezone=True),
            nullable=False,
            comment="按观察时来源保留天数计算的失效时间；到期不能用作有效证据",
        ),
        sa.Column(
            "before_payload", pg.JSONB(), comment="本次写入前的现金相关源值；首次INSERT为空，不将旧值观察时间倒填"
        ),
        sa.Column(
            "after_payload", pg.JSONB(), comment="本次写入后的现金相关源值；DELETE为空，不包含来源凭据或原始响应"
        ),
        sa.Column(
            "content_hash",
            sa.String(64),
            nullable=False,
            comment="种类及前后JSONB源值的SHA256，不含观察时间，不是审批签名",
        ),
        sa.CheckConstraint("kind IN ('NAV','DIVIDEND')", name="ck_cash_observation_kind"),
        sa.CheckConstraint("fund_code IN ('001632','006730','008888')", name="ck_cash_observation_fund"),
        sa.CheckConstraint("operation IN ('INSERT','UPDATE','DELETE')", name="ck_cash_observation_operation"),
        sa.CheckConstraint("expires_at > observed_at", name="ck_cash_observation_expiry"),
        sa.CheckConstraint("content_hash ~ '^[0-9a-f]{64}$'", name="ck_cash_observation_hash"),
        sa.CheckConstraint(
            "(operation='INSERT' AND before_payload IS NULL AND after_payload IS NOT NULL "
            "AND jsonb_typeof(after_payload)='object') OR "
            "(operation='UPDATE' AND before_payload IS NOT NULL AND after_payload IS NOT NULL "
            "AND jsonb_typeof(before_payload)='object' AND jsonb_typeof(after_payload)='object') OR "
            "(operation='DELETE' AND before_payload IS NOT NULL "
            "AND jsonb_typeof(before_payload)='object' AND after_payload IS NULL)",
            name="ck_cash_observation_payloads",
        ),
        comment="现金三试点的源值变化观察；只证明本地写入时刻，不代表历史首次公开或完整事件清单",
    )
    op.create_index(
        "ix_cash_observation_lookup",
        "cash_source_observation",
        ["source_id", "fund_code", "kind", "event_date", "observation_id"],
        unique=False,
    )
    op.create_index(
        "ix_cash_observation_expiry", "cash_source_observation", ["expires_at", "observation_id"], unique=False
    )
    # 白名单投影：没有经理姓名、账号、Token、原始API响应或其他预测未使用的信息。
    op.execute("""CREATE FUNCTION cash_observation_payload(row_data jsonb, data_kind text)
        RETURNS jsonb LANGUAGE sql IMMUTABLE STRICT AS $$
        SELECT CASE WHEN data_kind='NAV' THEN jsonb_build_object(
            'fund_code', row_data->>'fund_code', 'source_id', row_data->>'source_id',
            'nav_date', row_data->>'nav_date', 'ann_date', row_data->>'ann_date', 'unit_nav', row_data->>'unit_nav')
        ELSE jsonb_build_object(
            'fund_code', row_data->>'fund_code', 'source_id', row_data->>'source_id',
            'source_event_key', row_data->>'source_event_key', 'ann_date', row_data->>'ann_date',
            'implementation_ann_date', row_data->>'implementation_ann_date',
            'ex_date', row_data->>'ex_date', 'nav_ex_date', row_data->>'nav_ex_date',
            'cash_dividend', row_data->>'cash_dividend', 'base_unit', row_data->>'base_unit',
            'process_status', row_data->>'process_status') END; $$""")
    op.execute("""CREATE FUNCTION capture_cash_source_observation() RETURNS trigger LANGUAGE plpgsql AS $$
        DECLARE prior jsonb; following jsonb; identity_row jsonb; prior_identity jsonb;
            data_kind text := TG_ARGV[0]; source_uuid uuid; fund text; item_key text;
            business_date date; captured timestamptz; keep_days integer;
        BEGIN
            IF TG_OP <> 'INSERT' THEN prior := cash_observation_payload(to_jsonb(OLD), data_kind); END IF;
            IF TG_OP <> 'DELETE' THEN following := cash_observation_payload(to_jsonb(NEW), data_kind); END IF;
            identity_row := COALESCE(following, prior);
            fund := identity_row->>'fund_code';
            IF fund NOT IN ('001632','006730','008888') AND
                COALESCE(prior->>'fund_code','') NOT IN ('001632','006730','008888') THEN RETURN NULL; END IF;
            IF prior IS NOT NULL AND following IS NOT NULL THEN
                prior_identity := jsonb_build_array(prior->>'fund_code', prior->>'source_id',
                    COALESCE(prior->>'nav_date',prior->>'source_event_key'));
                IF prior_identity <> jsonb_build_array(fund,following->>'source_id',
                    COALESCE(following->>'nav_date',following->>'source_event_key')) THEN
                    RAISE EXCEPTION 'cash source identity cannot be reassigned' USING ERRCODE='55000';
                END IF;
                IF prior = following THEN RETURN NULL; END IF;
            END IF;
            source_uuid := (identity_row->>'source_id')::uuid;
            SELECT retention_days INTO keep_days FROM source_registry
                WHERE source_id=source_uuid AND source_code='TUSHARE_PRO_FUND';
            IF keep_days IS NULL OR keep_days <= 0 THEN RETURN NULL; END IF;
            item_key := COALESCE(identity_row->>'nav_date',identity_row->>'source_event_key');
            business_date := COALESCE(identity_row->>'nav_date',identity_row->>'nav_ex_date',
                identity_row->>'ex_date',identity_row->>'ann_date')::date;
            captured := clock_timestamp();
            INSERT INTO cash_source_observation(source_id,fund_code,kind,record_key,event_date,operation,
                observed_at,expires_at,before_payload,after_payload,content_hash)
            VALUES(source_uuid,fund,data_kind,item_key,business_date,TG_OP,captured,
                captured + keep_days * interval '1 day',prior,following,
                encode(sha256(convert_to(jsonb_build_object('kind',data_kind,'before',prior,'after',following)::text,'UTF8')),'hex'));
            RETURN NULL;
        END; $$""")
    op.execute("""CREATE FUNCTION protect_cash_source_observation() RETURNS trigger LANGUAGE plpgsql AS $$
        BEGIN
            IF TG_OP='DELETE' AND OLD.expires_at <= clock_timestamp() THEN RETURN OLD; END IF;
            RAISE EXCEPTION 'cash observations cannot be rewritten or removed before expiry' USING ERRCODE='55000';
        END; $$""")
    op.execute(
        "CREATE TRIGGER cash_observation_no_rewrite BEFORE UPDATE OR DELETE ON cash_source_observation "
        "FOR EACH ROW EXECUTE FUNCTION protect_cash_source_observation()"
    )
    op.execute(
        "CREATE TRIGGER cash_observation_no_truncate BEFORE TRUNCATE ON cash_source_observation "
        "FOR EACH STATEMENT EXECUTE FUNCTION protect_cash_source_observation()"
    )
    op.execute(
        "CREATE TRIGGER cash_nav_observation AFTER INSERT OR UPDATE OR DELETE ON nav_daily "
        "FOR EACH ROW EXECUTE FUNCTION capture_cash_source_observation('NAV')"
    )
    op.execute(
        "CREATE TRIGGER cash_dividend_observation AFTER INSERT OR UPDATE OR DELETE ON fund_dividend "
        "FOR EACH ROW EXECUTE FUNCTION capture_cash_source_observation('DIVIDEND')"
    )


def downgrade():
    op.execute("LOCK TABLE cash_source_observation IN ACCESS EXCLUSIVE MODE")
    op.execute("""DO $$ BEGIN IF EXISTS (SELECT 1 FROM cash_source_observation)
        THEN RAISE EXCEPTION 'cash source observations are not empty; downgrade refused'; END IF; END $$""")
    op.execute("DROP TRIGGER cash_nav_observation ON nav_daily")
    op.execute("DROP TRIGGER cash_dividend_observation ON fund_dividend")
    op.drop_table("cash_source_observation")
    op.execute("DROP FUNCTION protect_cash_source_observation()")
    op.execute("DROP FUNCTION capture_cash_source_observation()")
    op.execute("DROP FUNCTION cash_observation_payload(jsonb,text)")
