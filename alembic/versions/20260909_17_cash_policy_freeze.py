"""已确认规则快照；仅新增表，非空不降级，禁止原地更改或清空。"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql as pg

revision = "20260909_17"
down_revision = "20260909_16"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "cash_policy_freeze",
        sa.Column("freeze_id", pg.UUID(), primary_key=True, comment="不可变规则快照编号，非模型授权"),
        sa.Column("request_key", pg.UUID(), nullable=False, comment="首次冻结请求的重试凭证"),
        sa.Column("policy_version", sa.String(64), nullable=False, comment="规则版本，每版本仅允许一份内容"),
        sa.Column("policy_hash", sa.String(64), nullable=False, comment="规则正文SHA256，不是审批签名"),
        sa.Column("binding_hash", sa.String(64), nullable=False, comment="固定窗口、口径及比较约定的SHA256"),
        sa.Column("snapshot", pg.JSONB(), nullable=False, comment="当时已确认的完整规则和比较约定，不含模型或答案"),
        sa.Column("content_hash", sa.String(64), nullable=False, comment="身份、时间与完整快照的SHA256"),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
            comment="数据库时钟取得的实际冻结时间",
        ),
        sa.UniqueConstraint("request_key", name="uq_cash_policy_request"),
        sa.UniqueConstraint("policy_version", name="uq_cash_policy_version"),
        sa.CheckConstraint(
            "policy_hash ~ '^[0-9a-f]{64}$' AND binding_hash ~ '^[0-9a-f]{64}$' AND content_hash ~ '^[0-9a-f]{64}$'",
            name="ck_cash_policy_hashes",
        ),
        sa.CheckConstraint(
            "(snapshot->>'version' = 'CASH_POLICY_FREEZE_V1' "
            "AND snapshot#>>'{policy,approval_state}' = 'APPROVED' "
            "AND snapshot#>>'{policy,version}' = policy_version "
            "AND length(btrim(snapshot#>>'{policy,approval_reference}')) > 0) IS TRUE",
            name="ck_cash_policy_approved",
        ),
        comment="已确认的现金发布规则不可变快照；不代表模型获准发布或最终测试已完成",
    )
    op.execute("""CREATE FUNCTION cash_policy_freeze_immutable() RETURNS trigger LANGUAGE plpgsql AS $$
        BEGIN RAISE EXCEPTION 'cash policy freezes are immutable' USING ERRCODE='55000'; END; $$""")
    op.execute(
        "CREATE TRIGGER cash_policy_freeze_no_mutation BEFORE UPDATE OR DELETE OR TRUNCATE "
        "ON cash_policy_freeze FOR EACH STATEMENT EXECUTE FUNCTION cash_policy_freeze_immutable()"
    )


def downgrade():
    op.execute("LOCK TABLE cash_policy_freeze IN ACCESS EXCLUSIVE MODE")
    op.execute("""DO $$ BEGIN IF EXISTS (SELECT 1 FROM cash_policy_freeze)
        THEN RAISE EXCEPTION 'cash policy freezes are not empty; downgrade refused'; END IF; END $$""")
    op.drop_table("cash_policy_freeze")
    op.execute("DROP FUNCTION cash_policy_freeze_immutable()")
