"""为原预测保存统一风格的解释；不修改预测、研究或收益结果。"""

from alembic import op

revision = "20260930_32"
down_revision = "20260929_31"
branch_labels = None
depends_on = None


def upgrade():
    """短事务领取任务，租约恢复中断；成功解释只写一次，跨页面共享。"""
    op.execute("""
      CREATE TABLE prediction_narrative (
        source_kind varchar(8) NOT NULL CHECK(source_kind IN ('daily','multi')),
        source_id uuid NOT NULL,
        fund_code varchar(6) NOT NULL CHECK(fund_code ~ '^[0-9]{6}$'),
        source_hash char(64) NOT NULL,
        input_hash char(64) NOT NULL,
        style_version varchar(64) NOT NULL,
        state varchar(8) NOT NULL CHECK(state IN ('RUNNING','READY','FAILED')),
        owner_id uuid NOT NULL,
        lease_until timestamptz NOT NULL,
        attempts integer NOT NULL CHECK(attempts BETWEEN 1 AND 3),
        retry_after timestamptz,
        payload jsonb,
        content_hash char(64),
        provider_model varchar(128) NOT NULL,
        failure_code varchar(64),
        created_at timestamptz NOT NULL DEFAULT clock_timestamp(),
        updated_at timestamptz NOT NULL DEFAULT clock_timestamp(),
        PRIMARY KEY(source_kind, source_id),
        CHECK((state='READY') = (payload IS NOT NULL AND content_hash IS NOT NULL))
      );
      CREATE INDEX ix_prediction_narrative_running ON prediction_narrative(state,lease_until);
      COMMENT ON TABLE prediction_narrative IS '原预测的公共解释快照；无个人账户、持仓或凭据信息';
      COMMENT ON COLUMN prediction_narrative.source_id IS '一日原作业编号或多周期原预测编号';
      COMMENT ON COLUMN prediction_narrative.source_hash IS '原预测内容摘要，防止解释与历史记录错配';
      COMMENT ON COLUMN prediction_narrative.input_hash IS '送入文案生成的已核对事实摘要';
      COMMENT ON COLUMN prediction_narrative.style_version IS '统一行文规范版本；不因升级改写旧解释';
      COMMENT ON COLUMN prediction_narrative.owner_id IS '本次领取的随机租约标识，不是用户编号';
      COMMENT ON COLUMN prediction_narrative.attempts IS '原预测最多三次外部生成尝试，失败间隔至少十五分钟';
      COMMENT ON COLUMN prediction_narrative.payload IS '通过校验后的展示正文与事实引用';
      CREATE FUNCTION prediction_narrative_immutable() RETURNS trigger LANGUAGE plpgsql AS $$
      BEGIN
        IF OLD.state='READY' THEN RAISE EXCEPTION 'saved narrative is immutable'; END IF;
        RETURN NEW;
      END $$;
      CREATE TRIGGER prediction_narrative_guard BEFORE UPDATE OR DELETE ON prediction_narrative
        FOR EACH ROW EXECUTE FUNCTION prediction_narrative_immutable();
    """)


def downgrade():
    raise RuntimeError("保留已经展示过的解释；回退应用即可停用生成，不删除历史。")
