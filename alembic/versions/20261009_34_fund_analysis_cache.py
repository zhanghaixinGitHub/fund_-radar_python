"""综合资料解析缓存；沿用公共作业，成功正文只追加不覆盖。"""

from alembic import op

revision = "20261009_34"
down_revision = "20260930_33"
branch_labels = None
depends_on = None


def upgrade():
    op.execute("""
    CREATE TABLE fund_analysis_event_parse (
      cache_key char(64) PRIMARY KEY, stage varchar(16) NOT NULL,
      state varchar(16) NOT NULL CHECK(state IN ('RUNNING','READY','FAILED')),
      owner_token uuid NOT NULL, lease_until timestamptz NOT NULL,
      attempts integer NOT NULL CHECK(attempts BETWEEN 1 AND 2),
      payload jsonb, content_hash char(64), failure_code varchar(64),
      created_at timestamptz NOT NULL DEFAULT clock_timestamp(), finished_at timestamptz,
      CHECK(state<>'READY' OR (payload IS NOT NULL AND content_hash IS NOT NULL))
    );
    CREATE INDEX ix_fund_analysis_parse_lease ON fund_analysis_event_parse(state,lease_until);
    COMMENT ON TABLE fund_analysis_event_parse IS '公共资料解析和综合调用缓存，不含用户身份；READY正文不可覆盖';
    COMMENT ON COLUMN fund_analysis_event_parse.cache_key IS '原件及实际阅读范围、上下文、提示词和模型配置的联合摘要';
    COMMENT ON COLUMN fund_analysis_event_parse.stage IS 'EVENT解析、SYNTHESIS综合或AUDIT核对';
    COMMENT ON COLUMN fund_analysis_event_parse.state IS '有限运行状态，失败不冒充已解析';
    COMMENT ON COLUMN fund_analysis_event_parse.owner_token IS '本次工作者令牌，过期工作者不能提交';
    COMMENT ON COLUMN fund_analysis_event_parse.lease_until IS '当前调用的有界租约截止时间';
    COMMENT ON COLUMN fund_analysis_event_parse.attempts IS '同一输入最多两次调用，超时不自动重试';
    COMMENT ON COLUMN fund_analysis_event_parse.payload IS '原始JSON响应，只含公开资料';
    COMMENT ON COLUMN fund_analysis_event_parse.content_hash IS '响应规范化JSON摘要';
    COMMENT ON COLUMN fund_analysis_event_parse.failure_code IS '脱敏失败原因，不含供应商原始错误';
    COMMENT ON COLUMN fund_analysis_event_parse.created_at IS '本地首次创建时间';
    COMMENT ON COLUMN fund_analysis_event_parse.finished_at IS '本次实际完成时间';
    CREATE FUNCTION fund_analysis_cache_guard() RETURNS trigger LANGUAGE plpgsql AS $$
    BEGIN IF OLD.state='READY' THEN RAISE EXCEPTION 'immutable analysis cache'; END IF; RETURN NEW; END $$;
    CREATE TRIGGER tg_fund_analysis_cache_guard BEFORE UPDATE OR DELETE ON fund_analysis_event_parse
      FOR EACH ROW EXECUTE FUNCTION fund_analysis_cache_guard();
    """)


def downgrade():
    raise RuntimeError("保留已生成依据；回退只关闭分析路由，不删除缓存")
