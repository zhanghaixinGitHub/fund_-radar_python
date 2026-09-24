"""逐基金净值缺口、失败及自动重试状态；不修改既有净值或预测历史。"""

from alembic import op

revision = "20260924_29"
down_revision = "20260923_28"
branch_labels = None
depends_on = None


def upgrade():
    op.execute("""
      CREATE TABLE nav_sync_state (
        source_id uuid NOT NULL REFERENCES source_registry(source_id),
        fund_code varchar(6) NOT NULL REFERENCES fund_share_class(fund_code),
        sync_run_id uuid NOT NULL REFERENCES source_sync_run(sync_run_id),
        status varchar(32) NOT NULL CHECK(status IN ('SUCCEEDED','STATUS_UNKNOWN','NOT_PUBLISHED','SYNC_FAILED')),
        missing_dates jsonb NOT NULL DEFAULT '[]', reason varchar(512) NOT NULL,
        attempts integer NOT NULL DEFAULT 0 CHECK(attempts>=0),
        next_retry_at timestamptz, checked_at timestamptz NOT NULL DEFAULT clock_timestamp(),
        PRIMARY KEY(source_id,fund_code)
      );
      CREATE INDEX ix_nav_sync_state_retry ON nav_sync_state(next_retry_at) WHERE next_retry_at IS NOT NULL;
      COMMENT ON TABLE nav_sync_state IS '逐基金净值补拉状态；空响应仅记状态未知，失败持续保留到修复';
      COMMENT ON COLUMN nav_sync_state.missing_dates IS '境内估值日历中未取得的日期，含中间缺口';
      COMMENT ON COLUMN nav_sync_state.next_retry_at IS '自动补拉最早重试时刻；空值表示需修正配置后手动重试';
    """)


def downgrade():
    op.drop_table("nav_sync_state")
