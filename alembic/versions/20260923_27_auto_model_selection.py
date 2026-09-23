"""后台自动周期和完整发布；兼容旧记录，禁止删除原始证据。"""

from alembic import op

revision = "20260923_27"
down_revision = "20260922_26"
branch_labels = None
depends_on = None


def upgrade():
    op.execute("""
    CREATE TABLE prediction_auto_cycle (
      cycle_id uuid PRIMARY KEY, request_key char(64) NOT NULL UNIQUE,
      scope_hash char(64) NOT NULL, data_hash char(64) NOT NULL, protocol_hash char(64) NOT NULL,
      status varchar(32) NOT NULL CHECK(status IN ('WAITING_DATA','QUEUED','TRAINING','EVALUATING',
        'REPLAYING','DECIDING','VERIFYING_ADOPTION','COMPLETED','FAILED','CANCELLED','INTERRUPTED')),
      trigger_reason varchar(40) NOT NULL, spec jsonb NOT NULL, checkpoint jsonb NOT NULL DEFAULT '{}',
      result jsonb, error jsonb, lease_owner uuid, lease_until timestamptz,
      heartbeat_at timestamptz, retries integer NOT NULL DEFAULT 0,
      next_attempt_at timestamptz, cancel_requested boolean NOT NULL DEFAULT false,
      created_at timestamptz NOT NULL DEFAULT clock_timestamp(),
      updated_at timestamptz NOT NULL DEFAULT clock_timestamp(), finished_at timestamptz);
    CREATE UNIQUE INDEX ix_auto_active_scope ON prediction_auto_cycle(scope_hash)
      WHERE status IN ('QUEUED','TRAINING','EVALUATING','REPLAYING','DECIDING','INTERRUPTED');
    CREATE INDEX ix_auto_pending ON prediction_auto_cycle(status,next_attempt_at,created_at);
    CREATE TABLE prediction_auto_input (
      cycle_id uuid NOT NULL REFERENCES prediction_auto_cycle, input_key varchar(100) NOT NULL,
      content_hash char(64) NOT NULL, payload jsonb NOT NULL,
      created_at timestamptz NOT NULL DEFAULT clock_timestamp(), PRIMARY KEY(cycle_id,input_key));
    CREATE TABLE prediction_auto_event (
      event_id uuid PRIMARY KEY, cycle_id uuid NOT NULL REFERENCES prediction_auto_cycle,
      kind varchar(40) NOT NULL, payload jsonb NOT NULL,
      created_at timestamptz NOT NULL DEFAULT clock_timestamp());
    CREATE INDEX ix_auto_event_cycle ON prediction_auto_event(cycle_id,created_at);
    CREATE TABLE prediction_model_release (
      release_id uuid PRIMARY KEY, cycle_id uuid UNIQUE REFERENCES prediction_auto_cycle,
      previous_release_id uuid REFERENCES prediction_model_release, content_hash char(64) NOT NULL,
      manifest jsonb NOT NULL, reason jsonb NOT NULL,
      created_at timestamptz NOT NULL DEFAULT clock_timestamp());
    CREATE TABLE prediction_release_pointer (
      scope varchar(32) PRIMARY KEY, release_id uuid NOT NULL REFERENCES prediction_model_release,
      revision bigint NOT NULL CHECK(revision>0));
    CREATE TABLE prediction_release_receipt (
      release_id uuid NOT NULL REFERENCES prediction_model_release, receipt_hash char(64) NOT NULL,
      kind varchar(32) NOT NULL, payload jsonb NOT NULL,
      created_at timestamptz NOT NULL DEFAULT clock_timestamp(), PRIMARY KEY(release_id,receipt_hash));
    CREATE TABLE prediction_model_quarantine (
      model_id varchar(100) PRIMARY KEY REFERENCES model_artifact, reason jsonb NOT NULL,
      retry_after timestamptz NOT NULL, failure_count integer NOT NULL DEFAULT 1);
    ALTER TABLE model_route ADD COLUMN release_id uuid REFERENCES prediction_model_release;
    ALTER TABLE prediction_research_run ADD COLUMN auto_cycle_id uuid REFERENCES prediction_auto_cycle;
    CREATE UNIQUE INDEX ix_auto_research_shard ON prediction_research_run(auto_cycle_id,spec_hash)
      WHERE auto_cycle_id IS NOT NULL;
    CREATE INDEX ix_primary_prediction_effect ON fund_prediction_record(fund_code,horizon_id,generated_at)
      WHERE mode='LIVE' AND payload->>'role'='PRIMARY';
    CREATE INDEX ix_outcome_latest ON prediction_outcome(prediction_id,checked_at DESC,outcome_id);
    COMMENT ON TABLE prediction_auto_cycle IS '自动研究单一编排：冻结协议、范围、水位及恢复租约；不属于用户交易';
    COMMENT ON TABLE prediction_model_release IS '完整多周期发布清单，CAS原子切换后等待真实调用回执';
    """)
    for table in (
        "prediction_auto_input",
        "prediction_auto_event",
        "prediction_model_release",
        "prediction_release_receipt",
    ):
        op.execute(f"""CREATE TRIGGER immutable_{table} BEFORE UPDATE OR DELETE ON {table}
          FOR EACH ROW EXECUTE FUNCTION prediction_reject_evidence_mutation()""")


def downgrade():
    raise RuntimeError("回退应用及发布指针，保留全部周期/发布/原预测证据；本迁移为添加式兼容迁移")
