"""多周期不可变证据、模型路由和可恢复任务；不改旧一日/二十日记录。"""

from alembic import op

revision = "20260922_24"
down_revision = "20260912_23"
branch_labels = None
depends_on = None


def upgrade():
    op.execute("""
    CREATE TABLE prediction_horizon (
      horizon_id varchar(40) PRIMARY KEY, content_hash char(64) NOT NULL, payload jsonb NOT NULL,
      created_at timestamptz NOT NULL DEFAULT clock_timestamp());
    CREATE TABLE model_artifact (
      model_id varchar(100) PRIMARY KEY, content_hash char(64) NOT NULL,
      horizon_id varchar(40) NOT NULL, target_definition_id varchar(100) NOT NULL,
      asset_group varchar(60) NOT NULL, adapter varchar(60) NOT NULL,
      manifest jsonb NOT NULL, created_at timestamptz NOT NULL DEFAULT clock_timestamp());
    CREATE TABLE model_evaluation (
      evaluation_id uuid PRIMARY KEY, experiment_id varchar(100) NOT NULL,
      content_hash char(64) NOT NULL UNIQUE, payload jsonb NOT NULL,
      created_at timestamptz NOT NULL DEFAULT clock_timestamp());
    CREATE TABLE model_route (
      route_key varchar(200) PRIMARY KEY, model_id varchar(100) NOT NULL REFERENCES model_artifact,
      previous_model_id varchar(100) REFERENCES model_artifact, revision bigint NOT NULL CHECK(revision>0),
      shadow_ids jsonb NOT NULL DEFAULT '[]', updated_at timestamptz NOT NULL DEFAULT clock_timestamp());
    CREATE TABLE model_activation_event (
      event_id uuid PRIMARY KEY, route_key varchar(200) NOT NULL, revision bigint NOT NULL,
      previous_model_id varchar(100), model_id varchar(100), action varchar(40) NOT NULL,
      reason jsonb NOT NULL, created_at timestamptz NOT NULL DEFAULT clock_timestamp());
    CREATE INDEX ix_model_activation_route ON model_activation_event(route_key,created_at);
    CREATE TABLE prediction_fact_version (
      fact_id uuid PRIMARY KEY, fund_code varchar(32) NOT NULL, kind varchar(40) NOT NULL,
      source_key varchar(200) NOT NULL, payload_hash char(64) NOT NULL,
      effective_at timestamptz NOT NULL, published_at timestamptz,
      first_observed_at timestamptz NOT NULL, quality varchar(40) NOT NULL, payload jsonb NOT NULL,
      UNIQUE(fund_code,kind,source_key,payload_hash));
    CREATE INDEX ix_prediction_fact_asof ON prediction_fact_version(fund_code,kind,effective_at,first_observed_at);
    CREATE TABLE prediction_feature_snapshot (
      snapshot_id uuid PRIMARY KEY, fund_code varchar(32) NOT NULL, knowledge_cutoff timestamptz NOT NULL,
      content_hash char(64) NOT NULL UNIQUE, payload jsonb NOT NULL,
      created_at timestamptz NOT NULL DEFAULT clock_timestamp());
    CREATE TABLE prediction_generation_task (
      task_id uuid PRIMARY KEY, request_key varchar(200) NOT NULL UNIQUE,
      mode varchar(40) NOT NULL CHECK(mode IN ('LIVE','HISTORICAL_REPLAY','RECOMPUTED_AFTER_EVENT')),
      status varchar(30) NOT NULL, payload jsonb NOT NULL, result jsonb,
      lease_until timestamptz, created_at timestamptz NOT NULL DEFAULT clock_timestamp(), finished_at timestamptz);
    CREATE TABLE prediction_generation_item (
      task_id uuid NOT NULL REFERENCES prediction_generation_task, fund_code varchar(32) NOT NULL,
      horizon_id varchar(40) NOT NULL, status varchar(30) NOT NULL DEFAULT 'PENDING',
      prediction_id uuid, attempts integer NOT NULL DEFAULT 0, result jsonb,
      PRIMARY KEY(task_id,fund_code,horizon_id));
    CREATE TABLE prediction_attempt (
      attempt_id uuid PRIMARY KEY, task_id uuid NOT NULL, fund_code varchar(32) NOT NULL,
      horizon_id varchar(40) NOT NULL, payload jsonb NOT NULL,
      created_at timestamptz NOT NULL DEFAULT clock_timestamp());
    CREATE TABLE fund_prediction_record (
      prediction_id uuid PRIMARY KEY, period_key varchar(240) NOT NULL UNIQUE,
      fund_code varchar(32) NOT NULL, horizon_id varchar(40) NOT NULL,
      mode varchar(40) NOT NULL, model_id varchar(100) NOT NULL REFERENCES model_artifact,
      activation_revision bigint NOT NULL, generated_at timestamptz NOT NULL,
      content_hash char(64) NOT NULL, payload jsonb NOT NULL,
      stored_at timestamptz NOT NULL DEFAULT clock_timestamp());
    CREATE INDEX ix_prediction_fund_time ON fund_prediction_record(fund_code,generated_at DESC);
    CREATE TABLE prediction_outcome (
      outcome_id uuid PRIMARY KEY, prediction_id uuid NOT NULL REFERENCES fund_prediction_record,
      content_hash char(64) NOT NULL, payload jsonb NOT NULL,
      checked_at timestamptz NOT NULL DEFAULT clock_timestamp(), UNIQUE(prediction_id,content_hash));
    CREATE TABLE prediction_research_run (
      run_id uuid PRIMARY KEY, spec_hash char(64) NOT NULL, spec jsonb NOT NULL,
      status varchar(30) NOT NULL, checkpoint jsonb NOT NULL DEFAULT '{}', result jsonb,
      cancel_requested boolean NOT NULL DEFAULT false,
      created_at timestamptz NOT NULL DEFAULT clock_timestamp(),
      updated_at timestamptz NOT NULL DEFAULT clock_timestamp());
    CREATE TABLE prediction_evaluation_usage (
      usage_id uuid PRIMARY KEY, sample_hash char(64) NOT NULL, purpose varchar(40) NOT NULL,
      run_id uuid NOT NULL REFERENCES prediction_research_run, start_date date NOT NULL, end_date date NOT NULL,
      created_at timestamptz NOT NULL DEFAULT clock_timestamp());
    COMMENT ON TABLE prediction_fact_version IS '公共事实版本：实收时间与历史公开重建分开，禁止伪造过去接收时间';
    COMMENT ON TABLE fund_prediction_record IS '不可变多周期预测；原LIVE、历史回放、换模重算独立保存';
    COMMENT ON TABLE model_route IS '模型路由当前指针；批次冻结revision，切换采用事务CAS';
    CREATE FUNCTION prediction_reject_evidence_mutation() RETURNS trigger LANGUAGE plpgsql AS $$
    BEGIN RAISE EXCEPTION 'prediction evidence is append only'; END $$;
    """)
    for table in (
        "prediction_horizon",
        "model_artifact",
        "model_evaluation",
        "model_activation_event",
        "prediction_fact_version",
        "prediction_feature_snapshot",
        "prediction_attempt",
        "fund_prediction_record",
        "prediction_outcome",
        "prediction_evaluation_usage",
    ):
        op.execute(
            f"CREATE TRIGGER immutable_evidence BEFORE UPDATE OR DELETE ON {table} "
            "FOR EACH ROW EXECUTE FUNCTION prediction_reject_evidence_mutation()"
        )


def downgrade():
    raise RuntimeError("回退服务或模型指针，保留已生成预测和模型身份，不删除实验档案。")
