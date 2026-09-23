"""实时与人工研究冻结来源；仅添加证据表，不改写旧预测或历史任务。"""

from alembic import op

revision = "20260923_28"
down_revision = "20260923_27"
branch_labels = None
depends_on = None


def upgrade():
    op.execute("""
    CREATE TABLE prediction_task_input (
      generation_task_id uuid REFERENCES prediction_generation_task,
      research_run_id uuid REFERENCES prediction_research_run,
      fund_code varchar(6) NOT NULL,
      content_hash char(64) NOT NULL, payload jsonb NOT NULL,
      created_at timestamptz NOT NULL DEFAULT clock_timestamp(),
      CHECK(num_nonnulls(generation_task_id,research_run_id)=1),
      UNIQUE NULLS NOT DISTINCT(generation_task_id,research_run_id,fund_code));
    COMMENT ON TABLE prediction_task_input IS '实时及人工研究冻结来源和缺口；恢复不重读，新任务读取新来源';
    CREATE TRIGGER immutable_prediction_task_input BEFORE UPDATE OR DELETE ON prediction_task_input
      FOR EACH ROW EXECUTE FUNCTION prediction_reject_evidence_mutation();
    """)


def downgrade():
    raise RuntimeError("应用可回退，冻结来源证据保留，不删除历史记录")
