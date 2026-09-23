"""目标日期追加解析、任务中断事件和历史研究租约。"""

from alembic import op

revision = "20260922_25"
down_revision = "20260922_24"
branch_labels = None
depends_on = None


def upgrade():
    op.execute("""CREATE TABLE prediction_target_resolution (
      prediction_id uuid NOT NULL REFERENCES fund_prediction_record(prediction_id),
      resolution_hash char(64) NOT NULL, payload jsonb NOT NULL,
      created_at timestamptz NOT NULL DEFAULT clock_timestamp(), PRIMARY KEY(prediction_id,resolution_hash))""")
    op.execute("""CREATE TABLE prediction_task_event (
      event_id uuid PRIMARY KEY,task_id uuid NOT NULL,kind varchar(40) NOT NULL,payload jsonb NOT NULL,
      created_at timestamptz NOT NULL DEFAULT clock_timestamp())""")
    op.execute("CREATE INDEX ix_prediction_task_event ON prediction_task_event(task_id,created_at)")
    op.execute("ALTER TABLE prediction_research_run ADD COLUMN lease_until timestamptz")
    for table in ("prediction_target_resolution", "prediction_task_event"):
        op.execute(f"""CREATE TRIGGER immutable_{table} BEFORE UPDATE OR DELETE ON {table}
                       FOR EACH ROW EXECUTE FUNCTION prediction_reject_evidence_mutation()""")


def downgrade():
    raise RuntimeError("证据表不得自动删除；回退应用时保留研究及预测历史")
