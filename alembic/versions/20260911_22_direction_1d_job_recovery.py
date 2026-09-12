"""作业租约用于重启恢复；已成功预测原文禁止UPDATE。"""

from alembic import op

revision = "20260911_22"
down_revision = "20260911_21"
branch_labels = None
depends_on = None


def upgrade():
    op.execute("""
      ALTER TABLE direction_1d_job ADD COLUMN lease_until timestamptz;
      COMMENT ON COLUMN direction_1d_job.lease_until IS '队列或进程恢复租约，不修改已成功原文';
      CREATE FUNCTION direction_1d_job_guard() RETURNS trigger LANGUAGE plpgsql AS $$
      BEGIN IF OLD.state='SUCCEEDED' THEN RAISE EXCEPTION 'successful direction_1d job is immutable'; END IF;
      RETURN NEW; END $$;
      CREATE TRIGGER guard_direction_1d_job BEFORE UPDATE ON direction_1d_job
      FOR EACH ROW EXECUTE FUNCTION direction_1d_job_guard();
    """)


def downgrade():
    raise RuntimeError("停用任务但保留成功原文与恢复水位。")
