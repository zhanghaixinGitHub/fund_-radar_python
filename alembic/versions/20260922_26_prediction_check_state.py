"""到期核验轮转状态：失败原因与最近检查时刻独立保存，不修改原预测。"""

from alembic import op

revision = "20260922_26"
down_revision = "20260922_25"
branch_labels = None
depends_on = None


def upgrade():
    op.execute("""CREATE TABLE prediction_check_state (
      prediction_id uuid PRIMARY KEY REFERENCES fund_prediction_record(prediction_id),
      status varchar(40) NOT NULL,payload jsonb NOT NULL,
      last_attempt_at timestamptz NOT NULL DEFAULT clock_timestamp())""")
    op.execute("CREATE INDEX ix_prediction_check_rotation ON prediction_check_state(last_attempt_at)")


def downgrade():
    raise RuntimeError("应用回退保留核验水位与失败原因，不自动删除")
