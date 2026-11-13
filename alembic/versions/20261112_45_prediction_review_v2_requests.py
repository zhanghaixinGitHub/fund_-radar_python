"""第二版新窗口审阅请求；不将等待状态当成已生成方向。"""

from alembic import op

revision = "20261112_45"
down_revision = "20261112_44"
branch_labels = None
depends_on = None


def upgrade():
    op.execute("""CREATE TABLE prediction_review_v2_request (
        task_id uuid PRIMARY KEY,
        idempotency_key text NOT NULL UNIQUE,
        created_at timestamptz NOT NULL DEFAULT clock_timestamp(),
        payload jsonb NOT NULL
    )""")


def downgrade():
    raise RuntimeError("审阅请求保留，不自动删除历史记录")
