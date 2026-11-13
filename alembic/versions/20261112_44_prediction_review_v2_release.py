"""追加保存正式采用的第二版分析；不覆盖旧预测和试验。"""

import sqlalchemy as sa
from alembic import op

revision = "20261112_44"
down_revision = "20261102_43"
branch_labels = None
depends_on = None


def upgrade():
    op.execute(
        sa.text("""CREATE TABLE prediction_review_v2_release (
      release_id uuid PRIMARY KEY,
      version varchar(64) NOT NULL,
      policy_hash char(64) NOT NULL,
      content_hash char(64) NOT NULL UNIQUE,
      adopted_at timestamptz NOT NULL DEFAULT clock_timestamp(),
      archive jsonb NOT NULL CHECK (jsonb_typeof(archive)='object'),
      views jsonb NOT NULL CHECK (jsonb_typeof(views)='object'),
      CHECK (jsonb_typeof(archive->'cases')='array' AND jsonb_array_length(archive->'cases')>0)
    )""")
    )
    op.execute(
        sa.text(
            "CREATE INDEX ix_prediction_review_v2_release_adopted "
            "ON prediction_review_v2_release(adopted_at DESC,release_id DESC)"
        )
    )
    op.execute(
        sa.text("""CREATE FUNCTION reject_prediction_review_v2_release_mutation() RETURNS trigger
      LANGUAGE plpgsql AS $$ BEGIN RAISE EXCEPTION 'prediction review v2 release is append only'; END $$""")
    )
    op.execute(
        sa.text("""CREATE TRIGGER prediction_review_v2_release_append_only
      BEFORE UPDATE OR DELETE ON prediction_review_v2_release
      FOR EACH ROW EXECUTE FUNCTION reject_prediction_review_v2_release_mutation()""")
    )


def downgrade():
    if op.get_bind().scalar(sa.text("SELECT EXISTS(SELECT 1 FROM prediction_review_v2_release)")):
        raise RuntimeError("已采用的第二版分析必须保留，禁止删除")
    op.drop_table("prediction_review_v2_release")
    op.execute(sa.text("DROP FUNCTION reject_prediction_review_v2_release_mutation()"))
