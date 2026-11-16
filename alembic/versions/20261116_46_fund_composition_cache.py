"""公共持仓构成的版本化结果；保留来源版本，不保存个人金额。"""

from alembic import op

revision = "20261116_46"
down_revision = "20261112_45"
branch_labels = None
depends_on = None


def upgrade():
    op.execute("""CREATE TABLE fund_composition_cache (
        cache_key text PRIMARY KEY,
        payload jsonb NOT NULL,
        created_at timestamptz NOT NULL DEFAULT clock_timestamp()
    )""")


def downgrade():
    raise RuntimeError("已保存构成结果保留，不自动删除")
