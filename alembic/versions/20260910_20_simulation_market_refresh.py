"""按基金保存模拟账本公共行情刷新状态，避免用全市场成功状态代替单基金证据。"""

import sqlalchemy as sa
from alembic import op

revision = "20260910_20"
down_revision = "20260909_19"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "simulation_market_refresh",
        sa.Column(
            "fund_code",
            sa.String(32),
            sa.ForeignKey("fund_share_class.fund_code"),
            primary_key=True,
            comment="已登记基金份额代码，不包含用户身份",
        ),
        sa.Column("status", sa.String(16), nullable=False, comment="公共资料刷新状态"),
        sa.Column("attempted_at", sa.DateTime(timezone=True), nullable=False, comment="最近尝试时间及限频水位"),
        sa.Column("dividends_verified_at", sa.DateTime(timezone=True), comment="该基金净值和分红均刷新成功的时间"),
        sa.Column("message", sa.String(256), comment="不含凭据的刷新状态说明"),
        sa.CheckConstraint("status IN ('RUNNING','SUCCEEDED','FAILED')", name="ck_simulation_market_refresh_status"),
        comment="模拟持仓所需公共行情的单基金刷新水位，不保存个人交易",
    )


def downgrade() -> None:
    op.drop_table("simulation_market_refresh")
