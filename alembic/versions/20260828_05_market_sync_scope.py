"""将基金市场同步范围固化为启用份额，并清理旧任务命名。

Revision ID: 20260828_05
Revises: 20260827_04
Create Date: 2026-08-28 10:00:00
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "20260828_05"
down_revision: str | None = "20260827_04"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """保存来源精确代码，并将历史同步审计切换为市场语义。"""
    op.add_column("fund_share_class", sa.Column("source_fund_code", sa.String(length=16), nullable=True))
    op.create_unique_constraint(
        "uq_fund_share_class_source_fund_code", "fund_share_class", ["source_fund_code"]
    )
    op.create_index(
        "ix_fund_share_class_market_sync", "fund_share_class", ["source_code", "status", "source_fund_code"]
    )
    op.execute(
        """
        UPDATE source_sync_run
        SET sync_type = CASE sync_type
            WHEN 'FOCUSED_CATALOG' THEN 'MARKET_CATALOG'
            WHEN 'FOCUSED_NAV_HISTORY' THEN 'MARKET_NAV_HISTORY'
            WHEN 'FOCUSED_NAV_INCREMENTAL' THEN 'MARKET_NAV_INCREMENTAL'
            ELSE sync_type
        END
        WHERE sync_type IN ('FOCUSED_CATALOG', 'FOCUSED_NAV_HISTORY', 'FOCUSED_NAV_INCREMENTAL')
        """
    )


def downgrade() -> None:
    """恢复旧审计命名并移除来源精确代码；降级会失去已解析映射。"""
    op.execute(
        """
        UPDATE source_sync_run
        SET sync_type = CASE sync_type
            WHEN 'MARKET_CATALOG' THEN 'FOCUSED_CATALOG'
            WHEN 'MARKET_NAV_HISTORY' THEN 'FOCUSED_NAV_HISTORY'
            WHEN 'MARKET_NAV_INCREMENTAL' THEN 'FOCUSED_NAV_INCREMENTAL'
            ELSE sync_type
        END
        WHERE sync_type IN ('MARKET_CATALOG', 'MARKET_NAV_HISTORY', 'MARKET_NAV_INCREMENTAL')
        """
    )
    op.drop_index("ix_fund_share_class_market_sync", table_name="fund_share_class")
    op.drop_constraint("uq_fund_share_class_source_fund_code", "fund_share_class", type_="unique")
    op.drop_column("fund_share_class", "source_fund_code")
