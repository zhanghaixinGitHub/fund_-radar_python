"""新增基金基础资料、经理、规模、分红快照，并扩展净值字段。

Revision ID: 20260828_06
Revises: 20260828_05
Create Date: 2026-08-28 15:00:00
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "20260828_06"
down_revision: str | None = "20260828_05"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """以追加方式建立关注后完整详情的可追溯资料表。"""
    op.create_table(
        "fund_profile",
        sa.Column("fund_profile_id", postgresql.UUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column("fund_code", sa.String(length=32), nullable=False),
        sa.Column("source_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("management_company_name", sa.String(length=256)),
        sa.Column("custodian_name", sa.String(length=256)),
        sa.Column("found_date", sa.Date()),
        sa.Column("due_date", sa.Date()),
        sa.Column("list_date", sa.Date()),
        sa.Column("issue_date", sa.Date()),
        sa.Column("delist_date", sa.Date()),
        sa.Column("issue_amount", sa.Numeric(20, 8)),
        sa.Column("management_fee", sa.Numeric(12, 8)),
        sa.Column("custodian_fee", sa.Numeric(12, 8)),
        sa.Column("duration_year", sa.Numeric(12, 4)),
        sa.Column("par_value", sa.Numeric(20, 8)),
        sa.Column("min_purchase_amount", sa.Numeric(20, 8)),
        sa.Column("expected_return", sa.Numeric(20, 8)),
        sa.Column("benchmark", sa.String(length=512)),
        sa.Column("invest_type", sa.String(length=128)),
        sa.Column("source_fund_type", sa.String(length=128)),
        sa.Column("trustee_name", sa.String(length=256)),
        sa.Column("purchase_start_date", sa.Date()),
        sa.Column("redemption_start_date", sa.Date()),
        sa.Column("market", sa.String(length=8)),
        sa.Column("content_hash", sa.String(length=64), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.ForeignKeyConstraint(["fund_code"], ["fund_share_class.fund_code"]),
        sa.ForeignKeyConstraint(["source_id"], ["source_registry.source_id"]),
        sa.UniqueConstraint("fund_code", "source_id", name="uq_fund_profile_source_fund"),
    )
    op.create_index("ix_fund_profile_source_fund", "fund_profile", ["source_id", "fund_code"])

    op.add_column("nav_daily", sa.Column("ann_date", sa.Date()))
    op.add_column("nav_daily", sa.Column("accumulated_dividend", sa.Numeric(20, 8)))
    op.add_column("nav_daily", sa.Column("net_asset", sa.Numeric(24, 4)))
    op.add_column("nav_daily", sa.Column("total_net_asset", sa.Numeric(24, 4)))
    op.add_column("nav_daily", sa.Column("adjusted_nav", sa.Numeric(20, 8)))

    op.create_table(
        "fund_manager_assignment",
        sa.Column("fund_manager_assignment_id", postgresql.UUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column("fund_code", sa.String(length=32), nullable=False),
        sa.Column("source_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("source_record_key", sa.String(length=64), nullable=False),
        sa.Column("manager_name", sa.String(length=128), nullable=False),
        sa.Column("ann_date", sa.Date()),
        sa.Column("begin_date", sa.Date()),
        sa.Column("end_date", sa.Date()),
        sa.Column("education", sa.String(length=128)),
        sa.Column("content_hash", sa.String(length=64), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.ForeignKeyConstraint(["fund_code"], ["fund_share_class.fund_code"]),
        sa.ForeignKeyConstraint(["source_id"], ["source_registry.source_id"]),
        sa.UniqueConstraint("fund_code", "source_id", "source_record_key", name="uq_fund_manager_assignment_source"),
    )
    op.create_index("ix_fund_manager_assignment_fund_date", "fund_manager_assignment", ["fund_code", "ann_date", "begin_date"])

    op.create_table(
        "fund_share_snapshot",
        sa.Column("fund_code", sa.String(length=32), primary_key=True, nullable=False),
        sa.Column("trade_date", sa.Date(), primary_key=True, nullable=False),
        sa.Column("source_id", postgresql.UUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column("fund_share", sa.Numeric(24, 4), nullable=False),
        sa.Column("content_hash", sa.String(length=64), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.CheckConstraint("fund_share >= 0", name="ck_fund_share_snapshot_nonnegative"),
        sa.ForeignKeyConstraint(["fund_code"], ["fund_share_class.fund_code"]),
        sa.ForeignKeyConstraint(["source_id"], ["source_registry.source_id"]),
    )
    op.create_index("ix_fund_share_snapshot_fund_date", "fund_share_snapshot", ["fund_code", "trade_date"])

    op.create_table(
        "fund_dividend",
        sa.Column("fund_code", sa.String(length=32), primary_key=True, nullable=False),
        sa.Column("source_id", postgresql.UUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column("source_event_key", sa.String(length=64), primary_key=True, nullable=False),
        sa.Column("ann_date", sa.Date()),
        sa.Column("implementation_ann_date", sa.Date()),
        sa.Column("base_date", sa.Date()),
        sa.Column("process_status", sa.String(length=64)),
        sa.Column("record_date", sa.Date()),
        sa.Column("ex_date", sa.Date()),
        sa.Column("pay_date", sa.Date()),
        sa.Column("earnings_pay_date", sa.Date()),
        sa.Column("nav_ex_date", sa.Date()),
        sa.Column("cash_dividend", sa.Numeric(20, 8)),
        sa.Column("base_unit", sa.Numeric(24, 4)),
        sa.Column("distributable_earnings", sa.Numeric(24, 4)),
        sa.Column("earnings_amount", sa.Numeric(24, 4)),
        sa.Column("reinvestment_arrival_date", sa.Date()),
        sa.Column("base_year", sa.String(length=16)),
        sa.Column("content_hash", sa.String(length=64), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.ForeignKeyConstraint(["fund_code"], ["fund_share_class.fund_code"]),
        sa.ForeignKeyConstraint(["source_id"], ["source_registry.source_id"]),
    )
    op.create_index("ix_fund_dividend_fund_ann_date", "fund_dividend", ["fund_code", "ann_date"])


def downgrade() -> None:
    """删除新增资料结构；执行前必须先确认其中历史数据已完成归档。"""
    op.drop_index("ix_fund_dividend_fund_ann_date", table_name="fund_dividend")
    op.drop_table("fund_dividend")
    op.drop_index("ix_fund_share_snapshot_fund_date", table_name="fund_share_snapshot")
    op.drop_table("fund_share_snapshot")
    op.drop_index("ix_fund_manager_assignment_fund_date", table_name="fund_manager_assignment")
    op.drop_table("fund_manager_assignment")
    op.drop_column("nav_daily", "adjusted_nav")
    op.drop_column("nav_daily", "total_net_asset")
    op.drop_column("nav_daily", "net_asset")
    op.drop_column("nav_daily", "accumulated_dividend")
    op.drop_column("nav_daily", "ann_date")
    op.drop_index("ix_fund_profile_source_fund", table_name="fund_profile")
    op.drop_table("fund_profile")
