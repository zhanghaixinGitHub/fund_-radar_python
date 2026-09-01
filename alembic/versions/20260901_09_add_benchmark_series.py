"""新增 M3 候选模型回测的已授权业绩基准存储。

Revision ID: 20260901_09
Revises: 20260901_08
Create Date: 2026-09-01 20:30:00
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "20260901_09"
down_revision: str | None = "20260901_08"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """创建基准元数据和日序列；不预置或虚构任何外部基准数据。"""
    op.create_table(
        "benchmark_series",
        sa.Column("benchmark_code", sa.String(length=64), nullable=False),
        sa.Column("display_name", sa.String(length=128), nullable=False),
        sa.Column("fund_type", sa.String(length=32), nullable=False),
        sa.Column("source_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("status", sa.String(length=16), server_default=sa.text("'DRAFT'"), nullable=False),
        sa.Column("license_reference", sa.String(length=512), nullable=False),
        sa.Column("row_hash", sa.String(length=64), nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False
        ),
        sa.CheckConstraint("status IN ('DRAFT', 'ACTIVE', 'SUSPENDED')", name="ck_benchmark_series_status"),
        sa.ForeignKeyConstraint(["source_id"], ["source_registry.source_id"], name="fk_benchmark_series_source"),
        sa.PrimaryKeyConstraint("benchmark_code", name="pk_benchmark_series"),
    )
    op.create_index(
        "ix_benchmark_series_fund_type_status", "benchmark_series", ["fund_type", "status"]
    )
    op.create_index("ix_benchmark_series_source_status", "benchmark_series", ["source_id", "status"])
    op.create_table(
        "benchmark_nav_daily",
        sa.Column("benchmark_code", sa.String(length=64), nullable=False),
        sa.Column("nav_date", sa.Date(), nullable=False),
        sa.Column("closing_value", sa.Numeric(precision=20, scale=8), nullable=False),
        sa.Column("source_published_at", sa.DateTime(timezone=True)),
        sa.Column("row_hash", sa.String(length=64), nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False
        ),
        sa.CheckConstraint("closing_value > 0", name="ck_benchmark_nav_daily_closing_value_positive"),
        sa.ForeignKeyConstraint(
            ["benchmark_code"], ["benchmark_series.benchmark_code"], name="fk_benchmark_nav_daily_series"
        ),
        sa.PrimaryKeyConstraint("benchmark_code", "nav_date", name="pk_benchmark_nav_daily"),
    )
    op.create_index(
        "ix_benchmark_nav_daily_benchmark_date", "benchmark_nav_daily", ["benchmark_code", "nav_date"]
    )


def downgrade() -> None:
    """按依赖逆序删除基准表；执行前需确认不再保留回测证据。"""
    op.drop_index("ix_benchmark_nav_daily_benchmark_date", table_name="benchmark_nav_daily")
    op.drop_table("benchmark_nav_daily")
    op.drop_index("ix_benchmark_series_source_status", table_name="benchmark_series")
    op.drop_index("ix_benchmark_series_fund_type_status", table_name="benchmark_series")
    op.drop_table("benchmark_series")
