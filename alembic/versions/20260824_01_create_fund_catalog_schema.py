"""创建 M1 基金目录、已授权数据源与净值数据库结构。

Revision ID: 20260824_01
Revises:
Create Date: 2026-08-24 20:00:00
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "20260824_01"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """创建 AI 服务维护的表和索引，不登记或导入任何外部数据源。"""
    op.create_table(
        "source_registry",
        sa.Column("source_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("source_code", sa.String(length=64), nullable=False),
        sa.Column("display_name", sa.String(length=128), nullable=False),
        sa.Column("source_kind", sa.String(length=32), nullable=False),
        sa.Column("license_scope", sa.Text(), nullable=False),
        sa.Column("rate_limit_per_minute", sa.Integer(), nullable=False),
        sa.Column("retention_days", sa.Integer(), nullable=False),
        sa.Column("enabled", sa.Boolean(), server_default=sa.text("false"), nullable=False),
        sa.Column("last_success_at", sa.DateTime(timezone=True)),
        sa.Column("last_error_at", sa.DateTime(timezone=True)),
        sa.Column("last_error_summary", sa.String(length=512)),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False
        ),
        sa.CheckConstraint("rate_limit_per_minute > 0", name="ck_source_registry_rate_limit_positive"),
        sa.CheckConstraint("retention_days >= 0", name="ck_source_registry_retention_days_nonnegative"),
        sa.PrimaryKeyConstraint("source_id", name="pk_source_registry"),
        sa.UniqueConstraint("source_code", name="uq_source_registry_source_code"),
    )
    op.create_table(
        "fund_master",
        sa.Column("fund_master_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("fund_name", sa.String(length=256), nullable=False),
        sa.Column("manager_name", sa.String(length=256), nullable=False),
        sa.Column("fund_type", sa.String(length=32), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("established_date", sa.Date()),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False
        ),
        sa.PrimaryKeyConstraint("fund_master_id", name="pk_fund_master"),
        sa.UniqueConstraint("manager_name", "fund_name", name="uq_fund_master_manager_name"),
    )
    op.create_table(
        "fund_share_class",
        sa.Column("fund_code", sa.String(length=32), nullable=False),
        sa.Column("fund_master_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("share_class", sa.String(length=64), nullable=False),
        sa.Column("fund_name", sa.String(length=256), nullable=False),
        sa.Column("fund_type", sa.String(length=32), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("benchmark_code", sa.String(length=64)),
        sa.Column("risk_level", sa.String(length=32)),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False
        ),
        sa.ForeignKeyConstraint(["fund_master_id"], ["fund_master.fund_master_id"], name="fk_fund_share_class_master"),
        sa.PrimaryKeyConstraint("fund_code", name="pk_fund_share_class"),
    )
    op.create_index("ix_fund_share_class_type_status", "fund_share_class", ["fund_type", "status"])
    op.create_table(
        "nav_daily",
        sa.Column("fund_code", sa.String(length=32), nullable=False),
        sa.Column("nav_date", sa.Date(), nullable=False),
        sa.Column("source_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("unit_nav", sa.Numeric(precision=20, scale=8), nullable=False),
        sa.Column("accumulated_nav", sa.Numeric(precision=20, scale=8)),
        sa.Column("content_hash", sa.String(length=64), nullable=False),
        sa.Column("source_published_at", sa.DateTime(timezone=True)),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False
        ),
        sa.CheckConstraint("unit_nav >= 0", name="ck_nav_daily_unit_nav_nonnegative"),
        sa.CheckConstraint(
            "accumulated_nav IS NULL OR accumulated_nav >= 0", name="ck_nav_daily_accumulated_nav_nonnegative"
        ),
        sa.ForeignKeyConstraint(["fund_code"], ["fund_share_class.fund_code"], name="fk_nav_daily_share_class"),
        sa.ForeignKeyConstraint(["source_id"], ["source_registry.source_id"], name="fk_nav_daily_source"),
        sa.PrimaryKeyConstraint("fund_code", "nav_date", "source_id", name="pk_nav_daily"),
        postgresql_partition_by="RANGE (nav_date)",
    )
    op.execute("CREATE TABLE nav_daily_2026_08 PARTITION OF nav_daily FOR VALUES FROM ('2026-08-01') TO ('2026-09-01')")
    op.execute("CREATE TABLE nav_daily_default PARTITION OF nav_daily DEFAULT")
    op.create_index("ix_nav_daily_fund_date", "nav_daily", ["fund_code", "nav_date"])
    op.create_index("ix_nav_daily_source_date", "nav_daily", ["source_id", "nav_date"])


def downgrade() -> None:
    """按依赖逆序删除 AI 服务维护的 M1 表和索引。"""
    op.drop_index("ix_nav_daily_source_date", table_name="nav_daily")
    op.drop_index("ix_nav_daily_fund_date", table_name="nav_daily")
    op.drop_table("nav_daily")
    op.drop_index("ix_fund_share_class_type_status", table_name="fund_share_class")
    op.drop_table("fund_share_class")
    op.drop_table("fund_master")
    op.drop_table("source_registry")
