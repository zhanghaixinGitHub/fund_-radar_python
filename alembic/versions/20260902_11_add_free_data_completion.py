"""Add free-data completion source capabilities, cursors, and market-reference tables.

Revision ID: 20260902_11
Revises: 20260901_10
Create Date: 2026-09-02 14:30:00
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "20260902_11"
down_revision: str | None = "20260901_10"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """仅加性地创建免费市场数据治理对象，不预置或抓取任何外部数据。"""
    op.add_column(
        "source_registry",
        sa.Column(
            "authorized_api_names",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'[]'::jsonb"),
            nullable=False,
        ),
    )
    op.add_column("source_registry", sa.Column("authorization_verified_at", sa.DateTime(timezone=True)))
    op.create_check_constraint(
        "ck_source_registry_authorized_api_names_array",
        "source_registry",
        "jsonb_typeof(authorized_api_names) = 'array'",
    )
    op.add_column("source_sync_run", sa.Column("parent_sync_run_id", postgresql.UUID(as_uuid=True)))
    op.add_column("source_sync_run", sa.Column("requested_window_start", sa.Date()))
    op.add_column("source_sync_run", sa.Column("requested_window_end", sa.Date()))
    op.add_column("source_sync_run", sa.Column("data_as_of_date", sa.Date()))
    op.create_foreign_key(
        "fk_source_sync_run_parent",
        "source_sync_run",
        "source_sync_run",
        ["parent_sync_run_id"],
        ["sync_run_id"],
    )
    op.create_index("ix_source_sync_run_parent_started", "source_sync_run", ["parent_sync_run_id", "started_at"])
    op.create_table(
        "source_sync_cursor",
        sa.Column("source_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("dataset_code", sa.String(length=64), nullable=False),
        sa.Column("entity_key", sa.String(length=128), nullable=False),
        sa.Column("last_successful_data_date", sa.Date()),
        sa.Column("last_successful_published_at", sa.DateTime(timezone=True)),
        sa.Column("last_sync_run_id", postgresql.UUID(as_uuid=True)),
        sa.Column("consecutive_failure_count", sa.Integer(), server_default="0", nullable=False),
        sa.Column("last_error_summary", sa.String(length=512)),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "consecutive_failure_count >= 0",
            name="ck_source_sync_cursor_failure_count_nonnegative",
        ),
        sa.ForeignKeyConstraint(
            ["source_id"],
            ["source_registry.source_id"],
            name="fk_source_sync_cursor_source",
        ),
        sa.ForeignKeyConstraint(
            ["last_sync_run_id"],
            ["source_sync_run.sync_run_id"],
            name="fk_source_sync_cursor_run",
        ),
        sa.PrimaryKeyConstraint("source_id", "dataset_code", "entity_key", name="pk_source_sync_cursor"),
    )
    op.create_index(
        "ix_source_sync_cursor_dataset_updated",
        "source_sync_cursor",
        ["dataset_code", "updated_at"],
    )
    op.create_table(
        "fund_exchange_daily",
        sa.Column("fund_code", sa.String(length=32), nullable=False),
        sa.Column("trade_date", sa.Date(), nullable=False),
        sa.Column("source_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("open_price", sa.Numeric(precision=20, scale=8)),
        sa.Column("high_price", sa.Numeric(precision=20, scale=8)),
        sa.Column("low_price", sa.Numeric(precision=20, scale=8)),
        sa.Column("close_price", sa.Numeric(precision=20, scale=8), nullable=False),
        sa.Column("previous_close_price", sa.Numeric(precision=20, scale=8)),
        sa.Column("change_value", sa.Numeric(precision=20, scale=8)),
        sa.Column("change_percent", sa.Numeric(precision=16, scale=8)),
        sa.Column("volume", sa.Numeric(precision=24, scale=4)),
        sa.Column("amount", sa.Numeric(precision=24, scale=4)),
        sa.Column("source_published_at", sa.DateTime(timezone=True)),
        sa.Column("content_hash", sa.String(length=64), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.CheckConstraint("close_price > 0", name="ck_fund_exchange_daily_close_positive"),
        sa.ForeignKeyConstraint(
            ["fund_code"],
            ["fund_share_class.fund_code"],
            name="fk_fund_exchange_daily_fund",
        ),
        sa.ForeignKeyConstraint(
            ["source_id"],
            ["source_registry.source_id"],
            name="fk_fund_exchange_daily_source",
        ),
        sa.PrimaryKeyConstraint("fund_code", "trade_date", "source_id", name="pk_fund_exchange_daily"),
    )
    op.create_index(
        "ix_fund_exchange_daily_fund_date",
        "fund_exchange_daily",
        ["fund_code", "trade_date"],
    )
    op.create_table(
        "market_index_catalog",
        sa.Column("source_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("index_code", sa.String(length=64), nullable=False),
        sa.Column("display_name", sa.String(length=128), nullable=False),
        sa.Column("market", sa.String(length=32)),
        sa.Column("publisher", sa.String(length=128)),
        sa.Column("category", sa.String(length=128)),
        sa.Column("base_date", sa.Date()),
        sa.Column("list_date", sa.Date()),
        sa.Column("expiry_date", sa.Date()),
        sa.Column("row_hash", sa.String(length=64), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["source_id"], ["source_registry.source_id"], name="fk_market_index_catalog_source"),
        sa.PrimaryKeyConstraint("source_id", "index_code", name="pk_market_index_catalog"),
    )
    op.create_index(
        "ix_market_index_catalog_category",
        "market_index_catalog",
        ["category", "display_name"],
    )
    op.create_table(
        "market_index_classification",
        sa.Column("source_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("classification_code", sa.String(length=64), nullable=False),
        sa.Column("classification_name", sa.String(length=128), nullable=False),
        sa.Column("parent_classification_code", sa.String(length=64)),
        sa.Column("hierarchy_level", sa.Integer()),
        sa.Column("source_name", sa.String(length=128)),
        sa.Column("row_hash", sa.String(length=64), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["source_id"],
            ["source_registry.source_id"],
            name="fk_market_index_classification_source",
        ),
        sa.PrimaryKeyConstraint("source_id", "classification_code", name="pk_market_index_classification"),
    )
    op.create_index(
        "ix_market_index_classification_parent",
        "market_index_classification",
        ["source_id", "parent_classification_code"],
    )
    op.create_table(
        "index_weight_snapshot",
        sa.Column("source_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("index_code", sa.String(length=64), nullable=False),
        sa.Column("weight_date", sa.Date(), nullable=False),
        sa.Column("constituent_code", sa.String(length=32), nullable=False),
        sa.Column("weight", sa.Numeric(precision=12, scale=8), nullable=False),
        sa.Column("row_hash", sa.String(length=64), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "weight >= 0 AND weight <= 100",
            name="ck_index_weight_snapshot_range",
        ),
        sa.ForeignKeyConstraint(
            ["source_id"],
            ["source_registry.source_id"],
            name="fk_index_weight_snapshot_source",
        ),
        sa.PrimaryKeyConstraint(
            "source_id",
            "index_code",
            "weight_date",
            "constituent_code",
            name="pk_index_weight_snapshot",
        ),
    )
    op.create_index(
        "ix_index_weight_snapshot_index_date",
        "index_weight_snapshot",
        ["index_code", "weight_date"],
    )


def downgrade() -> None:
    """按依赖逆序删除本次新增对象；执行前须确认不再保留同步审计数据。"""
    op.drop_index("ix_index_weight_snapshot_index_date", table_name="index_weight_snapshot")
    op.drop_table("index_weight_snapshot")
    op.drop_index("ix_market_index_classification_parent", table_name="market_index_classification")
    op.drop_table("market_index_classification")
    op.drop_index("ix_market_index_catalog_category", table_name="market_index_catalog")
    op.drop_table("market_index_catalog")
    op.drop_index("ix_fund_exchange_daily_fund_date", table_name="fund_exchange_daily")
    op.drop_table("fund_exchange_daily")
    op.drop_index("ix_source_sync_cursor_dataset_updated", table_name="source_sync_cursor")
    op.drop_table("source_sync_cursor")
    op.drop_index("ix_source_sync_run_parent_started", table_name="source_sync_run")
    op.drop_constraint("fk_source_sync_run_parent", "source_sync_run", type_="foreignkey")
    op.drop_column("source_sync_run", "data_as_of_date")
    op.drop_column("source_sync_run", "requested_window_end")
    op.drop_column("source_sync_run", "requested_window_start")
    op.drop_column("source_sync_run", "parent_sync_run_id")
    op.drop_constraint("ck_source_registry_authorized_api_names_array", "source_registry", type_="check")
    op.drop_column("source_registry", "authorization_verified_at")
    op.drop_column("source_registry", "authorized_api_names")
