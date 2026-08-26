"""为 Tushare 基金同步补充来源归属与运行审计。

Revision ID: 20260826_03
Revises: 20260825_02
Create Date: 2026-08-26 18:00:00
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "20260826_03"
down_revision: str | None = "20260825_02"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_MANUAL_SOURCE_CODE = "MANUAL_PUBLISHER_VERIFIED_SAMPLE"


def upgrade() -> None:
    """标记目录来源并创建不保存原始响应的同步运行记录。"""
    op.add_column("fund_share_class", sa.Column("source_code", sa.String(length=64), nullable=True))
    op.execute(
        """
        DO $$
        BEGIN
            IF EXISTS (SELECT 1 FROM fund_share_class)
               AND NOT EXISTS (
                   SELECT 1 FROM source_registry WHERE source_code = 'MANUAL_PUBLISHER_VERIFIED_SAMPLE'
               ) THEN
                RAISE EXCEPTION 'Cannot backfill fund_share_class.source_code without manual source registry row';
            END IF;
        END $$;
        """
    )
    op.execute(
        "UPDATE fund_share_class SET source_code = 'MANUAL_PUBLISHER_VERIFIED_SAMPLE' WHERE source_code IS NULL"
    )
    op.alter_column("fund_share_class", "source_code", nullable=False)
    op.create_index("ix_fund_share_class_source_code", "fund_share_class", ["source_code"])

    op.create_table(
        "source_sync_run",
        sa.Column("sync_run_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("source_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("sync_type", sa.String(length=32), nullable=False),
        sa.Column("requested_nav_date", sa.Date()),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("fetched_count", sa.Integer(), server_default=sa.text("0"), nullable=False),
        sa.Column("created_count", sa.Integer(), server_default=sa.text("0"), nullable=False),
        sa.Column("updated_count", sa.Integer(), server_default=sa.text("0"), nullable=False),
        sa.Column("skipped_count", sa.Integer(), server_default=sa.text("0"), nullable=False),
        sa.Column("error_summary", sa.String(length=512)),
        sa.Column(
            "started_at", sa.DateTime(timezone=True), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False
        ),
        sa.Column("finished_at", sa.DateTime(timezone=True)),
        sa.CheckConstraint("fetched_count >= 0", name="ck_source_sync_run_fetched_nonnegative"),
        sa.CheckConstraint("created_count >= 0", name="ck_source_sync_run_created_nonnegative"),
        sa.CheckConstraint("updated_count >= 0", name="ck_source_sync_run_updated_nonnegative"),
        sa.CheckConstraint("skipped_count >= 0", name="ck_source_sync_run_skipped_nonnegative"),
        sa.ForeignKeyConstraint(["source_id"], ["source_registry.source_id"], name="fk_source_sync_run_source"),
        sa.PrimaryKeyConstraint("sync_run_id", name="pk_source_sync_run"),
    )
    op.create_index("ix_source_sync_run_source_started", "source_sync_run", ["source_id", "started_at"])
    op.create_index("ix_source_sync_run_status_finished", "source_sync_run", ["status", "finished_at"])


def downgrade() -> None:
    """删除同步运行记录与目录来源字段。"""
    op.drop_index("ix_source_sync_run_status_finished", table_name="source_sync_run")
    op.drop_index("ix_source_sync_run_source_started", table_name="source_sync_run")
    op.drop_table("source_sync_run")
    op.drop_index("ix_fund_share_class_source_code", table_name="fund_share_class")
    op.drop_column("fund_share_class", "source_code")
