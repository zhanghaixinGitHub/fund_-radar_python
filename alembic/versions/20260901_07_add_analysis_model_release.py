"""新增 M3 模型发布控制面，并将评分结果关联至批准发布。

Revision ID: 20260901_07
Revises: 20260828_06
Create Date: 2026-09-01 15:30:00
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "20260901_07"
down_revision: str | None = "20260828_06"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """以非破坏性方式增加模型发布闸门；不创建模型、评分或回测数据。"""
    op.create_table(
        "analysis_model_release",
        sa.Column("model_release_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("model_code", sa.String(length=128), nullable=False),
        sa.Column("model_version", sa.String(length=128), nullable=False),
        sa.Column("feature_version", sa.String(length=128), nullable=False),
        sa.Column("fund_type", sa.String(length=32), nullable=False),
        sa.Column("backtest_run_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("release_status", sa.String(length=16), server_default=sa.text("'DRAFT'"), nullable=False),
        sa.Column("effective_at", sa.DateTime(timezone=True)),
        sa.Column("suspended_at", sa.DateTime(timezone=True)),
        sa.Column("release_reason", sa.Text(), nullable=False),
        sa.Column("config_hash", sa.String(length=64), nullable=False),
        sa.Column(
            "wide_created_at", sa.DateTime(timezone=True), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False
        ),
        sa.Column(
            "wide_updated_at", sa.DateTime(timezone=True), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False
        ),
        sa.CheckConstraint(
            "release_status IN ('DRAFT', 'ELIGIBLE', 'ACTIVE', 'SUSPENDED', 'RETIRED')",
            name="ck_analysis_model_release_status",
        ),
        sa.ForeignKeyConstraint(
            ["backtest_run_id"], ["backtest_run.run_id"], name="fk_analysis_model_release_backtest"
        ),
        sa.PrimaryKeyConstraint("model_release_id", name="pk_analysis_model_release"),
        sa.UniqueConstraint("model_code", "model_version", "fund_type", name="uq_analysis_model_release_version"),
    )
    op.create_index(
        "uq_analysis_model_release_active",
        "analysis_model_release",
        ["model_code", "fund_type"],
        unique=True,
        postgresql_where=sa.text("release_status = 'ACTIVE'"),
    )

    op.add_column("forecast_result", sa.Column("model_release_id", postgresql.UUID(as_uuid=True), nullable=True))
    op.create_foreign_key(
        "fk_forecast_result_model_release",
        "forecast_result",
        "analysis_model_release",
        ["model_release_id"],
        ["model_release_id"],
    )
    op.create_check_constraint(
        "ck_forecast_result_scored_release",
        "forecast_result",
        "score_status <> 'SCORED' OR model_release_id IS NOT NULL",
    )
    op.create_index(
        "ix_forecast_result_release_scored",
        "forecast_result",
        ["model_release_id", "scored_at", "forecast_id"],
    )


def downgrade() -> None:
    """按依赖逆序删除控制面扩展；执行前应先暂停模型并确认无服务依赖新列。"""
    op.drop_index("ix_forecast_result_release_scored", table_name="forecast_result")
    op.drop_constraint("ck_forecast_result_scored_release", "forecast_result", type_="check")
    op.drop_constraint("fk_forecast_result_model_release", "forecast_result", type_="foreignkey")
    op.drop_column("forecast_result", "model_release_id")
    op.drop_index("uq_analysis_model_release_active", table_name="analysis_model_release")
    op.drop_table("analysis_model_release")
