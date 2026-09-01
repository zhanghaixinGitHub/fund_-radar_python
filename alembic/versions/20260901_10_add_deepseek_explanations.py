"""Add DeepSeek explanation snapshots and controlled explanation runs.

Revision ID: 20260901_10
Revises: 20260901_09
Create Date: 2026-09-01
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision = "20260901_10"
down_revision = "20260901_09"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """新增解释快照，并允许控制面排队受限的基金解释任务。"""
    op.drop_constraint("ck_analysis_run_type", "analysis_run", type_="check")
    op.create_check_constraint(
        "ck_analysis_run_type", "analysis_run", "run_type IN ('ROLLING_BACKTEST', 'FUND_EXPLANATION')"
    )
    op.create_table(
        "analysis_explanation_snapshot",
        sa.Column("explanation_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("forecast_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("model_release_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("fund_code", sa.String(length=32), nullable=False),
        sa.Column("as_of_date", sa.Date(), nullable=False),
        sa.Column("provider", sa.String(length=32), server_default="DEEPSEEK", nullable=False),
        sa.Column("provider_model", sa.String(length=128), nullable=False),
        sa.Column("prompt_version", sa.String(length=64), nullable=False),
        sa.Column("provider_request_id", sa.String(length=128), nullable=True),
        sa.Column("source_input_hash", sa.String(length=64), nullable=False),
        sa.Column("content_hash", sa.String(length=64), nullable=False),
        sa.Column("overview", sa.Text(), nullable=False),
        sa.Column("evidence", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("risk_notice", sa.Text(), nullable=False),
        sa.Column("data_gap", sa.Text(), nullable=False),
        sa.Column("disclaimer", sa.Text(), nullable=False),
        sa.Column("prompt_tokens", sa.Integer(), nullable=True),
        sa.Column("completion_tokens", sa.Integer(), nullable=True),
        sa.Column("generated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.CheckConstraint("provider = 'DEEPSEEK'", name="ck_analysis_explanation_provider"),
        sa.ForeignKeyConstraint(["forecast_id"], ["forecast_result.forecast_id"]),
        sa.ForeignKeyConstraint(["fund_code"], ["fund_share_class.fund_code"]),
        sa.ForeignKeyConstraint(["model_release_id"], ["analysis_model_release.model_release_id"]),
        sa.PrimaryKeyConstraint("explanation_id"),
        sa.UniqueConstraint("content_hash", name="uq_analysis_explanation_content_hash"),
        sa.UniqueConstraint(
            "forecast_id", "provider", "provider_model", "prompt_version",
            name="uq_analysis_explanation_forecast_provider_prompt",
        ),
    )
    op.create_index(
        "ix_analysis_explanation_fund_generated", "analysis_explanation_snapshot", ["fund_code", "generated_at", "explanation_id"]
    )
    op.create_index(
        "ix_analysis_explanation_release_generated", "analysis_explanation_snapshot", ["model_release_id", "generated_at"]
    )


def downgrade() -> None:
    """删除解释快照并恢复旧的单一控制面运行类型约束。"""
    op.drop_index("ix_analysis_explanation_release_generated", table_name="analysis_explanation_snapshot")
    op.drop_index("ix_analysis_explanation_fund_generated", table_name="analysis_explanation_snapshot")
    op.drop_table("analysis_explanation_snapshot")
    op.drop_constraint("ck_analysis_run_type", "analysis_run", type_="check")
    op.create_check_constraint("ck_analysis_run_type", "analysis_run", "run_type IN ('ROLLING_BACKTEST')")
