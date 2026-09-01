"""新增 M3-05 分析运行持久控制面，不自动创建评分或回测任务。

Revision ID: 20260901_08
Revises: 20260901_07
Create Date: 2026-09-01 17:20:00
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "20260901_08"
down_revision: str | None = "20260901_07"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """创建可重启查询的分析运行记录，不改写已有评分和发布数据。"""
    op.create_table(
        "analysis_run",
        sa.Column("analysis_run_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("run_type", sa.String(length=32), nullable=False),
        sa.Column("status", sa.String(length=16), server_default=sa.text("'QUEUED'"), nullable=False),
        sa.Column("fund_type", sa.String(length=32), nullable=False),
        sa.Column("config_hash", sa.String(length=64), nullable=False),
        sa.Column("request_payload", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("task_id", sa.String(length=128)),
        sa.Column("backtest_run_id", postgresql.UUID(as_uuid=True)),
        sa.Column("result_payload", postgresql.JSONB(astext_type=sa.Text())),
        sa.Column("failure_reason", sa.Text()),
        sa.Column("trace_id", sa.String(length=128), nullable=False),
        sa.Column("requested_at", sa.DateTime(timezone=True), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True)),
        sa.Column("finished_at", sa.DateTime(timezone=True)),
        sa.CheckConstraint("run_type IN ('ROLLING_BACKTEST')", name="ck_analysis_run_type"),
        sa.CheckConstraint("status IN ('QUEUED', 'RUNNING', 'COMPLETED', 'FAILED')", name="ck_analysis_run_status"),
        sa.ForeignKeyConstraint(["backtest_run_id"], ["backtest_run.run_id"], name="fk_analysis_run_backtest"),
        sa.PrimaryKeyConstraint("analysis_run_id", name="pk_analysis_run"),
    )
    op.create_index("ix_analysis_run_status_requested", "analysis_run", ["status", "requested_at"])
    op.create_index("ix_analysis_run_task_id", "analysis_run", ["task_id"])


def downgrade() -> None:
    """按依赖逆序删除运行记录；执行前需确认不再需要历史任务审计。"""
    op.drop_index("ix_analysis_run_task_id", table_name="analysis_run")
    op.drop_index("ix_analysis_run_status_requested", table_name="analysis_run")
    op.drop_table("analysis_run")
