"""创建 M2 事件治理与 M3 分析数据库结构。

Revision ID: 20260825_02
Revises: 20260824_01
Create Date: 2026-08-25 09:00:00
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "20260825_02"
down_revision: str | None = "20260824_01"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """创建 AI 服务维护的空 M2/M3 表及索引，不访问任何外部数据源。"""
    op.create_table(
        "news_item",
        sa.Column("news_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("source_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("title", sa.String(length=512), nullable=False),
        sa.Column("url", sa.String(length=2048), nullable=False),
        sa.Column("summary", sa.Text()),
        sa.Column("content_hash", sa.String(length=64), nullable=False),
        sa.Column("published_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column(
            "fetched_at", sa.DateTime(timezone=True), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False
        ),
        sa.Column("retention_until", sa.DateTime(timezone=True), nullable=False),
        sa.Column("active", sa.Boolean(), server_default=sa.text("true"), nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False
        ),
        sa.ForeignKeyConstraint(["source_id"], ["source_registry.source_id"], name="fk_news_item_source"),
        sa.PrimaryKeyConstraint("news_id", name="pk_news_item"),
        sa.UniqueConstraint("content_hash", name="uq_news_item_content_hash"),
    )
    op.create_index("ix_news_item_published_at", "news_item", ["published_at"])
    op.create_index("ix_news_item_retention_until", "news_item", ["retention_until"])
    op.create_table(
        "news_source_reference",
        sa.Column("reference_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("news_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("source_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("url", sa.String(length=2048), nullable=False),
        sa.Column("published_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False
        ),
        sa.ForeignKeyConstraint(["news_id"], ["news_item.news_id"], name="fk_news_source_reference_news"),
        sa.ForeignKeyConstraint(["source_id"], ["source_registry.source_id"], name="fk_news_source_reference_source"),
        sa.PrimaryKeyConstraint("reference_id", name="pk_news_source_reference"),
        sa.UniqueConstraint("source_id", "url", name="uq_news_source_reference_source_url"),
    )
    op.create_index("ix_news_source_reference_news", "news_source_reference", ["news_id"])
    op.create_table(
        "market_event",
        sa.Column("event_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("news_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("event_type", sa.String(length=64), nullable=False),
        sa.Column("summary", sa.Text(), nullable=False),
        sa.Column("explanation", sa.Text(), nullable=False),
        sa.Column("confidence", sa.Numeric(precision=5, scale=4), nullable=False),
        sa.Column("published_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("event_hash", sa.String(length=64), nullable=False),
        sa.Column("approval_status", sa.String(length=16), server_default=sa.text("'PENDING'"), nullable=False),
        sa.Column("approved_at", sa.DateTime(timezone=True)),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False
        ),
        sa.CheckConstraint("confidence >= 0 AND confidence <= 1", name="ck_market_event_confidence_range"),
        sa.CheckConstraint(
            "approval_status IN ('PENDING', 'APPROVED', 'REJECTED')", name="ck_market_event_approval_status"
        ),
        sa.ForeignKeyConstraint(["news_id"], ["news_item.news_id"], name="fk_market_event_news"),
        sa.PrimaryKeyConstraint("event_id", name="pk_market_event"),
        sa.UniqueConstraint("event_hash", name="uq_market_event_event_hash"),
    )
    op.create_index("ix_market_event_reviewed_published", "market_event", ["approval_status", "published_at"])
    op.create_table(
        "event_relation",
        sa.Column("relation_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("event_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("entity_type", sa.String(length=32), nullable=False),
        sa.Column("entity_id", sa.String(length=256), nullable=False),
        sa.Column("relevance_score", sa.Numeric(precision=5, scale=4), nullable=False),
        sa.Column("relation_reason", sa.Text(), nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False
        ),
        sa.CheckConstraint("relevance_score >= 0 AND relevance_score <= 1", name="ck_event_relation_relevance_range"),
        sa.CheckConstraint(
            "entity_type IN ('FUND_CODE', 'FUND_MANAGER', 'INDUSTRY', 'INDEX', 'COMPANY', 'POLICY_TOPIC')",
            name="ck_event_relation_entity_type",
        ),
        sa.ForeignKeyConstraint(["event_id"], ["market_event.event_id"], name="fk_event_relation_event"),
        sa.PrimaryKeyConstraint("relation_id", name="pk_event_relation"),
        sa.UniqueConstraint("event_id", "entity_type", "entity_id", name="uq_event_relation_entity"),
    )
    op.create_index("ix_event_relation_entity", "event_relation", ["entity_type", "entity_id"])
    op.create_table(
        "feature_snapshot",
        sa.Column("feature_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("fund_code", sa.String(length=32), nullable=False),
        sa.Column("as_of_date", sa.Date(), nullable=False),
        sa.Column("fund_type", sa.String(length=32), nullable=False),
        sa.Column("feature_version", sa.String(length=128), nullable=False),
        sa.Column("completeness", sa.Numeric(precision=5, scale=4), nullable=False),
        sa.Column("eligibility_status", sa.String(length=32), nullable=False),
        sa.Column("unavailable_reason", sa.Text()),
        sa.Column("feature_payload", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("feature_hash", sa.String(length=64), nullable=False),
        sa.Column(
            "computed_at", sa.DateTime(timezone=True), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False
        ),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False
        ),
        sa.CheckConstraint("completeness >= 0 AND completeness <= 1", name="ck_feature_snapshot_completeness_range"),
        sa.CheckConstraint(
            "eligibility_status IN ('SCORABLE', 'DATA_INSUFFICIENT', 'NOT_APPLICABLE')",
            name="ck_feature_snapshot_eligibility_status",
        ),
        sa.ForeignKeyConstraint(["fund_code"], ["fund_share_class.fund_code"], name="fk_feature_snapshot_fund"),
        sa.PrimaryKeyConstraint("feature_id", name="pk_feature_snapshot"),
        sa.UniqueConstraint("fund_code", "as_of_date", "feature_version", name="uq_feature_snapshot_business"),
    )
    op.create_index("ix_feature_snapshot_fund_as_of", "feature_snapshot", ["fund_code", "as_of_date"])
    op.create_table(
        "forecast_result",
        sa.Column("forecast_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("feature_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("fund_code", sa.String(length=32), nullable=False),
        sa.Column("as_of_date", sa.Date(), nullable=False),
        sa.Column("model_version", sa.String(length=128), nullable=False),
        sa.Column("feature_version", sa.String(length=128), nullable=False),
        sa.Column("score_status", sa.String(length=32), nullable=False),
        sa.Column("direction", sa.String(length=16)),
        sa.Column("directional_probability", sa.Numeric(precision=5, scale=4)),
        sa.Column("confidence", sa.Numeric(precision=5, scale=4)),
        sa.Column("risk_level", sa.String(length=32)),
        sa.Column("max_drawdown_estimate", sa.Numeric(precision=9, scale=6)),
        sa.Column("explanation", sa.Text(), nullable=False),
        sa.Column("result_hash", sa.String(length=64), nullable=False),
        sa.Column("scored_at", sa.DateTime(timezone=True), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False
        ),
        sa.CheckConstraint(
            "score_status IN ('SCORED', 'DATA_INSUFFICIENT', 'NOT_APPLICABLE', 'MODEL_REJECTED')",
            name="ck_forecast_result_score_status",
        ),
        sa.CheckConstraint(
            "direction IS NULL OR direction IN ('UP', 'DOWN', 'NEUTRAL')", name="ck_forecast_result_direction"
        ),
        sa.CheckConstraint(
            "directional_probability IS NULL OR (directional_probability >= 0 AND directional_probability <= 1)",
            name="ck_forecast_result_probability_range",
        ),
        sa.CheckConstraint(
            "confidence IS NULL OR (confidence >= 0 AND confidence <= 1)", name="ck_forecast_result_confidence_range"
        ),
        sa.CheckConstraint(
            "(score_status = 'SCORED' AND direction IS NOT NULL AND directional_probability IS NOT NULL "
            "AND confidence IS NOT NULL AND risk_level IS NOT NULL) OR "
            "(score_status <> 'SCORED' AND direction IS NULL AND "
            "directional_probability IS NULL AND confidence IS NULL)",
            name="ck_forecast_result_no_direction_without_score",
        ),
        sa.ForeignKeyConstraint(["feature_id"], ["feature_snapshot.feature_id"], name="fk_forecast_result_feature"),
        sa.ForeignKeyConstraint(["fund_code"], ["fund_share_class.fund_code"], name="fk_forecast_result_fund"),
        sa.PrimaryKeyConstraint("forecast_id", name="pk_forecast_result"),
        sa.UniqueConstraint("fund_code", "as_of_date", "model_version", name="uq_forecast_result_business"),
        sa.UniqueConstraint("result_hash", name="uq_forecast_result_result_hash"),
    )
    op.create_index("ix_forecast_result_fund_as_of", "forecast_result", ["fund_code", "as_of_date"])
    op.create_table(
        "backtest_run",
        sa.Column("run_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("fund_type", sa.String(length=32), nullable=False),
        sa.Column("strategy_version", sa.String(length=128), nullable=False),
        sa.Column("feature_version", sa.String(length=128), nullable=False),
        sa.Column("model_version", sa.String(length=128), nullable=False),
        sa.Column("window_start", sa.Date(), nullable=False),
        sa.Column("window_end", sa.Date(), nullable=False),
        sa.Column("train_end", sa.Date()),
        sa.Column("validation_end", sa.Date()),
        sa.Column("test_start", sa.Date()),
        sa.Column("test_end", sa.Date()),
        sa.Column("fee_rate", sa.Numeric(precision=9, scale=6), nullable=False),
        sa.Column("status", sa.String(length=16), server_default=sa.text("'PENDING'"), nullable=False),
        sa.Column(
            "publication_status", sa.String(length=16), server_default=sa.text("'NOT_EVALUATED'"), nullable=False
        ),
        sa.Column("metrics", postgresql.JSONB(astext_type=sa.Text())),
        sa.Column("baselines", postgresql.JSONB(astext_type=sa.Text())),
        sa.Column("failure_reason", sa.Text()),
        sa.Column("started_at", sa.DateTime(timezone=True)),
        sa.Column("finished_at", sa.DateTime(timezone=True)),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False
        ),
        sa.CheckConstraint(
            "status IN ('PENDING', 'RUNNING', 'COMPLETED', 'REJECTED', 'FAILED')", name="ck_backtest_run_status"
        ),
        sa.CheckConstraint(
            "publication_status IN ('NOT_EVALUATED', 'ELIGIBLE', 'INELIGIBLE')",
            name="ck_backtest_run_publication_status",
        ),
        sa.CheckConstraint("window_start < window_end", name="ck_backtest_run_window_order"),
        sa.CheckConstraint(
            "train_end IS NULL OR validation_end IS NULL OR test_start IS NULL OR test_end IS NULL OR "
            "(train_end < validation_end AND validation_end < test_start AND test_start <= test_end)",
            name="ck_backtest_run_split_order",
        ),
        sa.PrimaryKeyConstraint("run_id", name="pk_backtest_run"),
    )
    op.create_index("ix_backtest_run_status_created_at", "backtest_run", ["status", "created_at"])


def downgrade() -> None:
    """按依赖逆序删除 M2/M3 表和索引。"""
    op.drop_index("ix_backtest_run_status_created_at", table_name="backtest_run")
    op.drop_table("backtest_run")
    op.drop_index("ix_forecast_result_fund_as_of", table_name="forecast_result")
    op.drop_table("forecast_result")
    op.drop_index("ix_feature_snapshot_fund_as_of", table_name="feature_snapshot")
    op.drop_table("feature_snapshot")
    op.drop_index("ix_event_relation_entity", table_name="event_relation")
    op.drop_table("event_relation")
    op.drop_index("ix_market_event_reviewed_published", table_name="market_event")
    op.drop_table("market_event")
    op.drop_index("ix_news_source_reference_news", table_name="news_source_reference")
    op.drop_table("news_source_reference")
    op.drop_index("ix_news_item_retention_until", table_name="news_item")
    op.drop_index("ix_news_item_published_at", table_name="news_item")
    op.drop_table("news_item")
