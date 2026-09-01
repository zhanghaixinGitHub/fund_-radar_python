"""由 AI 服务维护的 M3 特征、评分结果和回测持久化模型。"""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from uuid import UUID, uuid4

from sqlalchemy import (
    CheckConstraint,
    Date,
    DateTime,
    ForeignKey,
    Index,
    Numeric,
    String,
    Text,
    UniqueConstraint,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import UUID as PostgreSQLUUID
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class FeatureSnapshot(Base):
    """某基金在指定估值日期的可复现、按品类适配的输入特征快照。"""

    __tablename__ = "feature_snapshot"
    __table_args__ = (
        CheckConstraint("completeness >= 0 AND completeness <= 1", name="ck_feature_snapshot_completeness_range"),
        CheckConstraint(
            "eligibility_status IN ('SCORABLE', 'DATA_INSUFFICIENT', 'NOT_APPLICABLE')",
            name="ck_feature_snapshot_eligibility_status",
        ),
        UniqueConstraint("fund_code", "as_of_date", "feature_version", name="uq_feature_snapshot_business"),
    )

    feature_id: Mapped[UUID] = mapped_column(PostgreSQLUUID(as_uuid=True), primary_key=True, default=uuid4)
    fund_code: Mapped[str] = mapped_column(String(32), ForeignKey("fund_share_class.fund_code"), nullable=False)
    as_of_date: Mapped[date] = mapped_column(Date, nullable=False)
    fund_type: Mapped[str] = mapped_column(String(32), nullable=False)
    feature_version: Mapped[str] = mapped_column(String(128), nullable=False)
    completeness: Mapped[Decimal] = mapped_column(Numeric(5, 4), nullable=False)
    eligibility_status: Mapped[str] = mapped_column(String(32), nullable=False)
    unavailable_reason: Mapped[str | None] = mapped_column(Text)
    feature_payload: Mapped[dict[str, object]] = mapped_column(JSONB, nullable=False)
    feature_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    computed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())


class AnalysisModelRelease(Base):
    """经过回测准入的模型发布控制面；只有 ACTIVE 发布可供后续评分任务消费。"""

    __tablename__ = "analysis_model_release"
    __table_args__ = (
        CheckConstraint(
            "release_status IN ('DRAFT', 'ELIGIBLE', 'ACTIVE', 'SUSPENDED', 'RETIRED')",
            name="ck_analysis_model_release_status",
        ),
        UniqueConstraint("model_code", "model_version", "fund_type", name="uq_analysis_model_release_version"),
        Index(
            "uq_analysis_model_release_active",
            "model_code",
            "fund_type",
            unique=True,
            postgresql_where=text("release_status = 'ACTIVE'"),
        ),
    )

    model_release_id: Mapped[UUID] = mapped_column(PostgreSQLUUID(as_uuid=True), primary_key=True, default=uuid4)
    model_code: Mapped[str] = mapped_column(String(128), nullable=False)
    model_version: Mapped[str] = mapped_column(String(128), nullable=False)
    feature_version: Mapped[str] = mapped_column(String(128), nullable=False)
    fund_type: Mapped[str] = mapped_column(String(32), nullable=False)
    backtest_run_id: Mapped[UUID] = mapped_column(
        PostgreSQLUUID(as_uuid=True), ForeignKey("backtest_run.run_id"), nullable=False
    )
    release_status: Mapped[str] = mapped_column(String(16), nullable=False, default="DRAFT", server_default="DRAFT")
    effective_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    suspended_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    release_reason: Mapped[str] = mapped_column(Text, nullable=False)
    config_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    wide_created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    wide_updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()
    )


class AnalysisRun(Base):
    """M3-05 受控分析任务的持久状态，供 Java 管理端安全查询。"""

    __tablename__ = "analysis_run"
    __table_args__ = (
        CheckConstraint("run_type IN ('ROLLING_BACKTEST', 'FUND_EXPLANATION')", name="ck_analysis_run_type"),
        CheckConstraint("status IN ('QUEUED', 'RUNNING', 'COMPLETED', 'FAILED')", name="ck_analysis_run_status"),
        Index("ix_analysis_run_status_requested", "status", "requested_at"),
        Index("ix_analysis_run_task_id", "task_id"),
    )

    analysis_run_id: Mapped[UUID] = mapped_column(PostgreSQLUUID(as_uuid=True), primary_key=True, default=uuid4)
    run_type: Mapped[str] = mapped_column(String(32), nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="QUEUED", server_default="QUEUED")
    fund_type: Mapped[str] = mapped_column(String(32), nullable=False)
    config_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    request_payload: Mapped[dict[str, object]] = mapped_column(JSONB, nullable=False)
    task_id: Mapped[str | None] = mapped_column(String(128))
    backtest_run_id: Mapped[UUID | None] = mapped_column(
        PostgreSQLUUID(as_uuid=True), ForeignKey("backtest_run.run_id")
    )
    result_payload: Mapped[dict[str, object] | None] = mapped_column(JSONB)
    failure_reason: Mapped[str | None] = mapped_column(Text)
    trace_id: Mapped[str] = mapped_column(String(128), nullable=False)
    requested_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class AnalysisExplanationSnapshot(Base):
    """已发布量化评分的 DeepSeek 解释快照；不保存原始提示词或外部原始响应。"""

    __tablename__ = "analysis_explanation_snapshot"
    __table_args__ = (
        CheckConstraint("provider = 'DEEPSEEK'", name="ck_analysis_explanation_provider"),
        UniqueConstraint(
            "forecast_id", "provider", "provider_model", "prompt_version",
            name="uq_analysis_explanation_forecast_provider_prompt",
        ),
        UniqueConstraint("content_hash", name="uq_analysis_explanation_content_hash"),
        Index("ix_analysis_explanation_fund_generated", "fund_code", "generated_at", "explanation_id"),
        Index("ix_analysis_explanation_release_generated", "model_release_id", "generated_at"),
    )

    explanation_id: Mapped[UUID] = mapped_column(PostgreSQLUUID(as_uuid=True), primary_key=True, default=uuid4)
    forecast_id: Mapped[UUID] = mapped_column(
        PostgreSQLUUID(as_uuid=True), ForeignKey("forecast_result.forecast_id"), nullable=False
    )
    model_release_id: Mapped[UUID] = mapped_column(
        PostgreSQLUUID(as_uuid=True), ForeignKey("analysis_model_release.model_release_id"), nullable=False
    )
    fund_code: Mapped[str] = mapped_column(String(32), ForeignKey("fund_share_class.fund_code"), nullable=False)
    as_of_date: Mapped[date] = mapped_column(Date, nullable=False)
    provider: Mapped[str] = mapped_column(String(32), nullable=False, default="DEEPSEEK", server_default="DEEPSEEK")
    provider_model: Mapped[str] = mapped_column(String(128), nullable=False)
    prompt_version: Mapped[str] = mapped_column(String(64), nullable=False)
    provider_request_id: Mapped[str | None] = mapped_column(String(128))
    source_input_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    overview: Mapped[str] = mapped_column(Text, nullable=False)
    evidence: Mapped[list[dict[str, str]]] = mapped_column(JSONB, nullable=False)
    risk_notice: Mapped[str] = mapped_column(Text, nullable=False)
    data_gap: Mapped[str] = mapped_column(Text, nullable=False)
    disclaimer: Mapped[str] = mapped_column(Text, nullable=False)
    prompt_tokens: Mapped[int | None] = mapped_column()
    completion_tokens: Mapped[int | None] = mapped_column()
    generated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())


class ForecastResult(Base):
    """可复现的 M3 评分结果；数据不足或不适用时不得产生方向结论。"""

    __tablename__ = "forecast_result"
    __table_args__ = (
        CheckConstraint(
            "score_status IN ('SCORED', 'DATA_INSUFFICIENT', 'NOT_APPLICABLE', 'MODEL_REJECTED')",
            name="ck_forecast_result_score_status",
        ),
        CheckConstraint(
            "direction IS NULL OR direction IN ('UP', 'DOWN', 'NEUTRAL')", name="ck_forecast_result_direction"
        ),
        CheckConstraint(
            "directional_probability IS NULL OR (directional_probability >= 0 AND directional_probability <= 1)",
            name="ck_forecast_result_probability_range",
        ),
        CheckConstraint(
            "confidence IS NULL OR (confidence >= 0 AND confidence <= 1)",
            name="ck_forecast_result_confidence_range",
        ),
        CheckConstraint(
            "(score_status = 'SCORED' AND direction IS NOT NULL AND directional_probability IS NOT NULL "
            "AND confidence IS NOT NULL AND risk_level IS NOT NULL) OR "
            "(score_status <> 'SCORED' AND direction IS NULL AND "
            "directional_probability IS NULL AND confidence IS NULL)",
            name="ck_forecast_result_no_direction_without_score",
        ),
        CheckConstraint(
            "score_status <> 'SCORED' OR model_release_id IS NOT NULL",
            name="ck_forecast_result_scored_release",
        ),
        UniqueConstraint("fund_code", "as_of_date", "model_version", name="uq_forecast_result_business"),
        UniqueConstraint("result_hash", name="uq_forecast_result_result_hash"),
        Index("ix_forecast_result_release_scored", "model_release_id", "scored_at", "forecast_id"),
    )

    forecast_id: Mapped[UUID] = mapped_column(PostgreSQLUUID(as_uuid=True), primary_key=True, default=uuid4)
    feature_id: Mapped[UUID] = mapped_column(
        PostgreSQLUUID(as_uuid=True), ForeignKey("feature_snapshot.feature_id"), nullable=False
    )
    fund_code: Mapped[str] = mapped_column(String(32), ForeignKey("fund_share_class.fund_code"), nullable=False)
    as_of_date: Mapped[date] = mapped_column(Date, nullable=False)
    model_version: Mapped[str] = mapped_column(String(128), nullable=False)
    feature_version: Mapped[str] = mapped_column(String(128), nullable=False)
    model_release_id: Mapped[UUID | None] = mapped_column(
        PostgreSQLUUID(as_uuid=True), ForeignKey("analysis_model_release.model_release_id")
    )
    score_status: Mapped[str] = mapped_column(String(32), nullable=False)
    direction: Mapped[str | None] = mapped_column(String(16))
    directional_probability: Mapped[Decimal | None] = mapped_column(Numeric(5, 4))
    confidence: Mapped[Decimal | None] = mapped_column(Numeric(5, 4))
    risk_level: Mapped[str | None] = mapped_column(String(32))
    max_drawdown_estimate: Mapped[Decimal | None] = mapped_column(Numeric(9, 6))
    explanation: Mapped[str] = mapped_column(Text, nullable=False)
    result_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    scored_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())


class BacktestRun(Base):
    """M3 滚动窗口回测任务及其结果发布准入结论。"""

    __tablename__ = "backtest_run"
    __table_args__ = (
        CheckConstraint(
            "status IN ('PENDING', 'RUNNING', 'COMPLETED', 'REJECTED', 'FAILED')", name="ck_backtest_run_status"
        ),
        CheckConstraint(
            "publication_status IN ('NOT_EVALUATED', 'ELIGIBLE', 'INELIGIBLE')",
            name="ck_backtest_run_publication_status",
        ),
        CheckConstraint("window_start < window_end", name="ck_backtest_run_window_order"),
        CheckConstraint(
            "train_end IS NULL OR validation_end IS NULL OR test_start IS NULL OR test_end IS NULL OR "
            "(train_end < validation_end AND validation_end < test_start AND test_start <= test_end)",
            name="ck_backtest_run_split_order",
        ),
    )

    run_id: Mapped[UUID] = mapped_column(PostgreSQLUUID(as_uuid=True), primary_key=True, default=uuid4)
    fund_type: Mapped[str] = mapped_column(String(32), nullable=False)
    strategy_version: Mapped[str] = mapped_column(String(128), nullable=False)
    feature_version: Mapped[str] = mapped_column(String(128), nullable=False)
    model_version: Mapped[str] = mapped_column(String(128), nullable=False)
    window_start: Mapped[date] = mapped_column(Date, nullable=False)
    window_end: Mapped[date] = mapped_column(Date, nullable=False)
    train_end: Mapped[date | None] = mapped_column(Date)
    validation_end: Mapped[date | None] = mapped_column(Date)
    test_start: Mapped[date | None] = mapped_column(Date)
    test_end: Mapped[date | None] = mapped_column(Date)
    fee_rate: Mapped[Decimal] = mapped_column(Numeric(9, 6), nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="PENDING", server_default="PENDING")
    publication_status: Mapped[str] = mapped_column(
        String(16), nullable=False, default="NOT_EVALUATED", server_default="NOT_EVALUATED"
    )
    metrics: Mapped[dict[str, object] | None] = mapped_column(JSONB)
    baselines: Mapped[dict[str, object] | None] = mapped_column(JSONB)
    failure_reason: Mapped[str | None] = mapped_column(Text)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
