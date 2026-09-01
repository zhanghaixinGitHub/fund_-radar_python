"""M3-04 基线评分、滚动回测与模型发布的受控数据访问。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.analysis import AnalysisModelRelease, BacktestRun, FeatureSnapshot, ForecastResult
from app.models.fund import FundShareClass, NavDaily, SourceRegistry


@dataclass(frozen=True)
class BacktestNavPoint:
    """单个基金、单个业务日的已登记净值输入。"""

    fund_code: str
    nav_date: date
    unit_nav: Decimal
    accumulated_nav: Decimal | None


@dataclass(frozen=True)
class ForecastResultUpsert:
    """一条符合数据库状态约束、可按业务键幂等保存的评分结果。"""

    feature_id: UUID
    fund_code: str
    as_of_date: date
    model_version: str
    feature_version: str
    model_release_id: UUID | None
    score_status: str
    direction: str | None
    directional_probability: Decimal | None
    confidence: Decimal | None
    risk_level: str | None
    max_drawdown_estimate: Decimal | None
    explanation: str
    result_hash: str


@dataclass(frozen=True)
class ForecastWriteStats:
    """评分写入的创建、更新与跳过计数。"""

    created_count: int = 0
    updated_count: int = 0
    skipped_count: int = 0


def list_latest_feature_snapshots(
    session: Session,
    *,
    fund_type: str,
    feature_version: str,
) -> tuple[FeatureSnapshot, ...]:
    """读取一个类别、一个特征版本下每只基金的最新快照。"""
    latest_by_fund = (
        select(
            FeatureSnapshot.fund_code.label("fund_code"),
            FeatureSnapshot.as_of_date.label("as_of_date"),
        )
        .where(
            FeatureSnapshot.fund_type == fund_type,
            FeatureSnapshot.feature_version == feature_version,
        )
        .distinct(FeatureSnapshot.fund_code)
        .order_by(FeatureSnapshot.fund_code.asc(), FeatureSnapshot.as_of_date.desc())
        .subquery()
    )
    return tuple(
        session.scalars(
            select(FeatureSnapshot)
            .join(
                latest_by_fund,
                (FeatureSnapshot.fund_code == latest_by_fund.c.fund_code)
                & (FeatureSnapshot.as_of_date == latest_by_fund.c.as_of_date),
            )
            .where(
                FeatureSnapshot.fund_type == fund_type,
                FeatureSnapshot.feature_version == feature_version,
            )
            .order_by(FeatureSnapshot.fund_code.asc(), FeatureSnapshot.feature_id.asc())
        ).all()
    )


def get_active_model_release(
    session: Session,
    *,
    model_code: str,
    fund_type: str,
) -> AnalysisModelRelease | None:
    """仅读取同一类别、同一模型代码唯一的 ACTIVE 发布。"""
    return session.scalar(
        select(AnalysisModelRelease)
        .where(
            AnalysisModelRelease.model_code == model_code,
            AnalysisModelRelease.fund_type == fund_type,
            AnalysisModelRelease.release_status == "ACTIVE",
        )
        .limit(1)
    )


def list_stock_backtest_nav_points(
    session: Session,
    *,
    source_code: str,
) -> tuple[BacktestNavPoint, ...]:
    """读取股票型试点的全量本地历史净值，不访问外部来源。"""
    rows = session.execute(
        select(
            NavDaily.fund_code,
            NavDaily.nav_date,
            NavDaily.unit_nav,
            NavDaily.accumulated_nav,
        )
        .join(FundShareClass, FundShareClass.fund_code == NavDaily.fund_code)
        .join(SourceRegistry, SourceRegistry.source_id == NavDaily.source_id)
        .where(
            FundShareClass.fund_type == "STOCK",
            FundShareClass.status == "ACTIVE",
            FundShareClass.source_code == source_code,
            SourceRegistry.source_code == source_code,
            SourceRegistry.enabled.is_(True),
        )
        .order_by(NavDaily.fund_code.asc(), NavDaily.nav_date.asc())
    ).all()
    return tuple(
        BacktestNavPoint(
            fund_code=fund_code,
            nav_date=nav_date,
            unit_nav=unit_nav,
            accumulated_nav=accumulated_nav,
        )
        for fund_code, nav_date, unit_nav, accumulated_nav in rows
    )


def upsert_forecast_results(
    session: Session,
    *,
    records: tuple[ForecastResultUpsert, ...],
    scored_at: datetime,
) -> ForecastWriteStats:
    """按基金、估值日和模型版本幂等保存结果，未变化时不推动评分时间。"""
    if not records:
        return ForecastWriteStats()

    business_keys = {(record.fund_code, record.as_of_date, record.model_version) for record in records}
    existing_by_key = {
        (result.fund_code, result.as_of_date, result.model_version): result
        for result in session.scalars(
            select(ForecastResult).where(
                ForecastResult.fund_code.in_({record.fund_code for record in records}),
                ForecastResult.as_of_date.in_({record.as_of_date for record in records}),
                ForecastResult.model_version.in_({record.model_version for record in records}),
            )
        ).all()
        if (result.fund_code, result.as_of_date, result.model_version) in business_keys
    }
    created_count = 0
    updated_count = 0
    skipped_count = 0
    for record in records:
        key = (record.fund_code, record.as_of_date, record.model_version)
        existing = existing_by_key.get(key)
        if existing is None:
            session.add(
                ForecastResult(
                    feature_id=record.feature_id,
                    fund_code=record.fund_code,
                    as_of_date=record.as_of_date,
                    model_version=record.model_version,
                    feature_version=record.feature_version,
                    model_release_id=record.model_release_id,
                    score_status=record.score_status,
                    direction=record.direction,
                    directional_probability=record.directional_probability,
                    confidence=record.confidence,
                    risk_level=record.risk_level,
                    max_drawdown_estimate=record.max_drawdown_estimate,
                    explanation=record.explanation,
                    result_hash=record.result_hash,
                    scored_at=scored_at,
                )
            )
            created_count += 1
            continue
        if existing.result_hash == record.result_hash:
            skipped_count += 1
            continue
        existing.feature_id = record.feature_id
        existing.feature_version = record.feature_version
        existing.model_release_id = record.model_release_id
        existing.score_status = record.score_status
        existing.direction = record.direction
        existing.directional_probability = record.directional_probability
        existing.confidence = record.confidence
        existing.risk_level = record.risk_level
        existing.max_drawdown_estimate = record.max_drawdown_estimate
        existing.explanation = record.explanation
        existing.result_hash = record.result_hash
        existing.scored_at = scored_at
        updated_count += 1
    return ForecastWriteStats(
        created_count=created_count,
        updated_count=updated_count,
        skipped_count=skipped_count,
    )


def find_model_release(session: Session, model_release_id: UUID) -> AnalysisModelRelease | None:
    """按主键查询模型发布记录，调用方负责状态变更授权。"""
    return session.get(AnalysisModelRelease, model_release_id)


def find_backtest_run(session: Session, run_id: UUID) -> BacktestRun | None:
    """按主键读取回测运行，用于发布闸门校验。"""
    return session.get(BacktestRun, run_id)
