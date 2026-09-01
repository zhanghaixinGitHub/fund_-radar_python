"""已发布评分解释快照的数据访问；仓储层不调用外部模型。"""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.analysis import AnalysisExplanationSnapshot, AnalysisModelRelease, FeatureSnapshot, ForecastResult


@dataclass(frozen=True)
class ExplanationSource:
    """生成解释前已经通过发布闸门的一组本地事实。"""

    forecast: ForecastResult
    feature: FeatureSnapshot
    release: AnalysisModelRelease


def find_latest_active_scored_source(session: Session, fund_code: str) -> ExplanationSource | None:
    """只读取当前 ACTIVE 发布关联的最新 SCORED 结果，禁止用候选或不足数据调用大模型。"""
    row = session.execute(
        select(ForecastResult, FeatureSnapshot, AnalysisModelRelease)
        .join(FeatureSnapshot, FeatureSnapshot.feature_id == ForecastResult.feature_id)
        .join(AnalysisModelRelease, AnalysisModelRelease.model_release_id == ForecastResult.model_release_id)
        .where(
            ForecastResult.fund_code == fund_code,
            ForecastResult.score_status == "SCORED",
            AnalysisModelRelease.release_status == "ACTIVE",
        )
        .order_by(ForecastResult.as_of_date.desc(), ForecastResult.scored_at.desc(), ForecastResult.forecast_id.desc())
        .limit(1)
    ).first()
    if row is None:
        return None
    forecast, feature, release = row
    return ExplanationSource(forecast=forecast, feature=feature, release=release)


def find_explanation_snapshot(
    session: Session,
    *,
    forecast_id: object,
    provider: str,
    provider_model: str,
    prompt_version: str,
) -> AnalysisExplanationSnapshot | None:
    """按可复现输入版本查询既有解释，避免同一评分重复产生外部调用费用。"""
    return session.scalar(
        select(AnalysisExplanationSnapshot)
        .where(
            AnalysisExplanationSnapshot.forecast_id == forecast_id,
            AnalysisExplanationSnapshot.provider == provider,
            AnalysisExplanationSnapshot.provider_model == provider_model,
            AnalysisExplanationSnapshot.prompt_version == prompt_version,
        )
        .limit(1)
    )


def find_latest_fund_explanation(
    session: Session,
    *,
    fund_code: str,
    model_release_id: object,
) -> AnalysisExplanationSnapshot | None:
    """读取当前发布版本最近成功的解释快照；读取本身不调用 DeepSeek。"""
    return session.scalar(
        select(AnalysisExplanationSnapshot)
        .where(
            AnalysisExplanationSnapshot.fund_code == fund_code,
            AnalysisExplanationSnapshot.model_release_id == model_release_id,
        )
        .order_by(
            AnalysisExplanationSnapshot.as_of_date.desc(),
            AnalysisExplanationSnapshot.generated_at.desc(),
            AnalysisExplanationSnapshot.explanation_id.desc(),
        )
        .limit(1)
    )
