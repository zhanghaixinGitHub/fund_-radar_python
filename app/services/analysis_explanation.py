"""已发布评分的 DeepSeek 解释快照服务；不参与数值评分、回测或模型发布。"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.core.logging import get_logger
from app.db.session import get_engine
from app.models.analysis import AnalysisExplanationSnapshot
from app.repositories.analysis_explanation import (
    ExplanationSource,
    find_explanation_snapshot,
    find_latest_active_scored_source,
    find_latest_fund_explanation,
)
from app.schemas.analysis_summary import InternalFundExplanation, InternalFundExplanationEvidence
from app.services.deepseek_explanation_client import (
    DEEPSEEK_EXPLANATION_PROMPT_VERSION,
    DEEPSEEK_PROVIDER,
    DeepSeekExplanationClient,
    DeepSeekExplanationContent,
    DeepSeekExplanationError,
)

logger = get_logger(__name__)


class FundExplanationSourceNotReadyError(RuntimeError):
    """目标基金尚无已发布且已评分的本地事实，禁止触发外部解释调用。"""


@dataclass(frozen=True)
class FundExplanationGeneration:
    """单次生成的持久化结果；reused 表示未发生新的外部调用。"""

    explanation_id: str
    provider_model: str
    reused: bool


def validate_deepseek_explanation_configuration() -> str:
    """在排队前验证本地启用状态与密钥，仅返回受控模型名。"""
    client = DeepSeekExplanationClient()
    client.validate_configuration()
    return client.model


def generate_fund_explanation(fund_code: str) -> FundExplanationGeneration:
    """为一条已发布评分生成或复用解释快照；失败时不写半成品记录。"""
    client = DeepSeekExplanationClient()
    client.validate_configuration()
    source = _required_active_scored_source(fund_code)
    existing = _find_existing_snapshot(source, client.model)
    if existing is not None:
        return FundExplanationGeneration(
            explanation_id=str(existing.explanation_id), provider_model=existing.provider_model, reused=True
        )

    facts = _facts_for_model(source)
    generated = client.generate(facts)
    snapshot = _persist_snapshot(source, client.model, facts, generated)
    logger.info(
        "analysis_explanation.generate_fund_explanation >>> completed, fund_code=%s, forecast_id=%s, explanation_id=%s",
        source.forecast.fund_code,
        source.forecast.forecast_id,
        snapshot.explanation_id,
    )
    return FundExplanationGeneration(
        explanation_id=str(snapshot.explanation_id), provider_model=snapshot.provider_model, reused=False
    )


def get_published_fund_explanation(
    session: Session,
    *,
    fund_code: str,
    model_release_id: object,
) -> InternalFundExplanation | None:
    """将现有解释快照映射为基金详情白名单；读取不会发起外部调用。"""
    snapshot = find_latest_fund_explanation(session, fund_code=fund_code, model_release_id=model_release_id)
    return _to_internal_snapshot(snapshot) if snapshot is not None else None


def _required_active_scored_source(fund_code: str) -> ExplanationSource:
    """读取唯一允许进入 DeepSeek 的输入事实；缺失时拒绝而非尝试补齐。"""
    with Session(get_engine()) as session:
        source = find_latest_active_scored_source(session, fund_code)
    if source is None:
        raise FundExplanationSourceNotReadyError("EXPLANATION_SOURCE_NOT_READY")
    return source


def _find_existing_snapshot(source: ExplanationSource, provider_model: str) -> AnalysisExplanationSnapshot | None:
    """按预测、模型与提示词版本复用成功快照，防止重复点击重复计费。"""
    with Session(get_engine()) as session:
        return find_explanation_snapshot(
            session,
            forecast_id=source.forecast.forecast_id,
            provider=DEEPSEEK_PROVIDER,
            provider_model=provider_model,
            prompt_version=DEEPSEEK_EXPLANATION_PROMPT_VERSION,
        )


def _facts_for_model(source: ExplanationSource) -> dict[str, object]:
    """构造最小、稳定且无个人信息的结构化事实，不发送完整数据库记录或原始序列。"""
    forecast = source.forecast
    feature = source.feature
    return {
        "fund_code": forecast.fund_code,
        "as_of_date": forecast.as_of_date.isoformat(),
        "model_version": forecast.model_version,
        "feature_version": forecast.feature_version,
        "score_status": forecast.score_status,
        "direction": forecast.direction,
        "directional_probability": _decimal_text(forecast.directional_probability),
        "confidence": _decimal_text(forecast.confidence),
        "risk_level": forecast.risk_level,
        "max_drawdown_estimate": _decimal_text(forecast.max_drawdown_estimate),
        "score_explanation": forecast.explanation,
        "feature_completeness": _decimal_text(feature.completeness),
        "feature_metrics": _allowed_feature_metrics(feature.feature_payload),
        "product_boundary": "仅解释已发布评分，不构成交易建议。",
    }


def _persist_snapshot(
    source: ExplanationSource,
    provider_model: str,
    facts: dict[str, object],
    generated: DeepSeekExplanationContent,
) -> AnalysisExplanationSnapshot:
    """写入完整解释快照；并发重复命中唯一键时返回已存在快照。"""
    source_hash = _stable_hash(facts)
    content_payload = {
        "overview": generated.overview,
        "evidence": generated.evidence,
        "risk_notice": generated.risk_notice,
        "data_gap": generated.data_gap,
        "disclaimer": generated.disclaimer,
    }
    content_hash = _stable_hash(
        {
            "forecast_id": str(source.forecast.forecast_id),
            "provider": DEEPSEEK_PROVIDER,
            "provider_model": provider_model,
            "prompt_version": DEEPSEEK_EXPLANATION_PROMPT_VERSION,
            "source_hash": source_hash,
            "content": content_payload,
        }
    )
    with Session(get_engine()) as session:
        snapshot = AnalysisExplanationSnapshot(
            forecast_id=source.forecast.forecast_id,
            model_release_id=source.release.model_release_id,
            fund_code=source.forecast.fund_code,
            as_of_date=source.forecast.as_of_date,
            provider=DEEPSEEK_PROVIDER,
            provider_model=provider_model,
            prompt_version=DEEPSEEK_EXPLANATION_PROMPT_VERSION,
            provider_request_id=generated.provider_request_id,
            source_input_hash=source_hash,
            content_hash=content_hash,
            overview=generated.overview,
            evidence=list(generated.evidence),
            risk_notice=generated.risk_notice,
            data_gap=generated.data_gap,
            disclaimer=generated.disclaimer,
            prompt_tokens=generated.prompt_tokens,
            completion_tokens=generated.completion_tokens,
            generated_at=datetime.now(UTC),
        )
        session.add(snapshot)
        try:
            session.commit()
            session.refresh(snapshot)
            return snapshot
        except IntegrityError:
            session.rollback()
            existing = find_explanation_snapshot(
                session,
                forecast_id=source.forecast.forecast_id,
                provider=DEEPSEEK_PROVIDER,
                provider_model=provider_model,
                prompt_version=DEEPSEEK_EXPLANATION_PROMPT_VERSION,
            )
            if existing is None:
                raise
            return existing


def _to_internal_snapshot(snapshot: AnalysisExplanationSnapshot) -> InternalFundExplanation:
    """把 JSONB 证据转换为严格响应对象；异常历史数据不对用户端披露。"""
    evidence = tuple(
        InternalFundExplanationEvidence(label=item["label"], detail=item["detail"])
        for item in snapshot.evidence
        if isinstance(item, dict)
        and isinstance(item.get("label"), str)
        and isinstance(item.get("detail"), str)
    )
    if not evidence:
        raise DeepSeekExplanationError("EXPLANATION_SNAPSHOT_INVALID")
    return InternalFundExplanation(
        explanation_id=snapshot.explanation_id,
        forecast_id=snapshot.forecast_id,
        as_of_date=snapshot.as_of_date,
        provider=snapshot.provider,
        provider_model=snapshot.provider_model,
        prompt_version=snapshot.prompt_version,
        overview=snapshot.overview,
        evidence=evidence,
        risk_notice=snapshot.risk_notice,
        data_gap=snapshot.data_gap,
        disclaimer=snapshot.disclaimer,
        generated_at=snapshot.generated_at,
    )


def _allowed_feature_metrics(feature_payload: dict[str, object]) -> dict[str, str]:
    """只发送固定数值特征白名单，避免未来特征扩展把未审查文本输入大模型。"""
    raw_metrics = feature_payload.get("metrics")
    if not isinstance(raw_metrics, dict):
        return {}
    allowed_keys = ("return_20d", "volatility_20d", "max_drawdown_60d", "nav_observation_count")
    return {
        key: _decimal_text(raw_metrics.get(key))
        for key in allowed_keys
        if _decimal_text(raw_metrics.get(key)) is not None
    }


def _decimal_text(value: object) -> str | None:
    """以有限十进制文本传递数值，避免 NaN、布尔或对象进入提示词。"""
    if isinstance(value, bool) or value is None:
        return None
    try:
        decimal_value = Decimal(str(value))
    except (ArithmeticError, ValueError):
        return None
    return format(decimal_value, "f") if decimal_value.is_finite() else None


def _stable_hash(payload: dict[str, Any]) -> str:
    """对持久化解释输入和内容计算稳定哈希，不记录原始请求体。"""
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()
