"""从 fund_ai 已持久化控制面构造基金详情需要的只读摘要。"""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.session import get_engine
from app.models.analysis import AnalysisModelRelease, BacktestRun
from app.models.fund import FundShareClass
from app.schemas.analysis_summary import (
    InternalBacktestSummary,
    InternalFundAnalysisSummary,
    InternalModelAnalysisSummary,
)


def get_fund_analysis_summary(fund_code: str) -> InternalFundAnalysisSummary:
    """读取基金的已发布模型摘要；候选和未准入模型始终不向用户端披露。"""
    with Session(get_engine()) as session:
        fund_type = session.scalar(
            select(FundShareClass.fund_type).where(FundShareClass.fund_code == fund_code)
        )
        if fund_type is None:
            return _unavailable(fund_code, None, "该基金暂无可用分析资料。")

        active_release = session.scalar(
            select(AnalysisModelRelease)
            .where(
                AnalysisModelRelease.fund_type == fund_type,
                AnalysisModelRelease.release_status == "ACTIVE",
            )
            .order_by(AnalysisModelRelease.effective_at.desc(), AnalysisModelRelease.model_release_id.desc())
            .limit(1)
        )
        if active_release is not None:
            return _published_summary(
                fund_code,
                fund_type,
                active_release,
                availability_status="ACTIVE",
                message="模型已发布；评分仅在特征完整且数据新鲜时展示。",
                session=session,
            )

        paused_release = session.scalar(
            select(AnalysisModelRelease)
            .where(
                AnalysisModelRelease.fund_type == fund_type,
                AnalysisModelRelease.release_status == "SUSPENDED",
            )
            .order_by(AnalysisModelRelease.suspended_at.desc(), AnalysisModelRelease.model_release_id.desc())
            .limit(1)
        )
        if paused_release is not None:
            return _published_summary(
                fund_code,
                fund_type,
                paused_release,
                availability_status="MODEL_PAUSED",
                message="模型当前已暂停；历史摘要只读保留，不生成新的方向性评分或提醒。",
                session=session,
            )
        return _unavailable(fund_code, fund_type, "当前没有已发布且可用于该基金类型的模型。")


def _published_summary(
    fund_code: str,
    fund_type: str,
    release: AnalysisModelRelease,
    *,
    availability_status: str,
    message: str,
    session: Session,
) -> InternalFundAnalysisSummary:
    """把已发布或暂停发布及其关联回测映射为固定字段，缺失字段保持空值。"""
    backtest = session.get(BacktestRun, release.backtest_run_id)
    return InternalFundAnalysisSummary(
        fund_code=fund_code,
        fund_type=fund_type,
        availability_status=availability_status,
        message=message,
        model=InternalModelAnalysisSummary(
            model_release_id=release.model_release_id,
            model_version=release.model_version,
            feature_version=release.feature_version,
            release_status=release.release_status,
            effective_at=release.effective_at,
            suspended_at=release.suspended_at,
        ),
        backtest=_to_backtest_summary(backtest) if backtest is not None else None,
    )


def _unavailable(fund_code: str, fund_type: str | None, message: str) -> InternalFundAnalysisSummary:
    """未存在合规发布模型时不返回候选版本、回测指标或失败原因。"""
    return InternalFundAnalysisSummary(
        fund_code=fund_code,
        fund_type=fund_type,
        availability_status="MODEL_UNAVAILABLE",
        message=message,
        model=None,
        backtest=None,
    )


def _to_backtest_summary(backtest: BacktestRun) -> InternalBacktestSummary:
    """从已落库 JSON 指标中仅白名单读取用户可读字段，避免契约漂移和意外泄露。"""
    metrics = backtest.metrics or {}
    baselines = backtest.baselines or {}
    return InternalBacktestSummary(
        run_id=backtest.run_id,
        status=backtest.status,
        publication_status=backtest.publication_status,
        window_start=backtest.window_start,
        window_end=backtest.window_end,
        test_start=backtest.test_start,
        test_end=backtest.test_end,
        data_cutoff=_as_date(metrics.get("data_cutoff")),
        fee_rate=backtest.fee_rate,
        sample_count=_as_int(metrics.get("sample_count")),
        rolling_fold_count=_as_int(metrics.get("rolling_fold_count")),
        annualized_return=_as_decimal(metrics.get("annualized_return")),
        max_drawdown=_as_decimal(metrics.get("max_drawdown")),
        volatility=_as_decimal(metrics.get("volatility")),
        hit_rate=_as_decimal(metrics.get("hit_rate")),
        long_hold_result=_as_decimal(baselines.get("long_hold_result")),
        dca_result=_as_decimal(baselines.get("dca_result")),
        benchmark_status=_as_text(baselines.get("benchmark_status")),
        benchmark_result=_as_decimal(baselines.get("benchmark_result")),
        completed_at=backtest.finished_at,
    )


def _as_date(value: Any) -> date | None:
    """只接受 ISO 纯日期，避免无时区时间字符串被错误解释为数据截至日。"""
    if not isinstance(value, str):
        return None
    try:
        return date.fromisoformat(value)
    except ValueError:
        return None


def _as_decimal(value: Any) -> Decimal | None:
    """将有限数值转换为 Decimal；JSON 缺失、布尔与非法文本均保持空值。"""
    if isinstance(value, bool) or value is None:
        return None
    try:
        decimal_value = Decimal(str(value))
    except (ArithmeticError, ValueError):
        return None
    return decimal_value if decimal_value.is_finite() else None


def _as_int(value: Any) -> int | None:
    """样本量与折数必须是非负整数，避免将浮点或布尔值当作计数展示。"""
    if isinstance(value, bool):
        return None
    if isinstance(value, int) and value >= 0:
        return value
    return None


def _as_text(value: Any) -> str | None:
    """状态字段只接受非空文本，保证前端不接收任意 JSON 对象。"""
    return value.strip() if isinstance(value, str) and value.strip() else None
