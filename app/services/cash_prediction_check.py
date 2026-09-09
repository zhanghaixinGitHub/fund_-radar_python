"""独立的生成前检查：重读冻结报告，核验身份/指纹并解释为什么不能执行预测。"""

from datetime import UTC, datetime
from decimal import Decimal
from zoneinfo import ZoneInfo

from sqlalchemy import text
from sqlalchemy.orm import Session

from app.db.session import get_nav_preview_engine
from app.repositories.cash_reinvestment_research import find_research
from app.repositories.watchlist_prediction import read_prediction_inputs
from app.schemas.cash_prediction_check import CashComparisonCheck, CashPredictionCheck, CashPredictionCheckRequest
from app.schemas.cash_reinvestment_research import CashStoredResearch
from app.schemas.historical_nav_evaluation import BaselineMetrics
from app.services.cash_reinvestment_research import BLOCKERS, FUNDS, restore_research
from app.services.historical_nav_calibration import WINDOWS
from app.services.historical_nav_storage import HistoricalNavStorageError

BASELINES = frozenset(("ALWAYS_UP", "TRAIN_UP_FREQUENCY", "MOMENTUM_20D", "FIXED_MOMENTUM_SCORE"))


def _metrics_check(metrics: BaselineMetrics) -> None:
    """成绩必须有可比较的有限值；不能把缺失/NaN/样本错配解释为改善。"""
    if (
        metrics.sample_count <= 0
        or not 0 <= metrics.correct_count <= metrics.sample_count
        or not 0 <= metrics.actual_up_count <= metrics.sample_count
        or not 0 <= metrics.predicted_up_count <= metrics.sample_count
        or any(not value.is_finite() or not 0 <= value <= 1 for value in (metrics.accuracy, metrics.brier_score))
        or abs(metrics.accuracy - Decimal(metrics.correct_count) / metrics.sample_count) > Decimal("0.00000001")
    ):
        raise ValueError("invalid comparison metrics")


def inspect_cash_research(
    stored: CashStoredResearch,
) -> tuple[tuple[str, ...], tuple[CashComparisonCheck, ...], tuple[str, ...]]:
    """只解释已有成绩，不重新拟合、挑窗口、改变门槛或为当前研究授予正式发布资格。"""
    report = stored.report
    if tuple(w.window for w in report.windows) != WINDOWS:
        raise ValueError("fixed research windows mismatch")
    checks, incomplete = [], []
    for window in report.windows:
        if window.status != "EVALUATED":
            incomplete.append(window.window.window_id)
            continue
        if window.after is None or window.model is None:
            raise ValueError("evaluated window lacks model or scores")
        if len(window.baselines) != len(BASELINES) or {b.baseline_id for b in window.baselines} != BASELINES:
            raise ValueError("comparison baselines missing or duplicated")
        candidate = {"ALL": window.after.validation, **{f.fund_code: f.metrics for f in window.after.per_fund}}
        if len(window.after.per_fund) != len(FUNDS) or set(candidate) != {"ALL", *FUNDS}:
            raise ValueError("candidate fund coverage mismatch")
        for baseline in window.baselines:
            groups = {"ALL": baseline.validation, **{f.fund_code: f.metrics for f in baseline.per_fund}}
            if len(baseline.per_fund) != len(FUNDS) or groups.keys() != candidate.keys():
                raise ValueError("baseline fund coverage mismatch")
            for scope, actual in candidate.items():
                control = groups[scope]
                _metrics_check(actual)
                _metrics_check(control)
                if (actual.sample_count, actual.actual_up_count) != (control.sample_count, control.actual_up_count):
                    raise ValueError("comparison populations mismatch")
                accuracy_delta = actual.accuracy - control.accuracy
                brier_delta = actual.brier_score - control.brier_score
                checks.append(
                    CashComparisonCheck(
                        window_id=window.window.window_id,
                        scope=scope,
                        baseline_id=baseline.baseline_id,
                        sample_count=actual.sample_count,
                        accuracy_delta=accuracy_delta,
                        brier_delta=brier_delta,
                        strict_gain_observed=accuracy_delta > 0 and brier_delta < 0,
                    )
                )
    # 这几项缺口是已冻结协议的事实，不由分数改善或请求中的“通过”布尔值替代。
    blockers = list(BLOCKERS)
    if incomplete:
        blockers.append("ROLLING_WINDOWS_INCOMPLETE")
    if any(not c.strict_gain_observed for c in checks):
        blockers.append("REPEATABLE_BASELINE_GAIN_NOT_ESTABLISHED")
    return tuple(blockers), tuple(checks), tuple(incomplete)


def check_cash_prediction_in_session(
    session: Session, request: CashPredictionCheckRequest, *, now: datetime
) -> CashPredictionCheck:
    """复用调用方的一致性事务；自己只读，供预检和拒绝回执保存共用同一份判断。"""
    # 除指定报告外，只复用基金/来源及日期元数据查询，不调用旧发布或评分链路。
    fund, source, _, _ = read_prediction_inputs(
        session, request.fund_code, now.astimezone(ZoneInfo("Asia/Shanghai")).date(), include_research=False
    )
    if fund is None:
        raise HistoricalNavStorageError("FUND_NOT_FOUND", "基金不存在。", 404)
    if fund.fund_type != "STOCK" or request.fund_code not in FUNDS:
        raise HistoricalNavStorageError("CASH_PREDICTION_NOT_APPLICABLE", "当前只检查三只股票型试点。", 409)
    if fund.status != "ACTIVE" or fund.source_code != "TUSHARE_PRO_FUND" or source is None or not source.enabled:
        raise HistoricalNavStorageError("CASH_SOURCE_UNAVAILABLE", "基金或来源当前不可用。", 409)
    stored = restore_research(find_research(session, run_id=request.research_run_id))
    if stored.report.report_hash != request.expected_report_hash:
        raise HistoricalNavStorageError("REPORT_HASH_MISMATCH", "报告指纹不一致，请重新核对指定研究记录。", 409)
    counts = stored.report.preparation.fund_counts.get(request.fund_code, {})
    if not any(counts.get(stage, 0) > 0 for stage in ("TRAIN", "VALIDATION")):
        raise HistoricalNavStorageError("FUND_RESEARCH_MISMATCH", "指定报告没有这只基金的有效研究资料。", 409)
    try:
        blockers, comparisons, incomplete = inspect_cash_research(stored)
    except (ValueError, TypeError, KeyError, ArithmeticError) as error:
        raise HistoricalNavStorageError("CASH_RESEARCH_CORRUPTED", "报告比较数据校验失败。", 503) from error
    return CashPredictionCheck(
        checked_at=now,
        fund_code=request.fund_code,
        research_run_id=stored.run_id,
        report_hash=stored.report.report_hash,
        blocking_codes=blockers,
        comparisons=comparisons,
        incomplete_window_ids=incomplete,
    )


def check_cash_prediction(request: CashPredictionCheckRequest) -> CashPredictionCheck:
    """一个只读一致性事务，无净值数值、无特征/答案读取、无推理调用；未通过不生成预测。"""
    with Session(get_nav_preview_engine()) as session, session.begin():
        session.execute(text("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY"))
        return check_cash_prediction_in_session(session, request, now=datetime.now(UTC))
