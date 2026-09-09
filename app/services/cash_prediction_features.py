"""预测用输入只计算已经知道的历史，不为得到输入去读取未来答案。"""

from datetime import date, datetime
from time import perf_counter
from zoneinfo import ZoneInfo

from sqlalchemy import text
from sqlalchemy.orm import Session

from app.db.session import get_nav_preview_engine
from app.repositories.cash_prediction_features import read_cash_history_inputs, validate_history_read_bounds
from app.repositories.cash_reinvestment_samples import CashDividend, CashNavPoint
from app.repositories.feature_snapshot import FeatureSourceReadiness
from app.repositories.historical_nav import HistoricalNavPreviewReadError, read_historical_nav_source
from app.repositories.trading_nav_window import NavDatePoint
from app.schemas.cash_prediction_features import CashPredictionFeature, CashPredictionFeatureRequest
from app.schemas.cash_reinvestment_samples import CashSampleIssue
from app.services.cash_reinvestment_samples import _hash, build_cash_history_feature
from app.services.trading_calendar import CalendarCoverageError, TradingCalendar, load_prediction_calendar
from app.services.trading_nav_window import build_history_date_window


def cash_history_bounds(calendar: TradingCalendar, request: CashPredictionFeatureRequest) -> tuple[date, date]:
    """预留61点历史和1日公告滞后；不计算未来20日，也不查询未来日历或价格。"""
    latest = calendar.at_or_before_index(request.cutoff_date)
    if latest < 61:
        raise CalendarCoverageError("已核验日历不能覆盖61日历史及1日起点滞后。")
    start = calendar.sessions[latest - 61]
    validate_history_read_bounds(start, request.cutoff_date)
    return start, request.cutoff_date


def build_cash_prediction_feature(
    request: CashPredictionFeatureRequest,
    source: FeatureSourceReadiness,
    nav: tuple[CashNavPoint, ...],
    events: tuple[CashDividend, ...],
    calendar: TradingCalendar,
) -> CashPredictionFeature:
    """纯计算入口；同cutoff和来源时，输出的payload/hash应与旧现金样本的历史部分完全相同。"""
    request = CashPredictionFeatureRequest.model_validate(request.model_dump())
    start, end = cash_history_bounds(calendar, request)
    if source.source_code != "TUSHARE_PRO_FUND":
        raise HistoricalNavPreviewReadError("CASH_SOURCE_UNSUPPORTED", "现金口径只核验既有Tushare来源。")
    if (
        len(nav) > 192
        or any(not start <= p.nav_date <= end for p in nav)
        or any(a.nav_date >= b.nav_date for a, b in zip(nav[:-1], nav[1:], strict=True))
        or len(events) > 100
        or len({e.event_key for e in events}) != len(events)
    ):
        raise ValueError("cash prediction history is oversized, duplicated, unordered or outside window")
    window = build_history_date_window(
        request.cutoff_date,
        tuple(NavDatePoint(p.nav_date, p.ann_date) for p in nav),
        calendar,
    )
    issues = [CashSampleIssue(day=p.nav_date, code=p.reason) for p in window.issues]
    if window.anchor_issue:
        issues.append(CashSampleIssue(day=window.anchor.nav_date if window.anchor else None, code=window.anchor_issue))
    feature = None
    if not issues:
        feature, calculation_issues = build_cash_history_feature(
            fund_code=request.fund_code,
            cutoff_date=request.cutoff_date,
            history_dates=window.dates,
            source=source,
            nav={p.nav_date: p for p in nav},
            events=events,
            calendar=calendar,
        )
        issues.extend(calculation_issues)
    sessions = set(calendar.sessions)
    return CashPredictionFeature(
        status="INPUT_READY" if feature else "DATA_INSUFFICIENT",
        fund_code=request.fund_code,
        cutoff_date=request.cutoff_date,
        anchor_nav_date=window.anchor.nav_date if window.anchor else None,
        anchor_lag_sessions=window.lag_sessions,
        history_dates=window.dates,
        input_issues=tuple(issues),
        feature_payload=feature,
        feature_hash=_hash(feature) if feature else None,
        ignored_non_trading_nav_dates=tuple(p.nav_date for p in nav if p.nav_date not in sessions),
    )


def prepare_cash_feature_read(request: CashPredictionFeatureRequest):
    """查询前完成日期和2025保护；生成器可在开库前复用，不因新增写入入口放宽。"""
    request = CashPredictionFeatureRequest.model_validate(request.model_dump())
    # 公告只有日期，所以当天尚未结束时不能声称知道当日结束的全部信息。
    if request.cutoff_date >= datetime.now(ZoneInfo("Asia/Shanghai")).date():
        raise HistoricalNavPreviewReadError("CUTOFF_DAY_NOT_CLOSED", "只能使用已经结束的自然日作为信息截止日。")
    calendar = load_prediction_calendar(request.cutoff_date)
    start, end = cash_history_bounds(calendar, request)  # 日历不足/2025保护必须在开库前拒绝。
    return request, calendar, start, end


def read_cash_prediction_feature_in_session(
    session: Session, request: CashPredictionFeatureRequest, *, deadline: float
) -> CashPredictionFeature:
    """只读调用方的一致性快照；生成器在同一事务中完成授权、输入读取和结果写入。"""
    request, calendar, start, end = prepare_cash_feature_read(request)
    source = read_historical_nav_source(session, fund_code=request.fund_code)
    if source.source_code != "TUSHARE_PRO_FUND":
        raise HistoricalNavPreviewReadError("CASH_SOURCE_UNSUPPORTED", "现金口径只核验既有Tushare来源。")
    nav, events = read_cash_history_inputs(
        session, fund_code=request.fund_code, source_id=source.source_id, start=start, cutoff=end
    )
    if perf_counter() >= deadline:
        raise TimeoutError("cash prediction feature read budget exceeded")
    return build_cash_prediction_feature(request, source, nav, events, calendar)


def read_cash_prediction_feature(request: CashPredictionFeatureRequest) -> CashPredictionFeature:
    """有界只读适配器，不作为额外HTTP或页面入口。"""
    prepare_cash_feature_read(request)  # 日历不足/2025保护在开库前拒绝。
    with Session(get_nav_preview_engine()) as session, session.begin():
        session.execute(text("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY"))
        return read_cash_prediction_feature_in_session(session, request, deadline=perf_counter() + 15)
