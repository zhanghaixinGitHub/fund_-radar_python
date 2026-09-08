"""独立的严格日历窗口规则，不修改旧构建器、旧标签、旧训练接口或原始净值。"""

from bisect import bisect_left
from datetime import date
from time import perf_counter

from sqlalchemy import text
from sqlalchemy.orm import Session

from app.db.session import get_nav_preview_engine
from app.repositories.feature_snapshot import FeatureSourceReadiness
from app.repositories.historical_nav import read_historical_nav_source
from app.repositories.trading_nav_window import NavDatePoint, read_nav_dates
from app.schemas.trading_nav_window import (
    CalendarProvenance,
    NavDateIssue,
    TradingNavWindowRequest,
    TradingNavWindowResponse,
)
from app.services.trading_calendar import CalendarCoverageError, TradingCalendar, load_calendar

LIMITATIONS = (
    "只核对日期，不生成特征/标签，不验证净值数值、累计/复权或分红口径。",
    "cutoff按自然日结束解释；不具备盘中时刻或历史首次公告版本回放证据。",
    "未来从cutoff之后开始数20个沪深市场交易日，不是从旧净值日计数，也不是20个自然日。",
    "日历是官方休市规则派生的静态研究快照，不是实时开停市、基金申赎或境外交易日历。",
    "来源非交易日净值保留但不参与计数；缺失交易日不补值、不顺延、不自动同步。",
    "2025只可检查日期元数据，不取净值数值、方向标签或模型分数；旧实验不变。",
)


def window_read_bounds(calendar: TradingCalendar, cutoff: date) -> tuple[date, date]:
    """先固定日期，再读库；为最多落后1场的起点预留完整61个历史交易日。"""
    latest = calendar.at_or_before_index(cutoff)
    future = calendar.future_sessions(cutoff)
    if latest < 61:
        raise CalendarCoverageError("已核验日历不足以覆盖完整61日历史及起点公告滞后，不向前猜日期。")
    return calendar.sessions[latest - 61], future[-1]


def _issues(
    dates: tuple[date, ...], by_date: dict[date, NavDatePoint], cutoff: date | None
) -> tuple[NavDateIssue, ...]:
    result = []
    for day in dates:
        point = by_date.get(day)
        reason = None
        if point is None:
            reason = "MISSING_NAV"
        elif point.ann_date is None:
            reason = "MISSING_ANN_DATE"
        elif point.ann_date < day:
            reason = "ANN_BEFORE_NAV_DATE"
        elif cutoff is not None and point.ann_date > cutoff:
            reason = "NOT_KNOWN_AT_CUTOFF"
        if reason:
            result.append(NavDateIssue(nav_date=day, reason=reason))
    return tuple(result)


def build_trading_nav_window(
    request: TradingNavWindowRequest,
    source: FeatureSourceReadiness,
    points: tuple[NavDatePoint, ...],
    calendar: TradingCalendar,
) -> TradingNavWindowResponse:
    """纯日期计算：历史只用截止时已知元数据，未来缺失不影响历史起点和窗口。"""
    cutoff = request.cutoff_date
    start, end = window_read_bounds(calendar, cutoff)
    if (
        len(points) > 192
        or any(not start <= p.nav_date <= end for p in points)
        or any(a.nav_date >= b.nav_date for a, b in zip(points[:-1], points[1:], strict=True))
    ):
        raise ValueError("NAV date points are oversized, unordered, duplicated or outside request")
    sessions = calendar.sessions
    session_set = set(sessions)
    by_date = {p.nav_date: p for p in points}
    known = tuple(
        p for p in points if p.nav_date in session_set and p.ann_date is not None and p.nav_date <= p.ann_date <= cutoff
    )
    anchor = known[-1] if known else None
    anchor_index = bisect_left(sessions, anchor.nav_date) if anchor else None
    lag = calendar.at_or_before_index(cutoff) - anchor_index if anchor else None
    anchor_issue = "NO_KNOWN_TRADING_NAV" if anchor is None else "STALE_ANCHOR_OVER_ONE_SESSION" if lag > 1 else None
    # 过旧起点的历史可能超出本次读取范围；直接拒绝，不把未查询的日期伪报成数据库缺数。
    history = sessions[anchor_index - 60 : anchor_index + 1] if anchor_issue is None and anchor_index >= 60 else ()
    if anchor_issue is None and len(history) != 61:
        anchor_issue = "CALENDAR_HISTORY_SHORTAGE"
    historical_issues = _issues(history, by_date, cutoff)
    future = calendar.future_sessions(cutoff)
    future_issues = _issues(future, by_date, None)
    # 完整未来元数据只是离线核验；绝不回写历史known/anchor/history。
    future_available = max(by_date[d].ann_date for d in future) if not future_issues else None
    legacy_future = tuple(p for p in points if anchor and p.nav_date > anchor.nav_date)
    used_years = {d.year for d in (*history, *future, cutoff)}
    notices = tuple(y for y in calendar.definition.years if y.year in used_years)
    return TradingNavWindowResponse(
        status="INPUT_DATES_INCOMPLETE"
        if anchor_issue or historical_issues
        else "FUTURE_DATES_INCOMPLETE"
        if future_issues
        else "WINDOW_DATES_COMPLETE",
        fund_code=request.fund_code,
        cutoff_date=cutoff,
        source_code=source.source_code,
        source_sync_run_id=source.source_sync_run_id,
        calendar=CalendarProvenance(
            **calendar.definition.model_dump(
                include={"version", "market", "coverage_start", "coverage_end", "reviewed_on", "construction"}
            ),
            content_hash=calendar.content_hash,
            source_urls=tuple(url for n in notices for url in n.sources),
            future_schedule_known_at_cutoff=all(
                n.announced_on <= cutoff for n in notices if n.year in {d.year for d in future}
            ),
        ),
        anchor_nav_date=anchor.nav_date if anchor else None,
        anchor_ann_date=anchor.ann_date if anchor else None,
        anchor_lag_sessions=lag,
        anchor_issue=anchor_issue,
        history_dates=history,
        history_issues=historical_issues,
        future_dates=future,
        future_end_date=end,
        future_issues=future_issues,
        future_navs_available_at=future_available,
        ignored_non_trading_nav_dates=tuple(p.nav_date for p in points if p.nav_date not in session_set),
        anchor_based_20th_trading_date=calendar.future_sessions(anchor.nav_date)[-1] if anchor else None,
        legacy_20th_nav_date_from_anchor=legacy_future[19].nav_date if len(legacy_future) >= 20 else None,
        limitations=LIMITATIONS,
    )


def preview_trading_nav_window(request: TradingNavWindowRequest) -> TradingNavWindowResponse:
    """旧只读池、一次一致性事务；基金/来源检查后只SELECT日期元数据，不取价格和样本表。"""
    calendar = load_calendar()
    start, end = window_read_bounds(calendar, request.cutoff_date)
    deadline = perf_counter() + 15
    with Session(get_nav_preview_engine()) as session, session.begin():
        session.execute(text("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY"))
        source = read_historical_nav_source(session, fund_code=request.fund_code)
        points = read_nav_dates(session, fund_code=request.fund_code, source_id=source.source_id, start=start, end=end)
    if perf_counter() >= deadline:
        raise TimeoutError("trading NAV date preview exceeded phase budget")
    return build_trading_nav_window(request, source, points, calendar)
