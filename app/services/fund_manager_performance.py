"""经理任期展示回报：单位净值与已实施现金分红自行连乘，不复用预测或评分结论。"""

import hashlib
import json
import logging
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from decimal import ROUND_HALF_UP, Decimal, localcontext
from functools import lru_cache
from pathlib import Path
from zoneinfo import ZoneInfo

from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.core.middleware import get_trace_id
from app.repositories.fund_manager_performance import (
    MAX_DIVIDENDS,
    MAX_NAV_ROWS,
    ManagerDividend,
    ManagerNavPoint,
    read_manager_performance_inputs,
)
from app.repositories.fund_read import FundWatchlistDetailSnapshot
from app.schemas.fund import InternalFundManagerPerformance, InternalFundManagerPeriod
from app.services.trading_calendar import load_calendar, load_current_calendar

HISTORY_FILE = Path(__file__).resolve().parents[1] / "data/calendars/cn_a_share_2015_2020_research_v1.json"
HISTORY_HASH = "798a999a94f791cc7f856324ba4968738eee051a4b379434e4a6f8cfedce68ae"
# 累计净值只用于现金变化核对；来源复权因子只用于拒绝无法解释的拆分/折算，均不作为收益分子分母。
CASH_TOLERANCE = Decimal("0.000001")
FACTOR_TOLERANCE = Decimal("0.001")  # 兼容旧来源4位净值舍入；超过10基点的未解释调整拒绝计算。
logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def manager_calendar() -> tuple[date, ...]:
    """复用已核验的交易所休市事实，不导入研究执行器，也不修改其范围或指纹。"""
    raw = HISTORY_FILE.read_bytes()
    if len(raw) > 65536:
        raise ValueError("manager calendar exceeds limit")
    definition = json.loads(raw)
    normalized = json.dumps(definition, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    if hashlib.sha256(normalized.encode()).hexdigest() != HISTORY_HASH:
        raise ValueError("manager calendar changed without review")
    start, end = map(date.fromisoformat, (definition["coverage_start"], definition["coverage_end"]))
    closed = set()
    for year in definition["years"]:
        for left, right in year["closed_ranges"]:
            left, right = date.fromisoformat(left), date.fromisoformat(right)
            closed.update(left + timedelta(days=offset) for offset in range((right - left).days + 1))
    older = tuple(start + timedelta(days=offset) for offset in range((end - start).days + 1))
    return tuple(day for day in older if day.weekday() < 5 and day not in closed) + (
        *load_calendar().sessions, *load_current_calendar().sessions,
    )


def unavailable(reason: str, *, co_managed: bool = False) -> InternalFundManagerPerformance:
    """不可计算是资料状态，不输出零回报或部分任期结果。"""
    return InternalFundManagerPerformance(
        status="UNAVAILABLE", reason=reason, includes_co_management=co_managed,
    )


def _positive(value: Decimal | None) -> bool:
    return value is not None and value.is_finite() and value > 0


def calculate_tenure_return(
    *, begin: date | None, end: date | None, latest_nav_date: date, today: date,
    navs: tuple[ManagerNavPoint, ...], dividends: tuple[ManagerDividend, ...],
    sessions: tuple[date, ...], co_managed: bool = False,
) -> InternalFundManagerPerformance:
    """首个任职净值收盘至任期末净值收盘，现金按净值除权日单位净值再投。

    任职起止自然日均纳入区间；交易日历检查应有开市日缺口，季末周末额外披露的净值也保留。
    在任经理仅算至最新已披露净值，离任经理必须覆盖完整任期；缺失、重复、分红冲突均拒绝。
    连乘(当日单位净值+当日每份现金)/前日单位净值，首日现金属于前一收益段，不重复计入。
    """
    def fail(reason: str) -> InternalFundManagerPerformance:
        return unavailable(reason, co_managed=co_managed)

    if begin is None or begin > today or (end is not None and end < begin):
        return fail("任职日期不完整或尚未生效")
    cutoff = end if end is not None and end <= today else min(latest_nav_date, today)
    if not sessions or begin < sessions[0] or cutoff > sessions[-1]:
        return fail("该任期的净值日历尚未核实")
    expected = tuple(day for day in sessions if begin <= day <= cutoff)
    if expected and expected[-1] > latest_nav_date:
        return fail("离任时的净值尚未补齐")
    selected = tuple(point for point in navs if begin <= point.nav_date <= cutoff)
    values = {point.nav_date: point for point in selected}
    if len(values) != len(selected) or any(day not in values for day in expected):
        return fail("任期内净值缺失或存在重复记录")
    # 基金可能在季末周末披露净值；不能将合法的额外披露误判为重复，也不能静默删去。
    dates = tuple(sorted(values))
    if len(dates) < 2:
        return fail("任期内有效净值不足两个日期")
    for day in dates:
        point = values[day]
        if not all(_positive(value) for value in (point.unit_nav, point.accumulated_nav, point.adjusted_nav)):
            return fail("任期内净值或分红核对资料不完整")
        if point.ann_date is None or not day <= point.ann_date <= today:
            return fail("部分净值的披露日期尚未核实")

    cash_by_date: dict[date, Decimal] = {}
    for event in dividends:
        possible = tuple(day for day in (event.ex_date, event.nav_ex_date) if day is not None)
        if possible and not any(dates[0] < day <= dates[-1] for day in possible):
            continue
        effective = event.nav_ex_date or event.ex_date
        if effective is None or (event.ex_date and event.nav_ex_date and event.ex_date != event.nav_ex_date):
            return fail("分红生效日期待核实")
        if effective not in dates[1:] or event.process_status != "实施":
            return fail("任期内部分分红尚未完成核对")
        if event.ann_date is None or max(event.ann_date, event.implementation_ann_date or event.ann_date) > today:
            return fail("分红披露日期待核实")
        if event.cash_dividend is None or not event.cash_dividend.is_finite() or event.cash_dividend < 0:
            return fail("分红金额待核实")
        if effective in cash_by_date:
            return fail("同日分红存在重复或多条记录")
        cash_by_date[effective] = event.cash_dividend

    with localcontext() as context:
        context.prec = 40
        growth = Decimal(1)
        for previous_day, day in zip(dates[:-1], dates[1:], strict=True):
            previous, point = values[previous_day], values[day]
            cash = cash_by_date.get(day, Decimal(0))
            # 没有事件时也要核对累计现金变化，不能把漏同步的分红默认为0。
            cash_change = (point.accumulated_nav - point.unit_nav) - (
                previous.accumulated_nav - previous.unit_nav
            )
            if abs(cash_change - cash) > CASH_TOLERANCE:
                return fail("净值与分红记录不一致，暂无法计算")
            factor_change = (point.adjusted_nav / point.unit_nav) / (
                previous.adjusted_nav / previous.unit_nav
            )
            if abs(factor_change - (1 + cash / point.unit_nav)) > FACTOR_TOLERANCE:
                return fail("存在尚未核实的净值调整，暂无法计算")
            growth *= (point.unit_nav + cash) / previous.unit_nav
        rate = (growth - 1).quantize(Decimal("0.000000000001"), rounding=ROUND_HALF_UP)
    return InternalFundManagerPerformance(
        status="AVAILABLE", return_rate=rate, start_date=dates[0], end_date=dates[-1],
        includes_co_management=co_managed,
    )


@dataclass(frozen=True)
class ManagerReturnBundle:
    """同一批输入生成的个人完整任期与团队阶段回报；不相加百分比，不重复读库。"""

    managers: tuple[InternalFundManagerPerformance, ...]
    periods: tuple[InternalFundManagerPeriod, ...]


def manager_periods(managers, today: date) -> tuple[tuple[date, date | None, tuple[str, ...]], ...]:
    """按加入日、离任次日切分团队；同名重复和续任不会制造额外阶段，最近阶段在前。"""
    events: dict[date, dict[str, int]] = {}

    def add(day: date, name: str, delta: int):
        if day > today:
            return
        changes = events.setdefault(day, {})
        changes[name] = changes.get(name, 0) + delta

    for item in managers:
        assignment = item.assignment
        name, begin, end = assignment.manager_name.strip(), assignment.begin_date, assignment.end_date
        if not name or begin is None or begin > today or (end is not None and end < begin):
            continue
        add(begin, name, 1)
        if end is not None and end < today:
            add(end + timedelta(days=1), name, -1)
    boundaries, active, periods = sorted(events), {}, []
    for index, begin in enumerate(boundaries):
        for name, delta in events[begin].items():
            count = active.get(name, 0) + delta
            if count > 0:
                active[name] = count
            else:
                active.pop(name, None)
        end = boundaries[index + 1] - timedelta(days=1) if index + 1 < len(boundaries) else None
        names = tuple(sorted(active))
        if periods and periods[-1][2] == names:
            periods[-1] = (periods[-1][0], end, names)
        else:
            periods.append((begin, end, names))
    return tuple(reversed(periods))


def build_manager_return_bundle(
    session: Session, snapshot: FundWatchlistDetailSnapshot, *, today: date | None = None,
) -> ManagerReturnBundle:
    """完整任期与团队阶段共用两次有界查询和同一公式，分别输出真实净值端点。"""
    today = today or datetime.now(ZoneInfo("Asia/Shanghai")).date()
    managers = snapshot.managers
    periods = manager_periods(managers, today)

    def all_missing(reason: str):
        return ManagerReturnBundle(
            tuple(unavailable(reason) for _ in managers),
            tuple(InternalFundManagerPeriod(
                begin_date=begin, end_date=end, manager_names=names,
                performance=unavailable(reason, co_managed=len(names) > 1),
            ) for begin, end, names in periods),
        )

    if not managers:
        return ManagerReturnBundle((), ())
    detail = snapshot.detail
    profile = snapshot.profile.profile if snapshot.profile else None
    if (detail.fund.fund_type not in {"STOCK", "MIXED", "BOND", "INDEX"}
            or (profile and "QDII" in (profile.source_fund_type or "").upper())):
        return all_missing("该基金的净值及分红口径尚待核实")
    if detail.source_code != "TUSHARE_PRO_FUND" or detail.nav_date is None:
        return all_missing("暂无可核验的同来源净值")
    if "MARKET_DETAIL_DIVIDEND" not in snapshot.succeeded_sync_types:
        return all_missing("分红资料尚未补齐")
    eligible = [item for item in managers if item.source_code == detail.source_code]
    starts = [item.assignment.begin_date for item in eligible if item.assignment.begin_date is not None]
    if not starts:
        return all_missing("任职日期暂缺")
    try:
        sessions = manager_calendar()
    except (ValueError, OSError):
        return all_missing("净值日历暂时不可用")
    start, end = min(starts), min(detail.nav_date, today)
    # 早于日历覆盖的任期逐条降级，其余任期仍可正常计算，不能把超长历史静默截断为完整任期。
    try:
        # 独立保存点使回报查询超时只影响本区块，不让事务失败连带阻断已经读到的基金资料。
        with session.begin_nested():
            navs, dividends = read_manager_performance_inputs(
                session, fund_code=detail.fund.fund_code, source_id=eligible[0].assignment.source_id,
                start=max(start, sessions[0]), end=end,
            )
    except SQLAlchemyError:
        logger.exception(
            "fund_manager_performance.build_manager_performances   >>> return input read failed, "
            "fund_code=%s, trace_id=%s", detail.fund.fund_code, get_trace_id(),
        )
        return all_missing("任期回报暂时不可用，请稍后重试")
    if len(navs) > MAX_NAV_ROWS or len(dividends) > MAX_DIVIDENDS:
        return all_missing("任期资料超出当前可核对范围")
    results = []
    for item in managers:
        assignment = item.assignment
        if item.source_code != detail.source_code:
            results.append(unavailable("任职与净值来源尚未完成核对"))
            continue
        co_managed = any(
            other.assignment.manager_name != assignment.manager_name
            and other.assignment.begin_date is not None and assignment.begin_date is not None
            and other.assignment.begin_date <= min(assignment.end_date or end, end)
            and (other.assignment.end_date or end) >= assignment.begin_date
            for other in eligible
        )
        results.append(calculate_tenure_return(
            begin=assignment.begin_date, end=assignment.end_date, latest_nav_date=end, today=today,
            navs=navs, dividends=dividends, sessions=sessions, co_managed=co_managed,
        ))
    stage_results = []
    for begin, period_end, names in periods:
        performance = calculate_tenure_return(
            begin=begin, end=period_end, latest_nav_date=end, today=today,
            navs=navs, dividends=dividends, sessions=sessions, co_managed=len(names) > 1,
        ) if names else unavailable("该阶段的管理人资料暂缺")
        if performance.reason:
            performance = performance.model_copy(update={"reason": performance.reason.replace(
                "任期", "阶段"
            ).replace("任职", "阶段").replace("离任时", "阶段末期")})
        stage_results.append(InternalFundManagerPeriod(
            begin_date=begin, end_date=period_end, manager_names=names, performance=performance,
        ))
    return ManagerReturnBundle(tuple(results), tuple(stage_results))


def build_manager_performances(
    session: Session, snapshot: FundWatchlistDetailSnapshot, *, today: date | None = None,
) -> tuple[InternalFundManagerPerformance, ...]:
    """兼容已有完整任期调用方；详情使用bundle一次取得两种回报，避免增加输入查询。"""
    return build_manager_return_bundle(session, snapshot, today=today).managers
