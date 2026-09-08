"""比较口径，不择优训练：固定算式、只读当前快照、明确无法核准的部分。"""

import hashlib
import json
from collections import defaultdict
from dataclasses import asdict
from datetime import date
from decimal import ROUND_HALF_UP, Decimal, localcontext
from time import perf_counter

from sqlalchemy import text
from sqlalchemy.orm import Session

from app.db.session import get_nav_preview_engine
from app.repositories.feature_snapshot import FeatureSourceReadiness
from app.repositories.historical_nav import HistoricalNavPreviewReadError, read_historical_nav_source
from app.repositories.nav_basis_audit import BasisDividend, BasisNavPoint, read_basis_inputs
from app.schemas.nav_basis_audit import (
    BasisAuditIssue,
    BasisPeriodComparison,
    DividendBasisComparison,
    NavBasisAuditRequest,
    NavBasisAuditResponse,
)
from app.services.trading_calendar import CalendarCoverageError, TradingCalendar, load_calendar

_ONE = Decimal(1)
_BPS = Decimal(10000)
_SOURCE_COLUMNS = ("unit_nav", "accumulated_nav", "adjusted_nav")
EVIDENCE_URLS = ("https://tushare.pro/document/2?doc_id=119", "https://tushare.pro/document/2?doc_id=120")
LIMITATIONS = (
    "供应商字段说明只标注adj_nav为复权单位净值，未给出该字段的完整复权算法。",
    "现金算式明确假设现有分红事件完整、现金于当日按单位净值再投，不含申赎费用、到账延迟或个人收益。",
    "无事件日暂按现金0作条件性对照；事件表无记录、累计分红为空不能证明历史没有分红。",
    "除现金分红外的拆分/折算及来源修订仍未完整核验；数值接近不等于供应商公式已确认。",
    "当前源表会覆盖更新，ann_date和当前同步水位不能证明历史首次公开版本。",
    "2025数值及测试成绩不开放；本接口不是cutoff时点回放，不制作20日标签或模型输入。",
    "三种口径不能自动互相替换；未发现大差异也不批准训练或发布。",
)


def audit_dates(calendar: TradingCalendar, request: NavBasisAuditRequest) -> tuple[date, ...]:
    """返回额外一个前置交易日+请求范围交易日；前置日仅用作首个日收益的分母。"""
    days = tuple(d for d in calendar.sessions if request.start_date <= d <= request.end_date)
    if not days:
        raise CalendarCoverageError("请求范围没有交易日，请选择包含交易日的日期范围。")
    index = calendar.at_or_before_index(days[0])
    if index < 1:
        raise CalendarCoverageError("日历缺少审计首日前置交易日。")
    return (calendar.sessions[index - 1], *days)


def _positive(value: Decimal | None) -> bool:
    return value is not None and value.is_finite() and value > 0


def _number(value: Decimal) -> str:
    # 固定精度只用于输出；差异阈值和连乘使用未舍入值，不按漂亮的小数挑口径。
    return format(value.quantize(Decimal("0.000000000001"), rounding=ROUND_HALF_UP), "f")


def _events_by_date(
    events: tuple[BasisDividend, ...], days: tuple[date, ...], issues: list[BasisAuditIssue]
) -> tuple[dict[date, Decimal], set[date]]:
    grouped = defaultdict(list)
    invalid = set()
    for event in events:
        day = event.nav_ex_date or event.ex_date
        reason = None
        if day is None:
            reason = "DIVIDEND_EFFECTIVE_DATE_MISSING"
        elif event.nav_ex_date and event.ex_date and event.nav_ex_date != event.ex_date:
            reason = "DIVIDEND_DATE_CONFLICT"
        elif day not in days:
            reason = "DIVIDEND_NOT_ON_REQUIRED_TRADING_DAY"
        elif event.process_status != "实施":
            reason = "DIVIDEND_NOT_IMPLEMENTED"
        elif event.ann_date is None or event.ann_date > day:
            reason = "DIVIDEND_ANN_DATE_INVALID"
        elif event.cash_dividend is None or not event.cash_dividend.is_finite() or event.cash_dividend < 0:
            reason = "DIVIDEND_CASH_INVALID"
        if reason:
            issues.append(BasisAuditIssue(day=day, code=reason))
            # 日期冲突时两天都不能按现金0计算，不能把一个有问题的事件当作不存在。
            invalid.update(d for d in (event.ex_date, event.nav_ex_date) if d is not None)
        if day is not None:
            grouped[day].append(event)
    valid = {}
    for day, group in grouped.items():
        if len(group) != 1:
            issues.append(BasisAuditIssue(day=day, code="MULTIPLE_DIVIDENDS_REQUIRE_REVIEW"))
            invalid.add(day)
        if day not in invalid:
            valid[day] = group[0].cash_dividend
    return valid, invalid


def build_nav_basis_audit(
    request: NavBasisAuditRequest,
    source: FeatureSourceReadiness,
    nav: tuple[BasisNavPoint, ...],
    events: tuple[BasisDividend, ...],
    calendar: TradingCalendar,
) -> NavBasisAuditResponse:
    """纯审计入口：限定日期后固定Decimal精度，调用方不能传训练参数或选择赢家。"""
    days = audit_dates(calendar, request)
    if (
        len(nav) > 386
        or len(events) > 100
        or any(not days[0] <= p.nav_date <= request.end_date for p in nav)
        or any(a.nav_date >= b.nav_date for a, b in zip(nav[:-1], nav[1:], strict=True))
        or any(e.ann_date and e.ann_date > date(2024, 12, 31) for e in events)
    ):
        raise ValueError("audit inputs exceed bounded scope, contain held-out values or are unordered")
    with localcontext() as context:
        context.prec = 40
        return _calculate(request, source, nav, events, calendar, days)


def _calculate(request, source, nav, events, calendar, days) -> NavBasisAuditResponse:
    by_date = {p.nav_date: p for p in nav}
    issues = []
    invalid_counts = {field: 0 for field in _SOURCE_COLUMNS}
    missing_dividend = 0
    for day in days:
        point = by_date.get(day)
        if point is None:
            issues.append(BasisAuditIssue(day=day, code="TRADING_NAV_MISSING"))
        elif point.ann_date is None or point.ann_date < day:
            issues.append(BasisAuditIssue(day=day, code="NAV_ANN_DATE_INVALID"))
        for field in _SOURCE_COLUMNS:
            if point is None or not _positive(getattr(point, field)):
                invalid_counts[field] += 1
                if point is not None:
                    issues.append(BasisAuditIssue(day=day, code=f"INVALID_{field.upper()}"))
        missing_dividend += point is None or point.accumulated_dividend is None

    cash_by_date, invalid_events = _events_by_date(events, days[1:], issues)
    comparisons, gaps = [], []
    cash_growth = _ONE
    for previous, day in zip(days[:-1], days[1:], strict=True):
        before, after = by_date.get(previous), by_date.get(day)
        if (
            day in invalid_events
            or before is None
            or after is None
            or not all(_positive(getattr(p, field)) for p in (before, after) for field in ("unit_nav", "adjusted_nav"))
        ):
            continue
        # 无已知事件日按现金0只是条件性计算，不证明分红事件表完整。
        cash = cash_by_date.get(day, Decimal(0))
        unit_return = after.unit_nav / before.unit_nav - _ONE
        cash_return = (after.unit_nav + cash) / before.unit_nav - _ONE
        adjusted_return = after.adjusted_nav / before.adjusted_nav - _ONE
        gap_bps = abs(cash_return - adjusted_return) * _BPS
        cash_growth *= _ONE + cash_return
        gaps.append(gap_bps)
        if gap_bps > 1 and day not in cash_by_date:
            issues.append(BasisAuditIssue(day=day, code="UNEXPLAINED_ADJUSTMENT_WITHOUT_DIVIDEND"))
        if day in cash_by_date:
            hypothesis = after.unit_nav / (before.unit_nav - cash) - _ONE if before.unit_nav > cash else None
            comparisons.append(
                DividendBasisComparison(
                    effective_date=day,
                    previous_trading_date=previous,
                    unit_nav_before=_number(before.unit_nav),
                    unit_nav_after=_number(after.unit_nav),
                    cash_per_share=_number(cash),
                    unit_nav_ratio_return=_number(unit_return),
                    cash_inclusive_day_return=_number(cash_return),
                    source_adjusted_day_return=_number(adjusted_return),
                    absolute_gap_bps=_number(gap_bps),
                    ex_cash_denominator_hypothesis_return=_number(hypothesis) if hypothesis is not None else None,
                )
            )
    # 三种源口径只作同端点数学对照；中途缺数仍能展示端点比值，但整体状态必须是不完整。
    base, end = by_date.get(days[0]), by_date.get(days[-1])
    ratios = [
        _number(getattr(end, field) / getattr(base, field) - _ONE)
        if base and end and _positive(getattr(base, field)) and _positive(getattr(end, field))
        else None
        for field in _SOURCE_COLUMNS
    ]
    source_snapshot = {
        "fund": request.fund_code,
        "source": source.source_code,
        "source_id": str(source.source_id),
        "source_sync_run_id": str(source.source_sync_run_id),
        "start": str(request.start_date),
        "end": str(request.end_date),
        "calendar": calendar.content_hash,
        "nav": [asdict(p) for p in nav],
        "dividends": sorted((asdict(e) for e in events), key=lambda e: json.dumps(e, default=str, sort_keys=True)),
    }
    snapshot = json.dumps(source_snapshot, default=str, sort_keys=True, separators=(",", ":"))
    count = sum(g > 1 for g in gaps)
    return NavBasisAuditResponse(
        status="AUDIT_INCOMPLETE" if issues else "DIFFERENCES_FOUND" if count else "NO_LARGE_DAILY_DIFFERENCE_FOUND",
        fund_code=request.fund_code,
        start_date=request.start_date,
        end_date=request.end_date,
        source_code=source.source_code,
        source_sync_run_id=source.source_sync_run_id,
        calendar_version=calendar.definition.version,
        calendar_hash=calendar.content_hash,
        snapshot_hash=hashlib.sha256(snapshot.encode()).hexdigest(),
        trading_day_count=len(days) - 1,
        raw_nav_count=len(nav),
        ignored_non_trading_dates=tuple(p.nav_date for p in nav if p.nav_date not in set(calendar.sessions)),
        invalid_nav_counts=invalid_counts,
        accumulated_dividend_missing_count=missing_dividend,
        dividend_record_count=len(events),
        checked_daily_pairs=len(gaps),
        daily_gap_over_threshold_count=count,
        max_daily_gap_bps=_number(max(gaps)) if gaps else None,
        issues=tuple(issues),
        period_comparison=BasisPeriodComparison(
            base_date=days[0],
            end_date=days[-1],
            unit_nav_ratio_return=ratios[0],
            accumulated_nav_ratio_return=ratios[1],
            source_adjusted_nav_ratio_return=ratios[2],
            cash_reinvestment_candidate_return=_number(cash_growth - _ONE)
            if not issues and len(gaps) == len(days) - 1
            else None,
        ),
        dividend_comparisons=tuple(comparisons),
        evidence_urls=EVIDENCE_URLS,
        limitations=LIMITATIONS,
    )


def audit_nav_basis(request: NavBasisAuditRequest) -> NavBasisAuditResponse:
    """复用小连接池，一次只读一致性事务读净值+事件；离开连接后计算，不请求供应商。"""
    calendar = load_calendar()
    dates = audit_dates(calendar, request)
    deadline = perf_counter() + 15
    with Session(get_nav_preview_engine()) as session, session.begin():
        session.execute(text("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY"))
        source = read_historical_nav_source(session, fund_code=request.fund_code)
        if source.source_code != "TUSHARE_PRO_FUND":
            raise HistoricalNavPreviewReadError("BASIS_SOURCE_UNSUPPORTED", "本次口径映射只核验了既有Tushare来源。")
        nav, dividends = read_basis_inputs(
            session,
            fund_code=request.fund_code,
            source_id=source.source_id,
            base_date=dates[0],
            start=request.start_date,
            end=request.end_date,
        )
    if perf_counter() >= deadline:
        raise TimeoutError("basis audit exceeded read phase budget")
    return build_nav_basis_audit(request, source, nav, dividends, calendar)
