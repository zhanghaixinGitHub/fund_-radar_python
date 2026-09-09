"""独立现金分红再投研究样本；输入先冻结，后附答案，不调用旧标签或训练流程。"""

import hashlib
import json
from collections import defaultdict
from datetime import date
from decimal import ROUND_HALF_UP, Decimal, localcontext
from time import perf_counter

from sqlalchemy import text
from sqlalchemy.orm import Session

from app.db.session import get_nav_preview_engine
from app.repositories.cash_reinvestment_samples import CashDividend, CashNavPoint, read_cash_sample_inputs
from app.repositories.feature_snapshot import FeatureSourceReadiness
from app.repositories.historical_nav import HistoricalNavPreviewReadError, read_historical_nav_source
from app.repositories.trading_nav_window import NavDatePoint
from app.schemas.cash_reinvestment_samples import (
    CashDirectionLabel,
    CashFeaturePayload,
    CashReturnPoint,
    CashSampleIssue,
    CashSampleRequest,
    CashSampleResponse,
)
from app.schemas.trading_nav_window import TradingNavWindowRequest

# 仅复用七项纯数学公式，不执行旧的口径选择、记录计数、标签附加和样本保存。
from app.services.historical_nav_samples import _build_metrics
from app.services.trading_calendar import TradingCalendar, load_calendar
from app.services.trading_nav_window import build_trading_nav_window, window_read_bounds

RESEARCH_END = date(2024, 12, 31)
LIMITATIONS = (
    "本版固定假设现金在生效日按当日单位净值再投；不是供应商adj_nav，不含到账延迟、费用或个人收益。",
    "历史只使用ann_date及已提供的实施公告日均已到达的事件；无可见事件日按现金0是研究假设，不是无分红证明。",
    "公告缺失或晚于cutoff的事件不进入过去特征；若后来才获知旧分红，也不会回写该历史输入。",
    "当前净值/事件会被来源覆盖修订，当前实施状态也不是历史状态档案；首次版本与事件完整性仍未核准。",
    "只处理明确、唯一的实施现金分红；拆分/折算等非现金调整尚未覆盖，不具备正式训练/发布资格。",
    "输入起点可早于cutoff；答案从截至日最近交易日开始算20段，答案分母即使尚未公告也只出现在答案中。",
    "日历为沪深市场静态研究规则，不代表基金申赎或境外市场；不跳过缺数或顺延终点。",
    "仅三试点2022–2024预览，2025数值/成绩不开放；不存样本、不重训、不发布，不能混入旧版本训练器。",
)


def cash_sample_bounds(calendar: TradingCalendar, request: CashSampleRequest) -> tuple[date, date]:
    """先固定日期再读库；即使cutoff在2024，只要未来窗口进入2025也拒绝。"""
    start, end = window_read_bounds(calendar, request.cutoff_date)
    if end > RESEARCH_END:
        raise HistoricalNavPreviewReadError(
            "TEST_PERIOD_PROTECTED", "完整20交易日窗口进入2025测试期，本轮不读取其数值。"
        )
    return start, end


def _number(value: Decimal) -> str:
    return format(value.quantize(Decimal("0.000000000001"), rounding=ROUND_HALF_UP), "f")


def _hash(payload) -> str:
    return hashlib.sha256(
        json.dumps(payload.model_dump(mode="json"), ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()


def _event_available_at(event: CashDividend) -> date | None:
    if event.ann_date is None:
        return None
    return max(event.ann_date, event.implementation_ann_date or event.ann_date)


def _events_for_period(
    events: tuple[CashDividend, ...], dates: tuple[date, ...], latest_available: date
) -> tuple[dict[date, CashDividend], list[CashSampleIssue]]:
    """事件只影响其生效所在区间；两个日期冲突、同日多条、未实施均拒收，不求和或补0。"""
    issues, groups = [], defaultdict(list)
    for event in events:
        possible_dates = tuple(d for d in (event.ex_date, event.nav_ex_date) if d is not None)
        if possible_dates and not any(dates[0] < d <= dates[-1] for d in possible_dates):
            continue  # 基准日现金属于更早一段，不计入本段；未来事件不影响历史。
        day = event.nav_ex_date or event.ex_date
        availability = _event_available_at(event)
        reason = None
        if day is None:
            reason = "DIVIDEND_EFFECTIVE_DATE_MISSING"
        elif event.ex_date and event.nav_ex_date and event.ex_date != event.nav_ex_date:
            reason = "DIVIDEND_DATE_CONFLICT"
        elif day not in dates[1:]:
            reason = "DIVIDEND_NOT_ON_TRADING_DAY"
        elif event.process_status != "实施":
            reason = "DIVIDEND_NOT_IMPLEMENTED"
        elif availability is None:
            reason = "DIVIDEND_ANN_DATE_MISSING"
        elif availability > latest_available:
            reason = "DIVIDEND_NOT_AVAILABLE_BY_LIMIT"
        elif event.cash_dividend is None or not event.cash_dividend.is_finite() or event.cash_dividend < 0:
            reason = "DIVIDEND_CASH_INVALID"
        if reason:
            issues.append(CashSampleIssue(day=day, code=reason))
        if day is not None:
            groups[day].append(event)
    for day, group in groups.items():
        if len(group) != 1:
            issues.append(CashSampleIssue(day=day, code="MULTIPLE_DIVIDENDS_REQUIRE_REVIEW"))
    # 任何相关疑点都拒绝整个序列，不能使用一部分事件算出貌似完整的收益。
    return ({day: group[0] for day, group in groups.items()} if not issues else {}), issues


def build_cash_return_series(
    dates: tuple[date, ...], nav: dict[date, CashNavPoint], events: tuple[CashDividend, ...], *, latest_available: date
) -> tuple[tuple[CashReturnPoint, ...], tuple[CashSampleIssue, ...]]:
    """确定日期上的现金日收益连乘；调用者先隔离当时可见事件，不能传累计或复权替代值。"""
    if not 2 <= len(dates) <= 61 or any(a >= b for a, b in zip(dates[:-1], dates[1:], strict=True)):
        raise ValueError("cash series needs 2..61 strictly increasing dates")
    cash_events, issues = _events_for_period(events, dates, latest_available)
    for day in dates:
        point = nav.get(day)
        reason = None
        if point is None:
            reason = "TRADING_NAV_MISSING"
        elif point.ann_date is None:
            reason = "NAV_ANN_DATE_MISSING"
        elif point.ann_date < day:
            reason = "NAV_ANN_BEFORE_NAV_DATE"
        elif point.ann_date > latest_available:
            reason = "NAV_NOT_AVAILABLE_BY_LIMIT"
        elif point.unit_nav is None or not point.unit_nav.is_finite() or point.unit_nav <= 0:
            reason = "UNIT_NAV_INVALID"
        if reason:
            issues.append(CashSampleIssue(day=day, code=reason))
    if issues:
        return (), tuple(sorted(issues, key=lambda issue: (issue.day or date.min, issue.code)))
    with localcontext() as context:
        context.prec = 40
        context.rounding = ROUND_HALF_UP
        index = Decimal(100)
        available = nav[dates[0]].ann_date
        output = []
        for offset, day in enumerate(dates):
            point, event = nav[day], cash_events.get(day)
            cash = event.cash_dividend if event else Decimal(0)
            daily = None
            if offset:
                daily = (point.unit_nav + cash) / nav[dates[offset - 1]].unit_nav - 1
                index *= 1 + daily  # 中途不舍入；输出才固定12位，便于复算。
            available = max(available, point.ann_date, _event_available_at(event) if event else date.min)
            output.append(
                CashReturnPoint(
                    nav_date=day,
                    unit_nav=_number(point.unit_nav),
                    cash_per_share=_number(cash),
                    daily_return=_number(daily) if daily is not None else None,
                    growth_index=_number(index),
                    available_at=available,
                    dividend_event_keys=(event.event_key,) if event else (),
                )
            )
        return tuple(output), ()


def build_cash_history_feature(
    *,
    fund_code: str,
    cutoff_date: date,
    history_dates: tuple[date, ...],
    source: FeatureSourceReadiness,
    nav: dict[date, CashNavPoint],
    events: tuple[CashDividend, ...],
    calendar: TradingCalendar,
) -> tuple[CashFeaturePayload | None, tuple[CashSampleIssue, ...]]:
    """样本与预测复用同一现金序列/指标公式；调用者先核验历史日期，绝不附加答案。"""
    if len(history_dates) != 61:
        raise ValueError("cash features require exactly 61 history sessions")
    known_events = tuple(
        e for e in events if (available := _event_available_at(e)) is not None and available <= cutoff_date
    )
    with localcontext() as context:
        context.prec = 40
        context.rounding = ROUND_HALF_UP
        history, issues = build_cash_return_series(history_dates, nav, known_events, latest_available=cutoff_date)
        if issues:
            return None, issues
        metrics = _build_metrics(tuple(Decimal(p.growth_index) for p in history))
        if metrics is None:
            return None, (CashSampleIssue(day=history_dates[-1], code="FLAT_HISTORY_POSITION_UNDEFINED"),)
        return CashFeaturePayload(
            fund_code=fund_code,
            cutoff_date=cutoff_date,
            anchor_nav_date=history_dates[-1],
            available_at=history[-1].available_at,
            source_code=source.source_code,
            source_sync_run_id=source.source_sync_run_id,
            calendar_version=calendar.definition.version,
            calendar_hash=calendar.content_hash,
            history_series=history,
            metrics=metrics,
        ), ()


def build_cash_reinvestment_sample(
    request: CashSampleRequest,
    source: FeatureSourceReadiness,
    nav: tuple[CashNavPoint, ...],
    events: tuple[CashDividend, ...],
    calendar: TradingCalendar,
) -> CashSampleResponse:
    """先构造并哈希历史输入，再处理独立标签；未来数值/事件不会改变已知输入。"""
    cash_sample_bounds(calendar, request)
    if len(events) > 100 or len({e.event_key for e in events}) != len(events):
        raise ValueError("cash dividend inputs oversized or have duplicate event keys")
    # 日期层自身还会校验净值是否有序、重复、越界或超量；这里不复用它的总体status选择输入。
    window = build_trading_nav_window(
        TradingNavWindowRequest(fundCode=request.fund_code, cutoffDate=request.cutoff_date),
        source,
        tuple(NavDatePoint(p.nav_date, p.ann_date) for p in nav),
        calendar,
    )
    by_date = {p.nav_date: p for p in nav}
    input_issues = [CashSampleIssue(day=i.nav_date, code=i.reason) for i in window.history_issues]
    if window.anchor_issue:
        input_issues.append(CashSampleIssue(day=window.anchor_nav_date, code=window.anchor_issue))
    features = None
    with localcontext() as context:
        context.prec = 40
        context.rounding = ROUND_HALF_UP
        if not input_issues:
            features, problems = build_cash_history_feature(
                fund_code=request.fund_code,
                cutoff_date=request.cutoff_date,
                history_dates=window.history_dates,
                source=source,
                nav=by_date,
                events=events,
                calendar=calendar,
            )
            input_issues.extend(problems)
        feature_hash = _hash(features) if features else None
        # 标签基准不是历史anchor。否则anchor落后一天时，会把未来20日收益算成21段。
        base = calendar.sessions[calendar.at_or_before_index(request.cutoff_date)]
        label, label_issues = None, []
        if features is None:
            label_issues.append(CashSampleIssue(day=base, code="INPUT_UNAVAILABLE"))
        elif not window.calendar.future_schedule_known_at_cutoff:
            label_issues.append(CashSampleIssue(day=base, code="FUTURE_CALENDAR_NOT_KNOWN_AT_CUTOFF"))
        else:
            label_series, problems = build_cash_return_series(
                (base, *window.future_dates), by_date, events, latest_available=RESEARCH_END
            )
            label_issues.extend(problems)
            if not label_issues:
                # 12位舍入后统一判方向，避免微小Decimal残差造成“显示0却标签1”。
                future_return = _number(Decimal(label_series[-1].growth_index) / Decimal(100) - 1)
                label = CashDirectionLabel(
                    label_base_date=base,
                    label_end_date=window.future_end_date,
                    label_available_at=label_series[-1].available_at,
                    future_return_20d=future_return,
                    label_up_20d=int(Decimal(future_return) > 0),
                    label_series=label_series,
                )
    return CashSampleResponse(
        status="INPUT_UNAVAILABLE"
        if features is None
        else "LABEL_UNAVAILABLE"
        if label is None
        else "RESEARCH_SAMPLE_READY",
        fund_code=request.fund_code,
        cutoff_date=request.cutoff_date,
        calendar=window.calendar,
        anchor_nav_date=window.anchor_nav_date,
        anchor_lag_sessions=window.anchor_lag_sessions,
        label_base_date=base,
        label_end_date=window.future_end_date,
        history_dates=window.history_dates,
        future_dates=window.future_dates,
        input_issues=tuple(input_issues),
        label_issues=tuple(label_issues),
        feature_payload=features,
        feature_hash=feature_hash,
        offline_label=label,
        label_hash=_hash(label) if label else None,
        ignored_non_trading_nav_dates=window.ignored_non_trading_nav_dates,
        limitations=LIMITATIONS,
    )


def preview_cash_reinvestment_sample(request: CashSampleRequest) -> CashSampleResponse:
    """同来源只读一致性事务，限定一只基金一个窗口；不查询样本表、审计全历史或调用模型。"""
    calendar = load_calendar()
    start, end = cash_sample_bounds(calendar, request)
    deadline = perf_counter() + 15
    with Session(get_nav_preview_engine()) as session, session.begin():
        session.execute(text("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY"))
        source = read_historical_nav_source(session, fund_code=request.fund_code)
        if source.source_code != "TUSHARE_PRO_FUND":
            raise HistoricalNavPreviewReadError("CASH_SOURCE_UNSUPPORTED", "现金字段映射仅核验既有Tushare来源。")
        nav, dividends = read_cash_sample_inputs(
            session, fund_code=request.fund_code, source_id=source.source_id, start=start, end=end
        )
    if perf_counter() >= deadline:
        raise TimeoutError("cash sample exceeded read phase budget")
    return build_cash_reinvestment_sample(request, source, nav, dividends, calendar)
