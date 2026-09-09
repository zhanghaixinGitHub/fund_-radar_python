"""现金样本的有界批量编排；日历先选题，一次快照分页读取，逐题复用单日构建器。"""

import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from datetime import date, timedelta
from time import perf_counter

from sqlalchemy import text
from sqlalchemy.orm import Session

from app.db.session import get_nav_preview_engine
from app.repositories.cash_reinvestment_samples import CashDividend, read_cash_dividends, read_cash_nav_window
from app.repositories.feature_snapshot import FeatureSourceReadiness
from app.repositories.historical_nav import HistoricalNavPreviewReadError, read_historical_nav_source
from app.schemas.cash_reinvestment_batch import CashBatchRequest, CashBatchResponse
from app.schemas.cash_reinvestment_samples import CashSampleRequest, CashSampleResponse
from app.services.cash_reinvestment_samples import build_cash_reinvestment_sample, cash_sample_bounds
from app.services.trading_calendar import CalendarCoverageError, TradingCalendar, load_calendar


@dataclass(frozen=True)
class CashSamplePlan:
    cutoff_date: date  # 这道题的信息截止日，计划阶段只看日历，不看净值是否存在。
    read_start: date  # 为61日历史及公告滞后预留的最早读取日期。
    read_end: date  # 截止之后第20个交易日，未来缺数不顺延。


def plan_cash_batch(request: CashBatchRequest, calendar: TradingCalendar) -> tuple[CashSamplePlan, ...]:
    """整段先验范围检查；任何题目进入2025即整次拒绝，不能因分页大小跳过后面的保护。"""
    plans = []
    for day in calendar.sessions:
        if request.start_date <= day <= request.end_date:
            start, end = cash_sample_bounds(calendar, CashSampleRequest(fundCode=request.fund_code, cutoffDate=day))
            plans.append(CashSamplePlan(day, start, end))
    if plans and (plans[-1].read_end - plans[0].read_start).days >= 192:
        raise CalendarCoverageError("批量合并资料窗口超过192自然日，请缩短日期范围。")
    return tuple(plans)


def _events_in_window(events: tuple[CashDividend, ...], plan: CashSamplePlan) -> tuple[CashDividend, ...]:
    """完全对应单日SQL的范围条件；否则较大页中邻题事件可能污染当前题目的问题列表。"""
    return tuple(
        event
        for event in events
        if (event.ex_date is None and event.nav_ex_date is None)
        or any(plan.read_start <= day <= plan.read_end for day in (event.ex_date, event.nav_ex_date) if day is not None)
    )


def _check_budget(deadline: float) -> None:
    if perf_counter() >= deadline:
        raise TimeoutError("cash batch preview exceeded request phase budget")


def summarize_cash_batch(
    request: CashBatchRequest,
    source: FeatureSourceReadiness,
    calendar: TradingCalendar,
    items: tuple[CashSampleResponse, ...],
    *,
    page_count: int,
) -> CashBatchResponse:
    """计数按题目而不是问题条数；哈希不含分页方式，便于直接对照不同pageSize。"""
    statuses = Counter(item.status for item in items)
    input_issues, label_issues = Counter(), Counter()
    for item in items:
        input_issues.update({issue.code for issue in item.input_issues})
        label_issues.update({issue.code for issue in item.label_issues})
    sessions = set(calendar.sessions)
    all_dates = tuple(
        request.start_date + timedelta(days=i) for i in range((request.end_date - request.start_date).days + 1)
    )
    response = CashBatchResponse(
        status="DRY_RUN_COMPLETED" if items else "NO_TRADING_CUTOFFS",
        fund_code=request.fund_code,
        start_date=request.start_date,
        end_date=request.end_date,
        skipped_cutoff_dates=tuple(day for day in all_dates if day not in sessions),
        source_code=source.source_code,
        source_sync_run_id=source.source_sync_run_id,
        calendar_version=calendar.definition.version,
        calendar_hash=calendar.content_hash,
        page_size=request.page_size,
        page_count=page_count,
        sample_count=len(items),
        input_available_count=sum(item.feature_payload is not None for item in items),
        label_available_count=sum(item.offline_label is not None for item in items),
        ready_count=statuses["RESEARCH_SAMPLE_READY"],
        input_unavailable_count=statuses["INPUT_UNAVAILABLE"],
        label_unavailable_count=statuses["LABEL_UNAVAILABLE"],
        input_issue_counts=dict(sorted(input_issues.items())),
        label_issue_counts=dict(sorted(label_issues.items())),
        items=items,
        batch_hash="",
    )
    content = response.model_dump(mode="json", exclude={"page_size", "page_count", "batch_hash"})
    digest = hashlib.sha256(
        json.dumps(content, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return response.model_copy(update={"batch_hash": digest})


def build_cash_batch_in_session(
    session: Session, request: CashBatchRequest, *, deadline: float
) -> CashBatchResponse:
    """调用者持有一致性事务；只读预览与保存共用算法，不嵌套独立快照。"""
    calendar = load_calendar()
    plans = plan_cash_batch(request, calendar)  # 2025保护在开数据库前检查所有计划。
    items, page_count = [], 0
    _check_budget(deadline)
    source = read_historical_nav_source(session, fund_code=request.fund_code)
    if source.source_code != "TUSHARE_PRO_FUND":
        raise HistoricalNavPreviewReadError("CASH_SOURCE_UNSUPPORTED", "现金字段映射仅核验既有Tushare来源。")
    _check_budget(deadline)
    # 分红整批最多100条，先读一次；上限不会因换页大小而变成另一组规则。
    events = (
        read_cash_dividends(
            session,
            fund_code=request.fund_code,
            source_id=source.source_id,
            start=plans[0].read_start,
            end=plans[-1].read_end,
        )
        if plans
        else ()
    )
    if len(events) > 100 or len({event.event_key for event in events}) != len(events):
        raise ValueError("cash batch events oversized or duplicate keys")
    for offset in range(0, len(plans), request.page_size):
        _check_budget(deadline)
        page = plans[offset : offset + request.page_size]
        # 一页合并一条净值SELECT，不逐题查来源、查分红、再调单日接口。
        nav = read_cash_nav_window(
            session,
            fund_code=request.fund_code,
            source_id=source.source_id,
            start=page[0].read_start,
            end=page[-1].read_end,
        )
        for plan in page:
            _check_budget(deadline)
            # 裁回该题与单日GET完全一致的读取边界，包括非交易日诊断行和冲突事件。
            own_nav = tuple(point for point in nav if plan.read_start <= point.nav_date <= plan.read_end)
            items.append(
                build_cash_reinvestment_sample(
                    CashSampleRequest(fundCode=request.fund_code, cutoffDate=plan.cutoff_date),
                    source,
                    own_nav,
                    _events_in_window(events, plan),
                    calendar,
                )
            )
        page_count += 1
    _check_budget(deadline)
    # 空范围也先完成基金和来源检查，但不读取任何净值/事件；不伪造周末题目。
    result = summarize_cash_batch(request, source, calendar, tuple(items), page_count=page_count)
    _check_budget(deadline)
    return result


def preview_cash_reinvestment_batch(request: CashBatchRequest) -> CashBatchResponse:
    """只读同一快照最多31自然日；预检在开库前执行，全部完成才返回。"""
    deadline = perf_counter() + 15
    plan_cash_batch(request, load_calendar())
    with Session(get_nav_preview_engine()) as session, session.begin():
        session.execute(text("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY"))
        return build_cash_batch_in_session(session, request, deadline=deadline)
