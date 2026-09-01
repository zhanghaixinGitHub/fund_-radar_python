"""将基金目录数据库模型转换为 Java 可读取的内部契约。"""

from datetime import date

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.session import get_engine
from app.models.fund import FundShareClass, SourceSyncRun
from app.repositories.fund_read import (
    FundDetailSnapshot,
    FundProfileSnapshot,
    FundSummarySnapshot,
    get_current_market_same_type_comparison,
    get_fund_profile_snapshot,
    get_fund_summary,
    list_fund_nav_history,
    list_fund_share_history,
    list_fund_summaries,
    list_fund_summaries_by_codes,
)
from app.repositories.fund_read import (
    get_fund_watchlist_detail as get_fund_watchlist_detail_snapshot,
)
from app.schemas.fund import (
    InternalFundDetail,
    InternalFundDividend,
    InternalFundManager,
    InternalFundNavHistory,
    InternalFundNavPoint,
    InternalFundPage,
    InternalFundSameTypeComparison,
    InternalFundSameTypeComparisonItem,
    InternalFundShareHistory,
    InternalFundShareSnapshot,
    InternalFundSummary,
    InternalFundWatchlistDetail,
)


def list_funds(
    keyword: str | None,
    fund_type: str | None,
    page_size: int,
    cursor: str | None,
    page: int | None,
) -> InternalFundPage:
    """分页读取真实目录样本，并返回与当前筛选一致的总记录数。

    ``page`` 存在时使用页码分页，供浏览器显示总页数和跳转；未传入时保持旧版游标
    语义。两种模式均只查询已落库数据，不调用外部数据源或触发同步。
    """
    with Session(get_engine()) as session:
        result = list_fund_summaries(session, keyword, fund_type, page_size, cursor, page)
    page_rows = result.rows if page is not None else result.rows[:page_size]
    total_pages = (result.total_count + page_size - 1) // page_size
    return InternalFundPage(
        items=tuple(_to_summary(snapshot) for snapshot in page_rows),
        next_cursor=page_rows[-1].fund.fund_code if page is None and len(result.rows) > page_size else None,
        page=page,
        page_size=page_size,
        total_count=result.total_count,
        total_pages=total_pages,
    )


def get_fund(fund_code: str) -> InternalFundDetail | None:
    """读取一只真实目录样本，并显式返回净值同步状态。"""
    with Session(get_engine()) as session:
        row = get_fund_summary(session, fund_code)
        summary = list_fund_summaries_by_codes(session, (fund_code,))
        profile = get_fund_profile_snapshot(session, fund_code)
    if row is None or not summary:
        return None
    return _to_detail(summary[0], row, profile)


def get_fund_watchlist_detail(fund_code: str) -> InternalFundWatchlistDetail | None:
    """返回完整详情的本地只读投影；调用方必须先完成当前用户关注关系校验。"""
    with Session(get_engine()) as session:
        snapshot = get_fund_watchlist_detail_snapshot(session, fund_code)
        summaries = list_fund_summaries_by_codes(session, (fund_code,))
    if snapshot is None or not summaries:
        return None
    basic = _to_detail(summaries[0], snapshot.detail, snapshot.profile)
    return InternalFundWatchlistDetail(
        basic=basic,
        managers_status=(
            "SYNCED" if "MARKET_DETAIL_MANAGER" in snapshot.succeeded_sync_types else "NOT_SYNCED"
        ),
        managers=tuple(
            InternalFundManager(
                manager_name=item.assignment.manager_name,
                ann_date=item.assignment.ann_date,
                begin_date=item.assignment.begin_date,
                end_date=item.assignment.end_date,
                education=item.assignment.education,
                data_source=item.source_code,
            )
            for item in snapshot.managers
        ),
        latest_share_status=(
            "SYNCED" if "MARKET_DETAIL_SHARE" in snapshot.succeeded_sync_types else "NOT_SYNCED"
        ),
        latest_share=(
            InternalFundShareSnapshot(
                trade_date=snapshot.latest_share.snapshot.trade_date,
                fund_share=snapshot.latest_share.snapshot.fund_share,
                data_source=snapshot.latest_share.source_code,
            )
            if snapshot.latest_share is not None
            else None
        ),
        dividends_status=(
            "SYNCED" if "MARKET_DETAIL_DIVIDEND" in snapshot.succeeded_sync_types else "NOT_SYNCED"
        ),
        dividends=tuple(
            InternalFundDividend(
                ann_date=item.dividend.ann_date,
                implementation_ann_date=item.dividend.implementation_ann_date,
                base_date=item.dividend.base_date,
                process_status=item.dividend.process_status,
                record_date=item.dividend.record_date,
                ex_date=item.dividend.ex_date,
                pay_date=item.dividend.pay_date,
                earnings_pay_date=item.dividend.earnings_pay_date,
                nav_ex_date=item.dividend.nav_ex_date,
                cash_dividend=item.dividend.cash_dividend,
                base_unit=item.dividend.base_unit,
                distributable_earnings=item.dividend.distributable_earnings,
                earnings_amount=item.dividend.earnings_amount,
                reinvestment_arrival_date=item.dividend.reinvestment_arrival_date,
                base_year=item.dividend.base_year,
                data_source=item.source_code,
            )
            for item in snapshot.dividends
        ),
    )


def get_funds_by_codes(fund_codes: tuple[str, ...]) -> tuple[InternalFundSummary, ...]:
    """批量返回指定基金的公开摘要；仅查询持久化目录与净值，不触发同步。"""
    with Session(get_engine()) as session:
        summaries = list_fund_summaries_by_codes(session, fund_codes)
    return tuple(_to_summary(snapshot) for snapshot in summaries)


def get_fund_nav_history(fund_code: str, start_date: date, end_date: date) -> InternalFundNavHistory | None:
    """读取一只已落库基金在明确日期窗口内的历史净值，不触发外部同步。"""
    with Session(get_engine()) as session:
        fund = session.get(FundShareClass, fund_code)
        if fund is None:
            return None
        rows = list_fund_nav_history(session, fund_code, start_date, end_date)
    return InternalFundNavHistory(
        fund_code=fund_code,
        items=tuple(
            InternalFundNavPoint(
                nav_date=row.nav_date,
                unit_nav=row.unit_nav,
                accumulated_nav=row.accumulated_nav,
            )
            for row in rows
        ),
    )


def get_fund_share_history(fund_code: str, start_date: date, end_date: date) -> InternalFundShareHistory | None:
    """读取关注后份额规模历史；只读本地快照且不触发完整资料同步。"""
    with Session(get_engine()) as session:
        fund = session.get(FundShareClass, fund_code)
        if fund is None:
            return None
        synced = session.scalar(
            select(SourceSyncRun.sync_run_id)
            .where(
                SourceSyncRun.status == "SUCCEEDED",
                SourceSyncRun.sync_type == "MARKET_DETAIL_SHARE",
            )
            .limit(1)
        ) is not None
        rows = list_fund_share_history(session, fund_code, start_date, end_date) if synced else ()
    return InternalFundShareHistory(
        fund_code=fund_code,
        status="SYNCED" if synced else "NOT_SYNCED",
        items=tuple(
            InternalFundShareSnapshot(
                trade_date=row.trade_date,
                fund_share=row.fund_share,
                data_source=row.source_code,
            )
            for row in rows
        ),
    )


def get_fund_same_type_comparison(fund_code: str) -> InternalFundSameTypeComparison | None:
    """返回当前基金市场范围内同类型、同净值日期的一月涨跌事实比较。"""
    with Session(get_engine()) as session:
        result = get_current_market_same_type_comparison(session, fund_code)
    if result is None:
        return None
    target = result.target
    if target.fund.status != "ACTIVE" or target.fund.source_code != "TUSHARE_PRO_FUND":
        return InternalFundSameTypeComparison(
            fund_code=fund_code,
            fund_type=target.fund.fund_type,
            scope="CURRENT_MARKET_ACTIVE_TUSHARE_PRO_FUND",
            status="OUT_OF_SCOPE",
            as_of_date=target.nav_date,
        )
    if target.nav_date is None or target.performance.month_change_rate is None:
        return InternalFundSameTypeComparison(
            fund_code=fund_code,
            fund_type=target.fund.fund_type,
            scope="CURRENT_MARKET_ACTIVE_TUSHARE_PRO_FUND",
            status="DATA_INSUFFICIENT",
            as_of_date=target.nav_date,
        )
    comparable = sorted(
        (
            item for item in result.items
            if item.summary.nav_date == target.nav_date
            and item.summary.performance.month_change_rate is not None
            and item.data_source is not None
        ),
        key=lambda item: (-item.summary.performance.month_change_rate, item.summary.fund.fund_code),
    )
    target_rank = next(
        (
            index
            for index, item in enumerate(comparable, start=1)
            if item.summary.fund.fund_code == fund_code
        ),
        None,
    )
    if target_rank is None:
        return InternalFundSameTypeComparison(
            fund_code=fund_code,
            fund_type=target.fund.fund_type,
            scope="CURRENT_MARKET_ACTIVE_TUSHARE_PRO_FUND",
            status="DATA_INSUFFICIENT",
            as_of_date=target.nav_date,
        )
    return InternalFundSameTypeComparison(
        fund_code=fund_code,
        fund_type=target.fund.fund_type,
        scope="CURRENT_MARKET_ACTIVE_TUSHARE_PRO_FUND",
        status="SYNCED",
        as_of_date=target.nav_date,
        target_rank=target_rank,
        comparable_count=len(comparable),
        items=tuple(
            InternalFundSameTypeComparisonItem(
                rank=index,
                fund_code=item.summary.fund.fund_code,
                fund_name=item.summary.fund.fund_name,
                fund_type=item.summary.fund.fund_type,
                as_of_date=item.summary.nav_date,
                month_change_rate=item.summary.performance.month_change_rate,
                data_source=item.data_source,
            )
            for index, item in enumerate(comparable, start=1)
        ),
    )


def _to_summary(snapshot: FundSummarySnapshot) -> InternalFundSummary:
    """将 ORM 行投影为稳定的内部目录摘要。"""
    fund = snapshot.fund
    performance = snapshot.performance
    return InternalFundSummary(
        fund_code=fund.fund_code,
        fund_name=fund.fund_name,
        fund_type=fund.fund_type,
        status=fund.status,
        as_of_date=snapshot.nav_date,
        day_change_rate=performance.day_change_rate,
        week_change_rate=performance.week_change_rate,
        month_change_rate=performance.month_change_rate,
    )


def _to_detail(
    summary: FundSummarySnapshot,
    detail: FundDetailSnapshot,
    profile: FundProfileSnapshot | None,
) -> InternalFundDetail:
    """将基础详情、同源最新净值和可选资料快照拼成兼容市场详情。"""
    profile_data = profile.profile if profile is not None else None
    return InternalFundDetail(
        **_to_summary(summary).model_dump(),
        nav_status="SYNCED" if detail.nav_date else "NOT_SYNCED",
        data_source=detail.source_code or detail.fund.source_code,
        unit_nav=detail.unit_nav,
        accumulated_nav=detail.accumulated_nav,
        nav_ann_date=detail.ann_date,
        accumulated_dividend=detail.accumulated_dividend,
        net_asset=detail.net_asset,
        total_net_asset=detail.total_net_asset,
        adjusted_nav=detail.adjusted_nav,
        profile_status="SYNCED" if profile_data is not None else "NOT_SYNCED",
        profile_data_source=profile.source_code if profile is not None else None,
        management_company_name=profile_data.management_company_name if profile_data is not None else None,
        custodian_name=profile_data.custodian_name if profile_data is not None else None,
        found_date=profile_data.found_date if profile_data is not None else None,
        due_date=profile_data.due_date if profile_data is not None else None,
        list_date=profile_data.list_date if profile_data is not None else None,
        issue_date=profile_data.issue_date if profile_data is not None else None,
        delist_date=profile_data.delist_date if profile_data is not None else None,
        issue_amount=profile_data.issue_amount if profile_data is not None else None,
        management_fee=profile_data.management_fee if profile_data is not None else None,
        custodian_fee=profile_data.custodian_fee if profile_data is not None else None,
        duration_year=profile_data.duration_year if profile_data is not None else None,
        par_value=profile_data.par_value if profile_data is not None else None,
        min_purchase_amount=profile_data.min_purchase_amount if profile_data is not None else None,
        expected_return=profile_data.expected_return if profile_data is not None else None,
        benchmark=profile_data.benchmark if profile_data is not None else None,
        invest_type=profile_data.invest_type if profile_data is not None else None,
        source_fund_type=profile_data.source_fund_type if profile_data is not None else None,
        trustee_name=profile_data.trustee_name if profile_data is not None else None,
        purchase_start_date=profile_data.purchase_start_date if profile_data is not None else None,
        redemption_start_date=profile_data.redemption_start_date if profile_data is not None else None,
        market=profile_data.market if profile_data is not None else None,
    )
