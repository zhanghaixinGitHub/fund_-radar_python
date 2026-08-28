"""将基金目录数据库模型转换为 Java 可读取的内部契约。"""

from datetime import date

from sqlalchemy.orm import Session

from app.db.session import get_engine
from app.models.fund import FundShareClass
from app.repositories.fund_read import (
    FundSummarySnapshot,
    get_fund_summary,
    list_fund_nav_history,
    list_fund_summaries,
    list_fund_summaries_by_codes,
)
from app.schemas.fund import (
    InternalFundDetail,
    InternalFundNavHistory,
    InternalFundNavPoint,
    InternalFundPage,
    InternalFundSummary,
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
    if row is None or not summary:
        return None
    fund = row.fund
    return InternalFundDetail(
        **_to_summary(summary[0]).model_dump(),
        nav_status="SYNCED" if row.nav_date else "NOT_SYNCED",
        data_source=row.source_code or fund.source_code,
        unit_nav=row.unit_nav,
        accumulated_nav=row.accumulated_nav,
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
