"""将基金目录数据库模型转换为 Java 可读取的内部契约。"""

from datetime import date

from sqlalchemy.orm import Session

from app.db.session import get_engine
from app.models.fund import FundShareClass
from app.repositories.fund_read import get_fund_summary, list_fund_nav_history, list_fund_summaries
from app.schemas.fund import (
    InternalFundDetail,
    InternalFundNavHistory,
    InternalFundNavPoint,
    InternalFundPage,
    InternalFundSummary,
)


def list_funds(keyword: str | None, page_size: int, cursor: str | None, page: int | None) -> InternalFundPage:
    """分页读取真实目录样本，并返回与当前筛选一致的总记录数。

    ``page`` 存在时使用页码分页，供浏览器显示总页数和跳转；未传入时保持旧版游标
    语义。两种模式均只查询已落库数据，不调用外部数据源或触发同步。
    """
    with Session(get_engine()) as session:
        result = list_fund_summaries(session, keyword, page_size, cursor, page)
    page_rows = result.rows if page is not None else result.rows[:page_size]
    total_pages = (result.total_count + page_size - 1) // page_size
    return InternalFundPage(
        items=tuple(_to_summary(fund, nav_date) for fund, nav_date in page_rows),
        next_cursor=page_rows[-1][0].fund_code if page is None and len(result.rows) > page_size else None,
        page=page,
        page_size=page_size,
        total_count=result.total_count,
        total_pages=total_pages,
    )


def get_fund(fund_code: str) -> InternalFundDetail | None:
    """读取一只真实目录样本，并显式返回净值同步状态。"""
    with Session(get_engine()) as session:
        row = get_fund_summary(session, fund_code)
    if row is None:
        return None
    fund = row.fund
    return InternalFundDetail(
        **_to_summary(fund, row.nav_date).model_dump(),
        nav_status="SYNCED" if row.nav_date else "NOT_SYNCED",
        data_source=row.source_code or fund.source_code,
        unit_nav=row.unit_nav,
        accumulated_nav=row.accumulated_nav,
    )


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


def _to_summary(fund: FundShareClass, nav_date: date | None) -> InternalFundSummary:
    """将 ORM 行投影为稳定的内部目录摘要。"""
    return InternalFundSummary(
        fund_code=fund.fund_code,
        fund_name=fund.fund_name,
        fund_type=fund.fund_type,
        status=fund.status,
        as_of_date=nav_date,
    )
