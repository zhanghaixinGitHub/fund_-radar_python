"""将基金目录数据库模型转换为 Java 可读取的内部契约。"""

from datetime import date

from sqlalchemy.orm import Session

from app.db.session import get_engine
from app.models.fund import FundShareClass
from app.repositories.fund_read import get_fund_summary, list_fund_summaries
from app.schemas.fund import InternalFundDetail, InternalFundPage, InternalFundSummary


def list_funds(keyword: str | None, page_size: int, cursor: str | None) -> InternalFundPage:
    """分页读取真实目录样本，不调用外部数据源或在读取时执行同步。"""
    with Session(get_engine()) as session:
        rows = list_fund_summaries(session, keyword, page_size, cursor)
    page_rows = rows[:page_size]
    return InternalFundPage(
        items=tuple(_to_summary(fund, nav_date) for fund, nav_date in page_rows),
        next_cursor=page_rows[-1][0].fund_code if len(rows) > page_size else None,
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


def _to_summary(fund: FundShareClass, nav_date: date | None) -> InternalFundSummary:
    """将 ORM 行投影为稳定的内部目录摘要。"""
    return InternalFundSummary(
        fund_code=fund.fund_code,
        fund_name=fund.fund_name,
        fund_type=fund.fund_type,
        status=fund.status,
        as_of_date=nav_date,
    )
