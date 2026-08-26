"""基金目录与最新有效净值的只读仓储。"""

from datetime import date

from sqlalchemy import Select, func, select
from sqlalchemy.orm import Session

from app.models.fund import FundShareClass, NavDaily


def list_fund_summaries(
    session: Session, keyword: str | None, page_size: int, cursor: str | None
) -> tuple[tuple[FundShareClass, date | None], ...]:
    """按基金代码稳定分页读取已落库目录，并附带每只基金的最新净值日期。

    目录与净值同步分开执行。尚无已授权净值来源时，仍返回经过核验的目录，
    但 latest_nav_date 为 ``None``，由服务层明确标识为未同步。
    """
    latest_nav = (
        select(NavDaily.fund_code, func.max(NavDaily.nav_date).label("latest_nav_date"))
        .group_by(NavDaily.fund_code)
        .subquery()
    )
    statement: Select[tuple[FundShareClass, date | None]] = (
        select(FundShareClass, latest_nav.c.latest_nav_date)
        .outerjoin(latest_nav, latest_nav.c.fund_code == FundShareClass.fund_code)
        .order_by(FundShareClass.fund_code.asc())
    )
    if keyword:
        normalized_keyword = f"%{keyword.strip()}%"
        statement = statement.where(
            FundShareClass.fund_code.ilike(normalized_keyword)
            | FundShareClass.fund_name.ilike(normalized_keyword)
        )
    if cursor:
        statement = statement.where(FundShareClass.fund_code > cursor)
    return tuple(session.execute(statement.limit(page_size + 1)).all())


def get_fund_summary(session: Session, fund_code: str) -> tuple[FundShareClass, date | None] | None:
    """返回单只基金份额及其最新净值日期；目录不存在时返回 ``None``。"""
    latest_nav = (
        select(func.max(NavDaily.nav_date))
        .where(NavDaily.fund_code == fund_code)
        .scalar_subquery()
    )
    return session.execute(
        select(FundShareClass, latest_nav).where(FundShareClass.fund_code == fund_code)
    ).one_or_none()
