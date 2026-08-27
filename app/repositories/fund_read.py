"""基金目录与最新有效净值的只读仓储。"""

from dataclasses import dataclass
from datetime import date
from decimal import Decimal

from sqlalchemy import Select, func, select, true
from sqlalchemy.orm import Session

from app.models.fund import FundShareClass, NavDaily, SourceRegistry


@dataclass(frozen=True)
class FundDetailSnapshot:
    """同一条最新净值记录的详情投影，避免净值数值与来源错配。"""

    fund: FundShareClass
    nav_date: date | None
    unit_nav: Decimal | None
    accumulated_nav: Decimal | None
    source_code: str | None


@dataclass(frozen=True)
class FundNavHistorySnapshot:
    """一条按来源稳定去重后的历史净值投影。"""

    nav_date: date
    unit_nav: Decimal
    accumulated_nav: Decimal | None


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


def get_fund_summary(session: Session, fund_code: str) -> FundDetailSnapshot | None:
    """返回最新净值快照；目录不存在时返回 ``None``。

    同一日期可能存在多个已登记来源，按最新净值日期倒序、来源代码正序确定展示记录，
    并从该同一行读取数值与来源，避免跨行拼接出错误的详情。
    """
    latest_nav = (
        select(
            NavDaily.nav_date.label("nav_date"),
            NavDaily.unit_nav.label("unit_nav"),
            NavDaily.accumulated_nav.label("accumulated_nav"),
            SourceRegistry.source_code.label("source_code"),
        )
        .select_from(NavDaily)
        .join(SourceRegistry, SourceRegistry.source_id == NavDaily.source_id)
        .where(NavDaily.fund_code == fund_code)
        .order_by(NavDaily.nav_date.desc(), SourceRegistry.source_code.asc())
        .limit(1)
        .subquery()
    )
    row = session.execute(
        select(
            FundShareClass,
            latest_nav.c.nav_date,
            latest_nav.c.unit_nav,
            latest_nav.c.accumulated_nav,
            latest_nav.c.source_code,
        )
        .outerjoin(latest_nav, true())
        .where(FundShareClass.fund_code == fund_code)
    ).one_or_none()
    if row is None:
        return None
    fund, nav_date, unit_nav, accumulated_nav, source_code = row
    return FundDetailSnapshot(
        fund=fund,
        nav_date=nav_date,
        unit_nav=unit_nav,
        accumulated_nav=accumulated_nav,
        source_code=source_code,
    )


def list_fund_nav_history(
    session: Session, fund_code: str, start_date: date, end_date: date
) -> tuple[FundNavHistorySnapshot, ...]:
    """按日期正序读取指定窗口历史净值，同日期多来源时按来源代码稳定选取一条。"""
    ranked_nav = (
        select(
            NavDaily.nav_date.label("nav_date"),
            NavDaily.unit_nav.label("unit_nav"),
            NavDaily.accumulated_nav.label("accumulated_nav"),
            func.row_number()
            .over(partition_by=NavDaily.nav_date, order_by=SourceRegistry.source_code.asc())
            .label("source_rank"),
        )
        .select_from(NavDaily)
        .join(SourceRegistry, SourceRegistry.source_id == NavDaily.source_id)
        .where(
            NavDaily.fund_code == fund_code,
            NavDaily.nav_date >= start_date,
            NavDaily.nav_date <= end_date,
        )
        .subquery()
    )
    rows = session.execute(
        select(ranked_nav.c.nav_date, ranked_nav.c.unit_nav, ranked_nav.c.accumulated_nav)
        .where(ranked_nav.c.source_rank == 1)
        .order_by(ranked_nav.c.nav_date.asc())
    ).all()
    return tuple(
        FundNavHistorySnapshot(nav_date=nav_date, unit_nav=unit_nav, accumulated_nav=accumulated_nav)
        for nav_date, unit_nav, accumulated_nav in rows
    )
