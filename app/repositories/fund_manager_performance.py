"""单基金、单来源任期回报的有界只读输入，不采集资料或改变已有研究样本。"""

from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.fund import FundDividend, NavDaily

MAX_NAV_ROWS = 5000
MAX_DIVIDENDS = 100


@dataclass(frozen=True)
class ManagerNavPoint:
    nav_date: date
    ann_date: date | None
    unit_nav: Decimal | None
    accumulated_nav: Decimal | None  # 仅核对现金变化，不拿累计净值比值当总回报。
    adjusted_nav: Decimal | None  # 仅检查未解释的调整，不用供应商复权值替代本次计算。


@dataclass(frozen=True)
class ManagerDividend:
    ex_date: date | None
    nav_ex_date: date | None
    cash_dividend: Decimal | None  # 来源每份现金派息，元；不按base_unit重复换算。
    process_status: str | None
    ann_date: date | None
    implementation_ann_date: date | None


def read_manager_performance_inputs(
    session: Session, *, fund_code: str, source_id: UUID, start: date, end: date
) -> tuple[tuple[ManagerNavPoint, ...], tuple[ManagerDividend, ...]]:
    """所有经理共用两次查询；多读一条用于拒绝超限，绝不截断后声称完整。"""
    navs = session.execute(
        select(
            NavDaily.nav_date, NavDaily.ann_date, NavDaily.unit_nav,
            NavDaily.accumulated_nav, NavDaily.adjusted_nav,
        )
        .where(
            NavDaily.fund_code == fund_code, NavDaily.source_id == source_id,
            NavDaily.nav_date.between(start, end),
        )
        .order_by(NavDaily.nav_date)
        .limit(MAX_NAV_ROWS + 1)
    ).all()
    # 无有效日期的事件也读取并拒收，不能把其静默过滤后当作没有分红。
    dividends = session.execute(
        select(
            FundDividend.ex_date, FundDividend.nav_ex_date, FundDividend.cash_dividend,
            FundDividend.process_status, FundDividend.ann_date, FundDividend.implementation_ann_date,
        )
        .where(FundDividend.fund_code == fund_code, FundDividend.source_id == source_id)
        .order_by(FundDividend.ann_date, FundDividend.source_event_key)
        .limit(MAX_DIVIDENDS + 1)
    ).all()
    return tuple(ManagerNavPoint(*row) for row in navs), tuple(ManagerDividend(*row) for row in dividends)
