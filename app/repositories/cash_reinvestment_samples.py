"""新版研究样本的窄字段只读输入；不复用会读取累计/供应商复权列的旧审计仓储。"""

from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from uuid import UUID

from sqlalchemy import and_, or_, select
from sqlalchemy.orm import Session

from app.models.fund import FundDividend, NavDaily


@dataclass(frozen=True)
class CashNavPoint:
    nav_date: date  # 单位净值所属日，不是当时已经可见的证明。
    ann_date: date | None  # 来源公告日；历史输入必须不晚于cutoff。
    unit_nav: Decimal | None  # 只读取单位净值；绝不回退为累计或供应商复权净值。


@dataclass(frozen=True)
class CashDividend:
    event_key: str  # 同来源事件标识，用于对照；重复同日事件仍拒绝求和。
    ann_date: date | None  # 原方案公告日。
    implementation_ann_date: date | None  # 有实施公告日时也必须已到，不能只看原方案公告。
    ex_date: date | None  # 来源除息日。
    nav_ex_date: date | None  # 来源净值除权日；与除息日同时存在且不同则拒收。
    cash_dividend: Decimal | None  # 每份现金分红，元；不得再按每十份换算。
    process_status: str | None  # 当前来源进度，只接受“实施”，但不宣称已恢复历史状态版本。


def read_cash_sample_inputs(
    session: Session, *, fund_code: str, source_id: UUID, start: date, end: date
) -> tuple[tuple[CashNavPoint, ...], tuple[CashDividend, ...]]:
    """单基金日期跨度不足192日，净值最多192行、事件100行；多取一行作拒收哨兵。"""
    return (
        read_cash_nav_window(session, fund_code=fund_code, source_id=source_id, start=start, end=end),
        read_cash_dividends(session, fund_code=fund_code, source_id=source_id, start=start, end=end),
    )


def _validate_bounds(start: date, end: date) -> None:
    if not date(2021, 1, 1) <= start <= end <= date(2024, 12, 31) or (end - start).days >= 192:
        raise ValueError("cash sample read bounds invalid or enter protected test period")


def read_cash_nav_window(
    session: Session, *, fund_code: str, source_id: UUID, start: date, end: date
) -> tuple[CashNavPoint, ...]:
    """一页样本共用所需净值窗口，一条有界SELECT；单日入口也复用同样字段和限制。"""
    _validate_bounds(start, end)
    nav = session.execute(
        select(NavDaily.nav_date, NavDaily.ann_date, NavDaily.unit_nav)
        .where(
            NavDaily.fund_code == fund_code,
            NavDaily.source_id == source_id,
            NavDaily.nav_date >= start,
            NavDaily.nav_date <= end,
        )
        .order_by(NavDaily.nav_date)
        .limit(193)
    ).all()
    if (
        len(nav) > 192
        or any(not start <= row[0] <= end for row in nav)
        or any(a[0] >= b[0] for a, b in zip(nav[:-1], nav[1:], strict=True))
    ):
        raise ValueError("cash NAV input oversized, unordered, duplicated or outside read bounds")
    return tuple(CashNavPoint(*row) for row in nav)


def read_cash_dividends(
    session: Session, *, fund_code: str, source_id: UUID, start: date, end: date
) -> tuple[CashDividend, ...]:
    """固定整个请求窗口读取一次事件；100条上限不随pageSize变化。"""
    _validate_bounds(start, end)
    # 任一生效日期相交都读出，不能漏掉冲突；无生效日的旧事件也可能相关，保守交给计算层拒收。
    # 不按公告<=cutoff截掉标签事件：同一事务读取后，历史与答案各自按可得日期隔离。
    dividends = session.execute(
        select(
            FundDividend.source_event_key,
            FundDividend.ann_date,
            FundDividend.implementation_ann_date,
            FundDividend.ex_date,
            FundDividend.nav_ex_date,
            FundDividend.cash_dividend,
            FundDividend.process_status,
        )
        .where(
            FundDividend.fund_code == fund_code,
            FundDividend.source_id == source_id,
            or_(FundDividend.ann_date <= date(2024, 12, 31), FundDividend.ann_date.is_(None)),
            or_(
                FundDividend.ex_date.between(start, end),
                FundDividend.nav_ex_date.between(start, end),
                and_(FundDividend.ex_date.is_(None), FundDividend.nav_ex_date.is_(None)),
            ),
        )
        .order_by(FundDividend.source_event_key)
        .limit(101)
    ).all()
    if len(dividends) > 100:
        raise ValueError("cash sample input row limit exceeded")
    return tuple(CashDividend(*row) for row in dividends)
