"""净值与分红同来源、有界读取；不采集数据、不查询样本/模型，也不打开2025数值。"""

from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from uuid import UUID

from sqlalchemy import and_, or_, select
from sqlalchemy.orm import Session

from app.models.fund import FundDividend, NavDaily


@dataclass(frozen=True)
class BasisNavPoint:
    nav_date: date  # 净值业务日。
    ann_date: date | None  # 来源公告日，不代表已保存历史首次版本。
    unit_nav: Decimal | None  # 单位净值，不含外加现金。
    accumulated_nav: Decimal | None  # 来源累计净值，不自行重算。
    adjusted_nav: Decimal | None  # 来源adj_nav原样映射，不自行赋予其现金再投含义。
    accumulated_dividend: Decimal | None  # 来源累计分红，缺失不补0。


@dataclass(frozen=True)
class BasisDividend:
    ann_date: date | None  # 分红方案公告日。
    ex_date: date | None  # 除息日。
    nav_ex_date: date | None  # 净值除权日，若与除息日冲突则停止该项对照。
    cash_dividend: Decimal | None  # 每份现金派息，来源单位元。
    process_status: str | None  # 只按“实施”计算；未知或预案不能当成已发生现金。


def read_basis_inputs(
    session: Session, *, fund_code: str, source_id: UUID, base_date: date, start: date, end: date
) -> tuple[tuple[BasisNavPoint, ...], tuple[BasisDividend, ...]]:
    """一次最多读取386条净值、100条事件；多取1条仅用于发现超限，绝不截断后假装齐全。"""
    if not date(2021, 1, 1) <= base_date < start <= end <= date(2024, 12, 31) or (end - base_date).days > 385:
        raise ValueError("basis audit read bounds invalid or include held-out values")
    nav_rows = session.execute(
        select(
            NavDaily.nav_date,
            NavDaily.ann_date,
            NavDaily.unit_nav,
            NavDaily.accumulated_nav,
            NavDaily.adjusted_nav,
            NavDaily.accumulated_dividend,
        )
        .where(
            NavDaily.fund_code == fund_code,
            NavDaily.source_id == source_id,
            NavDaily.nav_date >= base_date,
            NavDaily.nav_date <= end,
        )
        .order_by(NavDaily.nav_date)
        .limit(387)
    ).all()
    # 任一有效日落入范围都读出，以免两个日期冲突时漏掉；两个有效日均未知时按公告范围查疑点。
    div_rows = session.execute(
        select(
            FundDividend.ann_date,
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
                and_(
                    FundDividend.ex_date.is_(None),
                    FundDividend.nav_ex_date.is_(None),
                    or_(FundDividend.ann_date.between(start, end), FundDividend.ann_date.is_(None)),
                ),
            ),
        )
        .order_by(FundDividend.ann_date, FundDividend.source_event_key)
        .limit(101)
    ).all()
    if len(nav_rows) > 386 or len(div_rows) > 100:
        raise ValueError("basis audit read exceeded row limit")
    return tuple(BasisNavPoint(*r) for r in nav_rows), tuple(BasisDividend(*r) for r in div_rows)
