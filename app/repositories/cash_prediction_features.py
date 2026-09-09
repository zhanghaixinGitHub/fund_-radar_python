"""历史输入专用窄查询：不读取未来净值，尚未公告的价格在SQL层屏蔽。"""

from datetime import date
from uuid import UUID

from sqlalchemy import and_, case, or_, select
from sqlalchemy.orm import Session

from app.models.fund import FundDividend, NavDaily
from app.repositories.cash_reinvestment_samples import CashDividend, CashNavPoint


def validate_history_read_bounds(start: date, cutoff: date) -> None:
    """窗口有界且禁止触碰2025保留测试数值；新年份不能借此解封旧测试资料。"""
    if start < date(2021, 1, 1) or not 0 <= (cutoff - start).days < 192:
        raise ValueError("invalid or oversized cash history window")
    if start <= date(2025, 12, 31) and cutoff >= date(2025, 1, 1):
        raise ValueError("cash history intersects protected 2025 values")


def read_cash_history_inputs(
    session: Session, *, fund_code: str, source_id: UUID, start: date, cutoff: date
) -> tuple[tuple[CashNavPoint, ...], tuple[CashDividend, ...]]:
    """一份输入两条有界SELECT；未公告行只保留日期元数据来解释缺口，不带回价格。"""
    validate_history_read_bounds(start, cutoff)
    visible_price = and_(NavDaily.ann_date >= NavDaily.nav_date, NavDaily.ann_date <= cutoff)
    rows = session.execute(
        select(NavDaily.nav_date, NavDaily.ann_date, case((visible_price, NavDaily.unit_nav), else_=None))
        .where(
            NavDaily.fund_code == fund_code,
            NavDaily.source_id == source_id,
            NavDaily.nav_date >= start,
            NavDaily.nav_date <= cutoff,
        )
        .order_by(NavDaily.nav_date)
        .limit(193)
    ).all()
    if (
        len(rows) > 192
        or any(not start <= row[0] <= cutoff for row in rows)
        or any(a[0] >= b[0] for a, b in zip(rows[:-1], rows[1:], strict=True))
        or any((row[1] is None or not row[0] <= row[1] <= cutoff) and row[2] is not None for row in rows)
    ):
        raise ValueError("cash history is oversized, out of bounds, duplicated, unordered or contains hidden prices")
    # 两个生效日期任一落入窗口都读取，避免漏掉日期冲突；未知生效日的已知事件仍交计算层拒收。
    events = session.execute(
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
            FundDividend.ann_date <= cutoff,
            or_(FundDividend.implementation_ann_date.is_(None), FundDividend.implementation_ann_date <= cutoff),
            or_(
                FundDividend.ex_date.between(start, cutoff),
                FundDividend.nav_ex_date.between(start, cutoff),
                and_(FundDividend.ex_date.is_(None), FundDividend.nav_ex_date.is_(None)),
            ),
        )
        .order_by(FundDividend.source_event_key)
        .limit(101)
    ).all()
    if (
        len(events) > 100
        or len({e[0] for e in events}) != len(events)
        or any(e[1] is None or e[1] > cutoff or e[2] is not None and e[2] > cutoff for e in events)
        or any(
            any(d is not None for d in e[3:5]) and not any(d is not None and start <= d <= cutoff for d in e[3:5])
            for e in events
        )
    ):
        raise ValueError("cash history dividends are oversized, duplicated, outside window or not yet known")
    return tuple(CashNavPoint(*row) for row in rows), tuple(CashDividend(*row) for row in events)
