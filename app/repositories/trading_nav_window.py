"""只读取一个基金/来源的净值日期、公告日期，不取价格、未来收益或测试标签。"""

from dataclasses import dataclass
from datetime import date
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.fund import NavDaily


@dataclass(frozen=True)
class NavDatePoint:
    nav_date: date  # 源净值的业务日；可能是周末，交给独立日历判断。
    ann_date: date | None  # 源记录的公告日；无此信息不能当作已知历史。


def read_nav_dates(
    session: Session, *, fund_code: str, source_id: UUID, start: date, end: date
) -> tuple[NavDatePoint, ...]:
    """单条有界查询，不以数据库记录数决定日历终点，也不读取任何净值数值。"""
    if not 0 <= (end - start).days < 192:
        raise ValueError("date window exceeds bounded read range")
    rows = session.execute(
        select(NavDaily.nav_date, NavDaily.ann_date)
        .where(
            NavDaily.fund_code == fund_code,
            NavDaily.source_id == source_id,
            NavDaily.nav_date >= start,
            NavDaily.nav_date <= end,
        )
        .order_by(NavDaily.nav_date)
        .limit(193)
    ).all()
    if len(rows) > 192:
        raise ValueError("NAV date read exceeds expected bound")
    return tuple(NavDatePoint(*row) for row in rows)
