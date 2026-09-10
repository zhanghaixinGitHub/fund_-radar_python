"""页面实验 NAV V2 输入；有界、只读，不查询未来净值或保留测试数据。"""

from datetime import date
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.fund import NavDaily
from app.repositories.cash_prediction_features import read_known_cash_dividends, validate_history_read_bounds
from app.repositories.cash_reinvestment_samples import CashNavPoint


def read_experiment_history(
    session: Session, *, fund_code: str, source_id: UUID, dates: tuple[date, ...], cutoff: date
):
    """仅取固定61交易日；ann_date不否决NAV V2输入，原值仍进入输入指纹。"""
    validate_history_read_bounds(dates[0], cutoff)
    if len(dates) != 61 or len(set(dates)) != 61 or dates[-1] >= cutoff:
        raise ValueError("EXPERIMENT_HISTORY_BOUNDS")
    rows = session.execute(
        select(NavDaily.nav_date, NavDaily.ann_date, NavDaily.unit_nav)
        .where(
            NavDaily.fund_code == fund_code,
            NavDaily.source_id == source_id,
            NavDaily.nav_date.in_(dates),
        )
        .order_by(NavDaily.nav_date)
        .limit(62)
    ).all()
    if len(rows) > 61 or len({r[0] for r in rows}) != len(rows):
        raise ValueError("EXPERIMENT_HISTORY_DUPLICATED")
    events = read_known_cash_dividends(session, fund_code=fund_code, source_id=source_id, start=dates[0], cutoff=cutoff)
    return tuple(CashNavPoint(*r) for r in rows), events
