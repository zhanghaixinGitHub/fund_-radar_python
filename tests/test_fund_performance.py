"""基金列表涨跌率计算的离线单元测试。"""

from datetime import date
from decimal import Decimal

from app.repositories.fund_read import FundNavHistorySnapshot, _build_performance


def _point(
    nav_date: date, accumulated_nav: str, unit_nav: str | None = None
) -> FundNavHistorySnapshot:
    """构造一条最小历史净值点；默认单位净值与累计净值相同。"""
    unit_value = unit_nav if unit_nav is not None else accumulated_nav
    return FundNavHistorySnapshot(
        nav_date=nav_date,
        unit_nav=Decimal(unit_value),
        accumulated_nav=Decimal(accumulated_nav),
    )


def test_performance_uses_accumulated_nav_for_day_week_and_month() -> None:
    """累计净值可用时，三段涨跌率都不受单位净值分红除权影响。"""
    performance = _build_performance(
        (
            _point(date(2026, 7, 27), "1.0000", "0.9000"),
            _point(date(2026, 8, 19), "1.0500", "0.9450"),
            _point(date(2026, 8, 25), "1.0800", "0.9700"),
            _point(date(2026, 8, 26), "1.1000", "0.9800"),
        )
    )

    assert performance.day_change_rate == Decimal("1.1000") / Decimal("1.0800") - Decimal("1")
    assert performance.week_change_rate == Decimal("1.1000") / Decimal("1.0500") - Decimal("1")
    assert performance.month_change_rate == Decimal("1.1000") / Decimal("1.0000") - Decimal("1")


def test_performance_keeps_missing_baseline_empty() -> None:
    """新基金或历史不完整时不得伪造一周、一月涨跌率。"""
    performance = _build_performance((_point(date(2026, 8, 26), "1.1000"),))

    assert performance.day_change_rate is None
    assert performance.week_change_rate is None
    assert performance.month_change_rate is None
