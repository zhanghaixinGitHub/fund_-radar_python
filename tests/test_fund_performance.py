"""基金列表涨跌率计算的离线单元测试。"""

from datetime import date
from decimal import Decimal

import pytest
from app.repositories.fund_read import FundNavHistorySnapshot, _build_performance


def _point(nav_date: date, accumulated_nav: str, unit_nav: str | None = None) -> FundNavHistorySnapshot:
    """构造一条最小历史净值点；默认单位净值与累计净值相同。"""
    unit_value = unit_nav if unit_nav is not None else accumulated_nav
    return FundNavHistorySnapshot(
        nav_date=nav_date,
        unit_nav=Decimal(unit_value),
        accumulated_nav=Decimal(accumulated_nav),
        source_code="TUSHARE_PRO_FUND",
    )


def test_performance_uses_explicit_unit_nav_change_not_accumulated_ratio() -> None:
    """已有历史现金分红不能扩大本期分母；仅展示明确的单位净值变化。"""
    performance = _build_performance(
        (
            _point(date(2026, 7, 27), "1.0000", "0.9000"),
            _point(date(2026, 8, 19), "1.0500", "0.9450"),
            _point(date(2026, 8, 25), "1.0800", "0.9700"),
            _point(date(2026, 8, 26), "1.1000", "0.9800"),
        )
    )

    assert performance.day_change_rate == Decimal("0.9800") / Decimal("0.9700") - Decimal("1")
    assert performance.week_change_rate == Decimal("0.9800") / Decimal("0.9450") - Decimal("1")
    assert performance.month_change_rate == Decimal("0.9800") / Decimal("0.9000") - Decimal("1")


def test_performance_keeps_missing_baseline_empty() -> None:
    """新基金或历史不完整时不得伪造一周、一月涨跌率。"""
    performance = _build_performance((_point(date(2026, 8, 26), "1.1000"),))

    assert performance.day_change_rate is None
    assert performance.week_change_rate is None
    assert performance.month_change_rate is None


@pytest.mark.parametrize("value", ["0", "-1", "NaN", "Infinity"])
@pytest.mark.parametrize("invalid_at_end", [True, False])
def test_invalid_endpoint_remains_unknown(value: str, invalid_at_end: bool) -> None:
    """任何端点无效时保持未知，不以累计净值替代，也不伪装成零变化。"""
    before = value if not invalid_at_end else "1"
    after = value if invalid_at_end else "1"
    result = _build_performance(
        (
            _point(date(2026, 9, 1), "2", before),
            _point(date(2026, 9, 2), "2.1", after),
        )
    )
    assert result.day_change_rate is None


@pytest.mark.parametrize("source", [None, "OTHER_SOURCE"])
def test_source_change_is_not_compared(source: str | None) -> None:
    """来源缺失或变更时，不把两个不同来源净值拼成涨跌。"""
    result = _build_performance(
        (
            _point(date(2026, 9, 1), "2", "1"),
            FundNavHistorySnapshot(date(2026, 9, 2), Decimal("1.1"), Decimal("2.1"), source),
        )
    )
    assert result.day_change_rate is None


def test_cash_distribution_does_not_silently_become_total_return() -> None:
    """除息造成的单位净值下降仍是净值变化，页面说明它不代表含分红收益。"""
    result = _build_performance(
        (
            _point(date(2026, 9, 1), "2", "1"),
            _point(date(2026, 9, 2), "2", "0.9"),
        )
    )
    assert result.day_change_rate == Decimal("-0.1")
