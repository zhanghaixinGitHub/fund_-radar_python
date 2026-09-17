"""股票分布只对齐相邻T日；缺失不前填，来源范围和数值不能静默变化。"""

from datetime import date

import pytest
from app.services import direction_1d_sprint_stock_breadth_data as s


@pytest.fixture
def points(monkeypatch):
    monkeypatch.setattr(s.base, "calendar", lambda: ([date(2026, 9, d) for d in [15, 16, 17]], "fixture"))
    return {
        "2026-09-15": {
            "date": "2026-09-15",
            "available": True,
            "scope": "PROVIDER_RETURNED_SH_SZ_DAILY_QUOTES",
            "breadth": 0.5,
            "median_pct": 1.0,
            "iqr_pct": 2.0,
        }
    }


def test_missing_t_is_not_zero_or_prior_day_fill(points):
    value = s.extend({"original": 1}, "2026-09-16", "2026-09-17", points)
    assert value == {"original": 1, "stock_breadth": {"date": "2026-09-16", "available": False}}
    with pytest.raises(ValueError, match="ADJACENCY_INVALID"):
        s.extend({}, "2026-09-15", "2026-09-17", points)


@pytest.mark.parametrize(
    "field,value",
    [("date", "2026-09-16"), ("scope", "OTHER"), ("breadth", 1.1), ("median_pct", float("nan")), ("iqr_pct", -1)],
)
def test_malformed_feature_is_not_used(points, field, value):
    points["2026-09-15"][field] = value
    with pytest.raises(ValueError):
        s.extend({}, "2026-09-15", "2026-09-16", points)


def test_original_rows_preserved(points):
    row = {"y": 1, "market": {"original": 2}}
    added = row | {"market": s.extend(row["market"], "2026-09-15", "2026-09-16", points)}
    assert s.original_row(added) == row
