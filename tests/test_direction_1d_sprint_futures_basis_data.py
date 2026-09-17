"""价差特征严格滞后一交易日，春节等跨假期按实际交易日历处理。"""

import pytest
from app.services import direction_1d_sprint_futures_basis_data as d


def test_one_china_day_lag_across_long_holiday(monkeypatch):
    monkeypatch.setattr(d.base, "calendar", lambda: (["2025-01-24", "2025-01-27", "2025-02-05"], "hash"))
    values = {"2025-01-24": {"basis_pct": -0.8}, "2025-01-27": {"basis_pct": 2.0}}
    result = d.extend({"features": [1, 2, 3]}, "2025-01-27", "2025-02-05", values)
    assert result["futures_basis"]["date"] == "2025-01-24"
    assert result["futures_basis"]["basis_pct"] == -0.8


def test_missing_lagged_date_not_filled_with_later_value(monkeypatch):
    monkeypatch.setattr(d.base, "calendar", lambda: (["2025-01-24", "2025-01-27", "2025-02-05"], "hash"))
    result = d.extend({}, "2025-01-27", "2025-02-05", {"2025-01-27": {"basis_pct": 2.0}})
    assert result["futures_basis"] == {"date": "2025-01-24", "available": False}


def test_skip_target_day_is_rejected(monkeypatch):
    monkeypatch.setattr(d.base, "calendar", lambda: (["2025-01-24", "2025-01-27", "2025-02-05"], "hash"))
    with pytest.raises(ValueError, match="ADJACENCY"):
        d.extend({}, "2025-01-24", "2025-02-05", {})


def test_original_market_and_label_preserved():
    row = {"y": 1, "market": {"features": [1, 2, 3], "available": False}}
    assert d.original_row(row | {"market": row["market"] | {"futures_basis": {"available": False}}}) == row
