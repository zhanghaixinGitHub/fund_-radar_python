"""方向交互使用同日正手数，拒绝缺失/非法持仓，不做HTTP。"""

from datetime import date

import numpy as np
import pytest
from app.services import direction_1d_sprint_futures_activity_data as s


def quote(**kw):
    return {"open": 100.0, "close": 102.0, "low": 99.0, "high": 103.0, "vol": 200.0, "oi": 100.0} | kw


def test_same_contract_activity_and_direction():
    v = s.features(quote(), "IF2609.CFX")
    assert v["main_contract"] == "IF2609.CFX"
    assert v["log_volume_to_oi"] == np.log(2)
    assert np.isclose(v["return_activity"], 2 * np.log(2))
    assert np.isclose(v["location_activity"], 0.5 * np.log(2))


@pytest.mark.parametrize("field", ["vol", "oi"])
@pytest.mark.parametrize("value", [0, -1, None, True, float("nan"), float("inf")])
def test_invalid_quantity_not_treated_as_neutral(field, value):
    with pytest.raises(ValueError, match="ACTIVITY_INVALID"):
        s.features(quote(**{field: value}), "IF2609.CFX")


def test_neutral_volume_ratio_and_zero_range_are_observed_not_missing():
    v = s.features(quote(vol=100, open=100, close=100, high=100, low=100), "IF2609.CFX")
    assert v["zero_range"] and v["return_activity"] == v["location_activity"] == 0


def test_missing_date_no_previous_fill_and_original_fields_preserved(monkeypatch):
    monkeypatch.setattr(
        s.base, "calendar", lambda: ([date(2026, 9, 15), date(2026, 9, 16), date(2026, 9, 17)], "calendar")
    )
    points = {"2026-09-15": s.features(quote(), "IF2609.CFX")}
    original = {"available": True, "core": 3}
    v = s.extend(original, "2026-09-16", "2026-09-17", points)
    assert not v["futures_activity"]["available"]
    assert s.original_row({"market": v, "y": 1}) == {"market": original, "y": 1}
