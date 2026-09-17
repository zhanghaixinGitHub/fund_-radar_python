"""期权一日变化不跨缺日，不把晚到历史当作预测前来源。"""

from copy import deepcopy
from datetime import date

import numpy as np
import pytest
from app.services import direction_1d_sprint_option_change_data as s


@pytest.fixture(autouse=True)
def calendar(monkeypatch):
    monkeypatch.setattr(s.base, "calendar", lambda: ([date(2026, 9, d) for d in (14, 15, 16, 17)], "calendar"))


def point(v, oi):
    return {"log_put_call_volume": v, "log_put_call_oi": oi}


def test_adjacent_difference_and_missing_day():
    points = {"2026-09-14": point(-0.6, -0.5), "2026-09-16": point(-0.3, -0.4)}
    assert s.features("2026-09-16", points) is None
    points["2026-09-15"] = point(-0.4, -0.2)
    v = s.features("2026-09-16", points)
    assert v["previous_date"] == "2026-09-15"
    assert np.isclose(v["log_put_call_volume_change"], 0.1)
    assert np.isclose(v["log_put_call_oi_change"], -0.2)


@pytest.mark.parametrize("value", [None, True, float("nan"), float("inf")])
def test_invalid_ratio_not_zero(value):
    with pytest.raises(ValueError, match="CHANGE_VALUE_INVALID"):
        s.features("2026-09-16", {"2026-09-15": point(-0.3, -0.4), "2026-09-16": point(value, -0.5)})


@pytest.fixture
def live_sources(monkeypatch):
    source = {"at": "2026-09-16T11:01:00+08:00", "snapshot": {"rows": {"2026-09-15": point(-0.4, -0.2)}}}
    parent = {
        "at": "2026-09-17T08:15:00+08:00",
        "snapshot": {"points": {"2026-09-16": point(-0.3, -0.4)}},
        "available": True,
    }
    monkeypatch.setattr(s.old, "load_live", lambda *_: deepcopy(parent))
    monkeypatch.setattr(s.old, "history", lambda: deepcopy(source))
    return source, parent


def test_live_binds_actual_t_and_earlier_history(live_sources):
    history, parent = live_sources
    v = s.live("2026-09-16", "2026-09-17", "plan97")
    assert v["available"] and v["new_source_requests"] == 0
    assert v["previous_history_hash"] == s.base.digest(history) and v["parent_source_hash"] == s.base.digest(parent)
    assert np.isclose(v["snapshot"]["points"]["2026-09-16"]["log_put_call_oi_change"], -0.2)


def test_missing_previous_or_current_keeps_unavailable(live_sources):
    history, parent = live_sources
    parent["available"] = False
    assert not s.live("2026-09-16", "2026-09-17", "plan97")["available"]
    parent["available"] = True
    history["snapshot"]["rows"].clear()
    assert not s.live("2026-09-16", "2026-09-17", "plan97")["available"]


def test_late_or_already_backfilled_history_rejected(live_sources):
    history, _ = live_sources
    history["at"] = "2026-09-17T08:16:00+08:00"
    with pytest.raises(ValueError, match="PREVIOUS_HISTORY_NOT_SAVED"):
        s.live("2026-09-16", "2026-09-17", "plan97")
    history["at"] = "2026-09-16T11:01:00+08:00"
    history["snapshot"]["rows"]["2026-09-16"] = point(-0.3, -0.4)
    with pytest.raises(ValueError, match="LIVE_T_ALREADY_IN_HISTORY"):
        s.live("2026-09-16", "2026-09-17", "plan97")
