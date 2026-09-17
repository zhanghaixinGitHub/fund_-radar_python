"""联合源分别保留缺失和来源时间，不能把其中一个来源的记录代替另一个。"""

import pytest
from app.services import direction_1d_sprint_health_futures_data as d


def health():
    return {
        "available": True,
        "response_signal": 0.2,
        "new_us_dates": ["2026-09-16"],
        "max_prior_mature": "2026-09-15",
        "context_cutoff": "2026-09-16",
        "context_hash": "individual",
    }


def future():
    return {"available": True, "date": "2026-09-16", "intraday_return_pct": 0.3, "close_location": 0.4}


def test_joint_keeps_distinct_units_and_dates():
    value = d.joint(health(), future())
    assert value["available"] and value["response_signal"] == 0.2 and value["intraday_return_pct"] == 0.3
    assert value["date"] == "2026-09-16" and value["context_hash"] == "individual"


def test_missing_health_does_not_erase_observed_futures():
    h = health() | {"available": False, "response_signal": 0.0}
    value = d.joint(h, future())
    assert not value["available"] and value["if_available"] and value["intraday_return_pct"] == 0.3


def test_missing_future_cannot_be_reported_available():
    value = d.joint(health(), {"available": False, "date": "2026-09-16"})
    assert not value["available"] and value["response_available"] and value["close_location"] == 0


def test_live_preserves_both_source_hashes_and_later_capture(monkeypatch):
    plan = {"old": "plan"}
    monkeypatch.setattr(d.b, "read", lambda p: plan)
    h = {
        "at": "2026-09-17T08:14:10+08:00",
        "contexts": {},
        "points": {},
        "available": True,
        "context_manifest_hash": "ctx",
    }
    f = {"at": "2026-09-17T08:15:00+08:00", "snapshot": {"points": {}}, "available": False}
    monkeypatch.setattr(d.response, "live", lambda *a: h)
    monkeypatch.setattr(d.futures, "live", lambda *a: f)
    result = d.live("2026-09-16", "2026-09-17", d.b.digest(plan))
    assert result["response_source_hash"] == d.b.digest(h) and result["intraday_source_hash"] == d.b.digest(f)
    assert result["at"] == f["at"] and not result["available"] and result["new_source_requests"] == 0


def test_wrong_parent_plan_rejected(monkeypatch):
    monkeypatch.setattr(d.b, "read", lambda p: {"plan": "real"})
    with pytest.raises(ValueError, match="JOINT_PARENT_PLAN_CHANGED"):
        d.live("2026-09-16", "2026-09-17", "different")


def test_original_row_preserves_label_and_base_market():
    old = {"y": 0, "market": {"features": [1, 2, 3], "original_market": {}}}
    extended = old | {"market": old["market"] | {"health_futures": {"available": True}}}
    assert d.original_row(extended) == old
