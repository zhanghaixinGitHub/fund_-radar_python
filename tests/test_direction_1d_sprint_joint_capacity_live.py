"""联合视图保持实际来源身份；不接受旧目标、晚时点或改变的模型计划。"""

from copy import deepcopy

import pytest
from app.services import direction_1d_sprint_joint_capacity_live as s


@pytest.fixture
def actual_sources(monkeypatch):
    shared = {"t": "2026-09-16", "u": "2026-09-17", "at": "2026-09-17T08:15:00+08:00"}
    sources = {
        "futures": shared | {"snapshot": {"points": {"2026-09-16": {"price": 1}}}},
        "dollar": shared | {"points": {"2026-09-16": {"price": 2}}},
        "credit": shared | {"points": {"HYG": {"price": 3}, "IEF": {"price": 4}}},
    }
    monkeypatch.setattr(s.b, "read", lambda path: {"plan": path.parent.name})
    monkeypatch.setattr(s.futures, "live", lambda *args: deepcopy(sources["futures"]))
    monkeypatch.setattr(s.dollar, "load_live", lambda *args: deepcopy(sources["dollar"]))
    monkeypatch.setattr(s.credit, "load_live", lambda *args: deepcopy(sources["credit"]))
    monkeypatch.setattr(s.data, "features", lambda t, u, points: {"available": all(points.values())})
    return sources, s.b.digest({"plan": "round-113"})


def test_same_actual_sources_reproduce_identical_view_without_new_requests(actual_sources):
    sources, ph = actual_sources
    value = s.load_live("2026-09-16", "2026-09-17", ph)
    assert value == s.load_live("2026-09-16", "2026-09-17", ph)
    assert value["source_hashes"] == {k: s.b.digest(v) for k, v in sources.items()}
    assert value["new_source_requests"] == value["reserved_request_slots"] == 0
    sources["dollar"]["points"]["2026-09-16"]["price"] = 9
    changed = s.load_live("2026-09-16", "2026-09-17", ph)
    assert changed["receipt_hash"] != value["receipt_hash"]


@pytest.mark.parametrize("source", ["futures", "dollar", "credit"])
@pytest.mark.parametrize("change", ["old_target", "old_time", "late_time"])
def test_any_mismatched_or_outside_window_source_rejected(actual_sources, source, change):
    sources, ph = actual_sources
    if change == "old_target":
        sources[source]["u"] = "2026-09-16"
    else:
        sources[source]["at"] = "2026-09-16T08:15:00+08:00" if change == "old_time" else "2026-09-17T08:31:00+08:00"
    with pytest.raises(ValueError, match="SOURCE_(TARGET_CHANGED|TIME_INVALID)"):
        s.load_live("2026-09-16", "2026-09-17", ph)


def test_missing_actual_source_remains_unavailable(actual_sources):
    sources, ph = actual_sources
    sources["dollar"]["points"] = {}
    assert not s.load_live("2026-09-16", "2026-09-17", ph)["available"]


def test_different_model_plan_rejected_before_sources(actual_sources):
    with pytest.raises(ValueError, match="MODEL_PLAN_CHANGED"):
        s.load_live("2026-09-16", "2026-09-17", "different")
