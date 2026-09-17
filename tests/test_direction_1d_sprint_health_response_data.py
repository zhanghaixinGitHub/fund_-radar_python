"""同日不同基金使用不同上下文，实际源和更早冻结的统计不能互相替代。"""

from copy import deepcopy

import pytest
from app.services import direction_1d_sprint_health_response_data as d


def prior(beta):
    return {
        "cutoff": "2026-09-16",
        "count": 126,
        "available": True,
        "beta": beta,
        "max_u": "2026-09-14",
        "max_mature": "2026-09-15",
        "sample_hash": "past",
    }


def test_same_group_funds_have_their_own_signal(monkeypatch):
    monkeypatch.setattr(
        d.source.data,
        "features",
        lambda *a: {"available": True, "intraday_log_pct": 2.0, "new_us_dates": ["2026-09-16"]},
    )
    contexts = {"a": prior(0.1), "b": prior(-0.2)}
    first = d.extend({}, "2026-09-16", "2026-09-17", {}, "a", contexts)["health_response"]
    second = d.extend({}, "2026-09-16", "2026-09-17", {}, "b", contexts)["health_response"]
    assert first["response_signal"] == 0.2 and second["response_signal"] == -0.4
    assert first["context_hash"] != second["context_hash"]


def test_unknown_fund_cannot_borrow_another_context():
    with pytest.raises(ValueError, match="FUND_CONTEXT_MISSING"):
        d.extend({}, "2026-09-16", "2026-09-17", {}, "c", {"a": prior(0.1)})


def test_current_day_context_maturity_rejected(monkeypatch):
    monkeypatch.setattr(
        d.source.data,
        "features",
        lambda *a: {"available": True, "intraday_log_pct": 1.0, "new_us_dates": ["2026-09-16"]},
    )
    context = prior(0.1)
    context["max_mature"] = "2026-09-16"
    with pytest.raises(ValueError, match="NOT_MATURE"):
        d.extend({}, "2026-09-16", "2026-09-17", {}, "a", {"a": context})


@pytest.fixture
def source_and_context(monkeypatch):
    actual = {"at": "2026-09-17T08:14:10+08:00", "available": True, "points": {}}
    frozen = {"at": "2026-09-16T13:40:00+08:00", "cutoff": "2026-09-16", "contexts": {"a": prior(0.1)}}
    calls = []

    def loaded(t, u, ph):
        calls.append((t, u, ph))
        return actual

    monkeypatch.setattr(d.live_source, "load_live", loaded)
    monkeypatch.setattr(d, "current", lambda: frozen)
    return actual, frozen, calls


def test_live_reuses_parent_bytes_and_frozen_context(source_and_context):
    actual, frozen, calls = source_and_context
    value = d.live("2026-09-16", "2026-09-17", "r104-plan")
    assert calls == [("2026-09-16", "2026-09-17", "r104-plan")]
    assert value["health_source_hash"] == d.b.digest(actual)
    assert value["context_manifest_hash"] == d.b.digest(frozen) and value["new_source_requests"] == 0


def test_context_created_after_source_cannot_be_backdated(source_and_context):
    _, frozen, _ = source_and_context
    frozen["at"] = "2026-09-17T08:15:00+08:00"
    with pytest.raises(ValueError, match="CURRENT_CONTEXT_AFTER_CAPTURE"):
        d.live("2026-09-16", "2026-09-17", "r104-plan")


def test_original_row_removes_only_new_context():
    row = {"y": 1, "market": {"features": [1, 2, 3], "health_response": {"response_signal": 0.2}}}
    before = deepcopy(row)
    assert d.original_row(row) == {"y": 1, "market": {"features": [1, 2, 3]}}
    assert row == before
