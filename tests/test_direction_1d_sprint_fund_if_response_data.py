"""验证逐基金上下文选择、晚于实际采集的上下文拒绝和原始来源摘要绑定。"""

from copy import deepcopy
from datetime import date

import pytest
from app.services import direction_1d_sprint_fund_if_response_data as s


def context(beta):
    return {
        "cutoff": "2026-09-16",
        "available": True,
        "count": 126,
        "beta": beta,
        "max_u": "2026-09-14",
        "max_mature": "2026-09-15",
        "sample_hash": "fixture",
    }


def test_same_market_can_produce_different_fund_responses(monkeypatch):
    monkeypatch.setattr(s.b, "calendar", lambda: ([date(2026, 9, 16), date(2026, 9, 17)], "fixture"))
    points = {"2026-09-16": {"intraday_return_pct": 2.0, "close_location": 0.5}}
    contexts = {"a": context([0.4, 0.5]), "b": context([-0.4, -0.5])}
    a, z = [s.extend({}, "2026-09-16", "2026-09-17", points, code, contexts) for code in ("a", "b")]
    assert a["futures_intraday"] == z["futures_intraday"]
    assert a["fund_if_response"]["response_return"] == -z["fund_if_response"]["response_return"] == 0.8
    with pytest.raises(ValueError, match="FUND_CONTEXT_MISSING"):
        s.extend({}, "2026-09-16", "2026-09-17", points, "unknown", contexts)


@pytest.fixture
def live_input(monkeypatch):
    parent = {
        "at": "2026-09-17T08:14:20+08:00",
        "available": True,
        "snapshot": {"points": {"2026-09-16": {"intraday_return_pct": 2, "close_location": 0.5}}},
    }
    frozen = {"at": "2026-09-16T14:00:00+08:00", "cutoff": "2026-09-16", "contexts": {"a": context([0.4, 0.5])}}
    monkeypatch.setattr(s.source.data, "live", lambda *_: deepcopy(parent))
    monkeypatch.setattr(s.b, "read", lambda _: {"plan": "fixture"})
    monkeypatch.setattr(s, "current", lambda: frozen)
    return parent, frozen


def test_actual_if_snapshot_and_frozen_context_have_separate_hashes(live_input):
    parent, frozen = live_input
    value = s.live("2026-09-16", "2026-09-17", s.b.digest({"plan": "fixture"}))
    assert value["at"] == parent["at"] and value["parent_source_hash"] == s.b.digest(parent)
    assert value["context_manifest_hash"] == s.b.digest(frozen) and value["new_source_requests"] == 0


def test_context_saved_after_actual_quote_cannot_be_backdated(live_input):
    _, frozen = live_input
    frozen["at"] = "2026-09-17T08:15:00+08:00"
    with pytest.raises(ValueError, match="CONTEXT_AFTER_ACTUAL_CAPTURE"):
        s.live("2026-09-16", "2026-09-17", s.b.digest({"plan": "fixture"}))


def test_wrong_source_plan_is_rejected(live_input):
    with pytest.raises(ValueError, match="LIVE_PARENT_PLAN_CHANGED"):
        s.live("2026-09-16", "2026-09-17", "other-plan")


def test_original_identity_removes_both_added_layers_only():
    row = {"y": 1, "market": {"original": "kept", "futures_intraday": {}, "fund_if_response": {}}}
    assert s.original_row(row) == {"y": 1, "market": {"original": "kept"}}
