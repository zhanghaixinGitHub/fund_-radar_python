"""新增个体模型只能复用真实父来源，不能发请求、回填历史或接受不同统计截止。"""

import pytest
from app.services import direction_1d_sprint_fund_moneyflow_live as live


@pytest.fixture
def source(monkeypatch):
    plan = {"synthetic": "plan"}
    source = {
        "at": "2026-09-17T08:02:00+08:00",
        "receipt_hash": "receipt",
        "points": {"2026-09-16": {"available": True}},
        "available": True,
        "received_at": "2026-09-17T08:01:50+08:00",
    }
    current = {"cutoff": "2026-09-16", "contexts": {"fund": {"beta": [0.1, -0.2]}}}
    monkeypatch.setattr(live.b, "read", lambda path: plan)
    monkeypatch.setattr(live.parent, "load_live", lambda t, u, ph: source)
    monkeypatch.setattr(live.data, "reference", lambda: ({}, {}, {"binding": "synthetic"}, current))
    return plan, source, current


def test_reuses_exact_source_and_current_snapshot_without_new_requests(source):
    plan, parent, current = source
    value = live.load_live("2026-09-16", "2026-09-17", live.b.digest(plan))
    assert value["points"] == parent["points"] and value["parent_source_hash"] == live.b.digest(parent)
    assert value["frozen_current"] == current and value["current_contexts_hash"] == live.b.digest(current)
    assert value["new_source_requests"] == value["reserved_request_slots"] == 0


def test_model_plan_mismatch_rejected(source):
    with pytest.raises(ValueError, match="LIVE_MODEL_PLAN_CHANGED"):
        live.load_live("2026-09-16", "2026-09-17", "changed")


def test_late_or_changed_current_statistics_rejected(source):
    plan, _, current = source
    current["cutoff"] = "2026-09-17"
    with pytest.raises(ValueError, match="LIVE_CONTEXT_NOT_MATURE"):
        live.load_live("2026-09-16", "2026-09-17", live.b.digest(plan))


def test_parent_failure_not_replaced_by_history(source, monkeypatch):
    def fail(*a):
        raise RuntimeError("ACTUAL_PARENT_MISSING")

    monkeypatch.setattr(live.parent, "load_live", fail)
    with pytest.raises(RuntimeError, match="ACTUAL_PARENT_MISSING"):
        live.load_live("2026-09-16", "2026-09-17", live.b.digest(source[0]))


def test_missing_parent_stays_unavailable(source):
    plan, parent, _ = source
    parent.update(available=False, points={})
    value = live.load_live("2026-09-16", "2026-09-17", live.b.digest(plan))
    assert not value["available"] and value["points"] == {}
