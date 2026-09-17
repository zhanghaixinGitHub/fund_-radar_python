"""读取复用来源时沿用原父计划和实收时间；不允许新计划替换原请求身份。"""

from copy import deepcopy

import pytest
from app.services import direction_1d_sprint_stock_interaction_live as s


@pytest.fixture
def source(monkeypatch):
    value = {
        "at": "2026-09-17T08:15:00+08:00",
        "received_at": "2026-09-17T08:14:58+08:00",
        "receipt_hash": "actual-receipt",
        "points": {"2026-09-16": {"breadth": 0.4}},
        "available": True,
    }
    monkeypatch.setattr(s.b, "read", lambda path: {"plan": path.parent.name})
    calls = []

    def load(t, u, ph):
        calls.append((t, u, ph))
        return deepcopy(value)

    monkeypatch.setattr(s.parent, "load_live", load)
    return value, calls, s.b.digest({"plan": "round-114"})


def test_preserves_parent_receipt_and_plan_with_no_new_request(source):
    value, calls, ph = source
    actual = s.load_live("2026-09-16", "2026-09-17", ph)
    assert calls == [("2026-09-16", "2026-09-17", s.b.digest({"plan": "round-109"}))]
    assert actual["parent_source_hash"] == s.b.digest(value)
    assert actual["at"] == value["at"] and actual["receipt_hash"] == value["receipt_hash"]
    assert actual["new_source_requests"] == actual["reserved_request_slots"] == 0


def test_missing_parent_remains_missing_not_neutral(source):
    value, _, ph = source
    value.update(available=False, points={}, received_at=None)
    actual = s.load_live("2026-09-16", "2026-09-17", ph)
    assert not actual["available"] and not actual["points"] and actual["received_at"] is None


def test_changed_model_plan_rejected_before_parent(source):
    _, calls, _ = source
    with pytest.raises(ValueError, match="INTERACTION_MODEL_PLAN_CHANGED"):
        s.load_live("2026-09-16", "2026-09-17", "changed")
    assert not calls
