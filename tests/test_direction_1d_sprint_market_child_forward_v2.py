"""市场子模型使用已验证的无NAV输入父答案，仍单独约束真实保存和成熟结果。"""

from datetime import datetime

import pytest
from app.services import direction_1d_sprint as b
from app.services import direction_1d_sprint_market_child_forward_v2 as child
from app.services import direction_1d_sprint_market_gap_delta as s
from app.services import direction_1d_sprint_market_only_forward as parent_forward

from test_direction_1d_sprint_market_only_forward import forecast, observation
from test_direction_1d_sprint_market_only_forward import ready as parent_ready  # noqa: F401


@pytest.fixture
def ready(parent_ready, monkeypatch):  # noqa: F811
    clock, _, _ = parent_ready
    original = forecast(parent_ready)
    manifest = {"at": "2026-09-16T03:00:00+08:00", "plan_hash": "CHILD_PLAN", "model_sha256": "CHILD_MODEL"}
    b.save(s.root() / "result.json", manifest)
    monkeypatch.setattr(s, "models", lambda: (manifest, {}))
    monkeypatch.setattr(s, "live_answers", lambda *args: {"CHILD_CANDIDATE": {"prediction": 0, "research_score": 0.4}})
    monkeypatch.setattr(s, "live_market", lambda source: {"features": [1, -1, 1, 0.2, -0.3], "available": True})
    return clock, original


def test_shared_inputs_and_outcomes_keep_independent_immutable_child_answer(ready):
    clock, original = ready
    assert child.tick(s)["verified_forecasts"] == 1
    path = s.root() / "forward/2026-09-16/001000.json"
    saved = path.read_bytes()
    assert child.tick(s)["verified_forecasts"] == 1 and path.read_bytes() == saved
    assert not (b.ROOT / "forward").exists()
    clock[0] = datetime(2026, 9, 16, 20, tzinfo=b.ZONE)
    b.save(b.ROOT / "observations/20260916T190010.json", observation())
    parent_forward.observe_outcomes([original], b.now())
    report = child.report(s)
    assert report["matched"] == 1 and report["matched_forward_metrics"]["CHILD_CANDIDATE"]["accuracy"] == 0
    assert report["matched_forward_metrics"]["CANDIDATE"]["accuracy"] == 1
    assert not (s.root() / "outcomes").exists()


@pytest.mark.parametrize("fault", ["parent", "model", "plan", "input", "direction", "late"])
def test_child_rehashed_bindings_or_answer_are_rejected(ready, fault):
    child.tick(s)
    path = s.root() / "forward/2026-09-16/001000.json"
    value = b.read(path)
    receipt_path = s.root() / "receipts/2026-09-16/001000.json"
    receipt = b.read(receipt_path)
    if fault == "direction":
        value["answers"]["CHILD_CANDIDATE"]["prediction"] = 1
    elif fault == "late":
        receipt["readback_at"] = "2026-09-16T08:30:00+08:00"
    else:
        value[fault + "_hash"] = "CHANGED"
    b.save(path, value, replace=True)
    b.save(receipt_path, receipt | {"forecast_hash": b.digest(value)}, replace=True)
    report = child.report(s)
    assert report["verified_forecasts"] == 0 and len(report["invalid_forecasts"]) == 1


def test_missing_parent_is_not_synthesized_and_deadline_does_not_backfill(ready, monkeypatch):
    clock, _ = ready
    monkeypatch.setattr(child, "parent_context", lambda: {})
    assert child.tick(s)["verified_forecasts"] == 0
    clock[0] = datetime(2026, 9, 17, 13, tzinfo=b.ZONE)
    assert child.tick(s)["verified_forecasts"] == 0


def test_corrupt_shared_outcome_does_not_become_child_accuracy(ready):
    clock, original = ready
    child.tick(s)
    clock[0] = datetime(2026, 9, 16, 20, tzinfo=b.ZONE)
    b.save(b.ROOT / "observations/20260916T190010.json", observation())
    parent_forward.observe_outcomes([original], b.now())
    path = child.parent.root() / "outcomes/2026-09-16/001000.json"
    value = b.read(path)
    b.save(path, value | {"forecast_hash": "WRONG_PARENT"}, replace=True)
    with pytest.raises(ValueError, match="OUTCOME_CHANGED"):
        child.report(s)


def test_rehashed_extra_vector_is_not_accepted_even_with_same_answer(ready):
    child.tick(s)
    path = s.root() / "forward/2026-09-16/001000.json"
    value = b.read(path)
    value["market"]["features"][4] += 1
    b.save(path, value, replace=True)
    receipt = s.root() / "receipts/2026-09-16/001000.json"
    b.save(receipt, b.read(receipt) | {"forecast_hash": b.digest(value)}, replace=True)
    assert child.report(s)["verified_forecasts"] == 0


def test_extension_is_built_once_per_target_in_each_write_or_report_phase(ready, monkeypatch):
    _, original = ready
    second = original | {"code": "002000"}
    monkeypatch.setattr(
        child, "parent_context", lambda: {("001000", original["u"]): original, ("002000", original["u"]): second}
    )
    calls = []
    monkeypatch.setattr(
        s, "live_market", lambda source: calls.append(1) or {"features": [1, -1, 1, 0.2, -0.3], "available": True}
    )
    assert child.tick(s)["verified_forecasts"] == 2
    assert calls == [1, 1]
