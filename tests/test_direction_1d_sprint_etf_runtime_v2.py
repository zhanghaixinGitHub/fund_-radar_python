"""模型方向严格核对、浮点求和容差以及两个真实预测版本的完整证据链。"""

from copy import deepcopy
from datetime import datetime

import numpy as np
import pytest
from app.services import direction_1d_sprint as b
from app.services import direction_1d_sprint_etf_runtime_v2 as runtime
from app.services import direction_1d_sprint_us_etf_direct as s

from test_direction_1d_sprint_hk import ready as hk_ready  # noqa: F401
from test_direction_1d_sprint_sequence import ancestor_ready  # noqa: F401
from test_direction_1d_sprint_sequence import ready as sequence_ready  # noqa: F401
from test_direction_1d_sprint_us_etf_direct import ready as direct_ready  # noqa: F401


def answers():
    return {
        "fixed": {
            "research_score": 0.51,
            "prediction": 1,
            "baseline_prediction": 0,
            "flipped": True,
            "kind": "UNCALIBRATED_UP_SCORE",
        }
    }


def test_roundoff_is_allowed_but_direction_and_fields_are_exact():
    saved, expected = answers(), answers()
    saved["fixed"]["research_score"] = float(np.nextafter(0.51, 1.0))
    assert runtime.answers_match(saved, expected)
    for key, value in (("prediction", 0), ("baseline_prediction", 1), ("flipped", False), ("kind", "error")):
        changed = deepcopy(saved)
        changed["fixed"][key] = value
        assert not runtime.answers_match(changed, expected)
    assert not runtime.answers_match({}, expected)


@pytest.mark.parametrize("score", [0.51 + 1e-8, float("nan"), float("inf"), -0.1, 1.1, True, "0.51"])
def test_non_roundoff_or_invalid_score_rejected(score):
    saved = answers()
    saved["fixed"]["research_score"] = score
    assert not runtime.answers_match(saved, answers())


@pytest.fixture
def ready(direct_ready, monkeypatch):  # noqa: F811
    manifest = {"version": runtime.VERSION, "legacy_forecasts": {}, "marker": "runtime-not-model"}
    monkeypatch.setattr(runtime, "verify", lambda model: manifest)
    monkeypatch.setattr(s, "FIRST_TARGET", "2026-09-15")
    return direct_ready, manifest


def test_live_roundoff_keeps_original_bytes_and_future_outcome(ready, monkeypatch):
    (directory, original, _), manifest = ready
    assert runtime.tick(s)["verified_forecasts"] == 1
    path = s.root() / "forward/2026-09-15/001000.json"
    before = path.read_bytes()
    value = b.read(path)
    assert value["runtime_manifest_hash"] == b.digest(manifest)
    assert value["model_hash"] == "abc"
    original_answer = s.answer

    def roundoff(*args, **kwargs):
        out = original_answer(*args, **kwargs)
        return out | {"research_score": float(np.nextafter(out["research_score"], 1.0))}

    monkeypatch.setattr(s, "answer", roundoff)
    assert runtime.tick(s)["verified_forecasts"] == 1
    assert path.read_bytes() == before
    b.save(
        directory / "outcomes/2026-09-15/001000.json",
        {"y": 1, "actual_direction": "UP", "forecast_hash": b.digest(original)},
    )
    monkeypatch.setattr(b, "now", lambda: datetime(2026, 9, 15, 19, tzinfo=b.ZONE))
    assert runtime.report(s)["matched_forward_metrics"][s.CANDIDATES[0]]["accuracy"] == 1


@pytest.mark.parametrize(
    "change,pattern",
    [
        ("direction", "SAVED_ANSWER"),
        ("vector", "VECTOR"),
        ("runtime", "FORECAST_BINDING"),
        ("model", "PARENT_OR_MODEL"),
    ],
)
def test_rehashed_edits_still_fail(ready, change, pattern):
    runtime.tick(s)
    path = s.root() / "forward/2026-09-15/001000.json"
    value = b.read(path)
    if change == "direction":
        value["answers"][s.CANDIDATES[0]]["prediction"] = 0
    elif change == "vector":
        value["z"][0] += 1
    elif change == "runtime":
        value["runtime_manifest_hash"] = "changed"
    else:
        value["model_hash"] = "changed"
    b.save(path, value, replace=True)
    receipt = s.root() / "receipts/2026-09-15/001000.json"
    b.save(receipt, b.read(receipt) | {"forecast_hash": b.digest(value)}, replace=True)
    with pytest.raises(ValueError, match=pattern):
        runtime.report(s)


def test_late_readback_is_not_verified(ready, monkeypatch):
    save = b.save

    def late(path, value, **kwargs):
        save(path, value, **kwargs)
        if path.parent.parent == s.root() / "forward":
            monkeypatch.setattr(b, "now", lambda: datetime(2026, 9, 15, 8, 30, tzinfo=b.ZONE))

    monkeypatch.setattr(b, "save", late)
    result = runtime.tick(s)
    assert result["verified_forecasts"] == 0 and result["invalid_or_late"] == 1


def test_both_new_branches_execute_if_old_branch_fails(monkeypatch):
    from scripts import direction_1d_sprint_etf_runtime_v2 as entry

    assert entry.existing.__name__ == "scripts.direction_1d_sprint_convertible"
    monkeypatch.setattr(entry.existing, "run", lambda: (_ for _ in ()).throw(ValueError("OLD_FAILED")))
    calls = []
    monkeypatch.setattr(runtime, "tick", lambda model: calls.append(model))
    monkeypatch.setattr(b, "save", lambda *args, **kwargs: None)
    with pytest.raises(ValueError, match="INDEPENDENT_BRANCH_FAILED"):
        entry.run()
    assert calls == [runtime.r48, runtime.r49]


def test_unregistered_old_answer_rejected(ready):
    _, manifest = ready
    path = s.root() / "forward/2026-09-15/001000.json"
    with pytest.raises(ValueError, match="UNREGISTERED_LEGACY"):
        runtime.validate_runtime_binding(s, path, {"code": "001000"}, manifest)


def test_changed_runtime_code_cannot_bypass_frozen_manifest(tmp_path, monkeypatch):
    monkeypatch.setattr(b, "ROOT", tmp_path)
    monkeypatch.setattr(b, "calendar", lambda: ([], "calendar"))
    monkeypatch.setattr(runtime, "fingerprint", lambda: {"code": {"module": "changed"}})
    b.save(
        runtime.folder() / "plan.json",
        {
            "version": runtime.VERSION,
            "score_atol": runtime.SCORE_ATOL,
            "fingerprint": {"code": {"module": "frozen"}},
            "calendar_hash": "calendar",
        },
    )
    with pytest.raises(ValueError, match="CODE_OR_CALENDAR_CHANGED"):
        runtime.verify(s)
