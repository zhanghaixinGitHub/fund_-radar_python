"""基线纠错目标、自然错误比例、时间隔离和真实预测接入。"""

from datetime import datetime

import numpy as np
import pytest
from app.services import direction_1d_sprint as b
from app.services import direction_1d_sprint_correction as s
from app.services import direction_1d_sprint_return_target as regression

from test_direction_1d_sprint_sequence import (
    ancestor_ready,  # noqa: F401
    sample_rows,
)
from test_direction_1d_sprint_sequence import ready as sequence_ready  # noqa: F401


def rows():
    return [r | {"z": s.vector(r["x"])} for r in sample_rows()]


class ConstantModel:
    def __init__(self, score=0.3):
        self.score = score

    def predict_proba(self, x):
        return np.asarray([[1 - self.score, self.score]] * len(x))


def trained(score):
    return {"model": ConstantModel(score), "mean": [0.0] * 8, "scale": [1.0] * 8}


@pytest.mark.parametrize("move", [-0.01, 0.0, 0.01])
def test_correction_threshold_preserves_baseline_until_strictly_exceeded(move):
    z = [move, abs(move), 1, 0, 0, 0, 0, 0]
    base_direction = int(move >= 0)
    assert s.answer(z, s.CANDIDATES[0], trained(0.55))["prediction"] == base_direction
    flipped = s.answer(z, s.CANDIDATES[0], trained(0.551))
    assert flipped["prediction"] == 1 - base_direction and flipped["flipped"]
    assert flipped["kind"] == "UNCALIBRATED_BASELINE_ERROR_SCORE"


@pytest.mark.parametrize("name", s.CANDIDATES)
def test_future_labels_do_not_change_error_model_or_scaling(name, monkeypatch):
    monkeypatch.setattr(s, "active", lambda: None)
    samples = rows()
    before = s.fit(samples, name, "2023-06-10")
    for r in samples:
        if r["mature"] >= "2023-06-10":
            r["z"], r["y"] = [float("nan")] * 8, 1 - r["y"]
    after = s.fit(samples, name, "2023-06-10")
    assert before["fit_hash"] == after["fit_hash"]
    assert before["mean"] == after["mean"] and before["scale"] == after["scale"]
    assert before["max_mature_date"] < "2023-06-10"
    probe = np.asarray([[0.01, 0.01, 1, 0, 0, 0, 0, 0]])
    np.testing.assert_array_equal(before["model"].predict_proba(probe), after["model"].predict_proba(probe))


def test_error_target_keeps_natural_rate_instead_of_balancing_errors(monkeypatch):
    monkeypatch.setattr(s, "active", lambda: None)
    samples = rows()
    for i, r in enumerate(samples):
        direction = int(r["z"][0] >= 0)
        r["y"] = 1 - direction if i % 5 == 0 else direction
    result = s.fit(samples, s.CANDIDATES[0], "2023-07-01")
    expected = np.average([int(int(r["z"][0] >= 0) != r["y"]) for r in samples], weights=regression.weights(samples))
    assert result["natural_error_rate"] == pytest.approx(expected)
    assert result["natural_error_rate"] == pytest.approx(0.2)


def test_feature_units_and_invalid_scores_are_explicit():
    x = [0.0] * 32
    x[30], x[31], x[3], x[7], x[18], x[23] = 0.01, 1, 0.02, 0.04, 0.03, -0.02
    assert s.vector(x) == pytest.approx([1, 1, 1, 2, 0, 0, 3, -2])
    with pytest.raises(ValueError, match="CORRECTION_SCORE_INVALID"):
        s.answer(s.vector(x), s.CANDIDATES[0], trained(float("nan")))


@pytest.fixture
def ready(sequence_ready, monkeypatch):  # noqa: F811
    b.save(s.root() / "result.json", {"winner": s.CANDIDATES[0], "model_sha256": "abc"})
    bundle = {n: {"CN_EQUITY": trained(0.3)} for n in s.CANDIDATES}
    monkeypatch.setattr(s, "models", lambda: ({"model_sha256": "abc"}, bundle))
    return sequence_ready


def test_new_version_does_not_backfill_old_target(ready, monkeypatch):
    monkeypatch.setattr(s, "models", lambda: pytest.fail("old target must not load correction model"))
    assert s.tick()["verified_forecasts"] == 0


def test_true_predictions_are_immutable_and_outcomes_match(ready, monkeypatch):
    monkeypatch.setattr(s, "FIRST_TARGET", "2026-09-15")
    directory, original, _ = ready
    assert s.tick()["verified_forecasts"] == 1
    path = s.root() / "forward/2026-09-15/001000.json"
    before = path.read_bytes()
    s.tick()
    assert path.read_bytes() == before
    b.save(
        directory / "outcomes/2026-09-15/001000.json",
        {"y": 1, "actual_direction": "UP", "forecast_hash": b.digest(original)},
    )
    monkeypatch.setattr(b, "now", lambda: datetime(2026, 9, 15, 19, tzinfo=b.ZONE))
    assert s.report()["matched_forward_metrics"][s.CANDIDATES[0]]["accuracy"] == 1


def test_late_answer_readback_is_invalid(ready, monkeypatch):
    monkeypatch.setattr(s, "FIRST_TARGET", "2026-09-15")
    save = b.save

    def crossing(path, value, **kwargs):
        save(path, value, **kwargs)
        if path.parent.parent == s.root() / "forward":
            monkeypatch.setattr(b, "now", lambda: datetime(2026, 9, 15, 8, 30, tzinfo=b.ZONE))

    monkeypatch.setattr(b, "save", crossing)
    report = s.tick()
    assert report["verified_forecasts"] == 0 and report["invalid_or_late"] == 1
    monkeypatch.setattr(s, "models", lambda: pytest.fail("postdeadline model loading"))
    assert s.tick()["verified_forecasts"] == 0


def test_changed_saved_feature_is_rejected(ready, monkeypatch):
    monkeypatch.setattr(s, "FIRST_TARGET", "2026-09-15")
    assert s.tick()["verified_forecasts"] == 1
    path = s.root() / "forward/2026-09-15/001000.json"
    value = b.read(path)
    value["z"][0] += 1
    b.save(path, value, replace=True)
    receipt_path = s.root() / "receipts/2026-09-15/001000.json"
    b.save(receipt_path, b.read(receipt_path) | {"forecast_hash": b.digest(value)}, replace=True)
    with pytest.raises(ValueError, match="ROUND_13_VECTOR_CHANGED"):
        s.report()


def test_prior_runner_failure_still_attempts_correction_branch(monkeypatch):
    from scripts import direction_1d_sprint_correction as entry

    monkeypatch.setattr(entry.existing, "run", lambda: (_ for _ in ()).throw(ValueError("OLD_FAILED")))
    calls = []
    monkeypatch.setattr(entry.correction, "tick", lambda: calls.append(1))
    monkeypatch.setattr(b, "save", lambda *args, **kwargs: None)
    with pytest.raises(ValueError, match="INDEPENDENT_BRANCH_FAILED"):
        entry.run()
    assert calls == [1]
