"""验证日历输入的相邻交易日边界、配方对照、未来隔离及提前预测证据。"""

from datetime import datetime

import numpy as np
import pytest
from app.services import direction_1d_sprint as b
from app.services import direction_1d_sprint_calendar as s
from app.services import direction_1d_sprint_fund_response as previous

from test_direction_1d_sprint_sequence import (
    ancestor_ready,  # noqa: F401
    sample_rows,
)
from test_direction_1d_sprint_sequence import ready as sequence_ready  # noqa: F401


def rows():
    return [r | {"z": s.correction.vector(r["x"]) + [1.0, 0, 0, 0, 0, 1 / 14, 0, 0]} for r in sample_rows()]


class ConstantModel:
    def __init__(self, score):
        self.score = score

    def predict_proba(self, x):
        return np.asarray([[1 - self.score, self.score]] * len(x))


def trained(name, score):
    size = 10 if name == s.CANDIDATES[0] else 16
    return {"model": ConstantModel(score), "mean": [0.0] * size, "scale": [1.0] * size}


def test_calendar_uses_exchange_sessions_not_available_exam_rows():
    values = s.calendar_vectors()
    assert values["2026-09-14", "2026-09-15"] == [0, 1, 0, 0, 0, 1 / 14, 0, 0]
    # 国庆休市跨月；必须由交易日历识别首尾，不能按是否有净值或标签决定。
    assert values["2025-09-29", "2025-09-30"][-1] == 1
    assert values["2025-09-30", "2025-10-09"][5:] == [9 / 14, 1, 0]
    with pytest.raises(ValueError, match="CALENDAR_TARGET_NOT_ADJACENT"):
        s.vector([0.0] * 32, "2026-09-14", "2026-09-16")


def test_dataset_features_do_not_depend_on_target_answer(monkeypatch):
    original = {"t": "2026-09-14", "u": "2026-09-15", "x": [0.0] * 32, "y": 0}
    monkeypatch.setattr(s.overnight, "dataset", lambda: [original])
    before = s.dataset()[0][0]["z"]
    original["y"] = 1
    assert s.dataset()[0][0]["z"] == before


@pytest.mark.parametrize("name", s.CANDIDATES)
def test_unmatured_future_labels_and_inputs_do_not_change_fitted_model(name, monkeypatch):
    monkeypatch.setattr(s, "active", lambda: None)
    samples = rows()
    before = s.fit(samples, name, "2023-06-10")
    for r in samples:
        if r["mature"] >= "2023-06-10":
            r["z"], r["y"] = [float("nan")] * 16, 1 - r["y"]
    after = s.fit(samples, name, "2023-06-10")
    assert before["fit_hash"] == after["fit_hash"]
    assert before["mean"] == after["mean"] and before["scale"] == after["scale"]
    assert before["max_mature_date"] < "2023-06-10"
    probe = [np.zeros(len(before["mean"]))]
    np.testing.assert_array_equal(before["model"].predict_proba(probe), after["model"].predict_proba(probe))


def test_logistic_without_calendar_variation_matches_original_recipe(monkeypatch):
    monkeypatch.setattr(s, "active", lambda: None)
    samples = rows()
    current = s.fit(samples, s.CANDIDATES[0], "2023-06-10")
    control = previous.fit_lr(previous.selected(samples, "2023-06-10"))
    probe = samples[40]
    score = control["model"].predict_proba([probe["x"][30:32]])[0, 1]
    assert s.answer(probe["z"], s.CANDIDATES[0], current)["research_score"] == pytest.approx(score, abs=1e-10)


def test_correction_preserves_natural_error_prior(monkeypatch):
    monkeypatch.setattr(s, "active", lambda: None)
    samples = rows()
    for i, r in enumerate(samples):
        baseline = int(r["z"][0] >= 0)
        r["y"] = 1 - baseline if i % 5 == 0 else baseline
    result = s.fit(samples, s.CANDIDATES[1], "2023-07-01")
    assert result["weighted_target_rate"] == pytest.approx(0.2)


@pytest.mark.parametrize("move", [-0.01, 0.0, 0.01])
def test_strict_direction_and_correction_thresholds_are_unchanged(move):
    z = [move, abs(move), 1, 0, 0, 0, 0, 0] + [0.0] * 8
    lr, extra = s.CANDIDATES
    assert s.answer(z, lr, trained(lr, 0.5))["prediction"] == 0
    assert s.answer(z, lr, trained(lr, 0.501))["prediction"] == 1
    baseline = int(move >= 0)
    assert s.answer(z, extra, trained(extra, 0.55))["prediction"] == baseline
    assert s.answer(z, extra, trained(extra, 0.551))["prediction"] == 1 - baseline


@pytest.mark.parametrize("name", s.CANDIDATES)
def test_batch_and_single_predictions_match_and_bad_scores_fail(name):
    values = [r["z"] for r in rows()[:3]]
    model = trained(name, 0.6)
    assert s.batch_answers(values, name, model) == [s.answer(z, name, model) for z in values]
    with pytest.raises(ValueError, match="CALENDAR_SCORE_INVALID"):
        s.answer(values[0], name, trained(name, float("nan")))


@pytest.fixture
def ready(sequence_ready, monkeypatch):  # noqa: F811
    b.save(s.root() / "result.json", {"winner": s.CANDIDATES[0], "model_sha256": "abc"})
    bundle = {n: {"CN_EQUITY": trained(n, 0.8 if n == s.CANDIDATES[0] else 0.3)} for n in s.CANDIDATES}
    monkeypatch.setattr(s, "models", lambda: ({"model_sha256": "abc"}, bundle))
    return sequence_ready


def test_new_version_does_not_backfill_old_target(ready, monkeypatch):
    monkeypatch.setattr(s, "models", lambda: pytest.fail("old target must not load calendar model"))
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
    result = s.tick()
    assert result["verified_forecasts"] == 0 and result["invalid_or_late"] == 1


def test_changed_calendar_feature_is_rejected(ready, monkeypatch):
    monkeypatch.setattr(s, "FIRST_TARGET", "2026-09-15")
    assert s.tick()["verified_forecasts"] == 1
    path = s.root() / "forward/2026-09-15/001000.json"
    value = b.read(path)
    value["z"][-1] = 1
    b.save(path, value, replace=True)
    receipt_path = s.root() / "receipts/2026-09-15/001000.json"
    b.save(receipt_path, b.read(receipt_path) | {"forecast_hash": b.digest(value)}, replace=True)
    with pytest.raises(ValueError, match="ROUND_16_VECTOR_CHANGED"):
        s.report()


def test_calendar_revision_invalidates_frozen_model(tmp_path, monkeypatch):
    monkeypatch.setattr(b, "ROOT", tmp_path)
    b.save(s.root() / "result.json", {"fingerprint": {}, "calendar_hash": "previous-calendar"})
    monkeypatch.setattr(s, "fingerprint", lambda: {})
    with pytest.raises(ValueError, match="ROUND_16_MODEL_OR_CODE_CHANGED"):
        s.models()


def test_prior_runner_failure_still_attempts_calendar_branch(monkeypatch):
    from scripts import direction_1d_sprint_calendar as entry

    monkeypatch.setattr(entry.existing, "run", lambda: (_ for _ in ()).throw(ValueError("OLD_FAILED")))
    calls = []
    monkeypatch.setattr(entry.calendar_model, "tick", lambda: calls.append(1))
    monkeypatch.setattr(b, "save", lambda *args, **kwargs: None)
    with pytest.raises(ValueError, match="INDEPENDENT_BRANCH_FAILED"):
        entry.run()
    assert calls == [1]
