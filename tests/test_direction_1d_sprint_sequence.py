"""收益时序的截止边界、训练隔离、固定迭代以及真实预测完整性。"""

from datetime import date, datetime

import numpy as np
import pytest
from app.services import direction_1d_sprint as b
from app.services import direction_1d_sprint_fund_response as f
from app.services import direction_1d_sprint_overnight as o
from app.services import direction_1d_sprint_sequence as s
from app.services import direction_1d_sprint_sparse as sparse

from test_direction_1d_sprint_fund_response import ready as ancestor_ready  # noqa: F401
from test_direction_1d_sprint_fund_response import sample_rows


def inputs():
    returns = np.linspace(-0.015, 0.02, 60)
    values = np.r_[1.0, np.cumprod(1 + returns)]
    x = b.vector(values, True) + [0.0] * 12 + [0.01, 1]
    return values, x


def test_sequence_order_and_short_window_are_derived_from_past_nav_only():
    values, x = inputs()
    z = s.vector(x, values)
    expected = np.clip((values[1:] / values[:-1] - 1) / x[3], -5, 5)
    np.testing.assert_allclose(z[:60], expected)
    np.testing.assert_allclose(s.selected_features(z, s.CANDIDATES[0]), list(expected[-20:]) + [0.01, 1])
    with pytest.raises(ValueError, match="SEQUENCE_INPUT_INVALID"):
        s.vector(x, values[:-1])
    values[-1] *= 2
    with pytest.raises(ValueError, match="SEQUENCE_NAV_WINDOW_CHANGED"):
        s.vector(x, values)


def test_weighted_centered_variance_is_finite_for_constant_features():
    x = np.c_[np.full(7, 0.3), np.arange(7)]
    z, mean, scale = s.normalize_training(x, np.arange(1, 8))
    assert scale[0] == 1
    assert np.isfinite(z).all()
    assert mean[1] == pytest.approx(4)
    np.testing.assert_allclose(np.average(z, weights=np.arange(1, 8), axis=0), [0, 0], atol=1e-14)


@pytest.mark.parametrize("name", s.CANDIDATES)
def test_unmatured_labels_cannot_change_scaling_or_fitted_parameters(name, monkeypatch):
    monkeypatch.setattr(s, "active", lambda: None)
    rows = [r | {"z": [r["x"][30]] * 60 + r["x"][30:]} for r in sample_rows()]
    first = s.fit(rows, name, "2023-06-10")
    for row in rows:
        if row["mature"] >= "2023-06-10":
            row["z"], row["y"] = [float("nan")] * 62, 1 - row["y"]
    second = s.fit(rows, name, "2023-06-10")
    assert first["fit_hash"] == second["fit_hash"]
    assert first["max_mature_date"] < "2023-06-10"
    assert first["fit_dates"] <= 252
    assert first["mean"] == second["mean"] and first["scale"] == second["scale"]
    assert first["loss_curve"] == second["loss_curve"]
    assert len(first["loss_curve"]) == s.RECIPES[name][2]
    left = getattr(first["model"], "coefs_", getattr(first["model"], "coef_", None))
    right = getattr(second["model"], "coefs_", getattr(second["model"], "coef_", None))
    for a, c in zip(left, right, strict=True):
        np.testing.assert_array_equal(a, c)


def test_neural_training_checks_deadline_before_each_epoch(monkeypatch):
    calls = []

    def deadline():
        calls.append(1)
        if len(calls) == 3:
            raise ValueError("SPRINT_DEADLINE_REACHED")

    monkeypatch.setattr(s, "active", deadline)
    rows = [r | {"z": [0.0] * 60 + r["x"][30:]} for r in sample_rows()]
    with pytest.raises(ValueError, match="SPRINT_DEADLINE_REACHED"):
        s.fit(rows, s.CANDIDATES[1], "2023-06-10")
    assert len(calls) == 3


def test_checkpoint_reuses_completed_fit_and_refuses_interrupted_attempt(tmp_path, monkeypatch):
    monkeypatch.setattr(b, "ROOT", tmp_path)
    calls = []
    monkeypatch.setattr(s, "fit", lambda *args: calls.append(1) or {"value": 3})
    assert s.fit_checkpoint([], "n", "2025-01-01", "one") == {"value": 3}
    assert s.fit_checkpoint([], "n", "2025-01-01", "one") == {"value": 3}
    assert calls == [1]
    b.save(s.root() / "checkpoints/two.attempt.json", {"at": "started"})
    with pytest.raises(ValueError, match="SEQUENCE_PREVIOUS_FIT_INTERRUPTED"):
        s.fit_checkpoint([], "n", "2025-01-01", "two")


class ConstantModel:
    def predict_proba(self, x):
        return np.asarray([[0.7, 0.3]] * len(x))


@pytest.fixture
def ready(ancestor_ready, monkeypatch):  # noqa: F811
    directory, original, _ = ancestor_ready
    values, x = inputs()
    days = list(map(str, b.input_days(date(2026, 9, 14))))
    original["inputs"] = [{"date": d, "nav": float(v), "ann_date": d} for d, v in zip(days, values, strict=True)]
    b.save(directory / "forward/2026-09-15/001000.json", original, replace=True)
    source_path = o.root() / "forward/2026-09-15/001000.json"
    source = b.read(source_path) | {"x": x, "original_hash": b.digest(original)}
    b.save(source_path, source, replace=True)
    receipt_path = o.root() / "receipts/2026-09-15/001000.json"
    b.save(receipt_path, b.read(receipt_path) | {"forecast_hash": b.digest(source)}, replace=True)
    parent_path = sparse.root() / "forward/2026-09-15/001000.json"
    parent = b.read(parent_path) | {"original_hash": b.digest(original), "parent_hash": b.digest(source)}
    b.save(parent_path, parent, replace=True)
    receipt_path = sparse.root() / "receipts/2026-09-15/001000.json"
    b.save(receipt_path, b.read(receipt_path) | {"forecast_hash": b.digest(parent)}, replace=True)
    assert f.tick()["verified_forecasts"] == 1
    b.save(s.root() / "result.json", {"winner": s.CANDIDATES[0], "model_sha256": "abc"})
    bundle = {
        n: {
            "CN_EQUITY": {
                "model": ConstantModel(),
                "mean": [0.0] * (s.RECIPES[n][0] + 2),
                "scale": [1.0] * (s.RECIPES[n][0] + 2),
            }
        }
        for n in s.CANDIDATES
    }
    monkeypatch.setattr(s, "models", lambda: ({"model_sha256": "abc"}, bundle))
    return directory, original, source


def test_new_version_never_backfills_september15(ready, monkeypatch):
    monkeypatch.setattr(s, "models", lambda: pytest.fail("old target must not load new model"))
    assert s.tick()["verified_forecasts"] == 0


def test_live_predictions_are_immutable_and_mature_outcomes_paired(ready, monkeypatch):
    monkeypatch.setattr(s, "FIRST_TARGET", "2026-09-15")
    directory, original, _ = ready
    assert s.tick()["verified_forecasts"] == 1
    path = s.root() / "forward/2026-09-15/001000.json"
    saved = path.read_bytes()
    s.tick()
    assert path.read_bytes() == saved
    b.save(
        directory / "outcomes/2026-09-15/001000.json",
        {"y": 0, "actual_direction": "DOWN", "forecast_hash": b.digest(original)},
    )
    monkeypatch.setattr(b, "now", lambda: datetime(2026, 9, 15, 19, tzinfo=b.ZONE))
    assert s.report()["matched_forward_metrics"][s.CANDIDATES[0]]["accuracy"] == 1


def test_late_readback_invalidates_prediction(ready, monkeypatch):
    monkeypatch.setattr(s, "FIRST_TARGET", "2026-09-15")
    save = b.save

    def crossing(path, value, **kwargs):
        save(path, value, **kwargs)
        if path.parent.parent == s.root() / "forward":
            monkeypatch.setattr(b, "now", lambda: datetime(2026, 9, 15, 8, 30, tzinfo=b.ZONE))

    monkeypatch.setattr(b, "save", crossing)
    report = s.tick()
    assert report["verified_forecasts"] == 0 and report["invalid_or_late"] == 1
    monkeypatch.setattr(s, "models", lambda: pytest.fail("post deadline must not load model"))
    assert s.tick()["verified_forecasts"] == 0


def test_wrong_input_order_and_changed_vector_are_rejected(ready, monkeypatch):
    monkeypatch.setattr(s, "FIRST_TARGET", "2026-09-15")
    _, original, source = ready
    with pytest.raises(ValueError, match="SEQUENCE_LIVE_DATES_INVALID"):
        s.live_vector(source, original | {"inputs": original["inputs"][::-1]})
    assert s.tick()["verified_forecasts"] == 1
    path = s.root() / "forward/2026-09-15/001000.json"
    value = b.read(path)
    value["z"][0] += 1
    b.save(path, value, replace=True)
    receipt_path = s.root() / "receipts/2026-09-15/001000.json"
    b.save(receipt_path, b.read(receipt_path) | {"forecast_hash": b.digest(value)}, replace=True)
    with pytest.raises(ValueError, match="ROUND_12_VECTOR_CHANGED"):
        s.report()


def test_old_runner_failure_does_not_skip_sequence_branch(monkeypatch):
    from scripts import direction_1d_sprint_sequence as entry

    monkeypatch.setattr(entry.existing, "run", lambda: (_ for _ in ()).throw(ValueError("OLD_FAILED")))
    calls = []
    monkeypatch.setattr(entry.sequence, "tick", lambda: calls.append(1))
    monkeypatch.setattr(b, "save", lambda *args, **kwargs: None)
    with pytest.raises(ValueError, match="INDEPENDENT_BRANCH_FAILED"):
        entry.run()
    assert calls == [1]
