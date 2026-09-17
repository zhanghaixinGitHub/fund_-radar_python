"""验证时间隔离校准不会使用外层考试标签，且真实提前答案仍不可改写。"""

from datetime import date, datetime, timedelta

import numpy as np
import pytest
from app.services import direction_1d_sprint as b
from app.services import direction_1d_sprint_calibrated as s

from test_direction_1d_sprint_hk import ConstantModel
from test_direction_1d_sprint_hk import ready as hk_ready  # noqa: F401
from test_direction_1d_sprint_sequence import ancestor_ready  # noqa: F401
from test_direction_1d_sprint_sequence import ready as sequence_ready  # noqa: F401


def samples():
    rows = []
    for i in range(1000):
        day = date(2021, 1, 1) + timedelta(days=i)
        baseline = i % 2
        move = 1 if baseline else -1
        rows.append(
            {
                "code": "a",
                "family": "a",
                "group": "CN_EQUITY",
                "u": str(day),
                "mature": str(day + timedelta(days=2)),
                "y": 1 - baseline if i % 5 == 0 else baseline,
                "z": [move, 1, 1, np.sin(i), np.cos(i), 0, 0, 0, 0, 0, 0, 0],
            }
        )
    return rows


def test_inner_scores_use_only_prior_models_and_future_mutations_do_not_change_them(tmp_path, monkeypatch):
    monkeypatch.setattr(s, "active", lambda: None)
    monkeypatch.setattr(s.previous_round, "active", lambda: None)
    rows = samples()
    cutoff = "2023-06-01"
    monkeypatch.setattr(b, "ROOT", tmp_path / "first")
    before = s.calibration_rows(rows, cutoff)
    assert len({r["u"] for r in before}) == 126
    assert len({r["inner_cutoff"] for r in before}) == 2
    assert all(r["inner_cutoff"] <= r["u"] and r["mature"] < cutoff for r in before)
    for row in rows:
        if row["mature"] >= cutoff:
            row["y"], row["z"] = 1 - row["y"], [float("nan")] * 12
    monkeypatch.setattr(b, "ROOT", tmp_path / "second")
    after = s.calibration_rows(rows, cutoff)
    assert [(r["u"], r["error_label"]) for r in before] == [(r["u"], r["error_label"]) for r in after]
    np.testing.assert_allclose([r["raw_score"] for r in before], [r["raw_score"] for r in after], atol=1e-12, rtol=0)


def calibration_samples():
    return [
        {
            "code": "a",
            "family": "a",
            "group": "CN_EQUITY",
            "u": str(date(2023, 1, 1) + timedelta(days=i)),
            "mature": str(date(2023, 1, 3) + timedelta(days=i)),
            "raw_score": 0.8 if i % 5 == 0 else 0.2,
            "error_label": int(i % 5 == 0),
        }
        for i in range(126)
    ]


@pytest.mark.parametrize("name", s.CANDIDATES)
def test_calibrators_keep_natural_prior_and_are_monotone(name, monkeypatch):
    monkeypatch.setattr(s, "active", lambda: None)
    monkeypatch.setattr(s, "calibration_rows", lambda *args: calibration_samples())
    monkeypatch.setattr(s, "outer_model", lambda *args: {})
    result = s.fit([], name, "2023-06-01")
    assert result["weighted_error_rate"] == pytest.approx(26 / 126)
    assert not result["identity_fallback"]
    if name == s.CANDIDATES[0]:
        scores = result["calibrator"].predict_proba(s.score_input([0.1, 0.9]))[:, 1]
    else:
        scores = result["calibrator"].predict([0.1, 0.9])
    assert 0 <= scores[0] < scores[1] <= 1


def test_negative_platt_slope_is_recorded_and_falls_back_without_refit(monkeypatch):
    monkeypatch.setattr(s, "active", lambda: None)
    rows = calibration_samples()
    for r in rows:
        r["error_label"] = 1 - r["error_label"]
    monkeypatch.setattr(s, "calibration_rows", lambda *args: rows)
    monkeypatch.setattr(s, "outer_model", lambda *args: {})
    result = s.fit([], s.CANDIDATES[0], "2023-06-01")
    assert result["platt_slope"] < 0 and result["identity_fallback"]


def test_cached_calibration_data_with_unmatured_label_is_rejected(monkeypatch):
    rows = calibration_samples()
    rows[0]["mature"] = "2023-06-01"
    monkeypatch.setattr(s, "calibration_rows", lambda *args: rows)
    with pytest.raises(ValueError, match="CALIBRATION_TRAINING_DATES_CHANGED"):
        s.fit([], s.CANDIDATES[0], "2023-06-01")


class IsotonicConstant:
    def predict(self, values):
        return np.full(len(values), 0.2)


def model(name, fallback=False, raw=0.3):
    return {
        "base_model": {"model": ConstantModel(raw), "mean": [0.0] * 12, "scale": [1.0] * 12},
        "calibrator": ConstantModel(0.2) if name == s.CANDIDATES[0] else IsotonicConstant(),
        "identity_fallback": fallback,
    }


@pytest.mark.parametrize("name", s.CANDIDATES)
def test_strict_half_threshold_and_identity_fallback(name):
    z = [1.0] + [0.0] * 11
    assert s.answer(z, name, model(name, True, 0.5))["prediction"] == 1
    assert s.answer(z, name, model(name, True, 0.501))["prediction"] == 0
    assert s.answer(z, name, model(name))["research_score"] == pytest.approx(0.2)


@pytest.fixture
def ready(hk_ready, monkeypatch):  # noqa: F811
    b.save(s.root() / "result.json", {"winner": s.CANDIDATES[0], "model_sha256": "abc"})
    bundle = {name: {"CN_EQUITY": model(name)} for name in s.CANDIDATES}
    monkeypatch.setattr(s, "models", lambda: ({"model_sha256": "abc"}, bundle))
    return hk_ready


def test_old_target_is_not_backfilled(ready, monkeypatch):
    monkeypatch.setattr(s, "models", lambda: pytest.fail("old target model loading"))
    assert s.tick()["verified_forecasts"] == 0


def test_early_answers_include_fixed_controls_and_are_immutable(ready, monkeypatch):
    monkeypatch.setattr(s, "FIRST_TARGET", "2026-09-15")
    directory, original, _ = ready
    assert s.tick()["verified_forecasts"] == 1
    path = s.root() / "forward/2026-09-15/001000.json"
    before = path.read_bytes()
    assert set(b.read(path)["answers"]) == set(s.CANDIDATES) | {"HK_EXTRA12_ERR504", "HK_RAW05_ERR504"}
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

    def late(path, value, **kwargs):
        save(path, value, **kwargs)
        if path.parent.parent == s.root() / "forward":
            monkeypatch.setattr(b, "now", lambda: datetime(2026, 9, 15, 8, 30, tzinfo=b.ZONE))

    monkeypatch.setattr(b, "save", late)
    value = s.tick()
    assert value["verified_forecasts"] == 0 and value["invalid_or_late"] == 1


def test_interrupted_inner_fit_cannot_restart(tmp_path, monkeypatch):
    monkeypatch.setattr(b, "ROOT", tmp_path)
    b.save(s.root() / "inner-models/key.attempt.json", {"attempted": True})
    monkeypatch.setattr(s.previous_round, "fit", lambda *args: pytest.fail("unexpected refit"))
    with pytest.raises(ValueError, match="INTERRUPTED_NO_RETRY"):
        s.inner_model([], "2025-01-01", "key")


def test_prior_runner_failure_does_not_skip_calibrated_branch(monkeypatch):
    from scripts import direction_1d_sprint_calibrated as entry

    monkeypatch.setattr(entry.existing, "run", lambda: (_ for _ in ()).throw(ValueError("OLD_FAILED")))
    calls = []
    monkeypatch.setattr(entry.calibrated, "tick", lambda: calls.append(1))
    monkeypatch.setattr(b, "save", lambda *args, **kwargs: None)
    with pytest.raises(ValueError, match="INDEPENDENT_BRANCH_FAILED"):
        entry.run()
    assert calls == [1]
