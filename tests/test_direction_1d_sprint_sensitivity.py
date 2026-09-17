"""个体市场敏感度一日纠错：目标语义、自然错误权重、成熟边界及提前答案完整性。"""

from datetime import date, datetime, timedelta

import numpy as np
import pytest
from app.services import direction_1d_sprint as b
from app.services import direction_1d_sprint_sensitivity as s
from threadpoolctl import threadpool_limits

from test_direction_1d_sprint_hk import ConstantModel, markets
from test_direction_1d_sprint_hk import ready as hk_ready  # noqa: F401
from test_direction_1d_sprint_hk import rows as original_rows
from test_direction_1d_sprint_sequence import ancestor_ready  # noqa: F401
from test_direction_1d_sprint_sequence import ready as sequence_ready  # noqa: F401


def empty_prior():
    return {"beta": [0.0] * 4, "count": 0, "available_fraction": 0.0, "max_u": None, "max_mature": None}


def historical_rows():
    return [r | {"z": r["z"] + [0.0] * 9} for r in original_rows()]


@pytest.fixture(autouse=True)
def bounded_threads():
    with threadpool_limits(limits=2):
        yield


def test_original_inputs_and_next_trading_day_label_are_kept():
    x = [0.0] * 32
    assert (
        s.vector(x, "2026-09-14", "2026-09-15", markets(), empty_prior())
        == s.hk.vector(x, "2026-09-14", "2026-09-15", markets()) + [0.0] * 9
    )
    with pytest.raises(ValueError, match="TARGET_NOT_ADJACENT"):
        s.vector(x, "2026-09-14", "2026-09-16", markets(), empty_prior())


def test_same504mature_dates_are_used():
    rows = []
    for i in range(600):
        day = date(2022, 1, 1) + timedelta(days=i)
        rows.append({"code": "fund", "u": str(day), "mature": str(day + timedelta(days=1)), "y": i % 2})
    selected = s.training_rows(rows, "2024-01-01")
    assert len(selected) == 504 and selected[0]["u"] == rows[96]["u"]
    assert s.training_rows(rows, "2023-01-01")[-1]["mature"] < "2023-01-01"


@pytest.mark.parametrize("candidate", s.CANDIDATES)
def test_future_labels_and_features_cannot_change_fitting(monkeypatch, candidate):
    monkeypatch.setattr(s, "active", lambda: None)
    rows = historical_rows()
    before = s.fit(rows, candidate, "2023-06-10")
    for row in rows:
        if row["mature"] >= "2023-06-10":
            row["z"], row["y"] = [float("nan")] * 21, 1 - row["y"]
    after = s.fit(rows, candidate, "2023-06-10")
    assert before["fit_hash"] == after["fit_hash"]
    assert before["mean"] == after["mean"] and before["scale"] == after["scale"]
    assert before["max_mature_date"] < "2023-06-10"
    np.testing.assert_allclose(
        before["model"].predict_proba([np.zeros(21)]), after["model"].predict_proba([np.zeros(21)])
    )


def test_error_labels_keep_natural_prior(monkeypatch):
    monkeypatch.setattr(s, "active", lambda: None)
    rows = historical_rows()
    for i, row in enumerate(rows):
        # 基线始终上涨，实际仅20%上涨，因此自然错误率应为80%，不能平衡成50%。
        row["z"], row["y"] = [0.0] * 21, int(i % 5 == 0)
    fitted = s.fit(rows, s.CANDIDATES[0], "2023-07-01")
    assert fitted["weighted_error_rate"] == pytest.approx(0.8)
    assert fitted["model"].predict_proba([np.zeros(21)])[0, 1] == pytest.approx(0.8)
    assert s.answer([0.0] * 21, s.CANDIDATES[0], fitted)["prediction"] == 0


def trained(score):
    return {"model": ConstantModel(score), "mean": [0.0] * 21, "scale": [1.0] * 21}


@pytest.mark.parametrize("move", [-0.01, 0, 0.01])
@pytest.mark.parametrize("candidate", s.CANDIDATES)
def test_only_strict_error_threshold_flips_baseline(move, candidate):
    z = [move] + [0.0] * 20
    baseline = int(move >= 0)
    for score, flipped in [(0.3, False), (0.5, False), (0.55, False), (0.551, True), (0.7, True)]:
        choice = s.answer(z, candidate, trained(score))
        assert choice["prediction"] == (1 - baseline if flipped else baseline)
        assert choice["baseline_prediction"] == baseline
        assert choice["flipped"] == flipped
        assert choice["kind"] == "UNCALIBRATED_BASELINE_ERROR_SCORE"


@pytest.fixture
def ready(hk_ready, monkeypatch):  # noqa: F811
    b.save(s.root() / "result.json", {"winner": s.CANDIDATES[0], "model_sha256": "abc"})
    bundle = {name: {"CN_EQUITY": trained(0.3)} for name in s.CANDIDATES}
    monkeypatch.setattr(s, "models", lambda: ({"model_sha256": "abc"}, bundle))
    monkeypatch.setattr(
        s,
        "current_contexts",
        lambda: {
            "cutoff": "2026-09-15",
            "contexts": {hk_ready[1]["code"]: empty_prior()},
        },
    )
    return hk_ready


def test_old_target_is_not_backfilled(ready, monkeypatch):
    monkeypatch.setattr(s, "models", lambda: pytest.fail("old target model loading"))
    assert s.tick()["verified_forecasts"] == 0


def test_live_input_order_and_announcements_are_checked(ready):
    _, original, source = ready
    with pytest.raises(ValueError, match="SEQUENCE_LIVE_DATES_INVALID"):
        s.live_vector(source, original | {"inputs": original["inputs"][::-1]}, markets(), s.current_contexts())
    original["inputs"][0]["ann_date"] = "2026-09-16"
    with pytest.raises(ValueError, match="SEQUENCE_LIVE_DATES_INVALID"):
        s.live_vector(source, original, markets(), s.current_contexts())


def test_early_answers_are_immutable_and_match_outcomes(ready, monkeypatch):
    monkeypatch.setattr(s, "FIRST_TARGET", "2026-09-15")
    directory, original, _ = ready
    assert s.tick()["verified_forecasts"] == 1
    path = s.root() / "forward/2026-09-15/001000.json"
    before = path.read_bytes()
    assert set(b.read(path)["answers"]) == set(s.CANDIDATES)
    s.tick()
    assert path.read_bytes() == before
    b.save(
        directory / "outcomes/2026-09-15/001000.json",
        {"y": 1, "actual_direction": "UP", "forecast_hash": b.digest(original)},
    )
    monkeypatch.setattr(b, "now", lambda: datetime(2026, 9, 15, 19, tzinfo=b.ZONE))
    result = s.report()
    for name in s.CANDIDATES:
        assert result["matched_forward_metrics"][name]["accuracy"] == 1


def test_late_readback_is_rejected(ready, monkeypatch):
    monkeypatch.setattr(s, "FIRST_TARGET", "2026-09-15")
    save = b.save

    def late(path, value, **kwargs):
        save(path, value, **kwargs)
        if path.parent.parent == s.root() / "forward":
            monkeypatch.setattr(b, "now", lambda: datetime(2026, 9, 15, 8, 30, tzinfo=b.ZONE))

    monkeypatch.setattr(b, "save", late)
    result = s.tick()
    assert result["verified_forecasts"] == 0 and result["invalid_or_late"] == 1


def test_changed_saved_input_is_rejected(ready, monkeypatch):
    monkeypatch.setattr(s, "FIRST_TARGET", "2026-09-15")
    assert s.tick()["verified_forecasts"] == 1
    path = s.root() / "forward/2026-09-15/001000.json"
    value = b.read(path)
    value["z"][-1] += 1
    b.save(path, value, replace=True)
    receipt = s.root() / "receipts/2026-09-15/001000.json"
    b.save(receipt, b.read(receipt) | {"forecast_hash": b.digest(value)}, replace=True)
    with pytest.raises(ValueError, match="ROUND_36_VECTOR_CHANGED"):
        s.report()


def test_prior_failure_still_attempts_sensitivity_branch(monkeypatch):
    from scripts import direction_1d_sprint_sensitivity as entry

    assert entry.existing.__name__ == "scripts.direction_1d_sprint_linear_error"
    monkeypatch.setattr(entry.existing, "run", lambda: (_ for _ in ()).throw(ValueError("OLD_FAILED")))
    calls = []
    monkeypatch.setattr(entry.sensitivity, "tick", lambda: calls.append(1))
    monkeypatch.setattr(b, "save", lambda *args, **kwargs: None)
    with pytest.raises(ValueError, match="INDEPENDENT_BRANCH_FAILED"):
        entry.run()
    assert calls == [1]


@pytest.mark.parametrize("values", [[0.0] * 20, [0.0] * 22, [float("nan")] * 21])
def test_malformed_feature_shape_or_nonfinite_is_rejected(values):
    with pytest.raises(ValueError, match="SENSITIVITY_INPUT_INVALID"):
        s.answer(values, s.CANDIDATES[0], trained(0.3))


def test_single_error_class_cannot_be_trained(monkeypatch):
    monkeypatch.setattr(s, "active", lambda: None)
    rows = historical_rows()
    for row in rows:
        row["z"][0] = 1 if row["y"] else -1
    with pytest.raises(ValueError, match="SENSITIVITY_MODEL_ERROR_LABELS_INSUFFICIENT"):
        s.fit(rows, s.CANDIDATES[0], "2023-07-01")


def test_code_freeze_covers_context_and_prototype(monkeypatch):
    monkeypatch.setattr(s.previous_round, "fingerprint", lambda: {"code": {}})
    files = s.fingerprint()["code"]
    assert len(files) == 6
    assert "app/services/direction_1d_sprint_sensitivity_context.py" in files
    assert any(path.endswith("round-36/context-feasibility.py") for path in files)


def test_missing_fund_or_future_context_is_rejected(ready):
    _, original, source = ready
    with pytest.raises(ValueError, match="CONTEXT_FUND_MISSING"):
        s.live_vector(source, original, markets(), {"cutoff": "2026-09-15", "contexts": {}})
    with pytest.raises(ValueError, match="CONTEXT_FUTURE_CUTOFF"):
        s.live_vector(source, original, markets(), {"cutoff": "2026-09-16", "contexts": {}})


def test_rehashed_context_cannot_replace_model_bound_snapshot(tmp_path, monkeypatch):
    monkeypatch.setattr(b, "ROOT", tmp_path)
    p = {"current_fit_cutoff": "2026-09-15", "input_hashes": {"round-36/current-context-feasibility.json": "reference"}}
    monkeypatch.setattr(s, "plan", lambda: p)
    snapshot = {
        "plan_hash": b.digest(p),
        "cutoff": "2026-09-15",
        "reference_hash": "reference",
        "contexts": {"fund": empty_prior()},
    }
    b.save(s.root() / "current-context.json", snapshot)
    b.save(s.root() / "result.json", {"context_snapshot_hash": b.digest(snapshot)})
    assert s.current_contexts() == snapshot
    snapshot["contexts"] = {}
    b.save(s.root() / "current-context.json", snapshot, replace=True)
    with pytest.raises(ValueError, match="CURRENT_CONTEXT_CHANGED"):
        s.current_contexts()
