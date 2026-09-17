"""最新现金日期CNYA净值口径差的一日目标、原输入保留、成熟时点及未来证据绑定。"""

from datetime import datetime

import numpy as np
import pytest
from app.services import direction_1d_sprint as b
from app.services import direction_1d_sprint_cnya_neutral as s

from test_direction_1d_sprint_cnya_data import body
from test_direction_1d_sprint_hk import ConstantModel, markets
from test_direction_1d_sprint_hk import ready as hk_ready  # noqa: F401
from test_direction_1d_sprint_hk import rows as historical_rows
from test_direction_1d_sprint_sequence import ancestor_ready  # noqa: F401
from test_direction_1d_sprint_sequence import ready as sequence_ready  # noqa: F401


def test_code_freeze_covers_real_fxi_client_and_data_files(monkeypatch):
    monkeypatch.setattr(s.previous_round, "fingerprint", lambda: {"code": {}})
    code = s.fingerprint()["code"]
    assert len(code) == 4
    assert ".local-runs/direction-1d-sprint-20260914/round-45/feature-feasibility.py" in code
    assert "app/services/direction_1d_sprint_cnya_neutral.py" in code


def test_fxi_appends_three_known_inputs_and_target_stays_one_day():
    x = [0.0] * 32
    points = s.cnya_data.parse(body())
    z = s.vector(x, "2026-09-14", "2026-09-15", markets(), points)
    assert len(z) == 15 and z[:12] == s.hk.vector(x, "2026-09-14", "2026-09-15", markets())
    assert z[-3:] == s.cnya_data.features("2026-09-14", "2026-09-15", points)
    with pytest.raises(ValueError, match="NOT_ADJACENT"):
        s.vector(x, "2026-09-14", "2026-09-16", markets(), points)


def training_rows():
    return [r | {"z": r["z"] + [1.0, 0.1, 1.0]} for r in historical_rows()]


def test_future_labels_and_input_changes_do_not_change_fitting(monkeypatch):
    monkeypatch.setattr(s, "active", lambda: None)
    rows = training_rows()
    before = s.fit(rows, s.CANDIDATES[0], "2023-06-10")
    for row in rows:
        if row["mature"] >= "2023-06-10":
            row["z"], row["y"] = [float("nan")] * 15, 1 - row["y"]
    after = s.fit(rows, s.CANDIDATES[0], "2023-06-10")
    assert before["fit_hash"] == after["fit_hash"]
    assert before["mean"] == after["mean"] and before["scale"] == after["scale"]
    assert before["max_mature_date"] < "2023-06-10"
    np.testing.assert_allclose(
        before["model"].predict_proba([np.zeros(2)]), after["model"].predict_proba([np.zeros(2)]), atol=1e-12
    )


def test_direct_up_labels_natural_prior_and_fixed_recipe(monkeypatch):
    monkeypatch.setattr(s, "active", lambda: None)
    rows = training_rows()
    for i, row in enumerate(rows):
        row["z"], row["y"] = [0.0] * 15, int(i % 5 == 0)
    fitted = s.fit(rows, s.CANDIDATES[0], "2023-07-01")
    # 基线一直上涨，而标签仅20%上涨；直接模型不能把80%错误率误当UP率。
    assert fitted["weighted_up_rate"] == pytest.approx(0.2)
    assert "weighted_error_rate" not in fitted
    assert fitted["model"].n_features_in_ == 2
    params = fitted["model"].get_params()
    assert params["C"] == 0.1 and params["max_iter"] == 1500 and params["class_weight"] is None
    assert params["fit_intercept"] is False
    assert fitted["mean"] == [0.0, 0.0]
    assert s.answer([0.0] * 15, s.CANDIDATES[0], fitted)["research_score"] == 0.5


def trained(score):
    return {"model": ConstantModel(score), "mean": [0.0] * 2, "scale": [1.0] * 2}


@pytest.mark.parametrize("move", [-0.01, 0, 0.01])
def test_direct_threshold_does_not_depend_on_baseline(move):
    z = [move] + [0.0] * 14
    for score, prediction in ((0.49, 0), (0.5, 1), (0.55, 1), (0.7, 1)):
        choice = s.answer(z, s.CANDIDATES[0], trained(score))
        assert choice["prediction"] == prediction
        assert choice["flipped"] == (prediction != int(move >= 0))
        assert choice["kind"] == "UNCALIBRATED_UP_SCORE"


def test_only_two_selected_inputs_reach_estimator():
    z = list(range(15))
    assert s.selected_features(z, s.CANDIDATES[0]) == [0, 12]
    z[1:12] = [999] * 11
    z[13] = -999
    assert s.selected_features(z, s.CANDIDATES[0]) == [0, 12]


def test_nonconvergence_is_not_silently_accepted(monkeypatch):
    import warnings

    monkeypatch.setattr(s, "active", lambda: None)

    def fail(*args, **kwargs):
        warnings.warn("synthetic", s.ConvergenceWarning, stacklevel=2)

    monkeypatch.setattr(s.LogisticRegression, "fit", fail)
    with pytest.raises(s.ConvergenceWarning):
        s.fit(training_rows(), s.CANDIDATES[0], "2023-07-01")


@pytest.fixture
def ready(hk_ready, monkeypatch):  # noqa: F811
    b.save(s.root() / "result.json", {"winner": s.CANDIDATES[0], "model_sha256": "abc"})
    bundle = {s.CANDIDATES[0]: {"CN_EQUITY": trained(0.7)}}
    monkeypatch.setattr(s, "models", lambda: ({"model_sha256": "abc"}, bundle))
    input_value = {
        "at": "2026-09-15T07:15:00+08:00",
        "base": "2026-09-14",
        "target": "2026-09-15",
        "rows": s.cnya_data.parse(body()),
        "source": s.cnya_data.SOURCE,
    }
    monkeypatch.setattr(s.cnya_data, "capture", lambda at: input_value)
    monkeypatch.setattr(s.cnya_data, "load", lambda target: input_value)
    return hk_ready


def test_old_target_is_not_backfilled(ready, monkeypatch):
    monkeypatch.setattr(s, "models", lambda: pytest.fail("old target model loading"))
    assert s.tick()["verified_forecasts"] == 0


def test_live_input_order_and_announcements_are_checked(ready):
    _, original, source = ready
    with pytest.raises(ValueError, match="SEQUENCE_LIVE_DATES_INVALID"):
        s.live_vector(source, original | {"inputs": original["inputs"][::-1]}, markets(), s.cnya_data.parse(body()))
    original["inputs"][0]["ann_date"] = "2026-09-16"
    with pytest.raises(ValueError, match="SEQUENCE_LIVE_DATES_INVALID"):
        s.live_vector(source, original, markets(), s.cnya_data.parse(body()))


def test_early_answers_are_immutable_and_outcomes_match(ready, monkeypatch):
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


def test_changed_saved_fxi_value_is_rejected(ready, monkeypatch):
    monkeypatch.setattr(s, "FIRST_TARGET", "2026-09-15")
    assert s.tick()["verified_forecasts"] == 1
    path = s.root() / "forward/2026-09-15/001000.json"
    value = b.read(path)
    value["z"][-1] += 1
    b.save(path, value, replace=True)
    receipt = s.root() / "receipts/2026-09-15/001000.json"
    b.save(receipt, b.read(receipt) | {"forecast_hash": b.digest(value)}, replace=True)
    with pytest.raises(ValueError, match="ROUND_45_VECTOR_CHANGED"):
        s.report()


def test_prior_failure_still_attempts_fxi_branch(monkeypatch):
    from scripts import direction_1d_sprint_cnya_neutral as entry

    assert entry.existing.__name__ == "scripts.direction_1d_sprint_cnya_direct"
    monkeypatch.setattr(entry.existing, "run", lambda: (_ for _ in ()).throw(ValueError("OLD_FAILED")))
    calls = []
    monkeypatch.setattr(entry.cnya_neutral, "tick", lambda: calls.append(1))
    monkeypatch.setattr(b, "save", lambda *args, **kwargs: None)
    with pytest.raises(ValueError, match="INDEPENDENT_BRANCH_FAILED"):
        entry.run()
    assert calls == [1]


def test_no_center_shift_or_constant_indicator_can_create_hidden_bias(monkeypatch):
    monkeypatch.setattr(s, "active", lambda: None)
    rows = training_rows()
    for i, row in enumerate(rows):
        row["z"][0], row["z"][12], row["z"][14] = (i % 7) * 0.2, (i % 9) * 0.1, 1.0
    fitted = s.fit(rows, s.CANDIDATES[0], "2023-07-01")
    assert fitted["mean"] == [0.0, 0.0]
    assert all(v > 0 for v in fitted["training_feature_mean_not_subtracted"])
    zero = [0.0] * 15
    zero[14] = 1.0
    assert s.answer(zero, s.CANDIDATES[0], fitted)["research_score"] == 0.5
    positive, negative = zero[:], zero[:]
    positive[0], positive[12], negative[0], negative[12] = 0.3, 0.2, -0.3, -0.2
    a = s.answer(positive, s.CANDIDATES[0], fitted)["research_score"]
    c = s.answer(negative, s.CANDIDATES[0], fitted)["research_score"]
    assert a + c == pytest.approx(1.0, abs=1e-12)
