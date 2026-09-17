"""美债特征必须只影响有新数据的题目；模型学习成熟历史中的多数规则错误。"""

from datetime import datetime

import numpy as np
import pytest
from app.services import direction_1d_sprint as b
from app.services import direction_1d_sprint_us_yield as s

from test_direction_1d_sprint_etf_joint import ready as joint_ready  # noqa: F401
from test_direction_1d_sprint_etf_joint import vector as parent_vector
from test_direction_1d_sprint_hk import ConstantModel
from test_direction_1d_sprint_hk import ready as hk_ready  # noqa: F401
from test_direction_1d_sprint_hk import rows as historical_rows
from test_direction_1d_sprint_sequence import ancestor_ready  # noqa: F401
from test_direction_1d_sprint_sequence import ready as sequence_ready  # noqa: F401
from test_direction_1d_sprint_us_etf_direct import ready as direct_ready  # noqa: F401


def vector():
    return parent_vector((1, -1, -1)) + [10, -10, 0, 10, 20, 60, 80, 5.1, 1]


def trained(name, score=0.56):
    width = 3 if name == s.CANDIDATES[0] else 11
    return {
        "control": {"model": ConstantModel(0.56), "mean": [0.0] * 12, "scale": [1.0] * 12},
        "us_etf": {"model": ConstantModel(score), "mean": [0.0] * width, "scale": [1.0] * width},
    }


@pytest.mark.parametrize("name", s.CANDIDATES)
def test_error_threshold_uses_majority_direction_not_spx(name):
    z = vector()
    assert s.parent_sign(z) == 0 and z[0] > 0
    for score in (0.55, 0.550001):
        choice = s.answer(z, name, trained(name, score))
        assert choice["prediction"] == int(score > 0.55)
        assert choice["baseline_prediction"] == 0
        assert choice["kind"] == "UNCALIBRATED_MAJORITY_ERROR_SCORE"


@pytest.mark.parametrize("name", s.CANDIDATES)
def test_missing_new_rate_keeps_exact_parent_and_rejects_stale_values(name):
    z = vector()
    z[20:] = [0.0] * 9
    assert s.answer(z, name, trained(name)) == s.majority.answer(z[:20], s.majority.CANDIDATES[0], trained(name))
    z[20] = 5.0
    with pytest.raises(ValueError, match="UNAVAILABLE_VALUES"):
        s.answer(z, name, trained(name))


def test_ablation_shares_mature_rows_and_ignores_future_labels(monkeypatch):
    monkeypatch.setattr(s, "active", lambda: None)
    rows = []
    for i, r in enumerate(historical_rows()):
        z = r["z"] + [0.2, 0, 0, 0, 1, -0.3, 0, 1] + [float((i + j) % 5) for j in range(8)] + [1]
        rows.append(r | {"z": z})
    first = s.fit(rows, s.CANDIDATES[0], "2023-06-10")
    second = s.fit(rows, s.CANDIDATES[1], "2023-06-10")
    assert first["fit_hash"] == second["fit_hash"]
    assert second["training_target"] == "MAJORITY3_ERROR"
    assert (first["model"].n_features_in_, second["model"].n_features_in_) == (3, 11)
    for row in rows:
        if row["mature"] >= "2023-06-10":
            row["z"], row["y"] = [float("nan")] * 29, 1 - row["y"]
    after = s.fit(rows, s.CANDIDATES[1], "2023-06-10")
    assert after["fit_hash"] == second["fit_hash"]
    np.testing.assert_allclose(after["mean"], second["mean"])
    np.testing.assert_allclose(after["model"].feature_importances_, second["model"].feature_importances_)


@pytest.fixture
def ready(joint_ready, monkeypatch):  # noqa: F811
    b.save(s.root() / "result.json", {"winner": s.CANDIDATES[0], "model_sha256": "abc"})
    bundle = {n: {"CN_EQUITY": trained(n)} for n in s.CANDIDATES}
    monkeypatch.setattr(s, "models", lambda: ({"model_sha256": "abc"}, bundle))
    rate_input = {"rows": {}, "target": "2026-09-15"}
    monkeypatch.setattr(s.rates, "capture", lambda at: rate_input)
    monkeypatch.setattr(s.rates, "load", lambda target: rate_input)
    monkeypatch.setattr(s.rates, "features", lambda *args: vector()[20:])
    monkeypatch.setattr(s, "FIRST_TARGET", "2026-09-15")
    return joint_ready


def test_future_answers_keep_original_label_and_bind_rate_source(ready, monkeypatch):
    directory, original, _ = ready
    assert s.tick()["verified_forecasts"] == 1
    path = s.root() / "forward/2026-09-15/001000.json"
    raw = path.read_bytes()
    assert s.tick()["verified_forecasts"] == 1 and raw == path.read_bytes()
    value = b.read(path)
    b.save(
        directory / "outcomes/2026-09-15/001000.json",
        {"y": 1, "actual_direction": "UP", "forecast_hash": b.digest(original)},
    )
    monkeypatch.setattr(b, "now", lambda: datetime(2026, 9, 15, 19, tzinfo=b.ZONE))
    assert (
        s.report()["matched_forward_metrics"][s.CANDIDATES[0]]["accuracy"]
        == value["answers"][s.CANDIDATES[0]]["prediction"]
    )
    monkeypatch.setattr(s.rates, "load", lambda target: {"rows": {}, "tampered": True})
    with pytest.raises(ValueError, match="RATE_INPUT_CHANGED"):
        s.report()


def test_rehashed_prediction_direction_is_rejected(ready):
    s.tick()
    path = s.root() / "forward/2026-09-15/001000.json"
    value = b.read(path)
    value["answers"][s.CANDIDATES[0]]["prediction"] ^= 1
    b.save(path, value, replace=True)
    receipt = s.root() / "receipts/2026-09-15/001000.json"
    b.save(receipt, b.read(receipt) | {"forecast_hash": b.digest(value)}, replace=True)
    with pytest.raises(ValueError, match="SAVED_ANSWER_CHANGED"):
        s.report()


def test_late_readback_is_rejected(ready, monkeypatch):
    save = b.save

    def late(path, value, **kwargs):
        save(path, value, **kwargs)
        if path.parent.parent == s.root() / "forward":
            monkeypatch.setattr(b, "now", lambda: datetime(2026, 9, 15, 8, 30, tzinfo=b.ZONE))

    monkeypatch.setattr(b, "save", late)
    value = s.tick()
    assert value["verified_forecasts"] == 0 and value["invalid_or_late"] == 1


def test_entry_keeps_all_prior_rounds_and_isolates_failure(monkeypatch):
    from scripts import direction_1d_sprint_us_yield as entry

    assert entry.existing.__name__ == "scripts.direction_1d_sprint_etf_disagreement"
    monkeypatch.setattr(entry.existing, "run", lambda: (_ for _ in ()).throw(ValueError("OLD_FAILED")))
    calls = []
    monkeypatch.setattr(s, "tick", lambda: calls.append(1))
    monkeypatch.setattr(b, "save", lambda *args, **kwargs: None)
    with pytest.raises(ValueError, match="INDEPENDENT_BRANCH_FAILED"):
        entry.run()
    assert calls == [1]
