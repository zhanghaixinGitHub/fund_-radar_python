"""两融余额变化只加入成熟训练输入，保留原一日标签和未来来源绑定。"""

from datetime import datetime

import numpy as np
import pytest
from app.services import direction_1d_sprint as b
from app.services import direction_1d_sprint_margin_balance as s

from test_direction_1d_sprint_etf_joint import ready as joint_ready  # noqa: F401
from test_direction_1d_sprint_etf_joint import vector as parent_vector
from test_direction_1d_sprint_hk import ConstantModel
from test_direction_1d_sprint_hk import ready as hk_ready  # noqa: F401
from test_direction_1d_sprint_hk import rows as historical_rows
from test_direction_1d_sprint_sequence import ancestor_ready  # noqa: F401
from test_direction_1d_sprint_sequence import ready as sequence_ready  # noqa: F401
from test_direction_1d_sprint_us_etf_direct import ready as direct_ready  # noqa: F401


def vector():
    return parent_vector((1, -1, -1)) + [0.1, 0.5, 0.2, 0.7]


def trained(name, score=0.56):
    width = 7
    return {
        "control": {"model": ConstantModel(0.56), "mean": [0.0] * 12, "scale": [1.0] * 12},
        "us_etf": {"model": ConstantModel(score), "mean": [0.0] * width, "scale": [1.0] * width},
    }


def test_direct_up_threshold_not_error_flip():
    z = vector()
    name = s.CANDIDATES[0]
    for score in (0.499999, 0.5):
        answer = s.answer(z, name, trained(name, score))
        assert answer["prediction"] == int(score >= 0.5)
        assert answer["kind"] == "UNCALIBRATED_UP_SCORE"


def test_missing_equity_source_keeps_parent_hk():
    z = vector()
    z[17:20] = [0.0] * 3
    name = s.CANDIDATES[0]
    assert s.answer(z, name, trained(name)) == s.majority.answer(z[:20], s.majority.CANDIDATES[0], trained(name))


def test_only_mature_labels_and_train_scaling_are_used(monkeypatch):
    monkeypatch.setattr(s, "active", lambda: None)
    rows = [r | {"z": r["z"] + [0.2, 0, 0, 0, 1, -0.3, 0, 1, 0.1, 0.5, 0.2, 0.7]} for r in historical_rows()]
    name = s.CANDIDATES[0]
    before = s.fit(rows, name, "2023-06-10")
    for row in rows:
        if row["mature"] >= "2023-06-10":
            row["z"], row["y"] = [float("nan")] * 24, 1 - row["y"]
    after = s.fit(rows, name, "2023-06-10")
    assert after["fit_hash"] == before["fit_hash"] and after["model"].n_features_in_ == 7
    assert after["model"].fit_intercept is False and after["mean"] == [0.0] * 7
    np.testing.assert_allclose(after["model"].coef_, before["model"].coef_)


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
    from scripts import direction_1d_sprint_margin_balance as entry

    assert entry.existing.__name__ == "scripts.direction_1d_sprint_us_yield"
    monkeypatch.setattr(entry.existing, "run", lambda: (_ for _ in ()).throw(ValueError("OLD_FAILED")))
    calls = []
    monkeypatch.setattr(s, "tick", lambda: calls.append(1))
    monkeypatch.setattr(b, "save", lambda *args, **kwargs: None)
    with pytest.raises(ValueError, match="INDEPENDENT_BRANCH_FAILED"):
        entry.run()
    assert calls == [1]
