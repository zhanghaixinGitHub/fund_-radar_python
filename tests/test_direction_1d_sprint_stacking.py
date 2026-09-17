"""组合模型的成熟训练隔离、基础分数作用与真实预测证据绑定。"""

from datetime import datetime

import numpy as np
import pytest
from app.services import direction_1d_sprint as b
from app.services import direction_1d_sprint_stacking as s

from test_direction_1d_sprint_etf_joint import ready as joint_ready  # noqa: F401
from test_direction_1d_sprint_hk import ConstantModel
from test_direction_1d_sprint_hk import ready as hk_ready  # noqa: F401
from test_direction_1d_sprint_hk import rows as historical_rows
from test_direction_1d_sprint_sector_return import vector
from test_direction_1d_sprint_sequence import ancestor_ready  # noqa: F401
from test_direction_1d_sprint_sequence import ready as sequence_ready  # noqa: F401
from test_direction_1d_sprint_stacking_oof import head
from test_direction_1d_sprint_us_etf_direct import ready as direct_ready  # noqa: F401


class MetaProbe:
    n_features_in_ = 2

    def predict_proba(self, x):
        score = 1 / (1 + np.exp(-(x[:, 0] + x[:, 1])))
        return np.column_stack((1 - score, score))


def trained(name=None):
    return {
        "control": {"model": ConstantModel(0.56), "mean": [0.0] * 12, "scale": [1.0] * 12},
        "us_etf": {"model": MetaProbe(), "mean": [0.0, 0.0], "scale": [1.0, 1.0]},
        "ridge_base": head(),
    }


def test_meta_uses_both_vote_and_base_output():
    name = s.CANDIDATES[0]
    assert s.answer(vector(values=(2, -3, 4)), name, trained())["prediction"] == 1
    assert s.answer(vector(values=(-2, -3, 4)), name, trained())["prediction"] == 0


def test_meta_fit_excludes_not_yet_mature_rows(monkeypatch):
    monkeypatch.setattr(s, "active", lambda: None)
    rows = [r | {"z": r["z"] + [0.2, 0, 0, 0, 1, -0.3, 0, 1, 0.2, -0.3, 1, 0]} for r in historical_rows()]
    seen = []

    def features(selected):
        seen.append([r["u"] for r in selected])
        return np.asarray([[np.sin(i), np.cos(i)] for i, _ in enumerate(selected)]), "manifest"

    monkeypatch.setattr(s.oof, "training_features", features)
    before = s.fit(rows, s.CANDIDATES[0], "2023-06-10")
    for r in rows:
        if r["mature"] >= "2023-06-10":
            r["z"], r["y"] = [float("nan")] * 24, 1 - r["y"]
    after = s.fit(rows, s.CANDIDATES[0], "2023-06-10")
    assert before["fit_hash"] == after["fit_hash"] and seen[0] == seen[1]
    assert after["model"].n_features_in_ == 2 and after["model"].fit_intercept is False
    np.testing.assert_allclose(before["model"].coef_, after["model"].coef_)


@pytest.fixture
def ready(joint_ready, monkeypatch):  # noqa: F811
    monkeypatch.setattr(s.previous_round, "CODE_ORDER", ("001000",) + s.previous_round.CODE_ORDER[1:])
    b.save(s.root() / "result.json", {"winner": s.CANDIDATES[0], "model_sha256": "abc"})
    bundle = {n: {"CN_EQUITY": trained(n)} for n in s.CANDIDATES}
    from test_direction_1d_sprint_us_sector_etf_data import points

    sector_input = {"rows": points()}
    monkeypatch.setattr(s.sectors, "capture", lambda at: sector_input)
    monkeypatch.setattr(s.sectors, "load", lambda target: sector_input)
    monkeypatch.setattr(s, "models", lambda: ({"model_sha256": "abc"}, bundle))
    monkeypatch.setattr(s, "FIRST_TARGET", "2026-09-15")
    return joint_ready


def test_future_answer_is_bound_to_original_fund_and_unchanged(ready, monkeypatch):
    directory, original, _ = ready
    assert s.tick()["verified_forecasts"] == 1
    path = s.root() / "forward/2026-09-15/001000.json"
    before = path.read_bytes()
    value = b.read(path)
    assert value["z"][23] == 0 and s.tick()["verified_forecasts"] == 1 and before == path.read_bytes()
    b.save(
        directory / "outcomes/2026-09-15/001000.json",
        {"y": 1, "actual_direction": "UP", "forecast_hash": b.digest(original)},
    )
    monkeypatch.setattr(b, "now", lambda: datetime(2026, 9, 15, 19, tzinfo=b.ZONE))
    assert (
        s.report()["matched_forward_metrics"][s.CANDIDATES[0]]["accuracy"]
        == value["answers"][s.CANDIDATES[0]]["prediction"]
    )


def test_rehashed_fund_identity_cannot_change_saved_vector(ready):
    s.tick()
    path = s.root() / "forward/2026-09-15/001000.json"
    value = b.read(path)
    value["z"][23] = 1
    b.save(path, value, replace=True)
    receipt = s.root() / "receipts/2026-09-15/001000.json"
    b.save(receipt, b.read(receipt) | {"forecast_hash": b.digest(value)}, replace=True)
    with pytest.raises(ValueError, match="VECTOR_CHANGED"):
        s.report()


def test_late_forecast_readback_rejected(ready, monkeypatch):
    save = b.save

    def late(path, value, **kwargs):
        save(path, value, **kwargs)
        if path.parent.parent == s.root() / "forward":
            monkeypatch.setattr(b, "now", lambda: datetime(2026, 9, 15, 8, 30, tzinfo=b.ZONE))

    monkeypatch.setattr(b, "save", late)
    value = s.tick()
    assert value["verified_forecasts"] == 0 and value["invalid_or_late"] == 1


def test_entry_keeps_previous_round_and_attempts_new_after_failure(monkeypatch):
    from scripts import direction_1d_sprint_stacking as entry

    assert entry.existing.__name__ == "scripts.direction_1d_sprint_sector_return"
    monkeypatch.setattr(entry.existing, "run", lambda: (_ for _ in ()).throw(ValueError("OLD_FAILED")))
    calls = []
    monkeypatch.setattr(s, "tick", lambda: calls.append(1))
    monkeypatch.setattr(b, "save", lambda *args, **kw: None)
    with pytest.raises(ValueError, match="INDEPENDENT_BRANCH_FAILED"):
        entry.run()
    assert calls == [1]


def test_missing_sector_keeps_joint_majority_even_when_parent_sources_present():
    z = vector()
    z[20:23] = [0.0, 0.0, 0.0]
    for n in s.CANDIDATES:
        assert s.answer(z, n, trained(n)) == s.majority.answer(z[:20], s.majority.CANDIDATES[0], trained(n))


def test_rehashed_sector_input_rejected(ready, monkeypatch):
    s.tick()
    original = s.sectors.load("2026-09-15")
    changed = dict(original, at="changed")
    monkeypatch.setattr(s.sectors, "load", lambda target: changed)
    with pytest.raises(ValueError, match="STACKING_INPUT_CHANGED"):
        s.report()
