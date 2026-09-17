"""验证自身净值状态的消融、成熟样本隔离及未来输入和答案不可回写。"""

from datetime import datetime

import numpy as np
import pytest
from app.services import direction_1d_sprint as b
from app.services import direction_1d_sprint_etf_fund_state as s

from test_direction_1d_sprint_etf_joint import ready as joint_ready  # noqa: F401
from test_direction_1d_sprint_etf_joint import vector as parent_vector
from test_direction_1d_sprint_hk import ConstantModel
from test_direction_1d_sprint_hk import ready as hk_ready  # noqa: F401
from test_direction_1d_sprint_hk import rows as historical_rows
from test_direction_1d_sprint_sequence import ancestor_ready  # noqa: F401
from test_direction_1d_sprint_sequence import ready as sequence_ready  # noqa: F401
from test_direction_1d_sprint_us_etf_direct import ready as direct_ready  # noqa: F401


class StateProbe:
    def __init__(self, size):
        self.n_features_in_ = size

    def predict_proba(self, x):
        # 在同一市场行情下，仅基金状态候选可以因自身涨跌状态改变判断。
        score = 1 / (1 + np.exp(-x[:, 3])) if self.n_features_in_ == 6 else np.full(len(x), 0.5)
        return np.column_stack((1 - score, score))


def trained(name):
    return {
        "control": {"model": ConstantModel(0.56), "mean": [0.0] * 12, "scale": [1.0] * 12},
        "us_etf": {"model": StateProbe(len(s.FEATURE_INDICES[name])), "feature_indices": s.FEATURE_INDICES[name]},
    }


def vector(state=(1.0, -2.0, 3.0)):
    z = parent_vector((2.0, -3.0, 4.0))
    z[3:6] = state
    return z


def test_ablation_selects_existing_market_and_own_fund_features():
    assert s.selected_features(vector(), s.CANDIDATES[0]) == [2.0, -3.0, 4.0]
    assert s.selected_features(vector(), s.CANDIDATES[1]) == [2.0, -3.0, 4.0, 1.0, -2.0, 3.0]
    for n in s.CANDIDATES:
        a = s.answer(vector(), n, trained(n))
        c = s.answer(vector((-1.0, -2.0, 3.0)), n, trained(n))
        assert (a["prediction"] != c["prediction"]) == (n == s.CANDIDATES[1])
    assert s.answer(vector((0.0, 0.0, 0.0)), s.CANDIDATES[1], trained(s.CANDIDATES[1]))["prediction"] == 1


@pytest.mark.parametrize("state", [(6, 0, 0), (0, float("nan"), 0), (0, 0, float("inf"))])
def test_invalid_fund_state_fails(state):
    with pytest.raises(ValueError, match="INPUT_INVALID"):
        s.selected_features(vector(state), s.CANDIDATES[1])


def test_reordered_features_or_wrong_model_dimension_fail():
    name = s.CANDIDATES[1]
    head = trained(name)
    head["us_etf"]["feature_indices"] = tuple(reversed(s.FEATURE_INDICES[name]))
    with pytest.raises(ValueError, match="MODEL_SCHEMA_CHANGED"):
        s.answer(vector(), name, head)
    head = trained(name)
    head["us_etf"]["model"].n_features_in_ = 3
    with pytest.raises(ValueError, match="MODEL_SCHEMA_CHANGED"):
        s.answer(vector(), name, head)


@pytest.mark.parametrize("name", s.CANDIDATES)
def test_missing_source_keeps_exact_parent(name):
    z = vector()
    z[17:20] = [0.0] * 3
    assert s.answer(z, name, trained(name)) == s.majority.answer(z, s.majority.CANDIDATES[0], trained(name))


def test_both_models_use_same_mature_questions_and_raw_up_target(monkeypatch):
    monkeypatch.setattr(s, "active", lambda: None)
    rows = [r | {"z": r["z"] + [0.2, 0, 0, 0, 1, -0.3, 0, 1]} for r in historical_rows()]
    heads = [s.fit(rows, n, "2023-06-10") for n in s.CANDIDATES]
    assert heads[0]["fit_hash"] == heads[1]["fit_hash"]
    for r in rows:
        if r["mature"] >= "2023-06-10":
            r["z"], r["y"] = [float("nan")] * 20, 1 - r["y"]
    after = s.fit(rows, s.CANDIDATES[1], "2023-06-10")
    assert heads[1]["fit_hash"] == after["fit_hash"]
    assert after["max_mature_date"] < after["cutoff"] and after["training_target"] == "raw_NAV_UP"
    assert after["model"].n_features_in_ == 6 and after["model"].class_weight is None
    assert after["model"].n_estimators == 256 and after["model"].min_samples_leaf == 80
    exam = np.asarray([s.selected_features(vector(), s.CANDIDATES[1])])
    np.testing.assert_allclose(heads[1]["model"].predict_proba(exam), after["model"].predict_proba(exam))


@pytest.fixture
def ready(joint_ready, monkeypatch):  # noqa: F811
    b.save(s.root() / "result.json", {"winner": s.CANDIDATES[1], "model_sha256": "abc"})
    bundle = {n: {"CN_EQUITY": trained(n)} for n in s.CANDIDATES}
    monkeypatch.setattr(s, "models", lambda: ({"model_sha256": "abc"}, bundle))
    monkeypatch.setattr(s, "FIRST_TARGET", "2026-09-15")
    return joint_ready


def test_future_both_candidates_immutable_and_match_raw_outcome(ready, monkeypatch):
    directory, original, _ = ready
    assert s.tick()["verified_forecasts"] == 1
    path = s.root() / "forward/2026-09-15/001000.json"
    before = path.read_bytes()
    value = b.read(path)
    assert set(value["answers"]) == set(s.CANDIDATES) and len(value["z"]) == 20
    assert s.tick()["verified_forecasts"] == 1 and before == path.read_bytes()
    b.save(
        directory / "outcomes/2026-09-15/001000.json",
        {"y": 1, "actual_direction": "UP", "forecast_hash": b.digest(original)},
    )
    monkeypatch.setattr(b, "now", lambda: datetime(2026, 9, 15, 19, tzinfo=b.ZONE))
    result = s.report()
    for n in s.CANDIDATES:
        assert result["matched_forward_metrics"][n]["accuracy"] == value["answers"][n]["prediction"]


def test_rehashed_nav_state_cannot_change_saved_input(ready):
    s.tick()
    path = s.root() / "forward/2026-09-15/001000.json"
    value = b.read(path)
    value["z"][3] += 1
    b.save(path, value, replace=True)
    receipt = s.root() / "receipts/2026-09-15/001000.json"
    b.save(receipt, b.read(receipt) | {"forecast_hash": b.digest(value)}, replace=True)
    with pytest.raises(ValueError, match="VECTOR_CHANGED"):
        s.report()


def test_late_readback_rejected(ready, monkeypatch):
    save = b.save

    def late(path, value, **kwargs):
        save(path, value, **kwargs)
        if path.parent.parent == s.root() / "forward":
            monkeypatch.setattr(b, "now", lambda: datetime(2026, 9, 15, 8, 30, tzinfo=b.ZONE))

    monkeypatch.setattr(b, "save", late)
    result = s.tick()
    assert result["verified_forecasts"] == 0 and result["invalid_or_late"] == 1


def test_entry_attempts_new_branch_even_if_previous_fails(monkeypatch):
    from scripts import direction_1d_sprint_etf_fund_state as entry

    assert entry.existing.__name__ == "scripts.direction_1d_sprint_fund_pooling"
    monkeypatch.setattr(entry.existing, "run", lambda: (_ for _ in ()).throw(ValueError("OLD_FAILED")))
    calls = []
    monkeypatch.setattr(s, "tick", lambda: calls.append(1))
    monkeypatch.setattr(b, "save", lambda *args, **kw: None)
    with pytest.raises(ValueError, match="INDEPENDENT_BRANCH_FAILED"):
        entry.run()
    assert calls == [1]
