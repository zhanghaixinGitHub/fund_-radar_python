"""验证新增互联网价格确实参与一日推断，成熟训练与未来证据保持原题。"""

from datetime import datetime

import numpy as np
import pytest
from app.services import direction_1d_sprint as b
from app.services import direction_1d_sprint_china_internet_tree as s

from test_direction_1d_sprint_etf_joint import ready as joint_ready  # noqa: F401
from test_direction_1d_sprint_etf_joint import vector as parent_vector
from test_direction_1d_sprint_hk import ConstantModel
from test_direction_1d_sprint_hk import ready as hk_ready  # noqa: F401
from test_direction_1d_sprint_hk import rows as historical_rows
from test_direction_1d_sprint_sequence import ancestor_ready  # noqa: F401
from test_direction_1d_sprint_sequence import ready as sequence_ready  # noqa: F401
from test_direction_1d_sprint_us_etf_direct import ready as direct_ready  # noqa: F401


class LinearProbe:
    n_features_in_ = 4

    def predict_proba(self, x):
        scores = 1 / (1 + np.exp(-x[:, 3]))
        return np.column_stack((1 - scores, scores))


def trained(name=None):
    return {
        "control": {"model": ConstantModel(0.56), "mean": [0.0] * 12, "scale": [1.0] * 12},
        "us_etf": {"model": LinearProbe(), "mean": [0.0] * 4, "scale": [1.0] * 4, "code_order": s.CODE_ORDER},
    }


def vector(index=0, values=(2, -3, 4)):
    return parent_vector(values) + [0.2, 1.0, index]


def test_four_scaled_inputs_include_internet_signal():
    matrix = s.design([vector(0), vector(1)], [2, 3, 4, 0.2], s.CANDIDATES[0])
    np.testing.assert_allclose(matrix, [[1, -1, 1, 1]] * 2)
    z = vector()
    assert s.answer(z, s.CANDIDATES[0], trained())["prediction"] == 1
    z[20] = -0.2
    assert s.answer(z, s.CANDIDATES[0], trained())["prediction"] == 0
    z[20] = 0
    assert s.answer(z, s.CANDIDATES[0], trained())["research_score"] == 0.5


@pytest.mark.parametrize("index", [-1, 30, 0.5, True, float("nan")])
def test_invalid_index_fails(index):
    with pytest.raises(ValueError, match="INPUT_INVALID"):
        s.design([vector(index)], [1, 1, 1, 1], s.CANDIDATES[0])


def test_unknown_code_or_reordered_model_fails():
    with pytest.raises(ValueError, match="UNKNOWN_CODE"):
        s.fund_index("999999")
    model = trained()
    model["us_etf"]["code_order"] = tuple(reversed(s.CODE_ORDER))
    with pytest.raises(ValueError, match="MODEL_SCHEMA_CHANGED"):
        s.answer(vector(), s.CANDIDATES[0], model)


def test_missing_equity_keeps_parent_without_fund_adjustment():
    z = vector()
    z[17:20] = [0.0] * 3
    name = s.CANDIDATES[0]
    assert s.answer(z, name, trained()) == s.majority.answer(z[:20], s.majority.CANDIDATES[0], trained())


def test_mature_only_fit_keeps_three_scales_and_no_intercept(monkeypatch):
    monkeypatch.setattr(s, "active", lambda: None)
    rows = []
    for i, r in enumerate(historical_rows()):
        index = i % 2
        rows.append(r | {"code": s.CODE_ORDER[index], "z": r["z"] + [0.2, 0, 0, 0, 1, -0.3, 0, 1, 0.2, 1, index]})
    name = s.CANDIDATES[0]
    before = s.fit(rows, name, "2023-06-10")
    for r in rows:
        if r["mature"] >= "2023-06-10":
            r["z"], r["y"] = [float("nan")] * 23, 1 - r["y"]
    after = s.fit(rows, name, "2023-06-10")
    assert before["fit_hash"] == after["fit_hash"] and len(after["scale"]) == 4
    assert after["model"].n_features_in_ == 4 and after["model"].max_depth == 6
    grid = s.design([vector(0), vector(1)], [1.0] * 4, s.CANDIDATES[0])
    np.testing.assert_allclose(before["model"].predict_proba(grid), after["model"].predict_proba(grid))
    assert after["scale"] == [1.0] * 4 and after["model"].n_estimators == 256
    assert after["model"].min_samples_leaf == 80 and after["model"].class_weight is None


@pytest.fixture
def ready(joint_ready, monkeypatch):  # noqa: F811
    monkeypatch.setattr(s, "CODE_ORDER", ("001000",) + s.CODE_ORDER[1:])
    b.save(s.root() / "result.json", {"winner": s.CANDIDATES[0], "model_sha256": "abc"})
    bundle = {n: {"CN_EQUITY": trained(n)} for n in s.CANDIDATES}
    from test_direction_1d_sprint_china_internet_data import points

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
    assert value["z"][22] == 0 and s.tick()["verified_forecasts"] == 1 and before == path.read_bytes()
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
    value["z"][22] = 1
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
    from scripts import direction_1d_sprint_china_internet_tree as entry

    assert entry.existing.__name__ == "scripts.direction_1d_sprint_china_internet"
    monkeypatch.setattr(entry.existing, "run", lambda: (_ for _ in ()).throw(ValueError("OLD_FAILED")))
    calls = []
    monkeypatch.setattr(s, "tick", lambda: calls.append(1))
    monkeypatch.setattr(b, "save", lambda *args, **kw: None)
    with pytest.raises(ValueError, match="INDEPENDENT_BRANCH_FAILED"):
        entry.run()
    assert calls == [1]


def test_missing_sector_keeps_joint_majority_even_when_parent_sources_present():
    z = vector()
    z[20:22] = [0.0, 0.0]
    for n in s.CANDIDATES:
        assert s.answer(z, n, trained(n)) == s.majority.answer(z[:20], s.majority.CANDIDATES[0], trained(n))


def test_rehashed_sector_input_rejected(ready, monkeypatch):
    s.tick()
    original = s.sectors.load("2026-09-15")
    changed = dict(original, at="changed")
    monkeypatch.setattr(s.sectors, "load", lambda target: changed)
    with pytest.raises(ValueError, match="CHINA_INTERNET_INPUT_CHANGED"):
        s.report()
