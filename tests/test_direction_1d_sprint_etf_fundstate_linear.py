"""三项信号的规则、直接方向拟合、数据回退和未来证据链。"""

from datetime import datetime

import numpy as np
import pytest
from app.services import direction_1d_sprint as b
from app.services import direction_1d_sprint_etf_fundstate_linear as s

from test_direction_1d_sprint_hk import ConstantModel
from test_direction_1d_sprint_hk import ready as hk_ready  # noqa: F401
from test_direction_1d_sprint_hk import rows as historical_rows
from test_direction_1d_sprint_sequence import ancestor_ready  # noqa: F401
from test_direction_1d_sprint_sequence import ready as sequence_ready  # noqa: F401
from test_direction_1d_sprint_us_etf_direct import ready as direct_ready  # noqa: F401


def trained():
    return {
        "control": {"model": ConstantModel(0.56), "mean": [0.0] * 12, "scale": [1.0] * 12},
        "us_etf": {
            "model": ConstantModel(0.7),
            "mean": [0.0] * 6,
            "scale": [1.0] * 6,
            "feature_indices": (0, 12, 17, 3, 4, 5),
        },
    }


def vector(values):
    z = [0.0] * 20
    z[0], z[12], z[17] = values
    z[16] = z[19] = 1.0
    return z


@pytest.mark.parametrize("source", ["ETF", "CNYA", "BOTH"])
def test_any_unavailable_source_retains_hk_direction_for_every_candidate(source):
    z = vector((1, -1, -1))
    if source in ("ETF", "BOTH"):
        z[12:17] = [0.0] * 5
    if source in ("CNYA", "BOTH"):
        z[17:20] = [0.0] * 3
    model = trained()
    for name in s.CANDIDATES:
        assert s.answer(z, name, model) == s.hk.answer(z[:12], s.CONTROL, model["control"])


def test_mature_only_fit_and_scale_have_no_intercept(monkeypatch):
    monkeypatch.setattr(s, "active", lambda: None)
    rows = [r | {"z": r["z"] + [0.2, 0.0, 0.0, 0.0, 1.0, -0.3, 0.0, 1.0]} for r in historical_rows()]
    name = s.LEARNED[0]
    before = s.fit(rows, name, "2023-06-10")
    for row in rows:
        if row["mature"] >= "2023-06-10":
            row["z"], row["y"] = [float("nan")] * 20, 1 - row["y"]
    after = s.fit(rows, name, "2023-06-10")
    assert before["fit_hash"] == after["fit_hash"]
    assert before["model"].fit_intercept is False and before["mean"] == [0.0] * 6
    assert before["training_target"] == "raw_NAV_UP" and before["model"].n_features_in_ == 6
    np.testing.assert_allclose(before["model"].coef_, after["model"].coef_)


@pytest.fixture
def ready(direct_ready, monkeypatch):  # noqa: F811
    b.save(s.root() / "result.json", {"winner": s.CANDIDATES[0], "model_sha256": "abc"})
    bundle = {n: {"CN_EQUITY": trained()} for n in s.CANDIDATES}
    monkeypatch.setattr(s, "models", lambda: ({"model_sha256": "abc"}, bundle))
    points = {
        day: {"nav_per_share": nav, "non_fv_nav": 100.0, "available": True, "unavailable_reason": None}
        for day, nav in (("2026-09-11", 101.0), ("2026-09-14", 102.0))
    }
    input_value = {"rows": points, "target": "2026-09-15", "base": "2026-09-14"}
    b.save(s.cnya_data.root() / "2026-09-15/input.json", input_value)
    monkeypatch.setattr(s.cnya_data, "load", lambda target: input_value)
    monkeypatch.setattr(s.cnya_data, "capture", lambda at: pytest.fail("must reuse existing CNYA capture"))
    monkeypatch.setattr(s, "FIRST_TARGET", "2026-09-15")
    return direct_ready


def test_early_answers_are_bound_to_both_sources_and_immutable(ready, monkeypatch):
    directory, original, _ = ready
    assert s.tick()["verified_forecasts"] == 1
    path = s.root() / "forward/2026-09-15/001000.json"
    before = path.read_bytes()
    value = b.read(path)
    assert "cnya_input_hash" in value and "us_etf_input_hash" in value
    assert s.tick()["verified_forecasts"] == 1 and path.read_bytes() == before
    b.save(
        directory / "outcomes/2026-09-15/001000.json",
        {"y": 1, "actual_direction": "UP", "forecast_hash": b.digest(original)},
    )
    monkeypatch.setattr(b, "now", lambda: datetime(2026, 9, 15, 19, tzinfo=b.ZONE))
    assert (
        s.report()["matched_forward_metrics"][s.CANDIDATES[0]]["accuracy"]
        == value["answers"][s.CANDIDATES[0]]["prediction"]
    )


def test_changed_cnya_source_is_rejected_even_if_prediction_receipt_unchanged(ready, monkeypatch):
    s.tick()
    original = s.cnya_data.load("2026-09-15")
    monkeypatch.setattr(s.cnya_data, "load", lambda target: original | {"changed": True})
    with pytest.raises(ValueError, match="CNYA_INPUT_CHANGED"):
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
    result = s.tick()
    assert result["verified_forecasts"] == 0 and result["invalid_or_late"] == 1


def test_new_entry_preserves_verified_etf_runtime(monkeypatch):
    from scripts import direction_1d_sprint_etf_fundstate_linear as entry

    assert entry.existing.__name__ == "scripts.direction_1d_sprint_etf_patterns"
    monkeypatch.setattr(entry.existing, "run", lambda: (_ for _ in ()).throw(ValueError("OLD_FAILED")))
    calls = []
    monkeypatch.setattr(s, "tick", lambda: calls.append(1))
    monkeypatch.setattr(b, "save", lambda *args, **kwargs: None)
    with pytest.raises(ValueError, match="INDEPENDENT_BRANCH_FAILED"):
        entry.run()
    assert calls == [1]


def test_fund_state_can_change_direction_with_identical_market_inputs():
    class Probe:
        def predict_proba(self, values):
            score = 1 / (1 + np.exp(-values[:, 3]))
            return np.column_stack((1 - score, score))

    model = trained()
    model["us_etf"]["model"] = Probe()
    up, down = vector((1.0, -1.0, 1.0)), vector((1.0, -1.0, 1.0))
    up[3], down[3] = 2.0, -2.0
    assert s.answer(up, s.CANDIDATES[0], model)["prediction"] == 1
    assert s.answer(down, s.CANDIDATES[0], model)["prediction"] == 0
    np.testing.assert_array_equal(s.selected_features(up, s.CANDIDATES[0]), [1.0, -1.0, 1.0, 2.0, 0.0, 0.0])


def test_three_market_scales_remain_original_in_six_feature_model(monkeypatch):
    monkeypatch.setattr(s, "active", lambda: None)
    monkeypatch.setattr(s.reference, "active", lambda: None)
    rows = [r | {"z": r["z"] + [0.2, 0.0, 0.0, 0.0, 1.0, -0.3, 0.0, 1.0]} for r in historical_rows()]
    old = s.reference.fit(rows, s.reference.LEARNED[0], "2023-06-10")
    new = s.fit(rows, s.LEARNED[0], "2023-06-10")
    assert old["fit_hash"] == new["fit_hash"]
    np.testing.assert_allclose(old["scale"], new["scale"][:3], rtol=0, atol=1e-12)
