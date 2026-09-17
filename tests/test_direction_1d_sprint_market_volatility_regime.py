"""波动交互训练变换、旧模型子空间、未来版本绑定及成熟隔离。"""

import numpy as np
import pytest
from app.services import direction_1d_sprint as b
from app.services import direction_1d_sprint_market_child_forward_v2 as runtime
from app.services import direction_1d_sprint_market_volatility_regime as s

from test_direction_1d_sprint_market_fxi_interval import head as old_head
from test_direction_1d_sprint_market_lagged_volatility import value
from test_direction_1d_sprint_market_only import history
from test_direction_1d_sprint_market_only_forward import forecast
from test_direction_1d_sprint_market_only_forward import ready as parent_ready  # noqa: F401


class Linear:
    def __init__(self, coef):
        self.coef = np.asarray(coef)

    def predict_proba(self, x):
        up = 1 / (1 + np.exp(-(x @ self.coef)))
        return np.column_stack((1 - up, up))


def model_head(coef=(0.2, 0.3, 0.4, 0.1, 0.1, 0.1)):
    return {
        "model": Linear(coef),
        "mean": [0.0] * 6,
        "transform": "SIGN3_LOGVOL20_INTERACTION6",
        "logvol_center": [0.1, 0.2, 0.3],
        "interaction_scale": [1.0, 2.0, 3.0],
    }


def test_zero_interaction_is_exactly_original_sign_model():
    h = model_head((0.2, 0.3, 0.4, 0, 0, 0))
    new = s.batch_answers([value()], s.CANDIDATES[0], h)[0]
    old = s.prior_chain.batch_answers(
        [value()],
        s.CONTROLS[1],
        {"model": Linear([0.2, 0.3, 0.4]), "mean": [0.0] * 3, "scale": [1.0] * 3, "signed": True},
    )[0]
    assert new["research_score"] == old["research_score"] and new["prediction"] == old["prediction"]


def test_transform_preserves_signs_and_uses_frozen_head_center():
    v = value()
    h = model_head()
    x = s.transformed([v], h["logvol_center"], h["interaction_scale"])
    np.testing.assert_array_equal(x[0, :3], [-1, 1, 1])
    np.testing.assert_allclose(
        x[0, 3:], np.array([-1, 1, 1]) * (np.log([2, 3, 4]) - h["logvol_center"]) / h["interaction_scale"]
    )
    doubled = s.transformed([v, v], h["logvol_center"], h["interaction_scale"])
    np.testing.assert_array_equal(doubled[0], x[0])
    np.testing.assert_array_equal(doubled[1], x[0])


@pytest.mark.parametrize("name", s.BRANCHES)
def test_missing_market_does_not_enter_interaction_model(name):
    assert s.batch_answers([value(False)], name)[0]["prediction"] == (1 if name == "ALWAYS_UP" else 0)


@pytest.mark.parametrize("fault", ["center_nan", "scale_zero", "width", "model_mean", "model_transform"])
def test_invalid_transform_or_native_head_fails_closed(fault):
    h = model_head()
    if fault == "center_nan":
        h["logvol_center"][0] = float("nan")
    elif fault == "scale_zero":
        h["interaction_scale"][0] = 0
    elif fault == "width":
        h["interaction_scale"] = [1, 2]
    elif fault == "model_mean":
        h["mean"] = [0.0] * 3
    else:
        h["transform"] = "SOMETHING_ELSE"
    with pytest.raises(ValueError):
        s.batch_answers([value()], s.CANDIDATES[0], h)


def test_immature_labels_and_market_states_cannot_change_current_training(monkeypatch):
    monkeypatch.setattr(s, "active", lambda: None)
    rows = history()
    for i, r in enumerate(rows):
        original = r["market"]
        interval = s.data.interval.pair(original, original)
        scale = [2 + i % 7 / 10, 3 + i % 5 / 10, 4 + i % 3 / 10]
        captured = {
            "available": True,
            "scale": scale,
            "normalized_features": (np.asarray(original["features"]) / scale).tolist(),
        }
        r.update({"market3": original, "interval_market": interval, "market": s.data.paired(interval, captured)})
    cutoff = rows[180]["u"]
    a = s.fit(rows, s.CANDIDATES[0], cutoff)
    for r in rows:
        if r["mature"] >= cutoff:
            r["y"] ^= 1
            r["market"]["volatility_context"]["scale"] = [float("nan")] * 3
    z = s.fit(rows, s.CANDIDATES[0], cutoff)
    for k in ["fit_hash", "original_fit_hash", "weight_hash", "logvol_center", "interaction_scale"]:
        assert a[k] == z[k]
    np.testing.assert_array_equal(a["model"].coef_, z["model"].coef_)
    assert a["model"].n_features_in_ == 6 and a["model"].fit_intercept is False


def test_six_future_branches_bind_existing_volatility_context(parent_ready, monkeypatch):  # noqa: F811
    forecast(parent_ready)
    manifest = {"at": "2026-09-16T06:00:00+08:00", "plan_hash": "REGIME6", "model_sha256": "MODEL"}
    bundle = {s.CANDIDATES[0]: {"CN_EQUITY": model_head()}}
    bundle.update({n: {"CN_EQUITY": old_head(i == 1)} for i, n in enumerate(s.CONTROLS[:2])})
    b.save(s.root() / "result.json", manifest)
    monkeypatch.setattr(s, "models", lambda: (manifest, bundle))
    hash_value = ["FROZEN_R84_INPUT"]
    monkeypatch.setattr(s, "live_market", lambda source: value() | {"volatility_input_hash": hash_value[0]})
    assert runtime.tick(s)["verified_forecasts"] == 1
    saved = b.read(s.root() / "forward/2026-09-16/001000.json")
    assert len(saved["answers"]) == 6
    assert saved["answers"][s.CANDIDATES[0]]["route"] == "MARKET_VOLATILITY_REGIME_LR6"
    hash_value[0] = "CHANGED"
    assert runtime.report(s)["verified_forecasts"] == 0
    from scripts import direction_1d_sprint_market_volatility_regime as entry

    assert entry.existing.__name__ == "scripts.direction_1d_sprint_market_lagged_volatility"
