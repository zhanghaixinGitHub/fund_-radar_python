"""混合转换、声明名单与预算、成熟标签隔离，以及独立市场未来链。"""

import numpy as np
import pytest
from app.services import direction_1d_sprint as b
from app.services import direction_1d_sprint_market_child_forward_v2 as runtime
from app.services import direction_1d_sprint_market_sign_gap as s

from test_direction_1d_sprint_market_gap_delta import history, market
from test_direction_1d_sprint_market_only_forward import forecast
from test_direction_1d_sprint_market_only_forward import ready as parent_ready  # noqa: F401


def proposal():
    return {
        "candidates": ["EXPANDED_SIGN3_OPEN_GAP_LR4_504"],
        "controls": [
            "FROZEN_R70_MARKET_SIGN_LR3",
            "FROZEN_R74_OPEN_GAP_LR4",
            "MARKET_MAJORITY3",
            "SPX_SIGN",
            "ALWAYS_UP",
        ],
        "branch_count": 6,
        "preflight_branch_checks": 180,
        "max_development_fits": 12,
        "max_current_fits": 3,
    }


def test_actual_declared_names_counts_and_budget_match():
    s.validate_spec(proposal(), 30)
    assert len(s.BRANCHES) == 6


@pytest.mark.parametrize(
    "fault", ["missing_control", "wrong_name", "duplicate", "branch_count", "preflight_count", "fit_budget"]
)
def test_proposal_inconsistency_prevents_freezing(fault):
    p = proposal()
    if fault == "missing_control":
        p["controls"].pop(1)
    elif fault == "wrong_name":
        p["controls"][0] = "WRONG"
    elif fault == "duplicate":
        p["controls"][1] = p["controls"][0]
    elif fault == "branch_count":
        p["branch_count"] = 5
    elif fault == "preflight_count":
        p["preflight_branch_checks"] = 150
    else:
        p["max_development_fits"] = 24
    with pytest.raises(ValueError, match="DECLARED_BRANCHES_OR_BUDGET_CHANGED"):
        s.validate_spec(p, 30)


def test_mixed_transform_signs_only_first_three_and_keeps_gap_magnitude():
    x = np.array([[0, -30, 4, 20], [-1, 0, 4, -20]], dtype=float)
    z = s.transformed(x, [2, 3, 4, 10], "SIGN3_RAW_GAP")
    np.testing.assert_array_equal(z, [[1, -1, 1, 2], [-1, 1, 1, -2]])
    np.testing.assert_array_equal(s.transformed(x[:, :3], [2, 3, 4], "SIGN3"), z[:, :3])
    np.testing.assert_array_equal(s.transformed(x, [2, 3, 4, 10], "SCALED"), x / [2, 3, 4, 10])


def test_fourth_amplitude_is_not_accidentally_binary():
    z = s.transformed(np.array([[1, 1, 1, 0.1], [1, 1, 1, 0.9]]), [1, 1, 1, 1], "SIGN3_RAW_GAP")
    assert z[0, 3] == 0.1 and z[1, 3] == 0.9


@pytest.mark.parametrize("kind,scale", [("BAD", [1] * 4), ("SIGN3", [1] * 4), ("SIGN3_RAW_GAP", [1, 1, 1, 0])])
def test_bad_conversion_shape_or_scale_rejected(kind, scale):
    with pytest.raises(ValueError):
        s.transformed(np.zeros((1, 4)), scale, kind)


def test_immature_labels_do_not_change_fit_or_scaling(monkeypatch):
    monkeypatch.setattr(s, "active", lambda: None)
    rows = history()
    cutoff = rows[180]["u"]
    before = s.fit(rows, s.CANDIDATES[0], cutoff)
    for r in rows:
        if r["mature"] >= cutoff:
            r["y"] ^= 1
            r["market"]["features"] = [float("nan")] * 5
    after = s.fit(rows, s.CANDIDATES[0], cutoff)
    assert before["fit_hash"] == after["fit_hash"] and before["weight_hash"] == after["weight_hash"]
    np.testing.assert_allclose(before["model"].coef_, after["model"].coef_)
    assert before["transform"] == "SIGN3_RAW_GAP" and before["model"].fit_intercept is False
    assert before["model"].n_features_in_ == 4 and before["max_mature_date"] < cutoff


class Recorder:
    def __init__(self):
        self.inputs = []

    def predict_proba(self, x):
        self.inputs.append(x.copy())
        return np.tile([0.4, 0.6], (len(x), 1))


def models():
    return {
        name: {
            "CN_EQUITY": {
                "model": Recorder(),
                "mean": [0.0] * len(s.FEATURE_COLUMNS[name]),
                "scale": [1.0] * len(s.FEATURE_COLUMNS[name]),
                "transform": s.TRANSFORMS[name],
            }
        }
        for name in s.CANDIDATES + s.CONTROLS[:2]
    }


def test_unselected_fifth_feature_and_extra_vote_cannot_change_answers():
    heads = models()
    a = market((-1, -1, 1))
    a["features"][3:] = [20, 1000]
    before = s.answers(a, "CN_EQUITY", heads)
    a["features"][4] = -1000
    assert s.answers(a, "CN_EQUITY", heads) == before
    assert before["MARKET_MAJORITY3"]["prediction"] == 0
    np.testing.assert_array_equal(heads[s.CANDIDATES[0]]["CN_EQUITY"]["model"].inputs[0], [[-1, -1, 1, 20]])
    np.testing.assert_array_equal(heads[s.CONTROLS[0]]["CN_EQUITY"]["model"].inputs[0], [[-1, -1, 1]])


@pytest.mark.parametrize("name", s.BRANCHES)
def test_same_spx_fallback_without_accessing_a_model(name):
    a = market((-1, 0, 0), False)
    assert s.batch_answers([a], name)[0]["prediction"] == (1 if name == "ALWAYS_UP" else 0)


def test_six_future_branches_have_independent_market_parent(parent_ready, monkeypatch):  # noqa: F811
    forecast(parent_ready)
    manifest = {"at": "2026-09-16T03:00:00+08:00", "plan_hash": "MIXED", "model_sha256": "MODEL"}
    b.save(s.root() / "result.json", manifest)
    monkeypatch.setattr(s, "models", lambda: (manifest, models()))
    monkeypatch.setattr(s, "live_market", lambda source: market())
    assert runtime.tick(s)["verified_forecasts"] == 1
    value = b.read(s.root() / "forward/2026-09-16/001000.json")
    assert set(value["answers"]) == set(s.BRANCHES)
    assert not (b.ROOT / "forward").exists()


def test_entry_preserves_corrected_previous_version():
    from scripts import direction_1d_sprint_market_sign_gap as entry

    assert entry.existing.__name__ == "scripts.direction_1d_sprint_market_gap_ablation_v2"
