"""三种输入分支隔离、成熟样本约束及未来六分支实际保存逻辑。"""

import numpy as np
import pytest
from app.services import direction_1d_sprint as b
from app.services import direction_1d_sprint_market_child_forward_v2 as runtime
from app.services import direction_1d_sprint_market_lagged_volatility as s

from test_direction_1d_sprint_market_fxi_interval import head, paired
from test_direction_1d_sprint_market_only import history
from test_direction_1d_sprint_market_only_forward import forecast
from test_direction_1d_sprint_market_only_forward import ready as parent_ready  # noqa: F401


def value(available=True):
    market = paired(available)
    captured = {"available": True, "scale": [2.0, 3.0, 4.0], "normalized_features": [-0.1, 0.09999999999999999, 0.1]}
    if not available:
        captured = {"available": False, "reason": "PARENT_SOURCE_UNAVAILABLE"}
    return s.data.paired(market, captured)


def test_candidate_and_two_kinds_of_controls_get_their_own_features():
    v = value()
    candidate = s.batch_answers([v], s.CANDIDATES[0], head())[0]
    learned = s.batch_answers([v], s.CONTROLS[0], head())[0]
    assert candidate["research_score"] < learned["research_score"]
    assert candidate["prediction"] == learned["prediction"] == 1
    assert s.batch_answers([v], "MARKET_MAJORITY3")[0]["prediction"] == 0


@pytest.mark.parametrize("fault", ["scale", "features", "original", "availability"])
def test_changed_feature_definitions_are_rejected(fault):
    v = value()
    if fault == "scale":
        v["volatility_context"]["scale"][0] = -1
    elif fault == "features":
        v["features"][1] += 1
    elif fault == "original":
        v["original_market"] = paired()["original_market"] | {"features": [1, 1, 1]}
    else:
        v["available"] = False
    with pytest.raises(ValueError):
        s.batch_answers([v], s.CANDIDATES[0], head())


@pytest.mark.parametrize("name", s.BRANCHES)
def test_all_six_branches_preserve_missing_market_rule(name):
    assert s.batch_answers([value(False)], name)[0]["prediction"] == (1 if name == "ALWAYS_UP" else 0)


def test_future_labels_cannot_change_fit_and_context_missing_cannot_drop_rows(monkeypatch):
    monkeypatch.setattr(s.baseline, "active", lambda: None)
    rows = history()
    for row in rows:
        original = row["market"]
        interval = s.data.interval.pair(original, original)
        c = {
            "available": True,
            "scale": [2.0, 3.0, 4.0],
            "normalized_features": (np.asarray(original["features"]) / [2, 3, 4]).tolist(),
        }
        row.update({"market3": original, "interval_market": interval, "market": s.data.paired(interval, c)})
    cutoff = rows[180]["u"]
    a = s.fit(rows, s.CANDIDATES[0], cutoff)
    for row in rows:
        if row["mature"] >= cutoff:
            row["y"] ^= 1
            row["market"]["features"] = [float("nan")] * 3
    z = s.fit(rows, s.CANDIDATES[0], cutoff)
    for k in ("fit_hash", "original_fit_hash", "weight_hash", "scale"):
        assert a[k] == z[k]
    np.testing.assert_array_equal(a["model"].coef_, z["model"].coef_)
    rows[50]["market"]["volatility_context"]["available"] = False
    with pytest.raises(ValueError, match="TRAIN_CONTEXT_MISSING"):
        s.training_rows(rows, cutoff)


def test_frozen_budget_and_no_duplicate_sign_model():
    p = {
        "candidates": list(s.CANDIDATES),
        "controls": list(s.CONTROLS),
        "branch_count": 6,
        "preflight_branch_checks": 180,
        "max_development_fits": 12,
        "max_current_fits": 3,
        "reproductions": 1,
        "current_cutoff": "2026-09-16",
        "lookback_complete_dates": 20,
        "scale_floor_percent": 1e-6,
        "future_scan_bound_cn_dates": 80,
        "control_input_policy": "R82_LEARNED_INTERVAL_FIXED_RULES_ORIGINAL_CANDIDATE_NORMALIZED",
    }
    s.validate_spec(p, 30)
    with pytest.raises(ValueError):
        s.validate_spec(p | {"lookback_complete_dates": 60}, 30)


def test_live_context_hash_binds_forecast_and_no_context_means_no_new_answer(parent_ready, monkeypatch):  # noqa: F811
    forecast(parent_ready)
    manifest = {"at": "2026-09-16T06:30:00+08:00", "plan_hash": "VOL20", "model_sha256": "MODEL"}
    bundle = {n: {"CN_EQUITY": head(n == s.CONTROLS[1])} for n in s.CANDIDATES + s.CONTROLS[:2]}
    b.save(s.root() / "result.json", manifest)
    monkeypatch.setattr(s, "models", lambda: (manifest, bundle))
    monkeypatch.setattr(s.data, "capture", lambda at: None)
    assert s.tick()["verified_forecasts"] == 0
    context_hash = ["CONTEXT_A"]
    monkeypatch.setattr(s, "live_market", lambda source: value() | {"volatility_input_hash": context_hash[0]})
    assert runtime.tick(s)["verified_forecasts"] == 1
    saved = b.read(s.root() / "forward/2026-09-16/001000.json")
    assert len(saved["answers"]) == 6 and saved["market"]["volatility_input_hash"] == "CONTEXT_A"
    context_hash[0] = "CONTEXT_B"
    assert runtime.report(s)["verified_forecasts"] == 0
    from scripts import direction_1d_sprint_market_lagged_volatility as entry

    assert entry.existing.__name__ == "scripts.direction_1d_sprint_market_expanding_interval"
