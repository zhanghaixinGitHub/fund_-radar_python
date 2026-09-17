"""四输入变换、成熟隔离、历史/当前对照来源，以及真实未来独立版本。"""

import numpy as np
import pytest
from app.services import direction_1d_sprint as b
from app.services import direction_1d_sprint_market_child_forward_v2 as runtime
from app.services import direction_1d_sprint_market_nasdaq_relative as s

from test_direction_1d_sprint_market_only import history, market
from test_direction_1d_sprint_market_only_forward import forecast
from test_direction_1d_sprint_market_only_forward import ready as parent_ready  # noqa: F401


def market4(raw=(1.0, -0.5, 0.2, -0.4), available=True):
    return market(raw, available)


def test_mixed_transform_keeps_relative_amplitude_and_zero_sign_policy():
    x = np.asarray([[0, -4, 2, -0.4]])
    np.testing.assert_array_equal(s.transformed(x, [1, 2, 2, 0.2], True), [[1, -1, 1, -2]])
    np.testing.assert_array_equal(s.transformed(x, [1, 2, 2, 0.2], False), [[0, -2, 1, -2]])
    with pytest.raises(ValueError):
        s.transformed(x, [1, 2, 2, 0], False)


@pytest.mark.parametrize("name", s.CANDIDATES)
def test_immature_labels_cannot_change_scaling_or_new_feature_coefficient(monkeypatch, name):
    monkeypatch.setattr(s, "active", lambda: None)
    rows = history()
    for i, r in enumerate(rows):
        r["market3"] = r["market"]
        r["market"] = r["market"] | {"features": r["market"]["features"] + [(i % 11 - 5) / 10]}
    cutoff = rows[180]["u"]
    before = s.fit(rows, name, cutoff)
    for r in rows:
        if r["mature"] >= cutoff:
            r["y"] ^= 1
            r["market"] = market4((float("nan"), 0, 0, 0))
    after = s.fit(rows, name, cutoff)
    for k in ("fit_hash", "weight_hash", "scale", "original_fit_hash"):
        assert before[k] == after[k]
    np.testing.assert_array_equal(before["model"].coef_, after["model"].coef_)
    assert before["model"].fit_intercept is False and before["max_mature_date"] < cutoff


@pytest.mark.parametrize("name", s.BRANCHES)
def test_full_seven_branch_fallback_never_changes_scope(name):
    a = s.batch_answers([market4((-1, 0, 0, 3), False)], name)
    assert a[0]["prediction"] == (1 if name == "ALWAYS_UP" else 0)


def test_controls_use_same_cutoff_data_as_candidates():
    for i, name in enumerate(s.CONTROLS[:2]):
        hist, original = s.control_path(name, "CN_EQUITY", "1")
        assert hist.parent.parent == s.baseline.root() and original == s.baseline.CANDIDATES[i]
        live, original = s.control_path(name, "CN_EQUITY", "current")
        assert live.parent.parent == s.previous.root() and original == s.previous.CANDIDATES[i]


def test_declared_names_and_budget_prevent_an_extra_model():
    p = {
        "candidates": list(s.CANDIDATES),
        "controls": list(s.CONTROLS),
        "branch_count": 7,
        "preflight_branch_checks": 210,
        "max_development_fits": 24,
        "max_current_fits": 6,
        "reproductions": 1,
        "current_cutoff": "2026-09-16",
    }
    s.validate_spec(p, 30)
    for k, v in (("controls", list(s.CONTROLS[:-1])), ("max_development_fits", 48), ("current_cutoff", "2026-09-15")):
        with pytest.raises(ValueError):
            s.validate_spec(p | {k: v}, 30)


class Fixed:
    fit_intercept = False
    n_features_in_ = 4

    def predict_proba(self, x):
        return np.tile([0.4, 0.6], (len(x), 1))


def test_future_seven_branches_bind_new_input_hash(parent_ready, monkeypatch):  # noqa: F811
    forecast(parent_ready)
    manifest = {"at": "2026-09-16T05:20:00+08:00", "plan_hash": "NASDAQ", "model_sha256": "MODEL"}
    bundle = {
        n: {"CN_EQUITY": {"model": Fixed(), "mean": [0.0] * 4, "scale": [1.0] * 4, "signed_three": i == 1}}
        for i, n in enumerate(s.CANDIDATES)
    }
    bundle.update(
        {
            n: {"CN_EQUITY": {"model": Fixed(), "mean": [0.0] * 3, "scale": [1.0] * 3, "signed": i == 1}}
            for i, n in enumerate(s.CONTROLS[:2])
        }
    )
    b.save(s.root() / "result.json", manifest)
    monkeypatch.setattr(s, "models", lambda: (manifest, bundle))
    monkeypatch.setattr(s, "live_market", lambda source: market4() | {"nasdaq_input_hash": "SOURCE"})
    assert runtime.tick(s)["verified_forecasts"] == 1
    v = b.read(s.root() / "forward/2026-09-16/001000.json")
    assert set(v["answers"]) == set(s.BRANCHES) and v["market"]["nasdaq_input_hash"] == "SOURCE"
    monkeypatch.setattr(s, "live_market", lambda source: market4() | {"nasdaq_input_hash": "CHANGED"})
    assert len(runtime.report(s)["invalid_forecasts"]) == 1


def test_no_ixic_response_skips_new_prediction_without_stopping_older_branches(monkeypatch):
    monkeypatch.setattr(s.data, "capture", lambda at: None)
    monkeypatch.setattr(runtime, "report", lambda service: {"pending_input": True})
    monkeypatch.setattr(runtime, "tick", lambda service: pytest.fail("must not write unqualified answer"))
    assert s.tick() == {"pending_input": True}
    from scripts import direction_1d_sprint_market_nasdaq_relative as entry

    assert entry.existing.__name__ == "scripts.direction_1d_sprint_market_current_refresh"
