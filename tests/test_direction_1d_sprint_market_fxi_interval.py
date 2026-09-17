"""候选修改区间、旧控制保持原输入、成熟隔离与未来版本验证。"""

import numpy as np
import pytest
from app.services import direction_1d_sprint as b
from app.services import direction_1d_sprint_market_child_forward_v2 as runtime
from app.services import direction_1d_sprint_market_fxi_interval as s

from test_direction_1d_sprint_market_only import history, market
from test_direction_1d_sprint_market_only_forward import forecast
from test_direction_1d_sprint_market_only_forward import ready as parent_ready  # noqa: F401


def paired(available=True):
    original = market((-0.2, -0.3, 0.4), available)
    return s.data.pair(original, market((-0.2, 0.3, 0.4), available))


class FxiOnly:
    def predict_proba(self, x):
        up = 1 / (1 + np.exp(-x[:, 1]))
        return np.column_stack((1 - up, up))


def head(signed=False):
    return {"model": FxiOnly(), "mean": [0.0] * 3, "scale": [1.0] * 3, "signed": signed}


def test_all_old_controls_read_old_fxi_while_candidates_read_interval():
    value = paired()
    assert s.batch_answers([value], s.CANDIDATES[0], head())[0]["prediction"] == 1
    assert s.batch_answers([value], s.CANDIDATES[1], head(True))[0]["prediction"] == 1
    assert s.batch_answers([value], s.CONTROLS[0], head())[0]["prediction"] == 0
    assert s.batch_answers([value], s.CONTROLS[1], head(True))[0]["prediction"] == 0
    assert s.batch_answers([value], "MARKET_MAJORITY3")[0]["prediction"] == 0


def test_other_features_cannot_be_silently_replaced():
    value = paired()
    value["features"][2] = 9
    with pytest.raises(ValueError, match="OTHER_FEATURE_CHANGED"):
        s.batch_answers([value], s.CANDIDATES[0], head())


@pytest.mark.parametrize("name", s.CANDIDATES)
def test_immature_rows_cannot_change_new_interval_model(monkeypatch, name):
    monkeypatch.setattr(s.baseline, "active", lambda: None)
    rows = history()
    for i, r in enumerate(rows):
        r["market3"] = r["market"]
        new = r["market"] | {"features": r["market"]["features"].copy()}
        if i % 30 == 0:
            new["features"][1] *= 2
        r["market"] = s.data.pair(r["market3"], new)
    cutoff = rows[180]["u"]
    before = s.fit(rows, name, cutoff)
    for r in rows:
        if r["mature"] >= cutoff:
            r["y"] ^= 1
            r["market"]["features"][1] = float("nan")
    after = s.fit(rows, name, cutoff)
    for k in ["fit_hash", "original_fit_hash", "weight_hash", "scale"]:
        assert before[k] == after[k]
    np.testing.assert_array_equal(before["model"].coef_, after["model"].coef_)


@pytest.mark.parametrize("name", s.BRANCHES)
def test_all_seven_branches_keep_original_missing_source_rule(name):
    value = s.batch_answers([paired(False)], name)[0]
    assert value["prediction"] == (1 if name == "ALWAYS_UP" else 0)


def test_frozen_contract_requires_original_control_input():
    p = {
        "candidates": list(s.CANDIDATES),
        "controls": list(s.CONTROLS),
        "branch_count": 7,
        "preflight_branch_checks": 210,
        "max_development_fits": 24,
        "max_current_fits": 6,
        "reproductions": 1,
        "current_cutoff": "2026-09-16",
        "control_input_policy": "ALL_CONTROLS_USE_ORIGINAL_FXI_FEATURE",
    }
    s.validate_spec(p, 30)
    with pytest.raises(ValueError):
        s.validate_spec(p | {"control_input_policy": "NEW_FXI"}, 30)


def test_future_record_preserves_both_input_definitions(parent_ready, monkeypatch):  # noqa: F811
    forecast(parent_ready)
    manifest = {"at": "2026-09-16T05:50:00+08:00", "plan_hash": "FXI_INTERVAL", "model_sha256": "MODEL"}
    bundle = {n: {"CN_EQUITY": head(i == 1)} for group in [s.CANDIDATES, s.CONTROLS[:2]] for i, n in enumerate(group)}
    b.save(s.root() / "result.json", manifest)
    monkeypatch.setattr(s, "models", lambda: (manifest, bundle))
    monkeypatch.setattr(s, "live_market", lambda source: paired())
    assert runtime.tick(s)["verified_forecasts"] == 1
    value = b.read(s.root() / "forward/2026-09-16/001000.json")
    assert value["market"]["original_market"]["features"][1] < 0 < value["market"]["features"][1]
    assert value["answers"]["MARKET_MAJORITY3"]["prediction"] == 0
    from scripts import direction_1d_sprint_market_fxi_interval as entry

    assert entry.existing.__name__ == "scripts.direction_1d_sprint_market_nonnegative"
