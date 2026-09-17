"""扩大历史保留旧子集、成熟边界与学习对照的完整区间输入。"""

from datetime import date, timedelta

import numpy as np
import pytest
from app.services import direction_1d_sprint as b
from app.services import direction_1d_sprint_market_child_forward_v2 as runtime
from app.services import direction_1d_sprint_market_expanding_interval as s

from test_direction_1d_sprint_market_fxi_interval import head, paired
from test_direction_1d_sprint_market_only import market
from test_direction_1d_sprint_market_only_forward import forecast
from test_direction_1d_sprint_market_only_forward import ready as parent_ready  # noqa: F401


def history():
    out = []
    for i in range(900):
        day = date(2021, 1, 1) + timedelta(days=i)
        original = market(((i % 7 - 3) / 10, (i % 5 - 2) / 10, (i % 9 - 4) / 10))
        out.append(
            {
                "code": "001000",
                "family": "F",
                "group": "CN_EQUITY",
                "u": str(day),
                "mature": str(day + timedelta(days=1)),
                "y": int(i % 3 == 0),
                "market3": original,
                "market": s.data.pair(original, original),
            }
        )
    return out


def test_entire_mature_history_contains_exact_rolling_subset():
    rows = history()
    cutoff = rows[850]["u"]
    extended = s.training_rows(rows, cutoff)
    rolling = s.previous.training_rows(rows, cutoff)
    assert len(extended) == 849 and len(rolling) == 504
    assert all(r in extended for r in rolling) and extended[0] == rows[0]
    assert all(r["mature"] < cutoff and r["u"] < cutoff for r in extended)


@pytest.mark.parametrize("name", s.CANDIDATES)
def test_future_labels_still_cannot_change_expanding_fit(monkeypatch, name):
    monkeypatch.setattr(s, "active", lambda: None)
    rows = history()
    cutoff = rows[850]["u"]
    before = s.fit(rows, name, cutoff)
    for r in rows:
        if r["mature"] >= cutoff:
            r["y"] ^= 1
            r["market"]["features"] = [float("nan")] * 3
    after = s.fit(rows, name, cutoff)
    assert before["fit_hash"] == after["fit_hash"] and before["scale"] == after["scale"]
    np.testing.assert_array_equal(before["model"].coef_, after["model"].coef_)
    assert before["retained_rolling_rows"] == 504 and before["fit_dates"] == 849


def test_learned_controls_use_interval_but_fixed_majority_still_uses_old_fxi():
    value = paired()
    for i, n in enumerate(s.CONTROLS[:2]):
        assert s.batch_answers([value], n, head(i == 1))[0]["prediction"] == 1
    assert s.batch_answers([value], "MARKET_MAJORITY3")[0]["prediction"] == 0


@pytest.mark.parametrize("name", s.BRANCHES)
def test_all_branches_preserve_unavailable_source_fallback(name):
    assert s.batch_answers([paired(False)], name)[0]["prediction"] == (1 if name == "ALWAYS_UP" else 0)


def test_declared_window_and_control_contract_are_frozen():
    p = {
        "candidates": list(s.CANDIDATES),
        "controls": list(s.CONTROLS),
        "branch_count": 7,
        "preflight_branch_checks": 210,
        "max_development_fits": 24,
        "max_current_fits": 6,
        "reproductions": 1,
        "current_cutoff": "2026-09-16",
        "training_window": "ALL_PAST_MATURE_COMPLETE_MARKET_ROWS",
        "control_input_policy": "R82_LEARNED_CONTROLS_USE_INTERVAL_FIXED_RULES_USE_ORIGINAL",
    }
    s.validate_spec(p, 30)
    for k, v in [("training_window", "1000"), ("control_input_policy", "ORIGINAL_ALL")]:
        with pytest.raises(ValueError):
            s.validate_spec(p | {k: v}, 30)


def test_future_seven_branches_keep_the_r82_control_semantics(parent_ready, monkeypatch):  # noqa: F811
    forecast(parent_ready)
    manifest = {"at": "2026-09-16T06:00:00+08:00", "plan_hash": "EXPANDING_INTERVAL", "model_sha256": "MODEL"}
    bundle = {n: {"CN_EQUITY": head(i == 1)} for names in (s.CANDIDATES, s.CONTROLS[:2]) for i, n in enumerate(names)}
    b.save(s.root() / "result.json", manifest)
    monkeypatch.setattr(s, "models", lambda: (manifest, bundle))
    monkeypatch.setattr(s, "live_market", lambda source: paired())
    assert runtime.tick(s)["verified_forecasts"] == 1
    value = b.read(s.root() / "forward/2026-09-16/001000.json")
    assert (
        value["answers"][s.CONTROLS[0]]["prediction"] == 1 and value["answers"]["MARKET_MAJORITY3"]["prediction"] == 0
    )
    from scripts import direction_1d_sprint_market_expanding_interval as entry

    assert entry.existing.__name__ == "scripts.direction_1d_sprint_market_fxi_interval"
