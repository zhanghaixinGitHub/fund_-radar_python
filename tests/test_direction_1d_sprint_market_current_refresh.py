"""近期真实标签的成熟边界、旧控制隔离、一次拟合及未来回读。"""

from copy import deepcopy

import numpy as np
import pytest
from app.services import direction_1d_sprint as b
from app.services import direction_1d_sprint_market_child_forward_v2 as runtime
from app.services import direction_1d_sprint_market_current_refresh as s

from test_direction_1d_sprint_market_only import history, market
from test_direction_1d_sprint_market_only_forward import forecast
from test_direction_1d_sprint_market_only_forward import ready as parent_ready  # noqa: F401


def source():
    received = "2026-09-16T04:25:20+08:00"
    snapshot = {
        "at": "2026-09-16T04:25:00+08:00",
        "expires_at": "2027-01-01T00:00:00+08:00",
        "errors": [],
        "funds": [
            {
                "fund_code": "001000",
                "family": "F",
                "group": "CN_EQUITY",
                "rows": [
                    {"date": "2026-09-11", "ann_date": "2026-09-12", "nav": "1.00", "received_at": received},
                    {"date": "2026-09-14", "ann_date": "2026-09-15", "nav": "1.01", "received_at": received},
                    {"date": "2026-09-15", "ann_date": "2026-09-16", "nav": "1.02", "received_at": received},
                ],
            }
        ],
    }
    plan = {
        "at": "2026-09-16T04:40:00+08:00",
        "new_cutoff": "2026-09-16",
        "nav_source_ref": "FROZEN",
        "nav_snapshot_hash": b.digest(snapshot),
        "candidate_target_dates": ["2026-09-14", "2026-09-15"],
    }
    successor = {"2026-09-11": "2026-09-14", "2026-09-14": "2026-09-15", "2026-09-15": "2026-09-16"}
    values = {("2026-09-11", "2026-09-14"): market(), ("2026-09-14", "2026-09-15"): market()}
    return snapshot, [{"u": "2026-09-11"}], plan, successor, values


def test_actual_receipt_after_batch_start_before_freeze_is_valid_but_immature_target_is_excluded():
    args = source()
    del args[-1][("2026-09-14", "2026-09-15")]
    added, excluded = s.rebuild_added(*args)
    assert len(added) == 1 and added[0]["u"] == "2026-09-14" and added[0]["y"] == 1
    assert added[0]["mature"] == "2026-09-15" and added[0]["label_source_kind"] == "ACTUAL_RESPONSE_OBSERVATION"
    assert not added[0]["fund_nav_used_as_predictor"]
    assert excluded == [
        {"code": "001000", "u": "2026-09-15", "reason": "LABEL_NOT_MATURE_BEFORE_CUTOFF", "mature": "2026-09-16"}
    ]


@pytest.mark.parametrize(
    "field,value",
    [
        ("received_at", "2026-09-16T04:40:00+08:00"),
        ("received_at", "2026-09-16T04:30:00"),
        ("nav", "NaN"),
        ("nav", "0"),
        ("ann_date", "2026-09-17"),
    ],
)
def test_unreceived_future_or_bad_nav_cannot_enter_current_training(field, value):
    args = source()
    args[0]["funds"][0]["rows"][1][field] = value
    with pytest.raises(ValueError):
        s.rebuild_added(*args)


def test_late_announcement_excludes_previously_mature_label_and_flat_remains_non_up():
    args = source()
    args[0]["funds"][0]["rows"][1]["ann_date"] = "2026-09-16"
    assert not s.rebuild_added(*args)[0]
    args = source()
    args[0]["funds"][0]["rows"][1]["nav"] = "1.00"
    row = s.rebuild_added(*args)[0][0]
    assert row["y"] == 0 and row["actual_direction"] == "FLAT"


def test_duplicate_nav_dates_are_rejected():
    args = source()
    args[0]["funds"][0]["rows"].append(deepcopy(args[0]["funds"][0]["rows"][0]))
    with pytest.raises(ValueError, match="DUPLICATE_NAV_DATE"):
        s.rebuild_added(*args)


@pytest.mark.parametrize("name", s.CANDIDATES)
def test_unchanged_recipe_ignores_new_immature_labels(monkeypatch, name):
    monkeypatch.setattr(s.baseline, "active", lambda: None)
    rows = history()
    cutoff = rows[180]["u"]
    before = s.fit(rows, name, cutoff)
    for r in rows:
        if r["mature"] >= cutoff:
            r["y"] ^= 1
            r["market"] = market((float("nan"), 0, 0))
    after = s.fit(rows, name, cutoff)
    assert before["fit_hash"] == after["fit_hash"] and before["scale"] == after["scale"]
    np.testing.assert_array_equal(before["model"].coef_, after["model"].coef_)


@pytest.mark.parametrize("name", s.BRANCHES)
def test_missing_source_fallback_is_kept_for_all_seven_branches(name):
    result = s.batch_answers([market((-1, 0, 0), False)], name)
    assert result[0]["prediction"] == (1 if name == "ALWAYS_UP" else 0)


def test_freeze_rejects_development_fit_or_cutoff_change():
    p = {
        "candidates": list(s.CANDIDATES),
        "controls": list(s.CONTROLS),
        "branch_count": 7,
        "preflight_branch_checks": 210,
        "max_development_fits": 0,
        "max_current_fits": 6,
        "reproductions": 1,
        "current_cutoff": "2026-09-16",
        "old_current_cutoff": "2026-09-15",
    }
    s.validate_spec(p, 30)
    for k, v in (
        ("max_development_fits", 24),
        ("max_current_fits", 12),
        ("current_cutoff", "2026-09-17"),
        ("controls", list(s.CONTROLS[:-1])),
    ):
        with pytest.raises(ValueError, match="DECLARED_BRANCHES_OR_BUDGET_CHANGED"):
            s.validate_spec(p | {k: v}, 30)


def test_interrupted_checkpoint_cannot_silently_refit(tmp_path, monkeypatch):
    monkeypatch.setattr(b, "ROOT", tmp_path)
    b.save(s.root() / "checkpoints/current-CN_EQUITY-test.attempt.json", {})
    with pytest.raises(ValueError, match="FIT_ALREADY_ATTEMPTED"):
        s.fit_checkpoint([], "test", "2026-09-16", "current-CN_EQUITY", "PLAN")


class Fixed:
    def predict_proba(self, x):
        return np.tile([0.4, 0.6], (len(x), 1))


def test_future_refresh_preserves_parent_and_old_controls(parent_ready, monkeypatch):  # noqa: F811
    forecast(parent_ready)
    manifest = {"at": "2026-09-16T05:00:00+08:00", "plan_hash": "CURRENT_REFRESH", "model_sha256": "MODEL"}
    bundle = {
        n: {"CN_EQUITY": {"model": Fixed(), "mean": [0.0] * 3, "scale": [1.0] * 3, "signed": "SIGN" in n}}
        for n in s.CANDIDATES + s.CONTROLS[:2]
    }
    b.save(s.root() / "result.json", manifest)
    monkeypatch.setattr(s, "models", lambda: (manifest, bundle))
    monkeypatch.setattr(s, "live_market", lambda value: market())
    assert runtime.tick(s)["verified_forecasts"] == 1
    value = b.read(s.root() / "forward/2026-09-16/001000.json")
    assert set(value["answers"]) == set(s.BRANCHES)
    assert value["plan_hash"] == "CURRENT_REFRESH" and not (b.ROOT / "forward").exists()


def test_entry_keeps_r78_completed_branch():
    from scripts import direction_1d_sprint_market_current_refresh as entry

    assert entry.existing.__name__ == "scripts.direction_1d_sprint_market_odd_features"
