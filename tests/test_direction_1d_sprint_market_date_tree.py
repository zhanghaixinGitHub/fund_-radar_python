"""检验日期汇总的业务权重、训练时间隔离、真实叶节点单位和未来同题分支。"""

import copy

import numpy as np
import pytest
from app.services import direction_1d_sprint as b
from app.services import direction_1d_sprint_market_child_forward_v2 as runtime
from app.services import direction_1d_sprint_market_date_tree as s

from test_direction_1d_sprint_market_only import history, market
from test_direction_1d_sprint_market_only_forward import forecast
from test_direction_1d_sprint_market_only_forward import ready as parent_ready  # noqa: F401


def proposal():
    return {
        "candidates": ["EXPANDED_ROW_ETR3_504", "EXPANDED_DATE_ETR3_504"],
        "controls": [
            "FROZEN_R70_MARKET_LR3",
            "FROZEN_R70_MARKET_SIGN_LR3",
            "MARKET_MAJORITY3",
            "SPX_SIGN",
            "ALWAYS_UP",
        ],
        "branch_count": 7,
        "preflight_branch_checks": 210,
        "max_development_fits": 24,
        "max_current_fits": 6,
        "reproductions": 1,
    }


def test_names_counts_budget_and_no_extra_reproduction():
    p = proposal()
    s.validate_spec(p, 30)
    for key, value in (
        ("controls", p["controls"][:-1]),
        ("branch_count", 6),
        ("preflight_branch_checks", 180),
        ("max_development_fits", 25),
        ("max_current_fits", 7),
        ("reproductions", 2),
    ):
        with pytest.raises(ValueError, match="DECLARED_BRANCHES_OR_BUDGET_CHANGED"):
            s.validate_spec(p | {key: value}, 30)


def family_rows():
    rows = []
    for r in history()[:4]:
        rows.extend(
            [
                r | {"family": "A", "code": "A1", "y": 0},
                r | {"family": "A", "code": "A2", "y": 0},
                r | {"family": "B", "code": "B", "y": 1},
            ]
        )
    return rows


def test_product_share_duplicates_do_not_change_daily_up_fraction():
    rows = family_rows()
    x, y, weights, days = s.training_arrays(rows, "FAMILY_WEIGHTED_DATE")
    np.testing.assert_allclose(y, 0.5)
    assert len(x) == len(days) == 4 and len(set(days)) == 4
    without_extra_share = [r for r in rows if r["code"] != "A2"]
    other = s.training_arrays(without_extra_share, "FAMILY_WEIGHTED_DATE")
    np.testing.assert_array_equal(other[0], x)
    np.testing.assert_allclose(other[1], y)
    np.testing.assert_allclose(other[2], weights)


def test_daily_squared_loss_preserves_row_loss_up_to_fixed_label_variance():
    rows = family_rows()
    _, y, w, days = s.training_arrays(rows, "FUND_ROW")
    _, dy, dw, unique_days = s.training_arrays(rows, "FAMILY_WEIGHTED_DATE")
    positions = {day: i for i, day in enumerate(unique_days)}
    indices = [positions[day] for day in days]
    constant = np.sum(w * (y - dy[indices]) ** 2)
    for prediction in ([0.1, 0.7, 0.4, 0.9], [0.5] * 4, [0.9, 0.2, 0.8, 0.4]):
        dp = np.asarray(prediction)
        original_loss = np.sum(w * (y - dp[indices]) ** 2)
        daily_loss = np.sum(dw * (dy - dp) ** 2)
        assert original_loss == pytest.approx(daily_loss + constant)


def test_different_market_inputs_on_same_date_prevent_aggregation():
    rows = copy.deepcopy(family_rows())
    rows[1]["market"] = copy.deepcopy(rows[1]["market"])
    rows[1]["market"]["features"][0] += 1
    with pytest.raises(ValueError, match="DIFFERENT_INPUTS_SAME_DATE"):
        s.training_arrays(rows, "FAMILY_WEIGHTED_DATE")


def test_mixed_groups_and_nonbinary_labels_fail_closed():
    rows = family_rows()
    with pytest.raises(ValueError, match="MIXED_ASSET_GROUPS"):
        s.training_arrays(rows + [rows[0] | {"group": "CN_BOND"}], "FAMILY_WEIGHTED_DATE")
    with pytest.raises(ValueError, match="NONBINARY_LABEL"):
        s.training_arrays([rows[0] | {"y": 0.5}], "FAMILY_WEIGHTED_DATE")


@pytest.mark.parametrize("name", s.CANDIDATES)
def test_immature_rows_do_not_change_training_arrays_or_predictions(monkeypatch, name):
    monkeypatch.setattr(s, "active", lambda: None)
    rows = history()
    cutoff = rows[180]["u"]
    before = s.fit(rows, name, cutoff)
    for row in rows:
        if row["mature"] >= cutoff:
            row["y"] ^= 1
            row["market"] = market((float("nan"), 0, 0))
    after = s.fit(rows, name, cutoff)
    assert before["fit_hash"] == after["fit_hash"]
    assert before["training_arrays_hash"] == after["training_arrays_hash"]
    np.testing.assert_array_equal(
        s.tree_scores(before["model"], [[0, 0, 0], [1, -1, 1]]), s.tree_scores(after["model"], [[0, 0, 0], [1, -1, 1]])
    )
    assert before["max_mature_date"] < cutoff
    assert before["model"].get_params()["min_samples_leaf"] == 60
    if name == s.CANDIDATES[1]:
        assert before["fit_dates"] == before["estimator_training_units"]
        for tree in before["model"].estimators_:
            leaf = tree.tree_.children_left == -1
            assert min(tree.tree_.n_node_samples[leaf]) >= 60


@pytest.mark.parametrize("name", s.BRANCHES)
def test_missing_market_uses_same_spx_fallback_without_model(name):
    result = s.batch_answers([market((-1, 0, 0), False)], name)
    assert result[0]["prediction"] == (1 if name == "ALWAYS_UP" else 0)


def test_controls_delegate_to_exact_original_branch(monkeypatch):
    calls = []
    monkeypatch.setattr(s.baseline, "batch_answers", lambda values, name, head: calls.append(name) or [])
    for name in s.CONTROLS:
        s.batch_answers([], name)
    assert calls == list(s.baseline.CANDIDATES) + list(s.CONTROLS[2:])


class Fixed:
    n_features_in_ = 3

    @property
    def estimators_(self):
        return [self]

    def predict(self, x):
        return np.repeat(0.6, len(x))

    def predict_proba(self, x):
        return np.tile([0.4, 0.6], (len(x), 1))


def models():
    return {
        **{n: {"CN_EQUITY": {"model": Fixed(), "training_unit": s.TRAINING_UNITS[n]}} for n in s.CANDIDATES},
        **{
            n: {"CN_EQUITY": {"model": Fixed(), "mean": [0.0] * 3, "scale": [1.0] * 3, "signed": n == s.CONTROLS[1]}}
            for n in s.CONTROLS[:2]
        },
    }


def test_seven_future_branches_use_independent_verified_market_parent(parent_ready, monkeypatch):  # noqa: F811
    forecast(parent_ready)
    manifest = {"at": "2026-09-16T03:00:00+08:00", "plan_hash": "DATE_TREE", "model_sha256": "MODEL"}
    b.save(s.root() / "result.json", manifest)
    monkeypatch.setattr(s, "models", lambda: (manifest, models()))
    monkeypatch.setattr(s, "live_market", lambda source: market())
    assert runtime.tick(s)["verified_forecasts"] == 1
    value = b.read(s.root() / "forward/2026-09-16/001000.json")
    assert set(value["answers"]) == set(s.BRANCHES)
    assert len(value["market"]["features"]) == 3
    assert not (b.ROOT / "forward").exists()


def test_latest_entry_retains_previous_completed_round():
    from scripts import direction_1d_sprint_market_date_tree as entry

    assert entry.existing.__name__ == "scripts.direction_1d_sprint_market_sign_gap"
