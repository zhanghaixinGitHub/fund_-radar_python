"""约束损失的数值正确性、单调性、债券复用与真实未来分支。"""

import numpy as np
import pytest
from app.services import direction_1d_sprint as b
from app.services import direction_1d_sprint_market_child_forward_v2 as runtime
from app.services import direction_1d_sprint_market_nonnegative as s
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression

from test_direction_1d_sprint_market_only import history, market
from test_direction_1d_sprint_market_only_forward import forecast
from test_direction_1d_sprint_market_only_forward import ready as parent_ready  # noqa: F401


def problem():
    rng = np.random.default_rng(811)
    x = rng.normal(size=(500, 3))
    y = (x[:, 0] - 0.5 * x[:, 1] + rng.normal(size=500) > 0.1).astype(float)
    weights = rng.uniform(0.3, 2, len(x))
    return x, y, weights


def test_analytic_gradient_and_penalty_match_original_sklearn_recipe():
    x, y, w = problem()
    coef = np.array([0.3, -0.2, 0.1])
    loss, gradient = s.objective(coef, x, y, w)
    finite = []
    for i in range(3):
        delta = np.eye(3)[i] * 1e-6
        finite.append((s.objective(coef + delta, x, y, w)[0] - s.objective(coef - delta, x, y, w)[0]) / 2e-6)
    np.testing.assert_allclose(gradient, finite, atol=1e-9)
    original = LogisticRegression(C=0.1, fit_intercept=False, max_iter=5000, tol=1e-10).fit(x, y, sample_weight=w)
    solved = minimize(
        s.objective, np.zeros(3), args=(x, y, w), method="L-BFGS-B", jac=True, options={"ftol": 1e-14, "gtol": 1e-9}
    )
    assert solved.success
    np.testing.assert_allclose(solved.x, original.coef_[0], atol=1e-6)


def test_nonnegative_solution_satisfies_kkt_and_monotone_prediction():
    x, y, w = problem()
    coef, receipt = s.solve(x, y, w)
    assert min(coef) >= 0 and receipt["projected_kkt_residual"] <= 1e-6
    assert coef[1] == 0
    model = s.NonnegativeLogit(coef)
    before = model.predict_proba(x)[:, 1]
    for column in range(3):
        larger = x.copy()
        larger[:, column] += 2
        assert np.all(model.predict_proba(larger)[:, 1] >= before)


def test_immature_labels_do_not_change_model(monkeypatch):
    monkeypatch.setattr(s, "active", lambda: None)
    rows = history()
    cutoff = rows[180]["u"]
    old = s.fit(rows, s.CANDIDATES[0], cutoff)
    for r in rows:
        if r["mature"] >= cutoff:
            r["y"] ^= 1
            r["market"] = market((float("nan"), 0, 0))
    new = s.fit(rows, s.CANDIDATES[0], cutoff)
    assert old["fit_hash"] == new["fit_hash"] and old["scale"] == new["scale"]
    np.testing.assert_array_equal(old["model"].coef_, new["model"].coef_)


def test_bond_candidate_reuses_existing_head_and_never_fits(monkeypatch):
    original = {"model": s.NonnegativeLogit([0.1, 0.2, 0.3]), "mean": [0.0] * 3, "scale": [1.0] * 3, "signed": False}
    monkeypatch.setattr(s, "frozen_head", lambda *args: original)
    monkeypatch.setattr(s, "fit_checkpoint", lambda *args: pytest.fail("bond must not fit"))
    result = s.candidate_head([], s.CANDIDATES[0], "2026-09-16", "current", "CN_BOND", "PLAN")
    assert result["model"] is original["model"] and result["constraint"] == "BOND_REUSED_RAW_CONTROL"
    with pytest.raises(ValueError):
        s.fit([{"group": "CN_BOND"}], s.CANDIDATES[0], "2026-09-16")


@pytest.mark.parametrize("name", s.BRANCHES)
def test_missing_source_keeps_original_spx_fallback(name):
    result = s.batch_answers([market((-1, 0, 0), False)], name)
    assert result[0]["prediction"] == (1 if name == "ALWAYS_UP" else 0)


def test_declared_budget_and_bond_policy_verified():
    p = {
        "candidates": list(s.CANDIDATES),
        "controls": list(s.CONTROLS),
        "branch_count": 6,
        "preflight_branch_checks": 180,
        "max_development_fits": 8,
        "max_current_fits": 2,
        "reproductions": 1,
        "current_cutoff": "2026-09-16",
        "constrained_groups": list(s.CONSTRAINED_GROUPS),
        "bond_policy": "REUSE_SAME_CUTOFF_RAW_CONTROL",
    }
    s.validate_spec(p, 30)
    for k, v in [("max_development_fits", 12), ("bond_policy", "FIT"), ("constrained_groups", ["CN_BOND"])]:
        with pytest.raises(ValueError):
            s.validate_spec(p | {k: v}, 30)


def test_future_six_branches_are_independent_of_ixic(parent_ready, monkeypatch):  # noqa: F811
    forecast(parent_ready)
    manifest = {"at": "2026-09-16T05:25:00+08:00", "plan_hash": "NONNEG", "model_sha256": "MODEL"}
    basehead = {"model": s.NonnegativeLogit([0.2, 0.1, 0.4]), "mean": [0.0] * 3, "scale": [1.0] * 3, "signed": False}
    bundle = {s.CANDIDATES[0]: {"CN_EQUITY": basehead | {"constraint": "NONNEGATIVE"}}}
    bundle.update({n: {"CN_EQUITY": basehead | {"signed": i == 1}} for i, n in enumerate(s.CONTROLS[:2])})
    b.save(s.root() / "result.json", manifest)
    monkeypatch.setattr(s, "models", lambda: (manifest, bundle))
    monkeypatch.setattr(s, "live_market", lambda source: market())
    assert runtime.tick(s)["verified_forecasts"] == 1
    assert set(b.read(s.root() / "forward/2026-09-16/001000.json")["answers"]) == set(s.BRANCHES)
    from scripts import direction_1d_sprint_market_nonnegative as entry

    assert entry.existing.__name__ == "scripts.direction_1d_sprint_market_nasdaq_relative"
