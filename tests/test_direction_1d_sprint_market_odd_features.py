"""完整奇次基函数、无均值变换、成熟隔离与七个未来同题分支。"""

import numpy as np
import pytest
from app.services import direction_1d_sprint as b
from app.services import direction_1d_sprint_market_child_forward_v2 as runtime
from app.services import direction_1d_sprint_market_odd_features as s

from test_direction_1d_sprint_market_only import history, market
from test_direction_1d_sprint_market_only_forward import forecast
from test_direction_1d_sprint_market_only_forward import ready as parent_ready  # noqa: F401


def test_complete_degree_one_and_three_basis_in_declared_order():
    x = np.arctanh([[0.2, -0.3, 0.4]])
    expected = [0.2, -0.3, 0.4, 0.008, -0.027, 0.064, -0.012, 0.016, 0.018, 0.036, 0.032, -0.048, -0.024]
    z = s.basis(x, [1, 1, 1], 13)
    np.testing.assert_allclose(z[0], expected, atol=1e-15)
    np.testing.assert_allclose(s.basis(-x, [1, 1, 1], 13), -z, atol=1e-15)
    np.testing.assert_array_equal(s.basis(x, [1, 1, 1], 3), z[:, :3])
    np.testing.assert_array_equal(s.basis(np.zeros((1, 3)), [1, 1, 1], 13), np.zeros((1, 13)))


@pytest.mark.parametrize(
    "width,scale,x",
    [
        (12, [1, 1, 1], [[0, 0, 0]]),
        (3, [1, 0, 1], [[0, 0, 0]]),
        (13, [1, 1, 1], [[float("nan"), 0, 0]]),
        (13, [1, 1], [[0, 0, 0]]),
    ],
)
def test_bad_basis_or_raw_scale_fails_closed(width, scale, x):
    with pytest.raises(ValueError, match="BASIS_INVALID"):
        s.basis(x, scale, width)


def test_mean_subtraction_is_rejected():
    h = {"basis_width": 3, "raw_scale": [1, 1, 1], "basis_scale": [1, 1, 1], "mean": [0.1, 0, 0]}
    with pytest.raises(ValueError, match="SCALE_INVALID"):
        s.transformed([[0, 0, 0]], h)


@pytest.mark.parametrize("name", s.CANDIDATES)
def test_immature_labels_cannot_change_any_scale_or_coefficient(monkeypatch, name):
    monkeypatch.setattr(s, "active", lambda: None)
    rows = history()
    cutoff = rows[180]["u"]
    before = s.fit(rows, name, cutoff)
    for r in rows:
        if r["mature"] >= cutoff:
            r["y"] ^= 1
            r["market"] = market((float("nan"), 0, 0))
    after = s.fit(rows, name, cutoff)
    for k in ("fit_hash", "weight_hash", "raw_scale", "basis_scale"):
        assert before[k] == after[k]
    np.testing.assert_array_equal(before["model"].coef_, after["model"].coef_)
    assert before["model"].fit_intercept is False and before["max_mature_date"] < cutoff
    x = np.array([[2.0, -0.3, 4.0], [1.0, 1.0, -1.0], [0.0, 0.0, 0.0]])
    pos = before["model"].predict_proba(s.transformed(x, before))[:, 1]
    neg = before["model"].predict_proba(s.transformed(-x, before))[:, 1]
    np.testing.assert_allclose(pos + neg, 1, atol=1e-14)
    assert pos[-1] == 0.5


def test_pair_uses_same_samples_weights_raw_scale_and_shared_three_basis_scales(monkeypatch):
    monkeypatch.setattr(s, "active", lambda: None)
    rows = history()
    cutoff = rows[180]["u"]
    a, z = [s.fit(rows, n, cutoff) for n in s.CANDIDATES]
    for key in ("fit_hash", "weight_hash", "raw_scale"):
        assert a[key] == z[key]
    np.testing.assert_allclose(a["basis_scale"], z["basis_scale"][:3], atol=1e-15)
    assert a["model"].n_features_in_ == 3 and z["model"].n_features_in_ == 13


@pytest.mark.parametrize("name", s.BRANCHES)
def test_all_branches_keep_same_missing_source_fallback(name):
    result = s.batch_answers([market((-1, 0, 0), False)], name)
    assert result[0]["prediction"] == (1 if name == "ALWAYS_UP" else 0)


def test_declared_names_and_budget_are_verified_before_freeze():
    p = {
        "candidates": list(s.CANDIDATES),
        "controls": list(s.CONTROLS),
        "branch_count": 7,
        "preflight_branch_checks": 210,
        "max_development_fits": 24,
        "max_current_fits": 6,
        "reproductions": 1,
    }
    s.validate_spec(p, 30)
    for key, value in (
        ("candidates", list(s.CANDIDATES[:1])),
        ("controls", list(s.CONTROLS[:-1])),
        ("branch_count", 6),
        ("max_development_fits", 48),
        ("reproductions", 2),
    ):
        with pytest.raises(ValueError, match="DECLARED_BRANCHES_OR_BUDGET_CHANGED"):
            s.validate_spec(p | {key: value}, 30)


class Fixed:
    fit_intercept = False

    def __init__(self, width):
        self.n_features_in_ = width

    def predict_proba(self, x):
        return np.tile([0.4, 0.6], (len(x), 1))


def models():
    output = {}
    for n, width in s.BASIS_WIDTHS.items():
        output[n] = {
            "CN_EQUITY": {
                "model": Fixed(width),
                "basis_width": width,
                "raw_scale": [1.0] * 3,
                "basis_scale": [1.0] * width,
                "mean": [0.0] * width,
            }
        }
    for n in s.CONTROLS[:2]:
        output[n] = {
            "CN_EQUITY": {"model": Fixed(3), "mean": [0.0] * 3, "scale": [1.0] * 3, "signed": n == s.CONTROLS[1]}
        }
    return output


def test_future_seven_branches_keep_the_independent_market_parent(parent_ready, monkeypatch):  # noqa: F811
    forecast(parent_ready)
    manifest = {"at": "2026-09-16T03:00:00+08:00", "plan_hash": "ODD_BASIS", "model_sha256": "MODEL"}
    b.save(s.root() / "result.json", manifest)
    monkeypatch.setattr(s, "models", lambda: (manifest, models()))
    monkeypatch.setattr(s, "live_market", lambda source: market())
    assert runtime.tick(s)["verified_forecasts"] == 1
    value = b.read(s.root() / "forward/2026-09-16/001000.json")
    assert set(value["answers"]) == set(s.BRANCHES)
    assert not (b.ROOT / "forward").exists()


def test_entry_keeps_the_completed_odd_tree_branch():
    from scripts import direction_1d_sprint_market_odd_features as entry

    assert entry.existing.__name__ == "scripts.direction_1d_sprint_market_odd_tree"
