"""用数值梯度和时间切分验证新的训练目标；不拟合真实历史数据。"""

import numpy as np
import pytest
from app.services import direction_1d_sprint_market_block_newton as s
from scipy.optimize import check_grad
from scipy.special import expit


def test_analytic_gradient_matches_independent_finite_difference():
    rng = np.random.default_rng(17)
    x, y, w = rng.normal(size=(80, 3)), rng.integers(0, 2, 80), rng.uniform(0.2, 2, 80)
    blocks, coef = np.repeat(np.arange(8), 10), np.array([0.12, -0.32, 0.24])
    error = check_grad(
        lambda c: s.loss_gradient(c, x, y, w, blocks)[0], lambda c: s.loss_gradient(c, x, y, w, blocks)[1], coef
    )
    assert error < 1e-6


def test_identical_blocks_reduce_to_weighted_logistic():
    x0 = np.array([[1.0, 2.0, -0.2], [-0.5, -1.0, 1.0]])
    y0, w0, coef = np.array([1, 0]), np.array([2.0, 1.0]), np.array([0.2, 0.1, -0.1])
    x, y, w = np.tile(x0, (8, 1)), np.tile(y0, 8), np.tile(w0, 8)
    loss, gradient, _, adversary = s.loss_gradient(coef, x, y, w, np.repeat(np.arange(8), 2))
    z = x @ coef
    expected = np.average(np.logaddexp(0, z) - y * z, weights=w) + 10 / w.sum() * (coef @ coef) / 2
    expected_gradient = ((expit(z) - y) * w) @ x / w.sum() + 10 / w.sum() * coef
    assert np.isclose(loss, expected, atol=1e-12)
    assert np.allclose(gradient, expected_gradient, atol=1e-12)
    assert np.allclose(adversary, np.ones(8) / 8, atol=1e-12)


def test_bad_block_gets_more_weight_without_future_labels():
    x = np.ones((80, 3))
    y = np.ones(80)
    y[:10] = 0
    _, _, means, weights = s.loss_gradient(np.ones(3) * 0.2, x, y, np.ones(80), np.repeat(np.arange(8), 10))
    assert means[0] > max(means[1:]) and weights[0] > max(weights[1:])
    assert np.isclose(weights.sum(), 1.0)


def test_time_blocks_keep_same_day_together():
    rows = [{"u": f"{i:04d}"} for i in range(504) for _ in range(3)]
    blocks = s.time_blocks(rows)
    assert list(np.bincount(blocks)) == [189] * 8
    assert np.array_equal(blocks.reshape(504, 3)[:, 0], np.repeat(np.arange(8), 63))


def test_wrong_number_of_days_rejected():
    with pytest.raises(ValueError, match="DATES_NOT_504"):
        s.time_blocks([{"u": str(i)} for i in range(503)])


@pytest.mark.parametrize("bad", [float("nan"), 0.0, -1.0])
def test_bad_weights_rejected(bad):
    weights = np.ones(8)
    weights[0] = bad
    with pytest.raises(ValueError, match="OBJECTIVE_VALUES"):
        s.loss_gradient(np.zeros(3), np.ones((8, 3)), np.ones(8), weights, np.arange(8))


def test_missing_block_rejected():
    with pytest.raises(ValueError, match="OBJECTIVE_VALUES"):
        s.loss_gradient(np.zeros(3), np.ones((8, 3)), np.ones(8), np.ones(8), np.zeros(8, dtype=int))


def test_missing_market_uses_spx_and_keeps_row(monkeypatch):
    value = {"available": False, "features": [-0.2, 0.0, 0.0]}
    monkeypatch.setattr(s.baseline, "market_vector", lambda v: v["features"])
    monkeypatch.setattr(s.original, "batch_answers", lambda *_: [])
    result = s.candidate_answers([value], s.CANDIDATES[0], None)
    assert result == [{"prediction": 0, "kind": "FIXED_DIRECTION", "route": "SPX_SIGN_FALLBACK"}]


def test_exact_hessian_matches_finite_difference_of_gradient():
    rng = np.random.default_rng(92)
    x, y, w = rng.normal(size=(160, 3)), rng.integers(0, 2, 160), rng.uniform(0.1, 2, 160)
    blocks, coef = np.repeat(np.arange(8), 20), np.array([0.3, -0.2, 0.1])
    step = 1e-5
    numerical = np.column_stack(
        [
            (
                s.loss_gradient(coef + np.eye(3)[j] * step, x, y, w, blocks)[1]
                - s.loss_gradient(coef - np.eye(3)[j] * step, x, y, w, blocks)[1]
            )
            / (2 * step)
            for j in range(3)
        ]
    )
    actual = s.hessian(coef, x, y, w, blocks)
    assert np.allclose(actual, numerical, atol=1e-8, rtol=1e-7)
    assert np.linalg.eigvalsh(actual).min() >= 10 / w.sum() - 1e-10


@pytest.mark.parametrize("mode", ["correlated", "imbalanced", "different_scales", "signed"])
def test_newton_converges_on_difficult_synthetic_inputs(mode):
    rng = np.random.default_rng(9201)
    x = rng.normal(size=(504, 3))
    y = (x[:, 0] + rng.normal(size=504) > 0).astype(int)
    if mode == "correlated":
        x[:, 1] = x[:, 0] + 1e-9 * x[:, 1]
    elif mode == "imbalanced":
        y[:450] = 1
    elif mode == "different_scales":
        x[:, 1] *= 100
        x[:, 2] *= 0.001
    else:
        x = np.sign(x)
    weights, blocks = np.ones(504), np.repeat(np.arange(8), 63)
    solved = s.solve_newton(x, y, weights, blocks)
    assert solved.success and solved.nit < 100
    assert np.max(np.abs(s.loss_gradient(solved.x, x, y, weights, blocks)[1])) < 1e-8
    losses = [r["loss"] for r in solved.trace]
    assert all(b <= a + 1e-13 for a, b in zip(losses, losses[1:], strict=False))
