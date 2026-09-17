"""验证平方损失汇总的数学等价、核模型分数和真实提前预测边界。"""

from copy import deepcopy
from datetime import date, timedelta

import joblib
import numpy as np
import pytest
from app.services import direction_1d_sprint as b
from app.services import direction_1d_sprint_market_child_forward_v2 as runtime
from app.services import direction_1d_sprint_market_kernel_direction as s
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics.pairwise import rbf_kernel

from test_direction_1d_sprint_market_fxi_interval import head as lr_head
from test_direction_1d_sprint_market_fxi_interval import paired
from test_direction_1d_sprint_market_only_forward import forecast
from test_direction_1d_sprint_market_only_forward import ready as parent_ready  # noqa: F401


def head():
    return {
        "centers": [[0.0, 0.0, 0.0]],
        "dual": [0.2],
        "aggregate_weights": [1.0],
        "aggregate_targets": [0.5],
        "mean": [0.0] * 3,
        "scale": [1.0] * 3,
        "signed": False,
        "alpha": 10.0,
        "gamma": 1 / 3,
        "target": "2*y-1",
        "threshold": 0.0,
        "normal_equation_residual": 0.0,
        "fit_dates": 504,
        "fit_end": "2026-09-14",
        "max_mature_date": "2026-09-15",
        "cutoff": "2026-09-16",
    }


def test_aggregation_matches_independent_uncompressed_kernel_ridge():
    # 重复向量上保留相反标签和非均匀权重，避免只检验相同标签的平凡情况。
    rng = np.random.default_rng(17090)
    x = np.repeat(rng.normal(size=(23, 3)), 4, axis=0)
    y = rng.integers(0, 2, size=len(x))
    weights = rng.uniform(0.1, 3, size=len(x))
    centers, dual, totals, targets, residual = s.solve_kernel(x, y, weights)
    reference = KernelRidge(alpha=10, gamma=1 / 3, kernel="rbf").fit(x, 2 * y - 1, sample_weight=weights)
    exam = rng.normal(size=(37, 3))
    np.testing.assert_allclose(
        rbf_kernel(exam, centers, gamma=1 / 3) @ dual, reference.predict(exam), rtol=1e-11, atol=1e-12
    )
    assert len(centers) == 23 and residual < 1e-10
    assert np.sum(totals) == pytest.approx(np.sum(weights))
    # 任意函数取值下，聚合前后的平方目标差为常数，与函数本身无关。
    gaps = []
    for c in (-0.4, 0.2, 1.3):
        gaps.append(float(np.sum(weights * (2 * y - 1 - c) ** 2) - np.sum(totals * (targets - c) ** 2)))
    np.testing.assert_allclose(gaps, [gaps[0]] * 3, rtol=1e-12)


def test_fit_uses_only_504_mature_days_and_preserves_weights(monkeypatch):
    rng = np.random.default_rng(17)
    x = np.repeat(rng.normal(size=(504, 3)), 2, axis=0)
    rows = []
    for i, point in enumerate(x):
        day = str(date(2021, 1, 1) + timedelta(days=i // 2))
        rows.append(
            {
                "code": str(i % 2),
                "family": str(i % 2),
                "u": day,
                "mature": day,
                "y": int(point[0] + (i % 2) * 0.3 > 0),
                "market": {
                    "features": point.tolist(),
                    "available": True,
                    "etf_available": True,
                    "cnya_available": True,
                },
            }
        )
    monkeypatch.setattr(s.original, "training_rows", lambda values, cutoff: values)
    monkeypatch.setattr(s, "active", lambda: None)
    out = s.fit(rows, s.CANDIDATES[0], "2024-01-01")
    s.validate_head(out, s.CANDIDATES[0], "2024-01-01")
    weights = s.regression.weights(rows)
    assert len(out["centers"]) == 504 and out["fit_rows"] == 1008
    assert out["weight_hash"] == b.digest(weights.tolist()) and out["fit_hash"] == b.digest(rows)
    rows[0]["mature"] = "2024-01-01"
    with pytest.raises(ValueError, match="TRAINING_MATURITY_INVALID"):
        s.fit(rows, s.CANDIDATES[0], "2024-01-01")


@pytest.mark.parametrize(
    "fault",
    [
        {"dual": [float("nan")]},
        {"centers": [[1, 2]]},
        {"scale": [0, 1, 1]},
        {"alpha": 1},
        {"gamma": 0.5},
        {"threshold": 0.1},
        {"target": "return"},
        {"signed": True},
        {"normal_equation_residual": 1e-5},
        {"max_mature_date": "2026-09-16"},
        {"aggregate_weights": [0]},
        {"aggregate_targets": [2]},
    ],
)
def test_invalid_or_different_recipe_head_rejected(fault):
    with pytest.raises(ValueError):
        s.candidate_answers([paired()], s.CANDIDATES[0], head() | fault)


def test_margin_is_not_probability_and_keeps_zero_threshold():
    v = paired()
    for dual, expected in [(100.0, 1), (0.0, 1), (-1.0, 0)]:
        out = s.candidate_answers([v], s.CANDIDATES[0], head() | {"dual": [dual]})[0]
        assert out["prediction"] == expected and out["kind"] == "UNCALIBRATED_DIRECTION_MARGIN"
        if dual == 100:
            assert out["research_score"] > 1


def test_six_branches_preserve_fallback_controls_without_mutation():
    values = [paired(), paired(False)]
    before = deepcopy(values)
    heads = {s.CANDIDATES[0]: head()}
    controls = [lr_head(False), lr_head(True)]
    out = s.batch(values, heads, controls)
    assert len(out) == 6 and values == before
    old = s.reference.batch(values, controls)
    assert all(out[n] == old[n] for n in s.CONTROLS)
    assert out[s.CANDIDATES[0]][1]["route"] == "SPX_SIGN_FALLBACK"
    assert s.candidate_answers([paired(False)], s.CANDIDATES[0], None)[0]["prediction"] == 0


def test_modified_checkpoint_is_not_loaded_or_refit(parent_ready, monkeypatch):  # noqa: F811
    path = s.root() / "checkpoints" / f"2026-09-16-CN_EQUITY-{s.CANDIDATES[0]}.joblib"
    b.save(
        path.with_suffix(".json"),
        {"plan_hash": "P", "name": s.CANDIDATES[0], "cutoff": "2026-09-16", "group": "CN_EQUITY", "sha256": "wrong"},
    )
    joblib.dump(head(), path)
    monkeypatch.setattr(s, "fit", lambda *args: pytest.fail("must not refit"))
    with pytest.raises(ValueError, match="CHECKPOINT_CHANGED"):
        s.checkpoint([], s.CANDIDATES[0], "2026-09-16", "CN_EQUITY", "P")


def test_future_six_branches_save_once_and_reject_late_receipt(parent_ready, monkeypatch):  # noqa: F811
    forecast(parent_ready)
    manifest = {"at": "2026-09-16T06:50:00+08:00", "plan_hash": "RIDGE", "model_sha256": "RIDGE_HASH"}
    bundle = {n: {"CN_EQUITY": head()} for i, n in enumerate(s.CANDIDATES)}
    bundle.update({n: {"CN_EQUITY": lr_head(bool(i))} for i, n in enumerate(s.CONTROLS[:2])})
    b.save(s.root() / "result.json", manifest)
    monkeypatch.setattr(s, "models", lambda: (manifest, bundle))
    monkeypatch.setattr(s, "live_market", lambda source: paired())
    assert runtime.tick(s)["verified_forecasts"] == 1
    path = s.root() / "forward/2026-09-16/001000.json"
    assert len(b.read(path)["answers"]) == 6
    frozen = path.read_bytes()
    runtime.tick(s)
    assert path.read_bytes() == frozen
    receipt = s.root() / "receipts/2026-09-16/001000.json"
    b.save(receipt, b.read(receipt) | {"readback_at": "2026-09-16T08:30:00+08:00"}, replace=True)
    assert runtime.report(s)["verified_forecasts"] == 0
    from scripts import direction_1d_sprint_market_kernel_direction as entry

    assert entry.existing.__name__ == "scripts.direction_1d_sprint_market_fund_bias"
