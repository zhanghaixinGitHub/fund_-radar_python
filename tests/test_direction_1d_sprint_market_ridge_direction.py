"""验证带权平方目标、预测分数语义、冻结完整性及真实提前预测边界。"""

from copy import deepcopy
from datetime import date, timedelta

import joblib
import numpy as np
import pytest
from app.services import direction_1d_sprint as b
from app.services import direction_1d_sprint_market_child_forward_v2 as runtime
from app.services import direction_1d_sprint_market_ridge_direction as s
from sklearn.linear_model import Ridge

from test_direction_1d_sprint_market_fxi_interval import head as lr_head
from test_direction_1d_sprint_market_fxi_interval import paired
from test_direction_1d_sprint_market_only_forward import forecast
from test_direction_1d_sprint_market_only_forward import ready as parent_ready  # noqa: F401


def head(signed=False):
    return {
        "coef": [0.2, 0.4, 0.1],
        "mean": [0.0] * 3,
        "scale": [1.0] * 3,
        "signed": signed,
        "alpha": 10.0,
        "target": "2*y-1",
        "threshold": 0.0,
        "normal_equation_residual": 0.0,
        "fit_dates": 504,
        "fit_end": "2026-09-14",
        "max_mature_date": "2026-09-15",
        "cutoff": "2026-09-16",
    }


@pytest.mark.parametrize("signed", [False, True])
def test_closed_form_matches_independent_weighted_ridge_and_uses_direction_target(monkeypatch, signed):
    rng = np.random.default_rng(17088)
    x = rng.normal(size=(1008, 3))
    rows = []
    for i, point in enumerate(x):
        day = str(date(2021, 1, 1) + timedelta(days=i // 2))
        rows.append(
            {
                "code": str(i % 2),
                "family": str(i % 2),
                "u": day,
                "mature": day,
                "y": int(point[0] + point[1] > 0),
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
    out = s.fit(rows, s.CANDIDATES[int(signed)], "2024-01-01")
    weights = s.regression.weights(rows)
    transformed = np.where(x >= 0, 1.0, -1.0) if signed else x / out["scale"]
    y = np.asarray([r["y"] for r in rows])
    reference = Ridge(alpha=10, fit_intercept=False, solver="cholesky").fit(
        transformed, 2 * y - 1, sample_weight=weights
    )
    np.testing.assert_allclose(out["coef"], reference.coef_, rtol=1e-12, atol=1e-12)
    assert out["normal_equation_residual"] < 1e-10
    assert out["weight_hash"] == b.digest(weights.tolist()) and out["fit_hash"] == b.digest(rows)
    assert out["mean"] == [0.0] * 3
    rows[0]["mature"] = "2024-01-01"
    with pytest.raises(ValueError, match="TRAINING_MATURITY_INVALID"):
        s.fit(rows, s.CANDIDATES[int(signed)], "2024-01-01")


@pytest.mark.parametrize(
    "fault",
    [
        {"coef": [float("nan"), 0, 0]},
        {"coef": [1, 2]},
        {"scale": [0, 1, 1]},
        {"alpha": 1},
        {"threshold": 0.1},
        {"target": "return"},
        {"signed": True},
        {"normal_equation_residual": 1e-5},
        {"max_mature_date": "2026-09-16"},
    ],
)
def test_invalid_or_different_recipe_head_rejected(fault):
    with pytest.raises(ValueError):
        s.candidate_answers([paired()], s.CANDIDATES[0], head() | fault)


def test_margin_can_exceed_probability_bounds_and_keeps_zero_threshold():
    v = paired()
    x = np.asarray(v["features"])
    answers = s.candidate_answers([v], s.CANDIDATES[0], head() | {"coef": (x * 10).tolist()})
    assert answers[0]["research_score"] > 1 and answers[0]["prediction"] == 1
    assert answers[0]["kind"] == "UNCALIBRATED_DIRECTION_MARGIN"
    assert s.candidate_answers([v], s.CANDIDATES[0], head() | {"coef": [0, 0, 0]})[0]["prediction"] == 1
    assert s.candidate_answers([v], s.CANDIDATES[0], head() | {"coef": (-x).tolist()})[0]["prediction"] == 0


def test_seven_branches_preserve_fallback_and_frozen_controls_without_mutation():
    values = [paired(), paired(False)]
    before = deepcopy(values)
    heads = {n: head(bool(i)) for i, n in enumerate(s.CANDIDATES)}
    controls = [lr_head(False), lr_head(True)]
    out = s.batch(values, heads, controls)
    assert len(out) == 7 and values == before
    old = s.previous.batch(values, controls)
    assert all(out[n] == old[n] for n in s.CONTROLS)
    assert all(out[n][1]["route"] == "SPX_SIGN_FALLBACK" for n in s.CANDIDATES)
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


def test_future_seven_branches_save_once_and_reject_late_receipt(parent_ready, monkeypatch):  # noqa: F811
    forecast(parent_ready)
    manifest = {"at": "2026-09-16T06:50:00+08:00", "plan_hash": "RIDGE", "model_sha256": "RIDGE_HASH"}
    bundle = {n: {"CN_EQUITY": head(bool(i))} for i, n in enumerate(s.CANDIDATES)}
    bundle.update({n: {"CN_EQUITY": lr_head(bool(i))} for i, n in enumerate(s.CONTROLS[:2])})
    b.save(s.root() / "result.json", manifest)
    monkeypatch.setattr(s, "models", lambda: (manifest, bundle))
    monkeypatch.setattr(s, "live_market", lambda source: paired())
    assert runtime.tick(s)["verified_forecasts"] == 1
    path = s.root() / "forward/2026-09-16/001000.json"
    assert len(b.read(path)["answers"]) == 7
    frozen = path.read_bytes()
    runtime.tick(s)
    assert path.read_bytes() == frozen
    receipt = s.root() / "receipts/2026-09-16/001000.json"
    b.save(receipt, b.read(receipt) | {"readback_at": "2026-09-16T08:30:00+08:00"}, replace=True)
    assert runtime.report(s)["verified_forecasts"] == 0
    from scripts import direction_1d_sprint_market_ridge_direction as entry

    assert entry.existing.__name__ == "scripts.direction_1d_sprint_market_equal_mean"
