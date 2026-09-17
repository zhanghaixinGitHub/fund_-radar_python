"""期现价差增量模型的四维变换及缺失回退，不拟合真实历史。"""

from copy import deepcopy
from datetime import datetime

import numpy as np
import pytest
from app.services import direction_1d_sprint_futures_basis_forward as forward
from app.services import direction_1d_sprint_market_futures_basis as s
from sklearn.linear_model import LogisticRegression


def test_sign_three_inputs_keeps_basis_continuous():
    x = [[2.0, -0.1, 0.0, -0.8]]
    assert np.allclose(s.transform(x, [2.0, 2.0, 2.0, 0.4], True), [[1.0, -1.0, 0.0, -2.0]])
    assert np.allclose(s.transform(x, [2.0, 2.0, 2.0, 0.4], False), [[1.0, -0.05, 0.0, -2.0]])


@pytest.mark.parametrize("scale", [[1, 1, 1], [1, 0, 1, 1], [1, 1, 1, float("nan")]])
def test_bad_scale_rejected(scale):
    with pytest.raises(ValueError, match="TRANSFORM_INVALID"):
        s.transform([[1, 2, 3, 4]], scale, False)


def test_missing_extra_keeps_corresponding_parent_answer():
    values = [
        {"available": True, "futures_basis": {"available": False}},
        {"available": False, "futures_basis": {"available": True}},
    ]
    fallback = [{"prediction": 1, "route": "RAW_PARENT"}, {"prediction": 0, "route": "SPX_SIGN_FALLBACK"}]
    before = deepcopy(fallback)
    assert s.candidate_answers(values, s.CANDIDATES[0], None, fallback) == before
    assert fallback == before


@pytest.mark.parametrize("basis", [float("nan"), 21.0, True])
def test_bad_basis_not_silently_normalized(monkeypatch, basis):
    monkeypatch.setattr(s.baseline, "market_vector", lambda _: [1, 2, 3])
    with pytest.raises(ValueError, match="BASIS_MISSING_OR_INVALID"):
        s.vector({"futures_basis": {"available": True, "basis_pct": basis}})


@pytest.fixture
def head():
    m = LogisticRegression(C=0.1, fit_intercept=False, max_iter=1500, random_state=17)
    m.fit(
        np.array([[1.0, 1.0, 1.0, 0.1], [-1.0, -1.0, -1.0, -0.1], [1.0, -1.0, 1.0, 0.2], [-1.0, 1.0, -1.0, -0.2]]),
        [1, 0, 1, 0],
    )
    return {
        "model": m,
        "scale": [1.0] * 4,
        "mean": [0.0] * 4,
        "signed": False,
        "fit_dates": 504,
        "fit_end": "2026-09-14",
        "max_mature_date": "2026-09-15",
        "cutoff": "2026-09-16",
        "max_basis_date": "2026-09-10",
    }


def test_head_timing_and_schema(head):
    assert s.validate_head(head, s.CANDIDATES[0], "2026-09-16") is head["model"]


@pytest.mark.parametrize(
    "field,value",
    [
        ("cutoff", "2026-09-15"),
        ("max_basis_date", "2026-09-14"),
        ("fit_dates", 503),
        ("signed", True),
        ("mean", [1.0] * 4),
    ],
)
def test_head_changed_recipe_or_maturity_rejected(head, field, value):
    head[field] = value
    with pytest.raises(ValueError):
        s.validate_head(head, s.CANDIDATES[0], "2026-09-16")


@pytest.fixture
def actual_forecast(monkeypatch):
    source = {"market": {}}
    parent = {
        "code": "001021",
        "family": "A",
        "group": "CN_BOND",
        "t": "2026-09-16",
        "u": "2026-09-17",
        "at": "2026-09-17T08:14:00+08:00",
        "input_hash": s.base.digest(source),
    }
    extra = {"at": "2026-09-17T08:15:00+08:00"}
    history = {"at": "2026-09-16T09:13:00+08:00", "snapshot": {"rows": {}}}
    answers = {name: {"prediction": 1} for name in s.CANDIDATES}
    manifest = {"at": "2026-09-16T09:30:00+08:00", "model_sha256": "model", "plan_hash": "plan"}
    value = parent | {
        "at": "2026-09-17T08:16:00+08:00",
        "parent_hash": s.base.digest(parent),
        "model_hash": "model",
        "plan_hash": "plan",
        "r92_forecast_hash": s.base.digest(extra),
        "futures_history_hash": s.base.digest(history),
        "answers": answers,
        "status": "MODEL_NOT_RELEASED",
        "contract": forward.core.CONTRACT,
    }
    receipt = {"forecast_hash": s.base.digest(value), "status": "VERIFIED", "readback_at": "2026-09-17T08:16:01+08:00"}
    proofs = {
        forward.core.root() / "receipts/2026-09-17/001021.json": {
            "forecast_hash": s.base.digest(parent),
            "readback_at": "2026-09-17T08:14:01+08:00",
        },
        forward.prior_model.root() / "receipts/2026-09-17/001021.json": {
            "forecast_hash": s.base.digest(extra),
            "readback_at": "2026-09-17T08:15:01+08:00",
        },
    }
    monkeypatch.setattr(s.base, "read", lambda p: proofs[p])
    monkeypatch.setattr(s, "answers", lambda *_: answers)
    monkeypatch.setattr(s.data, "extend", lambda *_: {})
    monkeypatch.setattr(
        forward.core.evidence, "sprint_end", lambda: datetime.fromisoformat("2026-09-17T12:11:38+08:00")
    )
    return [value, receipt, parent, source, manifest, {}, extra, history], proofs


def test_actual_source_and_both_parent_receipts_precede_prediction(actual_forecast):
    args, _ = actual_forecast
    assert forward.validate(*args) == args[0]


def test_late_new_futures_snapshot_rejected_even_if_hashes_recomputed(actual_forecast):
    args, _ = actual_forecast
    args[-1]["at"] = "2026-09-17T08:17:00+08:00"
    args[0]["futures_history_hash"] = s.base.digest(args[-1])
    args[1]["forecast_hash"] = s.base.digest(args[0])
    with pytest.raises(ValueError, match="FUTURES_SOURCE_CHANGED_OR_LATE"):
        forward.validate(*args)


def test_r92_parent_must_finish_readback_first(actual_forecast):
    args, proofs = actual_forecast
    proofs[forward.prior_model.root() / "receipts/2026-09-17/001021.json"]["readback_at"] = "2026-09-17T08:17:00+08:00"
    with pytest.raises(ValueError, match="R92_PARENT_READBACK_LATE_OR_CHANGED"):
        forward.validate(*args)
