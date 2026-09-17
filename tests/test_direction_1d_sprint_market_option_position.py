"""期权成交持仓增量模型的五维变换及缺失回退，不拟合真实历史。"""

from copy import deepcopy
from datetime import datetime

import numpy as np
import pytest
from app.services import direction_1d_sprint_market_option_position as s
from app.services import direction_1d_sprint_option_position_forward as forward
from sklearn.linear_model import LogisticRegression


def test_sign_three_inputs_keeps_option_ratios_continuous():
    x = [[2.0, -0.1, 0.0, -0.8, 0.5]]
    assert np.allclose(s.transform(x, [2.0, 2.0, 2.0, 0.4, 0.5], True), [[1.0, -1.0, 0.0, -2.0, 1.0]])
    assert np.allclose(s.transform(x, [2.0, 2.0, 2.0, 0.4, 0.5], False), [[1.0, -0.05, 0.0, -2.0, 1.0]])


@pytest.mark.parametrize("scale", [[1, 1, 1], [1, 0, 1, 1], [1, 1, 1, float("nan")]])
def test_bad_scale_rejected(scale):
    with pytest.raises(ValueError, match="TRANSFORM_INVALID"):
        s.transform([[1, 2, 3, 4, 5]], scale, False)


def test_missing_extra_keeps_corresponding_parent_answer():
    values = [
        {"available": True, "option_position": {"available": False}},
        {"available": False, "option_position": {"available": True}},
    ]
    fallback = [{"prediction": 1, "route": "RAW_PARENT"}, {"prediction": 0, "route": "SPX_SIGN_FALLBACK"}]
    before = deepcopy(fallback)
    assert s.candidate_answers(values, s.CANDIDATES[0], None, fallback) == before
    assert fallback == before


@pytest.mark.parametrize("basis", [float("nan"), 21.0, True])
def test_bad_option_ratio_not_silently_normalized(monkeypatch, basis):
    monkeypatch.setattr(s.baseline, "market_vector", lambda _: [1, 2, 3])
    with pytest.raises(ValueError, match="OPTION_INVALID"):
        s.vector({"option_position": {"available": True, "log_put_call_volume": basis, "log_put_call_oi": 0.2}})


@pytest.fixture
def head():
    m = LogisticRegression(C=0.1, fit_intercept=False, max_iter=1500, random_state=17)
    m.fit(
        np.array(
            [
                [1.0, 1.0, 1.0, 0.1, 0.2],
                [-1.0, -1.0, -1.0, -0.1, -0.2],
                [1.0, -1.0, 1.0, 0.2, 0.1],
                [-1.0, 1.0, -1.0, -0.2, -0.1],
            ]
        ),
        [1, 0, 1, 0],
    )
    return {
        "model": m,
        "scale": [1.0] * 5,
        "mean": [0.0] * 5,
        "signed": True,
        "fit_dates": 504,
        "fit_end": "2026-09-14",
        "max_mature_date": "2026-09-15",
        "cutoff": "2026-09-16",
        "max_feature_date": "2026-09-10",
    }


def test_head_timing_and_schema(head):
    assert s.validate_head(head, s.CANDIDATES[0], "2026-09-16") is head["model"]


@pytest.mark.parametrize(
    "field,value",
    [
        ("cutoff", "2026-09-15"),
        ("max_feature_date", "2026-09-14"),
        ("fit_dates", 503),
        ("signed", False),
        ("mean", [1.0] * 5),
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
    extra = {"at": "2026-09-17T08:15:00+08:00", "plan_hash": "parent-plan", "basis_source_hash": "parent-source"}
    history = {
        "at": "2026-09-17T08:15:10+08:00",
        "t": "2026-09-16",
        "u": "2026-09-17",
        "plan_hash": "plan",
        "snapshot": {"points": {}},
    }
    answers = {name: {"prediction": 1} for name in s.CANDIDATES}
    manifest = {"at": "2026-09-16T09:30:00+08:00", "model_sha256": "model", "plan_hash": "plan"}
    value = parent | {
        "at": "2026-09-17T08:16:00+08:00",
        "parent_hash": s.base.digest(parent),
        "model_hash": "model",
        "plan_hash": "plan",
        "r96_forecast_hash": s.base.digest(extra),
        "option_source_hash": s.base.digest(history),
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


def test_late_new_option_snapshot_rejected_even_if_hashes_recomputed(actual_forecast):
    args, _ = actual_forecast
    args[-1]["at"] = "2026-09-17T08:17:00+08:00"
    args[0]["option_source_hash"] = s.base.digest(args[-1])
    args[1]["forecast_hash"] = s.base.digest(args[0])
    with pytest.raises(ValueError, match="OPTION_SOURCE_CHANGED_OR_LATE"):
        forward.validate(*args)


def test_r96_parent_must_finish_readback_first(actual_forecast):
    args, proofs = actual_forecast
    proofs[forward.prior_model.root() / "receipts/2026-09-17/001021.json"]["readback_at"] = "2026-09-17T08:17:00+08:00"
    with pytest.raises(ValueError, match="R96_PARENT_READBACK_LATE_OR_CHANGED"):
        forward.validate(*args)


@pytest.fixture
def identity_input():
    rows = [{"code": "001021", "y": 1, "market": {"original": 2, "option_position": {"log_put_call_volume": 0.1}}}]
    proof = {"at": "2026-09-16T09:00:00+08:00", "fresh_rows_hash": "source", "merged_rows": 1}
    return rows, proof, s.question_identity(rows, proof)


def test_only_verification_time_can_change(identity_input):
    rows, proof, expected = identity_input
    proof["at"] = "2026-09-16T09:30:00+08:00"
    assert s.verify_identity(rows, proof, expected) == expected


@pytest.mark.parametrize(
    "kind", ["source_hash", "source_count", "label", "original_feature", "basis", "extra_proof_key"]
)
def test_actual_question_or_source_changes_rejected(identity_input, kind):
    rows, proof, expected = identity_input
    if kind == "source_hash":
        proof["fresh_rows_hash"] = "changed"
    elif kind == "source_count":
        proof["merged_rows"] = 2
    elif kind == "label":
        rows[0]["y"] = 0
    elif kind == "original_feature":
        rows[0]["market"]["original"] = 3
    elif kind == "basis":
        rows[0]["market"]["option_position"]["log_put_call_volume"] = 0.2
    else:
        proof["new_field"] = True
    with pytest.raises(ValueError, match="QUESTION_IDENTITY_CHANGED"):
        s.verify_identity(rows, proof, expected)


def test_missing_new_feature_falls_back_to_sign_control_not_raw(monkeypatch):
    controls = {n: [{"prediction": int(n == s.CONTROLS[1])}] for n in s.CONTROLS}
    monkeypatch.setattr(s.reference, "batch", lambda *_: controls)
    result = s.batch([{"available": True, "option_position": {"available": False}}], {}, [])
    assert result[s.CANDIDATES[0]] == controls[s.CONTROLS[1]]
    assert result[s.CANDIDATES[0]] != controls[s.CONTROLS[0]]


def test_old_historical_snapshot_cannot_stand_in_for_actual_capture(actual_forecast):
    args, _ = actual_forecast
    args[-1]["at"] = "2026-09-16T09:13:00+08:00"
    args[0]["option_source_hash"] = s.base.digest(args[-1])
    args[1]["forecast_hash"] = s.base.digest(args[0])
    with pytest.raises(ValueError, match="OPTION_SOURCE_NOT_ACTUALLY_CAPTURED_IN_WINDOW"):
        forward.validate(*args)


def test_no_prediction_parent_yet_skips_without_fabrication(monkeypatch, tmp_path):
    parent = {"code": "001021", "t": "2026-09-16", "u": "2026-09-17"}
    monkeypatch.setattr(s, "models", lambda: ({"plan_hash": "new-plan"}, {}))
    monkeypatch.setattr(
        forward.prior_forward, "context", lambda: ({}, {}, {}, {(parent["code"], parent["u"]): parent}, {}, {})
    )
    monkeypatch.setattr(forward.prior_model, "root", lambda: tmp_path)
    result = forward.context()
    assert result[3] == result[4] == result[5] == {}


def test_option_source_must_match_new_model_plan(actual_forecast):
    args, _ = actual_forecast
    args[-1]["plan_hash"] = "different-plan"
    args[0]["option_source_hash"] = s.base.digest(args[-1])
    args[1]["forecast_hash"] = s.base.digest(args[0])
    with pytest.raises(ValueError, match="OPTION_SOURCE_SCOPE_CHANGED"):
        forward.validate(*args)
