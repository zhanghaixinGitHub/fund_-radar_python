"""医疗行业模型的四维变换及缺失回退，不拟合真实历史。"""

from copy import deepcopy
from datetime import datetime

import numpy as np
import pytest
from app.services import direction_1d_sprint_health_response_forward as f
from app.services import direction_1d_sprint_market_health_response as s
from sklearn.linear_model import LogisticRegression


def test_sign_three_inputs_keeps_intraday_continuous():
    x = [[2.0, -0.1, 0.0, -0.8]]
    assert np.allclose(s.transform(x, [2.0, 2.0, 2.0, 0.4], True), [[1.0, -1.0, 0.0, -2.0]])
    assert np.allclose(s.transform(x, [2.0, 2.0, 2.0, 0.4], False), [[1.0, -0.05, 0.0, -2.0]])


@pytest.mark.parametrize("scale", [[1, 1, 1], [1, 0, 1, 1], [1, 1, 1, float("nan")]])
def test_bad_scale_rejected(scale):
    with pytest.raises(ValueError, match="TRANSFORM_INVALID"):
        s.transform([[1, 2, 3, 4, 5]], scale, False)


def test_missing_extra_keeps_corresponding_parent_answer():
    values = [
        {"available": True, "health_response": {"available": False}},
        {"available": False, "health_response": {"available": True}},
    ]
    fallback = [{"prediction": 1, "route": "RAW_PARENT"}, {"prediction": 0, "route": "SPX_SIGN_FALLBACK"}]
    before = deepcopy(fallback)
    assert s.candidate_answers(values, s.CANDIDATES[0], None, fallback) == before
    assert fallback == before


@pytest.mark.parametrize("basis", [float("nan"), 61.0, True])
def test_bad_intraday_return_not_silently_normalized(monkeypatch, basis):
    monkeypatch.setattr(s.baseline, "market_vector", lambda _: [1, 2, 3])
    with pytest.raises(ValueError, match="INTRADAY_INVALID"):
        s.vector(
            {
                "health_response": {
                    "available": True,
                    "response_signal": basis,
                    "close_location": 0.2,
                    "ic_relative_intraday_pct": 0.1,
                    "ih_relative_intraday_pct": 0.2,
                }
            }
        )


@pytest.fixture
def head():
    m = LogisticRegression(C=0.1, fit_intercept=False, max_iter=1500, random_state=17)
    m.fit(
        np.array(
            [
                [1.0, 1.0, 1.0, 0.1],
                [-1.0, -1.0, -1.0, -0.1],
                [1.0, -1.0, 1.0, 0.2],
                [-1.0, 1.0, -1.0, -0.2],
            ]
        ),
        [1, 0, 1, 0],
    )
    return {
        "model": m,
        "scale": [1.0] * 4,
        "mean": [0.0] * 4,
        "signed": True,
        "fit_dates": 504,
        "fit_end": "2026-09-14",
        "max_mature_date": "2026-09-15",
        "cutoff": "2026-09-16",
        "max_feature_date": "2026-09-10",
        "max_prior_mature": "2026-09-10",
        "latest_context_cutoff": "2026-09-14",
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
        ("mean", [1.0] * 4),
    ],
)
def test_head_changed_recipe_or_maturity_rejected(head, field, value):
    head[field] = value
    with pytest.raises(ValueError):
        s.validate_head(head, s.CANDIDATES[0], "2026-09-16")


@pytest.fixture
def identity_input():
    rows = [{"code": "001021", "y": 1, "market": {"original": 2, "health_response": {"response_signal": 0.1}}}]
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
        rows[0]["market"]["health_response"]["response_signal"] = 0.2
    else:
        proof["new_field"] = True
    with pytest.raises(ValueError, match="QUESTION_IDENTITY_CHANGED"):
        s.verify_identity(rows, proof, expected)


def test_missing_new_feature_falls_back_to_sign_control_not_raw(monkeypatch):
    controls = {n: [{"prediction": int(n == s.CONTROLS[1])}] for n in s.CONTROLS}
    monkeypatch.setattr(s.reference, "batch", lambda *_: controls)
    result = s.batch([{"available": True, "health_response": {"available": False}}], {}, [])
    assert result[s.CANDIDATES[0]] == controls[s.CONTROLS[1]]
    assert result[s.CANDIDATES[0]] != controls[s.CONTROLS[0]]


def test_training_missing_signal_is_zero_but_keeps_original_three_inputs(monkeypatch):
    monkeypatch.setattr(s.baseline, "market_vector", lambda _: [1.0, -1.0, 1.0])
    assert s.vector({"health_response": {"available": False, "response_signal": 0.0}}) == [1.0, -1.0, 1.0, 0.0]
    with pytest.raises(ValueError, match="MISSING_INPUT_NOT_ZERO"):
        s.vector({"health_response": {"available": False, "response_signal": 1.0}})


@pytest.fixture
def future_record(monkeypatch):
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
    extra = {"answers": {"r104": {"prediction": 1}}, "plan_hash": "prior-plan", "health_source_hash": "health-source"}
    history = {
        "at": "2026-09-17T08:15:10+08:00",
        "t": parent["t"],
        "u": parent["u"],
        "points": {},
        "contexts": {},
        "health_source_hash": "health-source",
    }
    answers = {s.CANDIDATES[0]: {"prediction": 1}}
    manifest = {"at": "2026-09-16T13:00:00+08:00", "model_sha256": "model", "plan_hash": "plan"}
    value = parent | {
        "at": "2026-09-17T08:16:00+08:00",
        "parent_hash": s.base.digest(parent),
        "model_hash": "model",
        "plan_hash": "plan",
        "r104_forecast_hash": s.base.digest(extra),
        "response_source_hash": s.base.digest(history),
        "answers": answers,
        "status": "MODEL_NOT_RELEASED",
        "contract": f.core.CONTRACT,
    }
    receipt = {"forecast_hash": s.base.digest(value), "readback_at": "2026-09-17T08:16:01+08:00", "status": "VERIFIED"}
    proofs = {
        f.core.root() / "receipts/2026-09-17/001021.json": {
            "forecast_hash": s.base.digest(parent),
            "readback_at": "2026-09-17T08:14:01+08:00",
        },
        f.prior_model.root() / "receipts/2026-09-17/001021.json": {
            "forecast_hash": s.base.digest(extra),
            "readback_at": "2026-09-17T08:15:01+08:00",
        },
    }
    monkeypatch.setattr(s.base, "read", lambda p: proofs[p])
    original = deepcopy(history)
    monkeypatch.setattr(s.data, "live", lambda *a: original)
    monkeypatch.setattr(s.data, "extend", lambda *a: {})
    monkeypatch.setattr(s, "answers", lambda *a: answers)
    monkeypatch.setattr(f.core.evidence, "sprint_end", lambda: datetime.fromisoformat("2026-09-17T12:11:38+08:00"))
    return [value, receipt, parent, source, manifest, {}, extra, history], proofs


def test_actual_future_requires_both_parent_receipts_and_actual_source(future_record):
    args, _ = future_record
    assert f.validate(*args) == args[0]


def test_new_history_with_recomputed_hash_cannot_replace_actual_source(future_record):
    args, _ = future_record
    args[-1]["points"] = {"changed": 1}
    args[0]["response_source_hash"] = s.base.digest(args[-1])
    args[1]["forecast_hash"] = s.base.digest(args[0])
    with pytest.raises(ValueError, match="ACTUAL_RESPONSE_SOURCE_CHANGED"):
        f.validate(*args)


def test_parent_readback_must_finish_before_new_prediction(future_record):
    args, proofs = future_record
    proofs[f.prior_model.root() / "receipts/2026-09-17/001021.json"]["readback_at"] = "2026-09-17T08:17:00+08:00"
    with pytest.raises(ValueError, match="R104_PARENT_READBACK_LATE_OR_CHANGED"):
        f.validate(*args)


def test_late_source_rejected_even_with_recomputed_hashes(future_record):
    args, _ = future_record
    args[-1]["at"] = "2026-09-17T08:17:00+08:00"
    args[0]["response_source_hash"] = s.base.digest(args[-1])
    args[1]["forecast_hash"] = s.base.digest(args[0])
    with pytest.raises(ValueError, match="HEALTH_SOURCE_CHANGED_OR_LATE"):
        f.validate(*args)


def test_all_twenty_three_branches_preserved(monkeypatch):
    parent = {"u": "2026-09-17", "code": "001021", "answers": {f"core{i}": {"prediction": i % 2} for i in range(8)}}
    records = {
        f.base.ROOT / f"round-{n}/forward/2026-09-17/001021.json": {
            "answers": {f"r{n}-{i}": {"prediction": i % 2} for i in range(2 if n in (92, 94) else 1)}
        }
        for n in (92, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103)
    }
    monkeypatch.setattr(f.base, "read", lambda p: records[p])
    merged = f.combined_answers(
        parent, {"answers": {"r104": {"prediction": 1}}}, {"answers": {"r105": {"prediction": 0}}}
    )
    assert len(merged) == 23 and merged["r105"]["prediction"] == 0 and merged["r92-1"]["prediction"] == 1
