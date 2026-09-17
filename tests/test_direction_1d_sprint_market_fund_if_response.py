"""逐基金IF反应模型：时间、变换、已冻IF回退、双父原始来源和26分支验证。"""

from copy import deepcopy
from datetime import datetime

import numpy as np
import pytest
from app.services import direction_1d_sprint_fund_if_response_forward as forward
from app.services import direction_1d_sprint_market_fund_if_response as s
from sklearn.linear_model import LogisticRegression


def test_public_signs_and_four_continuous_inputs_scaled_without_centering():
    x = [[2.0, -1.0, 0.0, 2.0, -0.5, 4.0, -0.2]]
    assert np.allclose(
        s.transform(x, [2.0, 2.0, 2.0, 2.0, 0.5, 2.0, 0.1], True), [[1.0, -1.0, 0.0, 1.0, -1.0, 2.0, -2.0]]
    )


@pytest.mark.parametrize("scale", [[1] * 6, [1] * 6 + [0], [1] * 6 + [float("nan")]])
def test_invalid_seven_dimensional_scale_rejected(scale):
    with pytest.raises(ValueError):
        s.transform([[1] * 7], scale, True)


@pytest.mark.parametrize("value", [float("nan"), 61, True])
def test_response_bound_is_enforced(monkeypatch, value):
    monkeypatch.setattr(s.baseline, "market_vector", lambda _: [1, 2, 3])
    market = {
        "futures_intraday": {"available": True, "intraday_return_pct": 1, "close_location": 0.5},
        "fund_if_response": {"available": True, "response_return": value, "response_location": 0},
    }
    with pytest.raises(ValueError, match="RESPONSE_INPUT_INVALID"):
        s.vector(market)


def test_cold_context_keeps_if_inputs_and_zeroes_only_response(monkeypatch):
    monkeypatch.setattr(s.baseline, "market_vector", lambda _: [1, 2, 3])
    market = {
        "futures_intraday": {"available": True, "intraday_return_pct": 1, "close_location": 0.5},
        "fund_if_response": {"available": False, "response_return": 0, "response_location": 0},
    }
    assert s.vector(market) == [1, 2, 3, 1, 0.5, 0, 0]
    market["fund_if_response"]["response_location"] = 0.2
    with pytest.raises(ValueError, match="MISSING_RESPONSE_NOT_ZERO"):
        s.vector(market)


@pytest.fixture
def head():
    m = LogisticRegression(C=0.1, fit_intercept=False, max_iter=1500, random_state=17)
    m.fit(np.array([[1] * 7, [-1] * 7, [1, -1, 1, -1, 1, -1, 1], [-1, 1, -1, 1, -1, 1, -1]], dtype=float), [1, 0, 1, 0])
    return {
        "model": m,
        "scale": [1.0] * 7,
        "mean": [0.0] * 7,
        "signed": True,
        "fit_dates": 504,
        "fit_end": "2026-09-14",
        "max_mature_date": "2026-09-15",
        "cutoff": "2026-09-16",
        "max_feature_date": "2026-09-11",
        "latest_context_cutoff": "2026-09-14",
        "max_prior_mature": "2026-09-13",
    }


def test_head_includes_strictly_earlier_fund_context(head):
    assert s.validate_head(head, s.CANDIDATES[0], "2026-09-16") is head["model"]


@pytest.mark.parametrize(
    "field,value",
    [
        ("cutoff", "2026-09-15"),
        ("max_prior_mature", "2026-09-14"),
        ("fit_dates", 503),
        ("signed", False),
        ("mean", [1.0] * 7),
    ],
)
def test_changed_recipe_and_unmatured_context_fail(head, field, value):
    head[field] = value
    with pytest.raises(ValueError):
        s.validate_head(head, s.CANDIDATES[0], "2026-09-16")


def test_missing_response_uses_frozen_if_head_even_when_sign_disagrees(monkeypatch):
    controls = {n: [{"prediction": 0, "route": n}] for n in s.CONTROLS}
    expected = [{"prediction": 1, "route": s.fallback_model.CANDIDATES[0]}]
    monkeypatch.setattr(s.reference, "batch", lambda *_: controls)
    observed = []

    def prior_batch(values, heads, control_heads):
        observed.append(heads)
        return {s.fallback_model.CANDIDATES[0]: expected}

    monkeypatch.setattr(s.fallback_model, "batch", prior_batch)
    result = s.batch([{"available": True, "fund_if_response": {"available": False}}], {}, [], "frozen-if-head")
    assert result[s.CANDIDATES[0]] == expected and observed == [{s.fallback_model.CANDIDATES[0]: "frozen-if-head"}]
    assert result[s.CONTROLS[1]][0]["prediction"] == 0


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
        "parent_plan_hash": "parent-plan",
        "parent_source_hash": "parent-source",
        "previous_history_hash": "history",
        "previous_history_at": "2026-09-16T11:01:00+08:00",
        "snapshot": {"points": {}},
        "contexts": {},
    }
    answers = {name: {"prediction": 1} for name in s.CANDIDATES}
    manifest = {"at": "2026-09-16T09:30:00+08:00", "model_sha256": "model", "plan_hash": "plan"}
    value = parent | {
        "at": "2026-09-17T08:16:00+08:00",
        "parent_hash": s.base.digest(parent),
        "model_hash": "model",
        "plan_hash": "plan",
        "r107_forecast_hash": s.base.digest(extra),
        "fund_if_source_hash": s.base.digest(history),
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
    proofs[s.base.ROOT / "round-96/forward/2026-09-17/001021.json"] = {
        "plan_hash": "parent-plan",
        "intraday_source_hash": "parent-source",
    }
    monkeypatch.setattr(s.base, "read", lambda p: proofs[p])
    monkeypatch.setattr(s, "answers", lambda *_: answers)
    monkeypatch.setattr(s.data, "extend", lambda *_: {})
    monkeypatch.setattr(
        forward.core.evidence, "sprint_end", lambda: datetime.fromisoformat("2026-09-17T12:11:38+08:00")
    )
    expected_history = deepcopy(history)
    monkeypatch.setattr(s.data, "live", lambda *_: deepcopy(expected_history))
    return [value, receipt, parent, source, manifest, {}, extra, history], proofs


def test_actual_source_and_both_parent_receipts_precede_prediction(actual_forecast):
    args, _ = actual_forecast
    assert forward.validate(*args) == args[0]


def test_late_new_option_snapshot_rejected_even_if_hashes_recomputed(actual_forecast):
    args, _ = actual_forecast
    args[-1]["at"] = "2026-09-17T08:17:00+08:00"
    args[0]["fund_if_source_hash"] = s.base.digest(args[-1])
    args[1]["forecast_hash"] = s.base.digest(args[0])
    with pytest.raises(ValueError, match="OPTION_SOURCE_CHANGED_OR_LATE"):
        forward.validate(*args)


def test_r107_parent_must_finish_readback_first(actual_forecast):
    args, proofs = actual_forecast
    proofs[forward.prior_model.root() / "receipts/2026-09-17/001021.json"]["readback_at"] = "2026-09-17T08:17:00+08:00"
    with pytest.raises(ValueError, match="R107_PARENT_READBACK_LATE_OR_CHANGED"):
        forward.validate(*args)


@pytest.fixture
def identity_input():
    rows = [
        {
            "code": "001021",
            "y": 1,
            "market": {"original": 2, "fund_if_response": {"log_put_call_close_volume": 0.1}},
        }
    ]
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
        rows[0]["market"]["fund_if_response"]["log_put_call_close_volume"] = 0.2
    else:
        proof["new_field"] = True
    with pytest.raises(ValueError, match="QUESTION_IDENTITY_CHANGED"):
        s.verify_identity(rows, proof, expected)


def test_old_historical_snapshot_cannot_stand_in_for_actual_capture(actual_forecast):
    args, _ = actual_forecast
    args[-1]["at"] = "2026-09-16T09:13:00+08:00"
    args[0]["fund_if_source_hash"] = s.base.digest(args[-1])
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
    args[-1]["parent_plan_hash"] = "different-plan"
    args[0]["fund_if_source_hash"] = s.base.digest(args[-1])
    args[1]["forecast_hash"] = s.base.digest(args[0])
    with pytest.raises(ValueError, match="OPTION_SOURCE_SCOPE_CHANGED"):
        forward.validate(*args)


def test_all_twenty_six_branches_retained_after_new_parent_level(monkeypatch):
    parent = {"u": "2026-09-17", "code": "001021", "answers": {f"core{i}": {"prediction": i % 2} for i in range(8)}}
    records = {
        forward.base.ROOT / f"round-{n}/forward/2026-09-17/001021.json": {
            "answers": {f"r{n}-{i}": {"prediction": i % 2} for i in range(2 if n in (92, 94) else 1)}
        }
        for n in (92, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106)
    }
    monkeypatch.setattr(forward.base, "read", lambda path: records[path])
    merged = forward.combined_answers(
        parent, {"answers": {"r107": {"prediction": 1}}}, {"answers": {"r108": {"prediction": 0}}}
    )
    assert len(merged) == 26 and all(merged[k] == v for k, v in parent["answers"].items())
    assert merged["r92-1"] == {"prediction": 1} and merged["r108"] == {"prediction": 0}
    records[next(iter(records))]["answers"].clear()
    with pytest.raises(ValueError, match="FORWARD_BRANCH_COUNT_CHANGED"):
        forward.combined_answers(
            parent, {"answers": {"r107": {"prediction": 1}}}, {"answers": {"r108": {"prediction": 0}}}
        )


def test_actual_raw_derived_features_cannot_change_with_rehashed_envelope(actual_forecast):
    args, _ = actual_forecast
    args[-1]["snapshot"]["points"] = {"2026-09-16": {"tampered": True}}
    args[0]["fund_if_source_hash"] = s.base.digest(args[-1])
    args[1]["forecast_hash"] = s.base.digest(args[0])
    with pytest.raises(ValueError, match="ACTUAL_FUND_IF_SOURCE_CHANGED"):
        forward.validate(*args)
