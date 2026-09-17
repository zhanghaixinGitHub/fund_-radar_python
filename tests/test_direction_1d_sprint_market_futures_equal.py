"""验证固定组合、缺源回退、真实提前预测证据和完整对照，不拟合历史。"""

from copy import deepcopy
from datetime import datetime

import pytest
from app.services import direction_1d_sprint_futures_equal_forward as forward
from app.services import direction_1d_sprint_market_futures_equal as s


def member(i, score):
    return {
        "research_score": score,
        "prediction": int(score >= 0.5),
        "kind": "UNCALIBRATED_UP_SCORE",
        "route": s.MEMBERS[i],
    }


@pytest.mark.parametrize("a,b,score,pred", [(0.9, 0.3, 0.6, 1), (0.2, 0.8, 0.5, 1), (0.1, 0.5, 0.3, 0)])
def test_fixed_average_and_inclusive_tie(a, b, score, pred):
    result = s.combine(member(0, a), member(1, b), {"prediction": 0})
    assert result["research_score"] == pytest.approx(score)
    assert result["prediction"] == pred and result["route"] == s.CANDIDATES[0]


@pytest.mark.parametrize("which", [0, 1])
def test_missing_member_keeps_whole_original_sign_answer(which):
    fallback = {"prediction": 0, "route": "SPX_SIGN_FALLBACK", "kind": "FIXED_RULE"}
    parts = [member(0, 0.8), member(1, 0.6)]
    parts[which] = dict(fallback)
    before = deepcopy([parts, fallback])
    result = s.combine(*parts, fallback)
    assert result == fallback and result is not fallback
    assert [parts, fallback] == before


@pytest.mark.parametrize("bad", [True, float("nan"), float("inf"), -0.1, 1.1, "0.6"])
def test_invalid_score_fails_without_fallback(bad):
    a = member(0, 0.6) | {"research_score": bad}
    with pytest.raises(ValueError, match="MEMBER_SCORE_INVALID"):
        s.combine(a, member(1, 0.6), {"prediction": 1})


def test_inconsistent_direction_rejected():
    with pytest.raises(ValueError, match="MEMBER_DIRECTION_INVALID"):
        s.combine(member(0, 0.7) | {"prediction": 0}, member(1, 0.6), {"prediction": 0})


@pytest.fixture
def actual(monkeypatch):
    source = {"market": {}}
    parent = {
        "code": "001021",
        "family": "A",
        "group": "CN_BOND",
        "t": "2026-09-16",
        "u": "2026-09-17",
        "at": "2026-09-17T08:14:00+08:00",
        "input_hash": s.base.digest(source),
        "answers": {forward.FALLBACK: {"prediction": 0}},
    }
    proofs, records = {}, {}
    for n in (94, 96, 100):
        record = parent | {
            "at": "2026-09-17T08:15:00+08:00",
            "model_hash": str(n),
            "plan_hash": f"p{n}",
            "status": "MODEL_NOT_RELEASED",
            "contract": forward.core.CONTRACT,
            "answers": {s.MEMBERS[0]: member(0, 0.7)} if n == 94 else {s.MEMBERS[1]: member(1, 0.4)},
        }
        records[n] = record
        directory = s.base.ROOT / f"round-{n}"
        proofs[directory / "result.json"] = {
            "at": "2026-09-16T11:00:00+08:00",
            "model_sha256": str(n),
            "plan_hash": f"p{n}",
        }
        proofs[directory / "receipts/2026-09-17/001021.json"] = {
            "forecast_hash": s.base.digest(record),
            "status": "VERIFIED",
            "readback_at": "2026-09-17T08:15:01+08:00",
        }
    proofs[forward.core.root() / "receipts/2026-09-17/001021.json"] = {
        "forecast_hash": s.base.digest(parent),
        "readback_at": "2026-09-17T08:14:01+08:00",
    }
    parts = {n: records[n] for n in (94, 96)}
    manifest = {"at": "2026-09-16T12:00:00+08:00", "model_sha256": "model", "plan_hash": "plan"}
    value = parent | {
        "at": "2026-09-17T08:16:00+08:00",
        "parent_hash": s.base.digest(parent),
        "model_hash": "model",
        "plan_hash": "plan",
        "r100_forecast_hash": s.base.digest(records[100]),
        "member_forecast_hashes": {str(n): s.base.digest(v) for n, v in parts.items()},
        "answers": {s.CANDIDATES[0]: forward.answer(parent, parts)},
        "status": "MODEL_NOT_RELEASED",
        "contract": forward.core.CONTRACT,
    }
    receipt = {"forecast_hash": s.base.digest(value), "status": "VERIFIED", "readback_at": "2026-09-17T08:16:01+08:00"}
    monkeypatch.setattr(s.base, "read", lambda path: proofs[path])
    monkeypatch.setattr(
        forward.core.evidence, "sprint_end", lambda: datetime.fromisoformat("2026-09-17T12:11:38+08:00")
    )
    return [value, receipt, parent, source, manifest, (), records[100], parts], proofs


def test_valid_real_forecast(actual):
    args, _ = actual
    assert forward.validate(*args) == args[0]


@pytest.mark.parametrize("n", [94, 96, 100])
def test_each_member_and_parent_must_finish_readback_first(actual, n):
    args, proofs = actual
    proofs[s.base.ROOT / f"round-{n}/receipts/2026-09-17/001021.json"]["readback_at"] = "2026-09-17T08:17:00+08:00"
    with pytest.raises(ValueError, match="SOURCE_READBACK_LATE"):
        forward.validate(*args)


@pytest.mark.parametrize("field,new", [("input_hash", "changed"), ("u", "2026-09-18"), ("model_hash", "changed")])
def test_changed_member_cannot_be_masked_by_recomputed_local_hashes(actual, field, new):
    args, proofs = actual
    args[-1][94][field] = new
    args[0]["member_forecast_hashes"]["94"] = s.base.digest(args[-1][94])
    args[1]["forecast_hash"] = s.base.digest(args[0])
    proofs[s.base.ROOT / "round-94/receipts/2026-09-17/001021.json"]["forecast_hash"] = s.base.digest(args[-1][94])
    with pytest.raises(ValueError, match="SOURCE_(QUESTION|MODEL)_CHANGED"):
        forward.validate(*args)


def test_late_own_forecast_rejected(actual):
    args, _ = actual
    args[0]["at"] = "2026-09-17T08:30:00+08:00"
    args[1]["readback_at"] = "2026-09-17T08:30:01+08:00"
    args[1]["forecast_hash"] = s.base.digest(args[0])
    with pytest.raises(ValueError, match="FORWARD_LATE"):
        forward.validate(*args)


def test_no_new_parent_skips_without_fabrication(monkeypatch, tmp_path):
    parent = {"code": "001021", "t": "2026-09-16", "u": "2026-09-17"}
    monkeypatch.setattr(s, "models", lambda: ({"plan_hash": "new-plan"}, ()))
    monkeypatch.setattr(
        forward.prior_forward, "context", lambda: ({}, (), {}, {(parent["code"], parent["u"]): parent}, {}, {})
    )
    monkeypatch.setattr(forward.prior_model, "root", lambda: tmp_path)
    loaded = forward.context()
    assert loaded[3] == loaded[4] == loaded[5] == {}


def test_nineteen_branches_and_original_answers_retained(monkeypatch):
    parent = {"u": "2026-09-17", "code": "001021", "answers": {f"core{i}": {"prediction": i % 2} for i in range(8)}}
    records = {
        s.base.ROOT / f"round-{n}/forward/2026-09-17/001021.json": {
            "answers": {f"r{n}-{i}": {"prediction": i % 2} for i in range(2 if n in (92, 94) else 1)}
        }
        for n in (92, 94, 95, 96, 97, 98, 99)
    }
    monkeypatch.setattr(s.base, "read", lambda path: records[path])
    extra, value = {"answers": {"r100": {"prediction": 1}}}, {"answers": {s.CANDIDATES[0]: {"prediction": 0}}}
    merged = forward.combined_answers(parent, extra, value)
    assert len(merged) == 19 and all(merged[k] == v for k, v in parent["answers"].items())
    records[next(iter(records))]["answers"].clear()
    with pytest.raises(ValueError, match="FORWARD_BRANCH_COUNT_CHANGED"):
        forward.combined_answers(parent, extra, value)
