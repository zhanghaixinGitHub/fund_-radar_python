"""未来答案必须同时绑定实际来源、父答案、运行计划和事前时间；全程使用测试替身。"""

from copy import deepcopy
from datetime import datetime

import pytest
from app.services import direction_1d_sprint_market_stock_moneyflow as s
from app.services import direction_1d_sprint_stock_moneyflow_forward as f


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
    extra = {"answers": {"r116": {"prediction": 1}}}
    history = {"at": "2026-09-17T08:15:10+08:00", "t": parent["t"], "u": parent["u"], "points": {}}
    answers = {n: {"prediction": 1} for n in s.CANDIDATES}
    manifest = {"at": "2026-09-16T13:00:00+08:00", "model_sha256": "model", "plan_hash": "plan"}
    value = parent | {
        "at": "2026-09-17T08:16:00+08:00",
        "parent_hash": s.base.digest(parent),
        "model_hash": "model",
        "plan_hash": "plan",
        "r116_forecast_hash": s.base.digest(extra),
        "stock_moneyflow_source_hash": s.base.digest(history),
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
    monkeypatch.setattr(f.runtime, "validate_forecast_binding", lambda value: None)
    original = deepcopy(history)
    monkeypatch.setattr(f.live, "load_live", lambda *a: original)
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
    args[0]["stock_moneyflow_source_hash"] = s.base.digest(args[-1])
    args[1]["forecast_hash"] = s.base.digest(args[0])
    with pytest.raises(ValueError, match="ACTUAL_STOCK_MONEYFLOW_SOURCE_CHANGED"):
        f.validate(*args)


def test_parent_readback_must_finish_before_new_prediction(future_record):
    args, proofs = future_record
    proofs[f.prior_model.root() / "receipts/2026-09-17/001021.json"]["readback_at"] = "2026-09-17T08:17:00+08:00"
    with pytest.raises(ValueError, match="R116_PARENT_READBACK_LATE_OR_CHANGED"):
        f.validate(*args)


def test_late_source_rejected_even_with_recomputed_hashes(future_record):
    args, _ = future_record
    args[-1]["at"] = "2026-09-17T08:17:00+08:00"
    args[0]["stock_moneyflow_source_hash"] = s.base.digest(args[-1])
    args[1]["forecast_hash"] = s.base.digest(args[0])
    with pytest.raises(ValueError, match="STOCK_MONEYFLOW_SOURCE_CHANGED_OR_LATE"):
        f.validate(*args)


def test_all_thirty_six_branches_preserved(monkeypatch):
    parent = {"u": "2026-09-17", "code": "001021", "answers": {f"core{i}": {"prediction": i % 2} for i in range(8)}}
    records = {
        f.base.ROOT / f"round-{n}/forward/2026-09-17/001021.json": {
            "answers": {f"r{n}-{i}": {"prediction": i % 2} for i in range(2 if n in (92, 94, 113) else 1)}
        }
        for n in (
            92,
            94,
            95,
            96,
            97,
            98,
            99,
            100,
            101,
            102,
            103,
            104,
            105,
            106,
            107,
            108,
            109,
            110,
            111,
            112,
            113,
            114,
            115,
        )
    }
    monkeypatch.setattr(f.base, "read", lambda p: records[p])
    merged = f.combined_answers(
        parent,
        {"answers": {"r116": {"prediction": 1}}},
        {"answers": {"r117": {"prediction": 0}}},
    )
    assert len(merged) == 36 and merged["r117"]["prediction"] == 0 and merged["r92-1"]["prediction"] == 1


def test_old_source_cannot_be_relabelled_as_actual_future(future_record, monkeypatch):
    args, _ = future_record
    args[-1]["at"] = "2026-09-16T14:00:00+08:00"
    args[0]["stock_moneyflow_source_hash"] = s.base.digest(args[-1])
    args[1]["forecast_hash"] = s.base.digest(args[0])
    monkeypatch.setattr(f.live, "load_live", lambda *a: deepcopy(args[-1]))
    with pytest.raises(ValueError, match="STOCK_MONEYFLOW_SOURCE_NOT_ACTUALLY_CAPTURED_IN_WINDOW"):
        f.validate(*args)


def test_no_parent_prediction_cannot_create_new_future_record(monkeypatch, tmp_path):
    parent = {"code": "001021", "t": "2026-09-16", "u": "2026-09-17"}
    monkeypatch.setattr(f.runtime, "checked_models", lambda: ({"plan_hash": "new-plan"}, {}))
    monkeypatch.setattr(
        f.prior_forward, "context", lambda: ({}, {}, {}, {(parent["code"], parent["u"]): parent}, {}, {})
    )
    monkeypatch.setattr(f.prior_model, "root", lambda: tmp_path)
    actual = f.context()
    assert actual[3] == actual[4] == actual[5] == {}


@pytest.mark.parametrize(
    "stamp,early_sources", [("2026-09-17T08:00:00+08:00", True), ("2026-09-16T20:00:00+08:00", False)]
)
def test_time_sensitive_inputs_captured_before_long_parent_chain(monkeypatch, stamp, early_sources):
    calls = []
    monkeypatch.setattr(f.base, "now", lambda: datetime.fromisoformat(stamp))
    monkeypatch.setattr(f.base, "calendar", lambda: (["2026-09-16", "2026-09-17"], "calendar"))
    monkeypatch.setattr(f.base, "read", lambda p: {"path": str(p)})
    monkeypatch.setattr(f.runtime, "plan", lambda: {})
    monkeypatch.setattr(f.live.quote_live, "capture", lambda *a: calls.append(("quote", a)))
    monkeypatch.setattr(f.live, "capture", lambda *a: calls.append(("moneyflow", a)))
    monkeypatch.setattr(f.prior_forward, "run", lambda: calls.append(("parent", ())))
    monkeypatch.setattr(f, "context", lambda: ({}, {}, {}, {}, {}, {}))
    monkeypatch.setattr(f, "report", lambda loaded: loaded)
    f.run()
    assert [x[0] for x in calls] == (["quote", "moneyflow", "parent"] if early_sources else ["parent"])
    if early_sources:
        assert all(x[1][:2] == ("2026-09-16", "2026-09-17") for x in calls[:2])
