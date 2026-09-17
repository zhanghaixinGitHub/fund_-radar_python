"""独立接续的关键边界：旧模型不改、缺父拒绝、时间与来源绑定、先采集后完整父run。"""

from copy import deepcopy
from datetime import datetime
from types import SimpleNamespace

import pytest
from app.services import direction_1d_sprint_independent_candidates_v2 as m


def test_branches_preserved_without_untrained_placeholder():
    old = {f"old{i}": {"prediction": i % 2} for i in range(35)}
    new = {f"r{i}": {"prediction": i % 2} for i in (118, 119, 120, 121)}
    assert m.combine(old, new) == old | new
    assert "r117" not in m.combine(old, new)


@pytest.mark.parametrize("old_count,new_count,collide", [(34, 4, False), (35, 3, False), (35, 4, True)])
def test_branch_omission_or_collision_rejected(old_count, new_count, collide):
    old = {str(i): {} for i in range(old_count)}
    new = {(str(i) if collide else f"new{i}"): {} for i in range(new_count)}
    with pytest.raises(ValueError, match="BRANCH_COLLISION_OR_MISSING"):
        m.combine(old, new)


def test_no_historical_substitute_when_actual_snapshot_missing(monkeypatch):
    def forbidden(*args):
        raise AssertionError("Actual load forbidden before ready")

    monkeypatch.setattr(m, "SPECS", {118: (None, SimpleNamespace(ready=lambda u: False, load_live=forbidden), None)})
    assert m.actual_histories({"t": "2026-09-16", "u": "2026-09-17"}, {}) is None


def test_actual_source_outside_window_rejected(monkeypatch):
    live = SimpleNamespace(
        ready=lambda u: True, load_live=lambda *a: {"at": "2026-09-17T08:31:00+08:00"}, in_window=lambda *a: False
    )
    monkeypatch.setattr(m, "SPECS", {118: (None, live, None)})
    with pytest.raises(ValueError, match="SOURCE_OUTSIDE_WINDOW"):
        m.actual_histories({"t": "2026-09-16", "u": "2026-09-17"}, {118: ({"plan_hash": "p"}, {})})


def test_model_output_and_missing_fallback_used_without_changes(monkeypatch):
    seen, specs, loaded, histories = [], {}, {}, {}
    for n in m.ROUNDS:

        def extend(market, t, u, points, number=n):
            seen.append((number, market, t, u, points))
            return {"available": bool(points)}

        def answer(market, group, bundle, number=n):
            return {f"r{number}": {"prediction": 0, "fallback": not market["available"], "probability": 0.314}}

        model = SimpleNamespace(CANDIDATES=(f"r{n}",), data=SimpleNamespace(extend=extend), answers=answer)
        specs[n] = (model, None, None)
        loaded[n], histories[n] = ({}, {}), {"points": {} if n == 120 else {"day": n}}
    monkeypatch.setattr(m, "SPECS", specs)
    values = m.candidate_answers({"market": {"original": 1}}, {"t": "t", "u": "u", "group": "g"}, histories, loaded)
    assert len(seen) == 4 and values["r120"]["fallback"]
    assert not values["r118"]["fallback"] and values["r119"]["probability"] == 0.314


@pytest.fixture
def valid_case(monkeypatch, tmp_path):
    monkeypatch.setattr(m.b, "ROOT", tmp_path)
    source = {"market": {"synthetic": True}}
    q = {
        "code": "synthetic",
        "family": "f",
        "group": "g",
        "t": "2026-09-16",
        "u": "2026-09-17",
        "at": "2026-09-17T08:01:00+08:00",
        "input_hash": m.b.digest(source),
    }
    tail = {"synthetic": "r116"}
    plan = {"at": "2026-09-16T23:00:00+08:00"}
    loaded = {
        n: ({"at": "2026-09-16T21:00:00+08:00", "plan_hash": f"p{n}", "model_sha256": f"m{n}"}, {}) for n in m.ROUNDS
    }
    histories = {n: {"at": "2026-09-17T08:02:00+08:00", "points": {}, "available": False} for n in m.ROUNDS}
    old, new = {f"old{i}": {"prediction": 1} for i in range(35)}, {f"r{n}": {"prediction": 0} for n in m.ROUNDS}
    monkeypatch.setattr(m, "candidate_answers", lambda *a: deepcopy(new))
    monkeypatch.setattr(m.core.data, "deadline", lambda u: datetime.fromisoformat("2026-09-17T08:30:00+08:00"))
    monkeypatch.setattr(m.core.evidence, "sprint_end", lambda: datetime.fromisoformat("2026-09-17T12:11:38+08:00"))
    for folder, record in [(m.core.root(), q), (m.parent.model.root(), tail)]:
        m.b.save(
            folder / "receipts" / q["u"] / "synthetic.json",
            {"status": "VERIFIED", "forecast_hash": m.b.digest(record), "readback_at": "2026-09-17T08:03:00+08:00"},
        )
    v = {k: q[k] for k in ("code", "family", "group", "t", "u")} | {
        "at": "2026-09-17T08:04:00+08:00",
        "runtime_plan_hash": m.b.digest(plan),
        "parent_hash": m.b.digest(q),
        "r116_forecast_hash": m.b.digest(tail),
        "input_hash": q["input_hash"],
        "models": m.model_identities(loaded),
        "source_hashes": {str(n): m.b.digest(h) for n, h in histories.items()},
        "answers": old | new,
        "status": "MODEL_NOT_RELEASED",
        "contract": m.core.CONTRACT,
    }
    receipt = {"readback_at": "2026-09-17T08:04:01+08:00", "forecast_hash": m.b.digest(v), "status": "VERIFIED"}
    return [v, receipt, q, source, tail, old, histories, loaded, plan]


def test_complete_binding_passes(valid_case):
    m.validate(*valid_case)


@pytest.mark.parametrize(
    "field,value,reason",
    [
        ("input_hash", "changed", "INPUT_CHANGED"),
        ("runtime_plan_hash", "changed", "PLAN_OR_PARENT_CHANGED"),
        ("r116_forecast_hash", "changed", "PLAN_OR_PARENT_CHANGED"),
        ("models", {}, "FORECAST_MODEL_CHANGED"),
        ("source_hashes", {}, "SOURCE_CHANGED"),
        ("answers", {}, "ANSWERS_CHANGED"),
        ("at", "2026-09-17T08:30:00+08:00", "FORECAST_LATE"),
        ("code", "wrong", "QUESTION_CHANGED"),
        ("status", "RELEASED", "CONTRACT_CHANGED"),
    ],
)
def test_tampering_rejected_even_with_new_outer_receipt(valid_case, field, value, reason):
    valid_case[0][field] = value
    valid_case[1]["forecast_hash"] = m.b.digest(valid_case[0])
    with pytest.raises(ValueError, match=reason):
        m.validate(*valid_case)


def test_late_readback_rejected(valid_case):
    valid_case[1]["readback_at"] = "2026-09-17T08:30:00+08:00"
    with pytest.raises(ValueError, match="FORECAST_LATE"):
        m.validate(*valid_case)


def test_late_parent_receipt_rejected(valid_case):
    p = m.parent.model.root() / "receipts/2026-09-17/synthetic.json"
    receipt = m.b.read(p)
    receipt["readback_at"] = "2026-09-17T08:05:00+08:00"
    m.b.save(p, receipt, replace=True)
    with pytest.raises(ValueError, match="PARENT_RECEIPT_CHANGED_OR_LATE"):
        m.validate(*valid_case)


@pytest.mark.parametrize("index,reason", [(6, "SOURCE_CREATED_LATE"), (7, "MODEL_CREATED_LATE")])
def test_source_or_model_created_after_forecast_rejected(valid_case, index, reason):
    value = valid_case[index][118] if index == 6 else valid_case[index][118][0]
    value["at"] = "2026-09-17T08:05:00+08:00"
    with pytest.raises(ValueError, match=reason):
        m.validate(*valid_case)


def test_early_capture_keeps_original_quote_reservation(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(m.member_live, "in_window", lambda *a: True)
    monkeypatch.setattr(m.b, "read", lambda p: {})
    monkeypatch.setattr(m.member_live.quote_live, "live_root", lambda: tmp_path)
    monkeypatch.setattr(m.member_live.quote_live, "capture", lambda *a: calls.append("original_quote"))
    monkeypatch.setattr(m.member_live, "capture", lambda *a: calls.append("member_once"))
    monkeypatch.setattr(m.moneyflow_live.quote_live, "capture", lambda *a: calls.append("original_stocks"))
    monkeypatch.setattr(m.moneyflow_live, "capture", lambda *a: calls.append("moneyflow_once"))
    m.early_capture({120: ({"plan_hash": "p"}, {}), 121: ({"plan_hash": "p121"}, {})})
    assert calls == ["original_quote", "original_stocks", "moneyflow_once"]
    (tmp_path / "2026-09-17").mkdir()
    (tmp_path / "2026-09-17/snapshot.json").write_text("synthetic", encoding="utf-8")
    calls.clear()
    m.early_capture({120: ({"plan_hash": "p"}, {}), 121: ({"plan_hash": "p121"}, {})})
    assert calls == ["original_quote", "member_once", "original_stocks", "moneyflow_once"]


def test_run_never_skips_real_parent_and_stops_on_failure(monkeypatch):
    calls = []
    plan = {"parent_module_chain": [m.parent.__name__]}
    monkeypatch.setattr(m, "bindings", lambda: (plan, {}))
    monkeypatch.setattr(m, "early_capture", lambda x: calls.append("early_capture"))

    def fail():
        calls.append("real_parent_run")
        raise RuntimeError("PARENT_FAILED")

    monkeypatch.setattr(m, "Executor", lambda *a, **k: SimpleNamespace(modules=[m.parent], run=fail))
    monkeypatch.setattr(m, "write_and_report", lambda *a: calls.append("must_not_write"))
    with pytest.raises(RuntimeError, match="PARENT_FAILED"):
        m.execute("run")
    assert calls == ["early_capture", "real_parent_run"]
