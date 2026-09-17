"""复权输入与原净值严格配对、有界采集、原学习器对照及预测时间边界。"""

import json
from datetime import date, datetime

import numpy as np
import pytest
from app.integrations import tushare_sprint_adjusted_nav as client
from app.services import direction_1d_sprint as b
from app.services import direction_1d_sprint_adjusted as s
from app.services import direction_1d_sprint_sparse as sparse

from test_direction_1d_sprint_sequence import (
    ancestor_ready,  # noqa: F401
    inputs,
    sample_rows,
)
from test_direction_1d_sprint_sequence import ready as sequence_ready  # noqa: F401


def payload(original):
    return {
        "code": 0,
        "data": {
            "fields": s.FIELDS,
            "items": [
                ["001000.OF", r["date"].replace("-", ""), r["date"].replace("-", ""), r["nav"], r["nav"]]
                for r in original["inputs"]
            ],
        },
    }


def test_dividend_discontinuity_changes_input_only_not_raw_target():
    values, _ = inputs()
    raw = values.copy()
    raw[-4:] *= 0.8
    x = b.vector(raw, True) + [0.0] * 12 + [0.01, 1]
    days = list(map(str, b.input_days(date(2026, 9, 14))))
    original = {
        "base": "2026-09-14",
        "u": "2026-09-15",
        "inputs": [{"date": d, "nav": float(v)} for d, v in zip(days, raw, strict=True)],
    }
    points = {
        d: {"date": d, "nav": float(v), "adjusted_nav": float(a), "ann_date": d}
        for d, v, a in zip(days, raw, values, strict=True)
    }
    before = json.dumps(original)
    z = s.vector(x, original, points)
    assert z[3] == pytest.approx(b.vector(values, True)[0])
    assert z[3] != pytest.approx(x[0])
    assert json.dumps(original) == before
    points[days[-1]]["nav"] += 0.1
    with pytest.raises(ValueError, match="ADJUSTED_ORIGINAL_INPUT_CHANGED"):
        s.vector(x, original, points)


@pytest.mark.parametrize("kind", ["wrong_code", "duplicate", "nonfinite", "out_of_window"])
def test_parser_refuses_mismatched_or_invalid_source_data(kind):
    p = {"code": 0, "data": {"fields": s.FIELDS, "items": [["001000.OF", "20260914", "20260914", 1.0, 1.2]]}}
    rows = p["data"]["items"]
    if kind == "wrong_code":
        rows[0][0] = "002000.OF"
    elif kind == "duplicate":
        rows.append(rows[0])
    elif kind == "nonfinite":
        rows[0][-1] = float("nan")
    else:
        rows[0][2] = "20260915"
    with pytest.raises(ValueError):
        s.parse("001000.OF", json.dumps(p), "2026-09-01", "2026-09-14")


@pytest.mark.parametrize("adjusted,raw", [("ADJ_LR8_BAL252", "LR8_BAL252"), ("ADJ_TREE8_BAL252", "TREE8_BAL252")])
def test_identical_input_reproduces_original_learner_and_isolates_future_labels(adjusted, raw, monkeypatch):
    monkeypatch.setattr(s, "active", lambda: None)
    samples = [r | {"z": sparse.vector(r["x"], raw)} for r in sample_rows()]
    one = s.fit(samples, adjusted, "2023-06-10")
    control = sparse.fit(samples, raw, "2023-06-10")
    probe = samples[50]
    assert s.answer(probe["z"], adjusted, one)["research_score"] == pytest.approx(
        sparse.answer(probe["x"], raw, control)["research_score"], abs=1e-9
    )
    for r in samples:
        if r["mature"] >= "2023-06-10":
            r["z"], r["y"] = [float("nan")] * 8, 1 - r["y"]
    two = s.fit(samples, adjusted, "2023-06-10")
    assert one["fit_hash"] == two["fit_hash"]
    assert s.answer(probe["z"], adjusted, one) == s.answer(probe["z"], adjusted, two)


def test_adapter_rejects_unbounded_requests_before_network(monkeypatch):
    monkeypatch.setattr(client, "get_settings", lambda: pytest.fail("bad bounds should fail first"))
    for code, start, end in [
        ("RUT", date(2026, 9, 1), date(2026, 9, 14)),
        ("001000.OF", date(2021, 1, 1), date(2026, 9, 14)),
    ]:
        with pytest.raises(ValueError, match="ADJUSTED_QUERY_RANGE_INVALID"):
            client.fetch_adjusted_nav(code, start, end)


class ConstantModel:
    def predict_proba(self, x):
        return np.asarray([[0.7, 0.3]] * len(x))


@pytest.fixture
def ready(sequence_ready, monkeypatch):  # noqa: F811
    directory, original, source = sequence_ready
    history = b.read(directory / "history.json")
    history["funds"][0]["source_fund_code"] = "001000.OF"
    b.save(directory / "history.json", history, replace=True)
    b.save(s.root() / "result.json", {"winner": s.CANDIDATES[0], "model_sha256": "abc"})
    bundle = {n: {"CN_EQUITY": {"model": ConstantModel(), "mean": [0.0] * 8, "scale": [1.0] * 8}} for n in s.CANDIDATES}
    monkeypatch.setattr(s, "models", lambda: ({"model_sha256": "abc"}, bundle))
    monkeypatch.setattr(s.clock, "sleep", lambda *_: None)
    data = payload(original)
    calls = []
    monkeypatch.setattr(s, "fetch_adjusted_nav", lambda *args: calls.append(args) or json.dumps(data).encode())
    return directory, original, source, data, calls


def test_new_version_never_backfills_old_target(ready, monkeypatch):
    monkeypatch.setattr(s, "fetch_adjusted_nav", lambda *_: pytest.fail("old target cannot query"))
    assert s.tick()["verified_forecasts"] == 0


def test_future_capture_and_answers_are_immutable_and_outcomes_paired(ready, monkeypatch):
    monkeypatch.setattr(s, "FIRST_TARGET", "2026-09-15")
    directory, original, _, _, calls = ready
    assert s.tick()["verified_forecasts"] == 1
    path = s.root() / "forward/2026-09-15/001000.json"
    before = path.read_bytes()
    s.tick()
    assert path.read_bytes() == before and len(calls) == 1
    b.save(
        directory / "outcomes/2026-09-15/001000.json",
        {"y": 0, "actual_direction": "DOWN", "forecast_hash": b.digest(original)},
    )
    monkeypatch.setattr(b, "now", lambda: datetime(2026, 9, 15, 19, tzinfo=b.ZONE))
    assert s.report()["matched_forward_metrics"][s.CANDIDATES[0]]["accuracy"] == 1


def test_missing_data_gets_only_three_declared_attempts(ready, monkeypatch):
    monkeypatch.setattr(s, "FIRST_TARGET", "2026-09-15")
    ready[3]["data"]["items"].pop()
    for h, m in [(7, 15), (7, 20), (7, 45), (7, 55), (8, 15), (8, 20), (8, 30)]:
        monkeypatch.setattr(b, "now", lambda h=h, m=m: datetime(2026, 9, 15, h, m, tzinfo=b.ZONE))
        assert s.tick()["verified_forecasts"] == 0
    assert len(ready[4]) == 3


def test_response_arriving_at_deadline_cannot_save_answer(ready, monkeypatch):
    monkeypatch.setattr(s, "FIRST_TARGET", "2026-09-15")

    def late(*args):
        monkeypatch.setattr(b, "now", lambda: datetime(2026, 9, 15, 8, 30, tzinfo=b.ZONE))
        return json.dumps(ready[3]).encode()

    monkeypatch.setattr(s, "fetch_adjusted_nav", late)
    assert s.tick()["verified_forecasts"] == 0


def test_forged_parsed_values_cannot_disagree_with_saved_raw_response(ready, monkeypatch):
    monkeypatch.setattr(s, "FIRST_TARGET", "2026-09-15")
    assert s.tick()["verified_forecasts"] == 1
    path = s.root() / "forward/2026-09-15/001000.json"
    value = b.read(path)
    day = next(iter(value["context"]["points"]))
    value["context"]["points"][day]["adjusted_nav"] = "100"
    b.save(path, value, replace=True)
    receipt = s.root() / "receipts/2026-09-15/001000.json"
    b.save(receipt, b.read(receipt) | {"forecast_hash": b.digest(value)}, replace=True)
    with pytest.raises(ValueError, match="ROUND_14_PARSED_INPUT_CHANGED"):
        s.report()


def test_prior_failure_does_not_skip_new_branch(monkeypatch):
    from scripts import direction_1d_sprint_adjusted as entry

    monkeypatch.setattr(entry.existing, "run", lambda: (_ for _ in ()).throw(ValueError("OLD_FAILED")))
    calls = []
    monkeypatch.setattr(entry.adjusted, "tick", lambda: calls.append(1))
    monkeypatch.setattr(b, "save", lambda *args, **kwargs: None)
    with pytest.raises(ValueError, match="INDEPENDENT_BRANCH_FAILED"):
        entry.run()
    assert calls == [1]


def test_failed_capture_records_stack_and_does_not_repeat_same_slot(ready, monkeypatch):
    monkeypatch.setattr(s, "FIRST_TARGET", "2026-09-15")
    ready[3]["data"]["items"][0][0] = "wrong_code"
    with pytest.raises(ValueError, match="ADJUSTED_SOME_FUND_INPUTS_FAILED"):
        s.tick()
    error = b.read(s.root() / "live/2026-09-15/0700-errors.json")["001000"]
    assert error["stack"] and error["error"] == "ADJUSTED_CODE_OR_DATE_INVALID"
    assert s.tick()["verified_forecasts"] == 0
    assert len(ready[4]) == 1
