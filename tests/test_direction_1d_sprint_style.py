"""市场风格实验：未来标签隔离、同一隔夜时段、有限采集和实际落盘边界。"""

import json
from datetime import datetime

import numpy as np
import pytest
from app.services import direction_1d_sprint as b
from app.services import direction_1d_sprint_dual_us as d
from app.services import direction_1d_sprint_style as s

from test_direction_1d_sprint_dual_us import parent_ready  # noqa: F401
from test_direction_1d_sprint_dual_us import ready as previous_ready  # noqa: F401
from test_direction_1d_sprint_fund_response import sample_rows


def test_style_differences_and_equal_rule_use_decimal_returns():
    z = s.vector(-0.01, 0.02, 0.03, 0.04, 1)
    np.testing.assert_allclose(z, [-0.01, 0.03, 0.04, 0.05, 1])
    assert s.answer(z, "US4_EQUAL_SIGN") == {"prediction": 1, "research_score": None, "kind": "FIXED_RULE"}
    with pytest.raises(ValueError, match="STYLE_VECTOR_INVALID"):
        s.vector(0, 0, 0, 0, 0.5)


@pytest.mark.parametrize("name", s.LEARNED)
def test_unmatured_input_and_labels_do_not_change_fit(name):
    rows = [
        r | {"z": s.vector(r["x"][30], 0.01, -0.01, 0, r["x"][31]), "return_target": 1 if r["y"] else -1}
        for r in sample_rows()
    ]
    before = s.fit(rows, name, "2023-06-10")
    for row in rows:
        if row["mature"] >= "2023-06-10":
            row["y"] = 1 - row["y"]
            row["return_target"] = float("nan")
            row["z"] = [float("nan")] * 5
    after = s.fit(rows, name, "2023-06-10")
    assert before["fit_hash"] == after["fit_hash"]
    assert before["max_mature_date"] < "2023-06-10"
    left = s.answer([0.01, 0, 0, 0, 1], name, before)
    right = s.answer([0.01, 0, 0, 0, 1], name, after)
    assert left == right


class ConstantModel:
    def predict_proba(self, x):
        return np.asarray([[0.7, 0.3]] * len(x))

    def predict(self, x):
        return np.full(len(x), -0.3)


@pytest.fixture
def ready(previous_ready, monkeypatch):  # noqa: F811
    directory, original, payload = previous_ready
    b.save(d.root() / "result.json", {"winner": "LR3_US_BAL504", "model_sha256": "abc"}, replace=True)
    assert d.tick()["verified_forecasts"] == 1
    b.save(s.root() / "result.json", {"winner": s.CANDIDATES[0], "model_sha256": "style"})
    bundle = {n: {"CN_EQUITY": {"model": ConstantModel()}} for n in s.LEARNED}
    monkeypatch.setattr(s, "models", lambda: ({"model_sha256": "style"}, bundle))
    monkeypatch.setattr(s.overnight, "source", lambda: {"rate_limit_per_minute": 10})
    monkeypatch.setattr(s.walltime, "sleep", lambda *_: None)
    calls = []

    def fetch(code, *_):
        calls.append(code)
        response = json.loads(json.dumps(payload))
        for row in response["data"]["items"]:
            row[0] = code
        return json.dumps(response).encode()

    monkeypatch.setattr(s, "fetch_style", fetch)
    return directory, original, calls, fetch


def test_old_target_never_triggers_new_source_queries(ready):
    assert s.tick()["verified_forecasts"] == 0
    assert ready[2] == []


def test_actual_inputs_answers_and_outcomes_remain_paired_and_immutable(ready, monkeypatch):
    monkeypatch.setattr(s, "FIRST_TARGET", "2026-09-15")
    directory, original, calls, _ = ready
    assert s.tick()["verified_forecasts"] == 1
    assert calls == ["RUT", "DJI"]
    path = s.root() / "forward/2026-09-15/001000.json"
    before = path.read_bytes()
    assert len(b.read(path)["answers"]) == 4
    s.tick()
    assert path.read_bytes() == before and len(calls) == 2
    b.save(
        directory / "outcomes/2026-09-15/001000.json",
        {"y": 0, "actual_direction": "DOWN", "forecast_hash": b.digest(original)},
    )
    monkeypatch.setattr(b, "now", lambda: datetime(2026, 9, 15, 19, tzinfo=b.ZONE))
    report = s.report()
    assert report["matched_forward_metrics"]["STYLE_LR5_BAL252"]["accuracy"] == 1
    assert report["matched_forward_metrics"]["ORIGINAL7"]["count"] == 1


def test_incomplete_index_is_bounded_to_three_attempts_and_other_index_is_reused(ready, monkeypatch):
    monkeypatch.setattr(s, "FIRST_TARGET", "2026-09-15")
    calls, fetch = ready[2:]

    def incomplete(code, *args):
        payload = json.loads(fetch(code, *args))
        if code == "RUT":
            payload["data"]["items"].pop()
        return json.dumps(payload).encode()

    monkeypatch.setattr(s, "fetch_style", incomplete)
    for hour, minute in ((7, 15), (7, 20), (7, 45), (7, 55), (8, 15), (8, 20), (8, 30)):
        monkeypatch.setattr(b, "now", lambda h=hour, m=minute: datetime(2026, 9, 15, h, m, tzinfo=b.ZONE))
        assert s.tick()["verified_forecasts"] == 0
    assert calls.count("RUT") == 3 and calls.count("DJI") == 1


def test_input_arriving_at_deadline_does_not_generate_predictions(ready, monkeypatch):
    monkeypatch.setattr(s, "FIRST_TARGET", "2026-09-15")
    fetch = ready[3]

    def late(code, *args):
        payload = fetch(code, *args)
        monkeypatch.setattr(b, "now", lambda: datetime(2026, 9, 15, 8, 30, tzinfo=b.ZONE))
        return payload

    monkeypatch.setattr(s, "fetch_style", late)
    assert s.tick()["verified_forecasts"] == 0
    assert ready[2] == ["RUT"]


def test_save_crossing_cutoff_is_counted_as_late(ready, monkeypatch):
    monkeypatch.setattr(s, "FIRST_TARGET", "2026-09-15")
    save = b.save

    def crossing(path, value, **kwargs):
        save(path, value, **kwargs)
        if path.parent.parent == s.root() / "forward":
            monkeypatch.setattr(b, "now", lambda: datetime(2026, 9, 15, 8, 30, tzinfo=b.ZONE))

    monkeypatch.setattr(b, "save", crossing)
    result = s.tick()
    assert result["verified_forecasts"] == 0 and result["invalid_or_late"] == 1


def test_modified_observation_blocks_result_verification(ready, monkeypatch):
    monkeypatch.setattr(s, "FIRST_TARGET", "2026-09-15")
    assert s.tick()["verified_forecasts"] == 1
    path = next((s.root() / "live/2026-09-15").glob("RUT-*-response.json"))
    raw = b.read(path)
    raw["response"]["data"]["items"][0][2] += 1
    b.save(path, raw, replace=True)
    with pytest.raises(ValueError, match="ROUND_10_INPUT_EVIDENCE_CHANGED"):
        s.report()


def test_old_runner_failure_still_attempts_style_branch(monkeypatch):
    from scripts import direction_1d_sprint_style as entry

    monkeypatch.setattr(entry.existing, "run", lambda: (_ for _ in ()).throw(ValueError("OLD_FAILED")))
    calls = []
    monkeypatch.setattr(entry.style, "tick", lambda: calls.append(1))
    monkeypatch.setattr(b, "save", lambda *args, **kwargs: None)
    with pytest.raises(ValueError, match="INDEPENDENT_BRANCH_FAILED"):
        entry.run()
    assert calls == [1]
