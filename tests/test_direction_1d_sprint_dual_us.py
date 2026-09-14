"""验证双指数窗口一致、成熟标签隔离、三次有界采集及新信息截止前落盘。"""

import json
from datetime import datetime

import numpy as np
import pytest
from app.integrations.tushare_sprint_ixic import FIELDS
from app.services import direction_1d_sprint as b
from app.services import direction_1d_sprint_dual_us as d
from app.services import direction_1d_sprint_fund_response as f
from app.services import direction_1d_sprint_overnight as o
from app.services import direction_1d_sprint_sparse as s

from test_direction_1d_sprint_fund_response import ConstantModel, sample_rows
from test_direction_1d_sprint_fund_response import ready as parent_ready  # noqa: F401


def test_both_indices_share_exact_session_window_and_no_missing_fill():
    aligned = {"required_us_dates": ["2026-09-10", "2026-09-11"]}
    points = {"2026-09-10": {"close": 100}, "2026-09-11": {"close": 102, "pre_close": 100}}
    x = [0.0] * 32
    x[30], x[31] = 0.01, 1
    np.testing.assert_allclose(d.vector(x, aligned, points), [0.01, 0.02, 1])
    x[31] = 0
    with pytest.raises(ValueError, match="US_SESSION_COUNT_MISMATCH"):
        d.vector(x, aligned, points)
    points.pop("2026-09-11")
    with pytest.raises(ValueError, match="OVERNIGHT_INPUT_INCOMPLETE"):
        d.vector(x, aligned, points)


def test_fixed_rules_are_not_probabilities():
    assert d.answer([0.03, -0.01, 1], "IXIC_SIGN")["prediction"] == 0
    assert d.answer([0.03, -0.01, 1], "US_EQUAL_SIGN")["prediction"] == 1
    assert d.answer([0.03, -0.01, 1], "US_EQUAL_SIGN")["research_score"] is None


def test_unmatured_labels_cannot_change_fitted_model():
    rows = [r | {"z": [r["x"][30], r["x"][30] * 0.7, r["x"][31]]} for r in sample_rows()]
    before = d.fit(rows, "LR3_US_BAL504", "2023-06-10")
    for r in rows:
        if r["mature"] >= "2023-06-10":
            r["z"] = [float("nan")] * 3
            r["y"] = 1 - r["y"]
    after = d.fit(rows, "LR3_US_BAL504", "2023-06-10")
    assert before["fit_hash"] == after["fit_hash"]
    np.testing.assert_array_equal(before["model"][-1].coef_, after["model"][-1].coef_)


@pytest.fixture
def ready(parent_ready, monkeypatch):  # noqa: F811 - pytest按参数名注入复用的第五轮父证据fixture。
    directory, original, _ = parent_ready
    aligned = o.alignment("2026-09-14", "2026-09-15")
    source_path = o.root() / "forward/2026-09-15/001000.json"
    source = b.read(source_path)
    source["x"][31] = len(aligned["required_us_dates"]) - 1
    b.save(source_path, source, replace=True)
    receipt_path = o.root() / "receipts/2026-09-15/001000.json"
    b.save(receipt_path, b.read(receipt_path) | {"forecast_hash": b.digest(source)}, replace=True)
    parent_path = s.root() / "forward/2026-09-15/001000.json"
    parent = b.read(parent_path) | {"parent_hash": b.digest(source)}
    b.save(parent_path, parent, replace=True)
    receipt_path = s.root() / "receipts/2026-09-15/001000.json"
    b.save(receipt_path, b.read(receipt_path) | {"forecast_hash": b.digest(parent)}, replace=True)
    assert f.tick()["verified_forecasts"] == 1
    b.save(d.root() / "result.json", {"winner": "LR3_US_BAL504"})
    bundle = {n: {"CN_EQUITY": {"model": ConstantModel()}} for n in d.CANDIDATES[:2]}
    monkeypatch.setattr(d, "models", lambda: ({"model_sha256": "abc"}, bundle))
    payload = {
        "code": 0,
        "data": {
            "fields": FIELDS,
            "items": [
                ["IXIC", day.replace("-", ""), 100 + i, 99 + i] for i, day in enumerate(aligned["required_us_dates"])
            ],
        },
    }
    monkeypatch.setattr(d, "fetch_ixic", lambda *args: json.dumps(payload).encode())
    return directory, original, payload


def test_true_arrival_saved_answers_and_outcomes_are_immutable_and_paired(ready, monkeypatch):
    directory, original, _ = ready
    report = d.tick()
    assert report["verified_forecasts"] == 1 and report["pending"] == 1
    path = d.root() / "forward/2026-09-15/001000.json"
    before = path.read_bytes()
    d.tick()
    assert path.read_bytes() == before
    b.save(
        directory / "outcomes/2026-09-15/001000.json",
        {"y": 1, "actual_direction": "UP", "forecast_hash": b.digest(original)},
    )
    monkeypatch.setattr(b, "now", lambda: datetime(2026, 9, 15, 19, tzinfo=b.ZONE))
    metrics = d.report()["matched_forward_metrics"]
    assert metrics["LR3_US_BAL504"]["accuracy"] == 1
    assert metrics["SPX_SIGN"]["accuracy"] == 0
    assert metrics["ORIGINAL7"]["count"] == metrics["IXIC_SIGN"]["count"]


def test_incomplete_new_index_has_only_three_slot_attempts(ready, monkeypatch):
    payload = ready[2]
    payload["data"]["items"].pop()
    calls = []

    def fetch(*args):
        calls.append(1)
        return json.dumps(payload).encode()

    monkeypatch.setattr(d, "fetch_ixic", fetch)
    for hour, minute in [(7, 15), (7, 20), (7, 45), (7, 55), (8, 15), (8, 20), (8, 30)]:
        monkeypatch.setattr(b, "now", lambda h=hour, m=minute: datetime(2026, 9, 15, h, m, tzinfo=b.ZONE))
        if minute in (15, 45):
            with pytest.raises(ValueError, match="IXIC_CALENDAR_COVERAGE_INCOMPLETE"):
                d.tick()
        else:
            assert d.tick()["verified_forecasts"] == 0
    assert len(calls) == 3


def test_ixic_arriving_at_cutoff_cannot_generate_prediction(ready, monkeypatch):
    def late(*args):
        monkeypatch.setattr(b, "now", lambda: datetime(2026, 9, 15, 8, 30, tzinfo=b.ZONE))
        return json.dumps(ready[2]).encode()

    monkeypatch.setattr(d, "fetch_ixic", late)
    assert d.tick()["verified_forecasts"] == 0
    assert (d.root() / "live/2026-09-15/0700-response.json").exists()


def test_saved_prediction_crossing_cutoff_is_invalid(ready, monkeypatch):
    save = b.save

    def crossing(path, value, **kwargs):
        save(path, value, **kwargs)
        if path.parent.parent == d.root() / "forward":
            monkeypatch.setattr(b, "now", lambda: datetime(2026, 9, 15, 8, 30, tzinfo=b.ZONE))

    monkeypatch.setattr(b, "save", crossing)
    report = d.tick()
    assert report["verified_forecasts"] == 0 and report["invalid_or_late"] == 1


def test_post_deadline_no_new_query(ready, monkeypatch):
    monkeypatch.setattr(b, "now", lambda: datetime(2026, 9, 17, 13, tzinfo=b.ZONE))
    monkeypatch.setattr(d, "fetch_ixic", lambda *args: pytest.fail("unexpected query"))
    assert d.tick()["verified_forecasts"] == 0
