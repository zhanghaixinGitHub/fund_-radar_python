"""隔夜新增信息必须早于预测截止；官方休市、DST、预算和父答案绑定均实际验证。"""

import json
from datetime import datetime, timedelta

import numpy as np
import pytest
from app.services import direction_1d_sprint as b
from app.services import direction_1d_sprint_market as m
from app.services import direction_1d_sprint_overnight as o


def test_calendar_carter_early_close_and_dst():
    events = o.sessions()
    assert sum(d.startswith("2025") for d in events) == 250
    assert "2025-01-09" not in events
    assert events["2025-07-03"].isoformat() == "2025-07-04T01:00:00+08:00"
    assert events["2025-12-24"].isoformat() == "2025-12-25T02:00:00+08:00"
    assert events["2025-03-07"].hour == 5
    assert events["2025-03-10"].hour == 4


def test_no_session_is_zero_only_with_valid_baseline():
    a = o.alignment("2025-01-09", "2025-01-10")
    assert a["new_us_dates"] == []
    points = {"2025-01-08": {"close": 100, "pre_close": 99}}
    assert o.extend([0.1] * 30, a, points)[-2:] == [0, 0]
    with pytest.raises(ValueError, match="OVERNIGHT_INPUT_INCOMPLETE"):
        o.extend([0.1] * 30, a, {})


def test_china_holiday_accumulates_us_sessions_without_future_close():
    a = o.alignment("2025-01-27", "2025-02-05")
    assert len(a["new_us_dates"]) == 7
    assert a["required_us_dates"][-1] == "2025-02-04"
    points = {d: {"close": 100 + i, "pre_close": 99 + i} for i, d in enumerate(a["required_us_dates"])}
    first = o.extend([0.1] * 30, a, points)
    points["2025-02-05"] = {"close": 999999, "pre_close": 100}
    assert o.extend([0.1] * 30, a, points) == first
    assert first[-2] == pytest.approx(0.07)
    points[a["required_us_dates"][-1]]["pre_close"] = 1
    with pytest.raises(ValueError, match="OVERNIGHT_PREVIOUS_CLOSE_CHANGED"):
        o.extend([0.1] * 30, a, points)


def test_fit_does_not_use_unmatured_labels():
    days = b.calendar()[0][100:240]
    rows = [
        {"u": str(d), "mature": str(d + timedelta(days=2)), "family": "f", "y": i % 2, "x": [i / 100] + [0.1] * 31}
        for i, d in enumerate(days)
    ]
    cutoff = str(days[-1])
    first = o.fit(rows, "LR32_252", cutoff)
    for r in rows:
        if r["mature"] >= cutoff:
            r["y"] = 1 - r["y"]
    second = o.fit(rows, "LR32_252", cutoff)
    np.testing.assert_array_equal(first["model"][-1].coef_, second["model"][-1].coef_)


class ConstantModel:
    def predict_proba(self, x):
        return np.asarray([[0.2, 0.8]] * len(x))


def test_two_decimal_percentage_does_not_override_price_chain():
    body = {
        "code": 0,
        "data": {
            "fields": o.old_live.FIELDS,
            "items": [["SPX", "20260226", 6908.86, 6946.13, -0.54], ["SPX", "20260227", 6878.88, 6908.86, -0.43]],
        },
    }
    dates = ["2026-02-26", "2026-02-27"]
    result = o.validate_response(json.dumps(body).encode(), dates)
    assert result["status"] == "COMPLETE"
    assert result["two_decimal_rounding_dates"] == dates
    # 百分数看似正确也不能接纳前收盘和昨日收盘不一致的价格。
    body["data"]["items"][1] = ["SPX", "20260227", 6878.88, 6900.0, -0.31]
    with pytest.raises(ValueError, match="OVERNIGHT_PREVIOUS_CLOSE_CHANGED"):
        o.validate_response(json.dumps(body).encode(), dates)


def test_percentage_tolerance_is_not_expanded_to_integer_precision():
    body = {"code": 0, "data": {"fields": o.old_live.FIELDS, "items": [["SPX", "20260227", 100.1, 100, 0.0]]}}
    with pytest.raises(ValueError, match="OVERNIGHT_RETURN_INCONSISTENT"):
        o.validate_response(json.dumps(body).encode(), ["2026-02-27"])


def test_previous_close_only_accepts_observed_rounding_precision():
    assert o.same_previous_close(7718.60, 7718.5982)
    assert o.same_previous_close(6939.0295, 6939.03)
    assert not o.same_previous_close(7718.60, 7718.61)
    assert not o.same_previous_close(7718.6012, 7718.603)
    aligned = {"required_us_dates": ["2026-09-04", "2026-09-08"]}
    points = {"2026-09-04": {"close": 7718.60}, "2026-09-08": {"close": 7720.0, "pre_close": 7718.5982}}
    assert o.extend([0.0] * 30, aligned, points)[-2:] == [7720.0 / 7718.60 - 1, 1.0]
    points["2026-09-08"]["pre_close"] = 7718.61
    with pytest.raises(ValueError, match="OVERNIGHT_PREVIOUS_CLOSE_CHANGED"):
        o.extend([0.0] * 30, aligned, points)


@pytest.fixture
def ready(tmp_path, monkeypatch):
    monkeypatch.setattr(b, "ROOT", tmp_path)
    monkeypatch.setattr(b, "now", lambda: datetime(2026, 9, 14, 12, tzinfo=b.ZONE))
    b.initialize()
    b.save(tmp_path / "scope.json", {"codes": ["001000", "002000"]})
    b.save(tmp_path / "history.json", {"funds": [{"fund_code": "001000"}]})
    original = {
        "at": "2026-09-14T20:00:00+08:00",
        "base": "2026-09-14",
        "u": "2026-09-15",
        "code": "001000",
        "family": "f",
        "group": "CN_EQUITY",
        "answers": {"ORIGINAL7": {"prediction": 0}},
    }
    parent = {
        "at": "2026-09-14T20:02:00+08:00",
        "u": "2026-09-15",
        "code": "001000",
        "x": [0.1] * 30,
        "parent_hash": b.digest(original),
        "answers": {"TREE30_252": {"prediction": 0}},
    }
    b.save(tmp_path / "forward/2026-09-15/001000.json", original)
    b.save(m.root() / "forward/2026-09-15/001000.json", parent)
    b.save(
        m.root() / "receipts/2026-09-15/001000.json",
        {"status": "VERIFIED", "forecast_hash": b.digest(parent), "readback_at": "2026-09-14T20:03:00+08:00"},
    )
    b.save(o.root() / "result.json", {"winner": "TREE32_252"})
    monkeypatch.setattr(b, "now", lambda: datetime(2026, 9, 15, 7, 15, tzinfo=b.ZONE))
    monkeypatch.setattr(o, "source", lambda: None)
    monkeypatch.setattr(
        o, "models", lambda: ({"model_sha256": "abc"}, {"TREE32_252": {"CN_EQUITY": {"model": ConstantModel()}}})
    )
    a = o.alignment("2026-09-14", "2026-09-15")
    values = [
        ["SPX", d.replace("-", ""), 100 + i, 99 + i, round(((100 + i) / (99 + i) - 1) * 100, 4)]
        for i, d in enumerate(a["required_us_dates"])
    ]
    payload = {"code": 0, "data": {"fields": o.old_live.FIELDS, "items": values}}
    monkeypatch.setattr(o, "fetch_spx", lambda *args: json.dumps(payload).encode())
    return tmp_path, original, parent, payload


def test_forward_and_outcome_use_identical_saved_parents(ready, monkeypatch):
    root, original, parent, _ = ready
    result = o.tick()
    path = o.root() / "forward/2026-09-15/001000.json"
    before = path.read_bytes()
    value = b.read(path)
    assert value["parent_market_hash"] == b.digest(parent)
    assert value["x"][:30] == parent["x"]
    assert result["verified_forecasts"] == 1 and result["pending"] == 1
    o.tick()
    assert path.read_bytes() == before
    b.save(
        root / "outcomes/2026-09-15/001000.json",
        {"y": 1, "actual_direction": "UP", "forecast_hash": b.digest(original)},
    )
    monkeypatch.setattr(b, "now", lambda: datetime(2026, 9, 15, 19, tzinfo=b.ZONE))
    report = o.report()["matched_forward_metrics"]
    assert report["TREE32_252"]["accuracy"] == 1
    assert report["TREE30_252"]["accuracy"] == 0
    assert report["ORIGINAL7"]["count"] == 1


def test_incomplete_provider_data_has_three_slot_limit(ready, monkeypatch):
    payload = ready[3]
    payload["data"]["items"] = payload["data"]["items"][:-1]
    calls = []

    def query(*args):
        calls.append(1)
        return json.dumps(payload).encode()

    monkeypatch.setattr(o, "fetch_spx", query)
    for hour, minute in [(7, 15), (7, 20), (7, 45), (7, 55), (8, 15), (8, 20), (8, 30)]:
        monkeypatch.setattr(b, "now", lambda h=hour, m=minute: datetime(2026, 9, 15, h, m, tzinfo=b.ZONE))
        assert o.tick()["verified_forecasts"] == 0
    assert len(calls) == 3
    assert o.report()["missing_due_predictions"] == 1


def test_parent_late_receipt_is_rejected(ready):
    path = m.root() / "receipts/2026-09-15/001000.json"
    r = b.read(path) | {"readback_at": "2026-09-15T08:30:00+08:00"}
    b.save(path, r, replace=True)
    with pytest.raises(ValueError, match="PARENT_FORECAST_NOT_VERIFIED"):
        o.tick()


def test_persistence_crossing_deadline_is_not_a_success(ready, monkeypatch):
    save = b.save

    def crossing(path, payload, **kwargs):
        save(path, payload, **kwargs)
        if path.parent.parent == o.root() / "forward":
            monkeypatch.setattr(b, "now", lambda: datetime(2026, 9, 15, 8, 30, tzinfo=b.ZONE))

    monkeypatch.setattr(b, "save", crossing)
    result = o.tick()
    assert result["verified_forecasts"] == 0 and result["invalid_or_late"] == 1


def test_no_evening_or_post_checkpoint_provider_call(ready, monkeypatch):
    monkeypatch.setattr(o, "fetch_spx", lambda *args: pytest.fail("unexpected provider call"))
    for at in [datetime(2026, 9, 14, 21, tzinfo=b.ZONE), datetime(2026, 9, 17, 13, tzinfo=b.ZONE)]:
        monkeypatch.setattr(b, "now", lambda at=at: at)
        assert o.tick()["verified_forecasts"] == 0
