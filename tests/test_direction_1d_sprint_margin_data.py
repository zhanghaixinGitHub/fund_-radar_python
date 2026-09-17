"""两融日期滞后、来源口径、原始到达证据与三时槽上限。"""

import json
from datetime import date, datetime

import pytest
from app.services import direction_1d_sprint as b
from app.services import direction_1d_sprint_margin_data as d


def body(start=None, end=None):
    rows = []
    for i, text in enumerate(
        ("20260907", "20260908", "20260909", "20260910", "20260911", "20260914", "20260915", "20260916")
    ):
        if start is not None and not start.strftime("%Y%m%d") <= text <= end.strftime("%Y%m%d"):
            continue
        for ex, offset in [("SSE", 1000), ("SZSE", 2000)]:
            balance = offset + 10 * i
            rows.append([text, ex, balance, 10, 4.5, 5, balance + 5])
    return json.dumps({"code": 0, "data": {"fields": d.FIELDS, "items": rows}}).encode()


def test_six_days_end_before_base_and_ignore_repayment_and_later_data():
    points = d.parse(body())
    assert d.required("2026-09-15", "2026-09-16") == [
        "2026-09-07",
        "2026-09-08",
        "2026-09-09",
        "2026-09-10",
        "2026-09-11",
        "2026-09-14",
    ]
    expected = [100 * (1050 / 1040 - 1), 5.0, 100 * (2050 / 2040 - 1), 2.5]
    assert d.features("2026-09-15", "2026-09-16", points) == pytest.approx(expected)
    points["2026-09-14"]["SSE"]["rzche"] = 999
    points["2026-09-15"]["SSE"]["rzye"] = 99999
    assert d.features("2026-09-15", "2026-09-16", points) == pytest.approx(expected)
    with pytest.raises(ValueError, match="NOT_ADJACENT"):
        d.required("2026-09-15", "2026-09-17")
    del points["2026-09-11"]
    with pytest.raises(ValueError, match="REQUIRED_DATE_MISSING"):
        d.features("2026-09-15", "2026-09-16", points)


@pytest.mark.parametrize(
    "fault", ["nan", "negative", "bool", "duplicate", "fields", "missing_exchange", "balance", "pagination", "empty"]
)
def test_invalid_schema_and_balance_pairs_fail(fault):
    raw = json.loads(body())
    data = raw["data"]
    if fault in ("nan", "negative", "bool"):
        data["items"][0][2] = {"nan": float("nan"), "negative": -1, "bool": True}[fault]
    elif fault == "duplicate":
        data["items"].append(data["items"][0])
    elif fault == "fields":
        data["fields"][0] = "date"
    elif fault == "missing_exchange":
        data["items"].pop(0)
    elif fault == "balance":
        data["items"][0][-1] += 10
    elif fault == "pagination":
        data["has_more"] = True
    else:
        data["items"] = []
    with pytest.raises(ValueError):
        d.parse(json.dumps(raw).encode())


@pytest.fixture
def ready(tmp_path, monkeypatch):
    calendar = b.calendar()
    monkeypatch.setattr(b, "ROOT", tmp_path)
    monkeypatch.setattr(b, "calendar", lambda: calendar)
    clock = [datetime(2026, 9, 16, 7, 10, tzinfo=b.ZONE)]
    monkeypatch.setattr(b, "now", lambda: clock[0])
    monkeypatch.setattr(
        b, "window", lambda at: {"status": "OPEN", "base_nav_date": "2026-09-15", "target_nav_date": "2026-09-16"}
    )
    b.save(tmp_path / "protocol.json", {"deadline_at": "2026-09-17T12:11:38+08:00"})
    calls = []
    monkeypatch.setattr(d, "fetch_margin", lambda start, end: calls.append((start, end)) or body(start, end))
    return clock, calls


def test_shared_capture_and_hashed_raw_change(ready):
    _, calls = ready
    value = d.capture(b.now())
    assert d.capture(b.now()) == value and calls == [(date(2026, 9, 7), date(2026, 9, 14))]
    assert max(value["rows"]) == "2026-09-14"
    path = d.root() / "2026-09-16/raw/0700.response.json"
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="LIVE_RAW_CHANGED"):
        d.load("2026-09-16")


def test_rehashed_parsed_balances_cannot_replace_source(ready):
    value = d.capture(b.now())
    value["rows"]["2026-09-14"]["SSE"]["rzye"] += 100
    b.save(d.root() / "2026-09-16/input.json", value, replace=True)
    with pytest.raises(ValueError, match="LIVE_PARSED_INPUT_CHANGED"):
        d.load("2026-09-16")


def test_failed_slots_cannot_repeat_and_budget_three(ready, monkeypatch):
    clock, calls = ready

    def fail(*args):
        calls.append(1)
        raise ValueError("SOURCE_UNAVAILABLE")

    monkeypatch.setattr(d, "fetch_margin", fail)
    for hour, minute in ((7, 0), (7, 30), (8, 0)):
        clock[0] = datetime(2026, 9, 16, hour, minute, tzinfo=b.ZONE)
        with pytest.raises(ValueError, match="SOURCE_UNAVAILABLE"):
            d.capture(b.now())
        with pytest.raises(ValueError, match="SLOT_OR_BUDGET_EXHAUSTED"):
            d.capture(b.now())
    assert calls == [1, 1, 1]


def test_late_source_is_preserved_but_not_input(ready, monkeypatch):
    clock, _ = ready

    def late(start, end):
        clock[0] = datetime(2026, 9, 16, 8, 30, tzinfo=b.ZONE)
        return body(start, end)

    monkeypatch.setattr(d, "fetch_margin", late)
    with pytest.raises(ValueError, match="LIVE_RESPONSE_LATE"):
        d.capture(b.now())
    assert not (d.root() / "2026-09-16/input.json").exists()


def test_late_source_readback_rejected(ready, monkeypatch):
    clock, _ = ready
    original = d.load

    def late(target):
        value = original(target)
        clock[0] = datetime(2026, 9, 16, 8, 30, tzinfo=b.ZONE)
        return value

    monkeypatch.setattr(d, "load", late)
    with pytest.raises(ValueError, match="LIVE_READBACK_LATE"):
        d.capture(b.now())
