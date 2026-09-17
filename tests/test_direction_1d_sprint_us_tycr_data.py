"""美债日期、百分数和基点换算、原始证据、时槽预算及截止边界。"""

import json
from contextlib import nullcontext
from datetime import date, datetime
from types import SimpleNamespace

import pytest
from app.integrations import tushare_sprint_us_tycr as client
from app.services import direction_1d_sprint as b
from app.services import direction_1d_sprint_us_tycr_data as d
from pydantic import SecretStr


def body(start=None, end=None):
    values = {
        "20260910": [4.0, 4.4, 4.5, 4.8, 5.0],
        "20260911": [4.1, 4.5, 4.6, 4.9, 5.1],
        "20260914": [4.2, 4.6, 4.7, 5.0, 5.2],
        "20260915": [4.3, 4.5, 4.7, 5.1, 5.4],
        "20260916": [20.0] * 5,
    }
    rows = [
        [day, *row]
        for day, row in values.items()
        if start is None or start.strftime("%Y%m%d") <= day <= end.strftime("%Y%m%d")
    ]
    return json.dumps({"code": 0, "data": {"fields": d.FIELDS, "items": rows}}).encode()


def test_unit_conversion_and_new_york_publication_timing():
    points = d.parse(body())
    assert d.required("2026-09-15", "2026-09-16") == ["2026-09-14", "2026-09-15"]
    expected = [10, -10, 0, 10, 20, 60, 80, 5.1, 1]
    assert d.features("2026-09-15", "2026-09-16", points) == pytest.approx(expected)
    points["2026-09-16"]["m3"] = 25
    assert d.features("2026-09-15", "2026-09-16", points) == pytest.approx(expected)
    assert d.publication("2025-01-08").astimezone(b.ZONE).hour == 7
    assert d.publication("2025-07-08").astimezone(b.ZONE).hour == 6
    assert d.required("2025-01-09", "2025-01-10")[-1] == "2025-01-09"
    with pytest.raises(ValueError, match="NOT_ADJACENT"):
        d.required("2026-09-15", "2026-09-17")
    del points["2026-09-15"]
    with pytest.raises(ValueError, match="REQUIRED_DATE_MISSING"):
        d.features("2026-09-15", "2026-09-16", points)


def test_zero_nominal_yield_is_valid_and_paginated_response_rejected():
    raw = json.loads(body())
    raw["data"]["items"][0][1] = 0.0
    assert d.parse(json.dumps(raw).encode())["2026-09-10"]["m3"] == 0
    raw["data"]["has_more"] = True
    with pytest.raises(ValueError):
        d.parse(json.dumps(raw).encode())


@pytest.mark.parametrize("fault", ["field", "missing", "nan", "negative", "high", "bool", "duplicate", "date", "empty"])
def test_invalid_provider_response_is_rejected(fault):
    value = json.loads(body())
    data = value["data"]
    if fault == "field":
        data["fields"][1] = "ON"
    elif fault == "missing":
        data["items"][0].pop()
    elif fault == "duplicate":
        data["items"].append(data["items"][0])
    elif fault == "empty":
        data["items"] = []
    elif fault == "date":
        data["items"][0][0] = "2026910"
    else:
        data["items"][0][1] = {"nan": float("nan"), "negative": -1, "high": 31, "bool": True}[fault]
    with pytest.raises(ValueError):
        d.parse(json.dumps(value).encode())


def test_outside_query_dates_and_raw_size_are_rejected():
    with pytest.raises(ValueError, match="OUTSIDE_REQUEST"):
        d.parse(body(), "2026-09-14", "2026-09-15")
    with pytest.raises(ValueError, match="RAW_SIZE"):
        d.parse(b"x" * (d.MAX_BYTES + 1))


@pytest.fixture
def ready(tmp_path, monkeypatch):
    calendar = b.calendar()
    rate_calendar = b.read(d.source_folder() / "calendar.json")
    monkeypatch.setattr(b, "ROOT", tmp_path)
    monkeypatch.setattr(b, "calendar", lambda: calendar)
    b.save(d.source_folder() / "calendar.json", rate_calendar)
    clock = [datetime(2026, 9, 16, 7, 10, tzinfo=b.ZONE)]
    monkeypatch.setattr(b, "now", lambda: clock[0])
    monkeypatch.setattr(
        b, "window", lambda at: {"status": "OPEN", "base_nav_date": "2026-09-15", "target_nav_date": "2026-09-16"}
    )
    b.save(tmp_path / "protocol.json", {"deadline_at": "2026-09-17T12:11:38+08:00"})
    calls = []
    monkeypatch.setattr(d, "fetch_us_tycr", lambda start, end: calls.append((start, end)) or body(start, end))
    return clock, calls


def test_shared_capture_and_changed_raw_are_checked(ready):
    _, calls = ready
    value = d.capture(b.now())
    assert d.capture(b.now()) == value
    assert calls == [(date(2026, 9, 14), date(2026, 9, 15))]
    path = d.root() / "2026-09-16/raw/0700.response.json"
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="LIVE_RAW_CHANGED"):
        d.load("2026-09-16")


def test_rehashed_parsed_values_do_not_replace_original(ready):
    value = d.capture(b.now())
    value["rows"]["2026-09-15"]["m3"] += 0.1
    b.save(d.root() / "2026-09-16/input.json", value, replace=True)
    with pytest.raises(ValueError, match="LIVE_PARSED_INPUT_CHANGED"):
        d.load("2026-09-16")


def test_failed_slot_not_retried_and_budget_is_three(ready, monkeypatch):
    clock, calls = ready

    def fail(*args):
        calls.append(1)
        raise ValueError("SOURCE_UNAVAILABLE")

    monkeypatch.setattr(d, "fetch_us_tycr", fail)
    for hour, minute in [(7, 0), (7, 30), (8, 0)]:
        clock[0] = datetime(2026, 9, 16, hour, minute, tzinfo=b.ZONE)
        with pytest.raises(ValueError, match="SOURCE_UNAVAILABLE"):
            d.capture(b.now())
        with pytest.raises(ValueError, match="SLOT_OR_BUDGET_EXHAUSTED"):
            d.capture(b.now())
    assert calls == [1, 1, 1]


def test_response_after_deadline_is_preserved_but_ineligible(ready, monkeypatch):
    clock, _ = ready

    def late(start, end):
        clock[0] = datetime(2026, 9, 16, 8, 30, tzinfo=b.ZONE)
        return body(start, end)

    monkeypatch.setattr(d, "fetch_us_tycr", late)
    with pytest.raises(ValueError, match="LIVE_RESPONSE_LATE"):
        d.capture(b.now())
    assert (d.root() / "2026-09-16/raw/0700.response.json").exists()
    assert not (d.root() / "2026-09-16/input.json").exists()


def test_source_readback_crossing_deadline_is_rejected(ready, monkeypatch):
    clock, _ = ready
    original = d.load

    def late(target):
        value = original(target)
        clock[0] = datetime(2026, 9, 16, 8, 30, tzinfo=b.ZONE)
        return value

    monkeypatch.setattr(d, "load", late)
    with pytest.raises(ValueError, match="LIVE_READBACK_LATE"):
        d.capture(b.now())


def test_history_snapshot_cannot_be_changed_by_rehashing(tmp_path, monkeypatch):
    points = d.raw_history()
    source = d.source_folder()
    snapshot = b.read(source / "history.json")
    qualified = b.read(source / "qualification-result.json")
    calendar = b.read(source / "calendar.json")
    monkeypatch.setattr(b, "ROOT", tmp_path)
    monkeypatch.setattr(d, "raw_history", lambda: points)
    b.save(d.source_folder() / "history.json", snapshot)
    b.save(d.source_folder() / "qualification-result.json", qualified)
    b.save(d.source_folder() / "calendar.json", calendar)
    assert d.history() == points
    snapshot["rows"]["2025-01-02"]["m3"] += 0.1
    b.save(d.source_folder() / "history.json", snapshot, replace=True)
    b.save(
        d.source_folder() / "qualification-result.json", qualified | {"history_hash": b.digest(snapshot)}, replace=True
    )
    with pytest.raises(ValueError, match="HISTORY_CHANGED"):
        d.history()


@pytest.mark.parametrize(
    "status,limit,error", [(200, 524288, None), (301, 524288, "HTTP_FAILED"), (200, 3, "RESPONSE_LIMIT")]
)
def test_client_endpoint_query_and_size_limits(monkeypatch, status, limit, error):
    start, end = date(2026, 9, 14), date(2026, 9, 15)

    class Response:
        status_code = status

        def iter_bytes(self):
            yield body(start, end)

    class Client:
        def __init__(self, **kwargs):
            assert kwargs["follow_redirects"] is False

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def stream(self, method, url, json):
            assert method == "POST" and url == "https://api.tushare.pro"
            assert json["api_name"] == "us_tycr"
            assert json["params"] == {"start_date": "20260914", "end_date": "20260915"}
            return nullcontext(Response())

    monkeypatch.setattr(
        client,
        "get_settings",
        lambda: SimpleNamespace(tushare_api_url=client.URL, tushare_token=SecretStr("test-only-token")),
    )
    monkeypatch.setattr(client.httpx, "Client", Client)
    monkeypatch.setattr(client, "MAX_BYTES", limit)
    if error:
        with pytest.raises(ValueError, match=error):
            client.fetch_us_tycr(start, end)
    else:
        assert client.fetch_us_tycr(start, end) == body(start, end)
