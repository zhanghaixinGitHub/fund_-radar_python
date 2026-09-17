"""Shibor日期、百分数和基点换算、原始证据、时槽预算及截止边界。"""

import hashlib
import json
from contextlib import nullcontext
from datetime import date, datetime
from types import SimpleNamespace

import pytest
from app.integrations import tushare_sprint_shibor as client
from app.services import direction_1d_sprint as b
from app.services import direction_1d_sprint_shibor_data as d
from pydantic import SecretStr


def body(start=None, end=None):
    values = {
        "20260910": [0.8, 1.3, 1.4, 1.5, 1.8, 1.9, 2.0, 2.3],
        "20260911": [0.9, 1.4, 1.5, 1.6, 1.9, 2.0, 2.1, 2.4],
        "20260914": [1.0, 1.5, 1.6, 1.7, 2.0, 2.1, 2.2, 2.5],
        "20260915": [1.2, 1.4, 1.7, 1.8, 2.1, 2.2, 2.3, 2.4],
        "20260916": [80.0] * 8,
    }
    rows = [
        [day, *row]
        for day, row in values.items()
        if start is None or start.strftime("%Y%m%d") <= day <= end.strftime("%Y%m%d")
    ]
    return json.dumps({"code": 0, "data": {"fields": d.FIELDS, "items": rows}}).encode()


def test_unit_conversion_and_exact_adjacent_trading_dates():
    points = d.parse(body())
    assert d.required("2026-09-15", "2026-09-16") == ["2026-09-14", "2026-09-15"]
    assert d.required("2026-09-14", "2026-09-15") == ["2026-09-11", "2026-09-14"]
    expected = [20, -10, 10, -10, 1.2, 20, 70, 30]
    assert d.features("2026-09-15", "2026-09-16", points) == pytest.approx(expected)
    points["2026-09-16"]["on"] = 90
    assert d.features("2026-09-15", "2026-09-16", points) == pytest.approx(expected)
    assert d.parse(body())["2026-09-16"]["on"] == 80
    with pytest.raises(ValueError, match="NOT_ADJACENT"):
        d.required("2026-09-15", "2026-09-17")
    del points["2026-09-14"]
    with pytest.raises(ValueError, match="REQUIRED_DATE_MISSING"):
        d.features("2026-09-15", "2026-09-16", points)


@pytest.mark.parametrize("fault", ["field", "missing", "nan", "negative", "zero", "bool", "duplicate", "date", "empty"])
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
        data["items"][0][1] = {"nan": float("nan"), "negative": -1, "zero": 0, "bool": True}[fault]
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
    monkeypatch.setattr(b, "ROOT", tmp_path)
    monkeypatch.setattr(b, "calendar", lambda: calendar)
    clock = [datetime(2026, 9, 16, 7, 10, tzinfo=b.ZONE)]
    monkeypatch.setattr(b, "now", lambda: clock[0])
    monkeypatch.setattr(
        b, "window", lambda at: {"status": "OPEN", "base_nav_date": "2026-09-15", "target_nav_date": "2026-09-16"}
    )
    b.save(tmp_path / "protocol.json", {"deadline_at": "2026-09-17T12:11:38+08:00"})
    calls = []
    monkeypatch.setattr(d, "fetch_shibor", lambda start, end: calls.append((start, end)) or body(start, end))
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
    value["rows"]["2026-09-15"]["on"] += 0.1
    b.save(d.root() / "2026-09-16/input.json", value, replace=True)
    with pytest.raises(ValueError, match="LIVE_PARSED_INPUT_CHANGED"):
        d.load("2026-09-16")


def test_failed_slot_not_retried_and_budget_is_three(ready, monkeypatch):
    clock, calls = ready

    def fail(*args):
        calls.append(1)
        raise ValueError("SOURCE_UNAVAILABLE")

    monkeypatch.setattr(d, "fetch_shibor", fail)
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

    monkeypatch.setattr(d, "fetch_shibor", late)
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
    monkeypatch.setattr(b, "ROOT", tmp_path)
    folder = tmp_path / "shibor-feasibility-v1"
    folder.mkdir()
    (folder / "acquire_history.py").write_bytes(b"synthetic source")
    plan = {"script_sha256": hashlib.sha256(b"synthetic source").hexdigest()}
    b.save(folder / "history-plan.json", plan)
    points = {}
    for year in range(2021, 2027):
        start, end = f"{year}0101", f"{year}1231" if year < 2026 else "20260914"
        raw = json.dumps({"code": 0, "data": {"fields": d.FIELDS, "items": [[f"{year}0104", *([1.0] * 8)]]}}).encode()
        request = {"api_name": "shibor", "params": {"start_date": start, "end_date": end}, "plan_hash": b.digest(plan)}
        b.save(folder / f"attempt-{year}.json", request)
        (folder / f"response-{year}.json").write_bytes(raw)
        b.save(
            folder / f"receipt-{year}.json",
            {
                "request_hash": b.digest(request),
                "response_sha256": hashlib.sha256(raw).hexdigest(),
                "bytes": len(raw),
                "rows": 1,
            },
        )
        points.update(d.parse(raw))
    snapshot = {"plan_hash": b.digest(plan), "rows": points}
    b.save(folder / "history.json", snapshot)
    coverage = {"history_hash": b.digest(snapshot), "plan_hash": b.digest(plan)}
    b.save(folder / "coverage-result.json", coverage)
    assert d.history() == points
    snapshot["rows"]["2025-01-04"]["on"] = 5
    b.save(folder / "history.json", snapshot, replace=True)
    b.save(folder / "coverage-result.json", coverage | {"history_hash": b.digest(snapshot)}, replace=True)
    with pytest.raises(ValueError, match="HISTORICAL_PARSED_CHANGED"):
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
            assert json["api_name"] == "shibor"
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
            client.fetch_shibor(start, end)
    else:
        assert client.fetch_shibor(start, end) == body(start, end)
