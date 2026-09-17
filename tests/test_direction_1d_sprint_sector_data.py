"""验证行业输入的日期/单位、证据链、只读客户端与十五次请求预算。"""

import hashlib
import json
from contextlib import nullcontext
from datetime import date, datetime
from types import SimpleNamespace

import pytest
from app.integrations import tushare_sprint_sector as client
from app.services import direction_1d_sprint as b
from app.services import direction_1d_sprint_sector_data as d
from pydantic import SecretStr


def body(code=d.CODES[0], start=None, end=None):
    values = {"20260911": (100, 99), "20260914": (102, 100), "20260915": (105, 102), "20260916": (300, 105)}
    rows = [
        [code, day, *v]
        for day, v in values.items()
        if start is None or start.strftime("%Y%m%d") <= day <= end.strftime("%Y%m%d")
    ]
    return json.dumps({"code": 0, "data": {"fields": d.FIELDS, "items": rows}}).encode()


def points():
    return {code: d.parse(body(code), code) for code in d.CODES}


def test_exact_day_percent_units_reference_and_no_future_use():
    value = points()
    x = [0.0] * 32
    x[23] = 0.01
    expected = [100 * (105 / 102 - 1 - 0.01)] * 5
    assert d.required("2026-09-14", "2026-09-15") == ["2026-09-11", "2026-09-14"]
    assert d.features(x, "2026-09-15", "2026-09-16", value) == pytest.approx(expected)
    for code in d.CODES:
        value[code]["2026-09-16"]["close"] = float("nan")
    assert d.features(x, "2026-09-15", "2026-09-16", value) == pytest.approx(expected)
    x[18] = 999
    assert d.features(x, "2026-09-15", "2026-09-16", value) == pytest.approx(expected)
    assert points()[d.CODES[0]]["2026-09-16"]["close"] == 300
    with pytest.raises(ValueError, match="NOT_ADJACENT"):
        d.features(x, "2026-09-15", "2026-09-17", value)


def test_order_clipping_missing_base_and_preclose():
    value = points()
    for i, code in enumerate(d.CODES):
        value[code]["2026-09-15"]["close"] = 102 * (1 + (i - 2) * 0.3)
    assert d.features([0.0] * 32, "2026-09-15", "2026-09-16", value) == [-20, -20, 0, 20, 20]
    value[d.CODES[0]]["2026-09-15"]["pre_close"] = 99
    with pytest.raises(ValueError, match="PRECLOSE_DISCONTINUITY"):
        d.features([0.0] * 32, "2026-09-15", "2026-09-16", value)
    del value[d.CODES[0]]["2026-09-14"]
    with pytest.raises(ValueError, match="REQUIRED_DATE_MISSING"):
        d.features([0.0] * 32, "2026-09-15", "2026-09-16", value)


@pytest.mark.parametrize(
    "fault", ["field", "code", "width", "nan", "negative", "zero", "bool", "duplicate", "date", "empty"]
)
def test_invalid_raw_is_rejected(fault):
    value = json.loads(body())
    data = value["data"]
    if fault == "field":
        data["fields"][2] = "CLOSE"
    elif fault == "code":
        data["items"][0][0] = d.CODES[1]
    elif fault == "width":
        data["items"][0].pop()
    elif fault == "duplicate":
        data["items"].append(data["items"][0])
    elif fault == "empty":
        data["items"] = []
    elif fault == "date":
        data["items"][0][1] = "2026911"
    else:
        data["items"][0][2] = {"nan": float("nan"), "negative": -1, "zero": 0, "bool": True}[fault]
    with pytest.raises(ValueError):
        d.parse(json.dumps(value).encode(), d.CODES[0])


def test_outside_request_and_size_rejected():
    with pytest.raises(ValueError, match="OUTSIDE_REQUEST"):
        d.parse(body(), d.CODES[0], "2026-09-14", "2026-09-15")
    with pytest.raises(ValueError, match="RAW_SIZE"):
        d.parse(b"x" * (d.MAX_BYTES + 1), d.CODES[0])


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
    monkeypatch.setattr(d.market, "active", lambda: {"rate_limit_per_minute": 60})
    monkeypatch.setattr(d.sleep_time, "sleep", lambda seconds: None)
    calls = []
    monkeypatch.setattr(
        d, "fetch_sector", lambda code, start, end: calls.append((code, start, end)) or body(code, start, end)
    )
    return clock, calls


def test_shared_capture_and_raw_integrity(ready):
    _, calls = ready
    value = d.capture(b.now())
    assert d.capture(b.now()) == value and len(calls) == 5
    assert all(start == date(2026, 9, 14) and end == date(2026, 9, 15) for _, start, end in calls)
    path = d.root() / f"2026-09-16/raw/0700-{d.CODES[0]}.response.json"
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="LIVE_RAW_CHANGED"):
        d.load("2026-09-16")


@pytest.mark.parametrize("fault", ["parsed", "request_date", "source", "slot", "late_input"])
def test_rehashed_inputs_cannot_override_original_evidence(ready, fault):
    value = d.capture(b.now())
    code = d.CODES[0]
    if fault == "parsed":
        value["rows"][code]["2026-09-15"]["close"] += 1
    elif fault == "request_date":
        path = d.root() / f"2026-09-16/requests/0700-{code}.json"
        request = b.read(path)
        request["required_dates"][0] = "2026-09-11"
        b.save(path, request, replace=True)
    elif fault == "source":
        value["source"] = "UNKNOWN"
    elif fault == "slot":
        value["raw_refs"][code]["slot"] = "../bad"
    else:
        value["at"] = "2026-09-16T08:30:00+08:00"
    b.save(d.root() / "2026-09-16/input.json", value, replace=True)
    with pytest.raises(ValueError):
        d.load("2026-09-16")


def test_failed_slots_not_retried_and_total_budget15(ready, monkeypatch):
    clock, calls = ready

    def fail(*args):
        calls.append(args)
        raise ValueError("SOURCE_UNAVAILABLE")

    monkeypatch.setattr(d, "fetch_sector", fail)
    for hour, minute in [(7, 0), (7, 30), (8, 0)]:
        clock[0] = datetime(2026, 9, 16, hour, minute, tzinfo=b.ZONE)
        for _ in range(2):
            with pytest.raises(ValueError, match="LIVE_INPUT_INCOMPLETE"):
                d.capture(b.now())
    assert len(calls) == 15


def test_partial_success_is_reused_within_slot(ready, monkeypatch):
    _, calls = ready

    def partial(code, start, end):
        calls.append(code)
        if code == d.CODES[1]:
            raise ValueError("SOURCE_UNAVAILABLE")
        return body(code, start, end)

    monkeypatch.setattr(d, "fetch_sector", partial)
    for _ in range(2):
        with pytest.raises(ValueError, match="LIVE_INPUT_INCOMPLETE"):
            d.capture(b.now())
    assert len(calls) == 5


def test_late_response_preserved_but_not_eligible(ready, monkeypatch):
    clock, _ = ready

    def late(code, start, end):
        clock[0] = datetime(2026, 9, 16, 8, 30, tzinfo=b.ZONE)
        return body(code, start, end)

    monkeypatch.setattr(d, "fetch_sector", late)
    with pytest.raises(ValueError, match="LIVE_INPUT_INCOMPLETE"):
        d.capture(b.now())
    assert (d.root() / f"2026-09-16/raw/0700-{d.CODES[0]}.response.json").exists()
    assert not (d.root() / "2026-09-16/input.json").exists()


def test_readback_late_rejected(ready, monkeypatch):
    clock, _ = ready
    original = d.load

    def late(target):
        value = original(target)
        clock[0] = datetime(2026, 9, 16, 8, 30, tzinfo=b.ZONE)
        return value

    monkeypatch.setattr(d, "load", late)
    with pytest.raises(ValueError, match="LIVE_READBACK_LATE"):
        d.capture(b.now())


@pytest.mark.parametrize(
    "status,limit,error", [(200, 1048576, None), (301, 1048576, "HTTP_FAILED"), (200, 3, "RESPONSE_LIMIT")]
)
def test_client_official_query_and_byte_limits(monkeypatch, status, limit, error):
    code, start, end = d.CODES[0], date(2026, 9, 14), date(2026, 9, 15)

    class Response:
        status_code = status

        def iter_bytes(self):
            yield body(code, start, end)

    class Client:
        def __init__(self, **kwargs):
            assert kwargs["follow_redirects"] is False

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def stream(self, method, url, json):
            assert method == "POST" and url == client.URL
            assert json["api_name"] == "index_daily"
            assert json["params"] == {"ts_code": code, "start_date": "20260914", "end_date": "20260915"}
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
            client.fetch_sector(code, start, end)
    else:
        assert client.fetch_sector(code, start, end) == body(code, start, end)


@pytest.mark.parametrize(
    "url", ["http://api.tushare.pro", "https://api.tushare.pro.evil.invalid", "https://api.tushare.pro/?x=1"]
)
def test_client_rejects_other_endpoint_before_network(monkeypatch, url):
    monkeypatch.setattr(
        client, "get_settings", lambda: SimpleNamespace(tushare_api_url=url, tushare_token=SecretStr("test-only-token"))
    )
    monkeypatch.setattr(client.httpx, "Client", lambda **kwargs: pytest.fail("network unexpectedly called"))
    with pytest.raises(ValueError, match="OFFICIAL_ENDPOINT"):
        client.fetch_sector(d.CODES[0], date(2026, 9, 14), date(2026, 9, 15))


def test_historical_snapshot_rehash_cannot_replace_raw(tmp_path, monkeypatch):
    calendar = b.calendar()
    monkeypatch.setattr(b, "ROOT", tmp_path)
    monkeypatch.setattr(b, "calendar", lambda: calendar)
    folder = tmp_path / "sector-data-feasibility-v1"
    folder.mkdir()
    for name in ("acquire.py", "check_coverage.py"):
        (folder / name).write_bytes(b"synthetic-script")
    plan = {"codes": list(d.CODES), "script_sha256": hashlib.sha256(b"synthetic-script").hexdigest()}
    cp = {"script_sha256": plan["script_sha256"], "acquisition_plan_hash": b.digest(plan), "calendar_hash": calendar[1]}
    merged = {}
    for code in d.CODES:
        merged[code] = {}
        pieces = [("probe-" + code, "20250102", "20250110", "20250106")]
        pieces += [
            (f"{code}-{y}", f"{y}0101", f"{y}1231" if y < 2026 else "20260914", f"{y}0106") for y in range(2021, 2027)
        ]
        for label, start, end, day in pieces:
            request = {
                "api_name": "index_daily",
                "plan_hash": b.digest(plan),
                "params": {"ts_code": code, "start_date": start, "end_date": end},
            }
            raw = json.dumps({"code": 0, "data": {"fields": d.FIELDS, "items": [[code, day, 100, 99]]}}).encode()
            b.save(folder / f"attempt-{label}.json", request)
            (folder / f"response-{label}.json").write_bytes(raw)
            b.save(
                folder / f"receipt-{label}.json",
                {
                    "request_hash": b.digest(request),
                    "sha256": hashlib.sha256(raw).hexdigest(),
                    "bytes": len(raw),
                    "rows": 1,
                    "provider_code": 0,
                },
            )
            if not label.startswith("probe-"):
                merged[code].update(d.parse(raw, code))
    snapshot = {"plan_hash": b.digest(plan), "points": merged}
    coverage = {
        "plan_hash": b.digest(cp),
        "status": "FULL_COVERAGE_NOT_TRAINED",
        "history_hash": b.digest(snapshot),
        "features_hash": "fixture",
    }
    proposal = {
        "coverage_result_hash": b.digest(coverage),
        "history_hash": b.digest(snapshot),
        "features_hash": "fixture",
    }
    for name, value in (("plan", plan), ("coverage-plan", cp), ("history", snapshot), ("coverage-result", coverage)):
        b.save(folder / (name + ".json"), value)
    b.save(tmp_path / "round-37/proposal-before-training.json", proposal)
    assert d.history() == merged
    snapshot["points"][d.CODES[0]]["2025-01-06"]["close"] = 123
    coverage["history_hash"] = b.digest(snapshot)
    proposal.update(history_hash=b.digest(snapshot), coverage_result_hash=b.digest(coverage))
    b.save(folder / "history.json", snapshot, replace=True)
    b.save(folder / "coverage-result.json", coverage, replace=True)
    b.save(tmp_path / "round-37/proposal-before-training.json", proposal, replace=True)
    with pytest.raises(ValueError, match="HISTORICAL_PARSED_CHANGED"):
        d.history()
