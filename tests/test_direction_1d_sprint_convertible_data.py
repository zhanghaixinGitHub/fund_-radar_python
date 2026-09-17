"""自身基准特征的单位、历史边界与真实收盘时点。"""

import hashlib
import json
from contextlib import nullcontext
from datetime import date, datetime
from types import SimpleNamespace

import pytest
from app.integrations import tushare_sprint_convertible as client
from app.services import direction_1d_sprint as b
from app.services import direction_1d_sprint_convertible_data as d
from pydantic import SecretStr


def points():
    dates = d.required("2026-09-14", "2026-09-15")
    return {code: {day: {"close": 100.0 + i, "pre_close": 99.0 + i} for i, day in enumerate(dates)} for code in d.CODES}


def test_composite_weights_are_applied_daily_before_five_day_compounding():
    dates = d.required("2026-09-14", "2026-09-15")
    p = {}
    for code, r in zip(d.CODES, [0.01, 0.02, 0.001], strict=True):
        p[code] = {
            day: {"close": 100 * (1 + r) ** i, "pre_close": 100 * (1 + r) ** (i - 1)} for i, day in enumerate(dates)
        }
    x = [0.0] * 32
    x[7], x[0], x[23], x[24] = 0.01, 0.05, 0.005, 0.02
    one = 0.8 * 0.01 + 0.1 * 0.02 + 0.1 * 0.001
    five = (1 + one) ** 5 - 1
    expected = [(one - 0.005) * 100, (five - 0.02) * 100, (0.01 - one) * 100, (0.05 - five) * 100, 1.0]
    assert d.features(x, "005284", "2026-09-14", "2026-09-15", p) == pytest.approx(expected)
    wrong = 0.8 * ((1.01) ** 5 - 1) + 0.1 * ((1.02) ** 5 - 1) + 0.1 * ((1.001) ** 5 - 1)
    assert abs(five - wrong) > 1e-5


def test_target_prices_cannot_enter_features():
    p = points()
    before = d.features([0.0] * 32, "005284", "2026-09-14", "2026-09-15", p)
    p[d.CODES[0]]["2026-09-15"] = {"close": float("nan"), "pre_close": -999}
    assert d.features([0.0] * 32, "005284", "2026-09-14", "2026-09-15", p) == before
    with pytest.raises(ValueError, match="NOT_ADJACENT"):
        d.features([0.0] * 32, "005284", "2026-09-14", "2026-09-16", p)


def test_missing_or_discontinuous_history_does_not_forward_fill():
    p = points()
    p[d.CODES[0]]["2026-09-14"]["pre_close"] = 1
    with pytest.raises(ValueError, match="DISCONTINUITY"):
        d.features([0.0] * 32, "005284", "2026-09-14", "2026-09-15", p)
    p = points()
    del p[d.CODES[0]]["2026-09-14"]
    with pytest.raises(ValueError, match="MISSING"):
        d.features([0.0] * 32, "005284", "2026-09-14", "2026-09-15", p)


def test_exact_provider_code_and_raw_schema():
    raw = json.dumps(
        {"code": 0, "data": {"fields": d.FIELDS, "items": [[d.CODES[0], "20260914", 105.0, 104.0]]}}
    ).encode()
    assert d.parse(raw, d.CODES[0])["2026-09-14"]["close"] == 105
    with pytest.raises(ValueError):
        d.parse(raw, "000990.SH")
    with pytest.raises(ValueError):
        client.fetch_convertible("000990.SH", date(2026, 9, 1), date(2026, 9, 14))


def body(code, start, end):
    days = [d for d in b.calendar()[0] if start <= d <= end]
    return json.dumps(
        {
            "code": 0,
            "data": {
                "fields": d.FIELDS,
                "items": [[code, day.strftime("%Y%m%d"), 100.0 + i, 99.0 + i] for i, day in enumerate(days)],
            },
        }
    ).encode()


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
        d, "fetch_convertible", lambda code, start, end: calls.append((code, start, end)) or body(code, start, end)
    )
    return clock, calls


def test_shared_capture_and_raw_integrity(ready):
    _, calls = ready
    value = d.capture(b.now())
    assert d.capture(b.now()) == value and len(calls) == 3
    assert all(
        start == date.fromisoformat(d.required("2026-09-15", "2026-09-16")[0]) and end == date(2026, 9, 15)
        for _, start, end in calls
    )
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


def test_failed_slots_not_retried_and_total_budget9(ready, monkeypatch):
    clock, calls = ready

    def fail(*args):
        calls.append(args)
        raise ValueError("SOURCE_UNAVAILABLE")

    monkeypatch.setattr(d, "fetch_convertible", fail)
    for hour, minute in [(7, 0), (7, 30), (8, 0)]:
        clock[0] = datetime(2026, 9, 16, hour, minute, tzinfo=b.ZONE)
        for _ in range(2):
            with pytest.raises(ValueError, match="LIVE_INPUT_INCOMPLETE"):
                d.capture(b.now())
    assert len(calls) == 9


def test_late_response_preserved_but_not_eligible(ready, monkeypatch):
    clock, _ = ready

    def late(code, start, end):
        clock[0] = datetime(2026, 9, 16, 8, 30, tzinfo=b.ZONE)
        return body(code, start, end)

    monkeypatch.setattr(d, "fetch_convertible", late)
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
            client.fetch_convertible(code, start, end)
    else:
        assert client.fetch_convertible(code, start, end) == body(code, start, end)


@pytest.mark.parametrize(
    "url", ["http://api.tushare.pro", "https://api.tushare.pro.evil.invalid", "https://api.tushare.pro/?x=1"]
)
def test_client_rejects_other_endpoint_before_network(monkeypatch, url):
    monkeypatch.setattr(
        client, "get_settings", lambda: SimpleNamespace(tushare_api_url=url, tushare_token=SecretStr("test-only-token"))
    )
    monkeypatch.setattr(client.httpx, "Client", lambda **kwargs: pytest.fail("network unexpectedly called"))
    with pytest.raises(ValueError, match="OFFICIAL_ENDPOINT"):
        client.fetch_convertible(d.CODES[0], date(2026, 9, 14), date(2026, 9, 15))


def test_two_source_history_rehash_cannot_replace_raw_prices(tmp_path, monkeypatch):
    monkeypatch.setattr(b, "ROOT", tmp_path)
    names = {"000832.CSI": "中证可转换债券指数", "000906.SH": "中证800指数", "H11001.CSI": "中证全债指数"}
    for supplement, codes in ((False, [d.CODES[0], d.CODES[2]]), (True, [d.CODES[1]])):
        folder = tmp_path / ("convertible-csi800-supplement-v1" if supplement else "convertible-benchmark-data-v1")
        folder.mkdir()
        (folder / "acquire.py").write_bytes(b"synthetic")
        plan = {"script_sha256": hashlib.sha256(b"synthetic").hexdigest()}
        identities = {"plan_hash": b.digest(plan)}
        rows = {code.split(".")[0]: {"ts_code": code, "fullname": names[code]} for code in codes}
        identities["row" if supplement else "rows"] = rows["000906"] if supplement else rows
        histories = {}
        for code in codes:
            symbol = code.split(".")[0]
            pieces = [("probe" if supplement else "probe-" + symbol, "20250102", "20250110", "20250106")]
            pieces += [
                (
                    str(y) if supplement else f"{symbol}-{y}",
                    f"{y}0101",
                    f"{y}1231" if y < 2026 else "20260914",
                    f"{y}0106",
                )
                for y in range(2021, 2027)
            ]
            points = {}
            for label, start, end, day in pieces:
                request = {
                    "plan_hash": b.digest(plan),
                    "url": d.URL,
                    "api": "index_daily",
                    "fields": d.FIELDS,
                    "params": {"ts_code": code, "start_date": start, "end_date": end},
                }
                raw = json.dumps(
                    {"code": 0, "data": {"fields": d.FIELDS, "items": [[code, day, 100.0, 99.0]]}}
                ).encode()
                b.save(folder / f"attempt-{label}.json", request)
                (folder / f"response-{label}.json").write_bytes(raw)
                b.save(
                    folder / f"receipt-{label}.json",
                    {
                        "request_hash": b.digest(request),
                        "status": "OK",
                        "bytes": len(raw),
                        "sha256": hashlib.sha256(raw).hexdigest(),
                    },
                )
                if not label.startswith("probe"):
                    points.update(d.parse(raw, code))
            histories[code] = points
        snapshot = {"plan_hash": b.digest(plan), "points": histories[codes[0]] if supplement else histories}
        result = {
            "plan_hash": b.digest(plan),
            "history_hash": b.digest(snapshot),
            "status": "ACQUIRED_NOT_TRAINED",
            "series": {code.split(".")[0]: {"status": "ACQUIRED"} for code in codes},
        }
        for name, value in (
            ("plan", plan),
            ("identity" if supplement else "identities", identities),
            ("history", snapshot),
            ("result", result),
        ):
            b.save(folder / f"{name}.json", value)
    assert len(d.history()) == 3
    snapshot["points"]["2025-01-06"]["close"] = 123
    result["history_hash"] = b.digest(snapshot)
    b.save(folder / "history.json", snapshot, replace=True)
    b.save(folder / "result.json", result, replace=True)
    with pytest.raises(ValueError, match="HISTORICAL_PARSED_CHANGED"):
        d.history()
