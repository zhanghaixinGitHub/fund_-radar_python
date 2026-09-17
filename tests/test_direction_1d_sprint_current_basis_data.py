"""T日期现价差：模拟HTTP与时钟检查真实可用性边界，不访问供应商。"""

import json
from datetime import date, datetime, timedelta
from types import SimpleNamespace

import pytest
from app.services import direction_1d_sprint_current_basis_data as s

T, U = "2026-09-16", "2026-09-17"


def test_feature_uses_exact_t_and_does_not_fill_older_day(monkeypatch):
    monkeypatch.setattr(s.base, "calendar", lambda: ([date(2026, 9, 15), date(2026, 9, 16), date(2026, 9, 17)], "c"))
    rows = {"2026-09-15": {"basis_pct": 9.0}, T: {"basis_pct": 0.1}, U: {"basis_pct": 8.0}}
    assert s.extend({}, T, U, rows)["futures_basis"]["basis_pct"] == 0.1
    del rows[T]
    assert s.extend({}, T, U, rows)["futures_basis"] == {"date": T, "available": False}
    with pytest.raises(ValueError, match="ADJACENCY_INVALID"):
        s.extend({}, "2026-09-15", U, rows)


@pytest.mark.parametrize(
    "at,expected",
    [
        ("2026-09-17T07:59:59+08:00", False),
        ("2026-09-17T08:00:00+08:00", True),
        ("2026-09-17T08:29:59+08:00", True),
        ("2026-09-17T08:30:00+08:00", False),
        ("2026-09-16T08:14:00+08:00", False),
    ],
)
def test_fixed_real_capture_window(at, expected):
    assert s.in_window(T, U, datetime.fromisoformat(at)) is expected


@pytest.fixture
def fake_capture(monkeypatch, tmp_path):
    clock = [datetime.fromisoformat("2026-09-17T08:14:00+08:00")]
    calls = []
    rows = {
        "fut_daily": ["IF.CFX", "20260916", 4440, 4460, 4500, 4400, 4450, 4449, 100, 200],
        "fut_mapping": ["IF.CFX", "20260916", "IF2609.CFX"],
        "index_daily": ["000300.SH", "20260916", 4440],
    }
    status = [200]

    class Response:
        def __init__(self, api):
            self.api = api
            self.status_code = status[0]

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return None

        def iter_bytes(self):
            yield json.dumps(
                {"code": 0, "data": {"fields": s.parser.FIELDS[self.api], "items": [rows[self.api]]}}
            ).encode()

    class Client:
        def __init__(self, **_):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return None

        def stream(self, method, url, json):
            assert method == "POST" and url == "https://api.tushare.pro"
            assert json["params"]["start_date"] == json["params"]["end_date"] == "20260916"
            calls.append(json["api_name"])
            return Response(json["api_name"])

    contract = {
        "IF2609.CFX": {
            "list_date": "20260119",
            "delist_date": "20260918",
            "exchange": "CFFEX",
            "quote_unit": "指数点",
            "multiplier": 300,
        }
    }
    monkeypatch.setattr(s, "live_root", lambda: tmp_path)
    monkeypatch.setattr(s.base, "now", lambda: clock[0])
    monkeypatch.setattr(s.time, "sleep", lambda seconds: clock.__setitem__(0, clock[0] + timedelta(seconds=seconds)))
    monkeypatch.setattr(
        s,
        "get_settings",
        lambda: SimpleNamespace(
            tushare_api_url="https://api.tushare.pro",
            tushare_token=SimpleNamespace(get_secret_value=lambda: "fake-secret-12345"),
        ),
    )
    monkeypatch.setattr(s.httpx, "Client", Client)
    monkeypatch.setattr(s, "contracts", lambda: contract)
    return clock, calls, status, rows, tmp_path, contract


def test_three_actual_receipts_and_no_repeat(fake_capture):
    clock, calls, _, _, folder, _ = fake_capture
    value = s.capture(T, U, "frozen-plan")
    assert value["available"] and len(value["snapshot"]["reservations"]) == 3
    assert set(calls) == set(s.parser.FIELDS)
    assert clock[0].second >= 14
    assert s.load_live(T, U, "frozen-plan") == value
    assert s.capture(T, U, "frozen-plan") == value and len(calls) == 3
    assert all(b"fake-secret" not in p.read_bytes() for p in folder.rglob("*.*"))


def test_http_failure_keeps_explicit_missing_snapshot_and_no_retry(fake_capture):
    _, calls, status, _, _, _ = fake_capture
    status[0] = 429
    value = s.capture(T, U, "frozen-plan")
    assert not value["available"] and value["snapshot"]["points"] == {}
    assert value["errors"] and len(calls) == 1
    s.capture(T, U, "frozen-plan")
    assert len(calls) == 1


def test_reserved_interrupted_request_is_not_retried(fake_capture):
    clock, calls, _, _, folder, _ = fake_capture
    q = s.queries(T)[0]
    s.base.save(
        folder / U / "requests" / (q["api"] + ".json"),
        {"at": clock[0].isoformat(), "query": q, "plan_hash": "frozen-plan", "max_retries": 0},
    )
    value = s.capture(T, U, "frozen-plan")
    assert not calls and not value["available"] and value["errors"] == ["RESERVED_BATCH_NOT_RETRIED"]


def test_outside_window_does_not_call_http(fake_capture):
    clock, calls, *_ = fake_capture
    clock[0] -= timedelta(days=1)
    assert s.capture(T, U, "frozen-plan") is None and not calls


def test_changed_raw_response_is_rejected(fake_capture):
    *_, folder, _ = fake_capture
    s.capture(T, U, "frozen-plan")
    path = folder / U / "raw/fut_daily.json"
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="LIVE_RAW_CHANGED"):
        s.load_live(T, U, "frozen-plan")


def test_expired_or_unknown_contract_cannot_produce_feature(fake_capture):
    _, calls, _, _, _, contracts = fake_capture
    contracts["IF2609.CFX"]["delist_date"] = "20260915"
    value = s.capture(T, U, "frozen-plan")
    assert len(calls) == 3 and not value["available"]


def test_late_receipt_rejected_even_if_snapshot_hashes_rebuilt(fake_capture):
    *_, folder, _ = fake_capture
    s.capture(T, U, "frozen-plan")
    path = folder / U / "responses/index_daily.json"
    meta = s.base.read(path)
    meta["received_at"] = "2026-09-17T08:30:00+08:00"
    s.base.save(path, meta, replace=True)
    with pytest.raises(ValueError, match="LIVE_RESPONSE_LATE"):
        s.load_live(T, U, "frozen-plan")
