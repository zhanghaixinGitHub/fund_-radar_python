"""具体IF期限价差源的实际时间与单次请求边界；HTTP均为模拟，不访问供应商。"""

import json
from datetime import date, datetime
from types import SimpleNamespace

import pytest
from app.services import direction_1d_sprint_futures_curve_data as s

T, U = "2026-09-16", "2026-09-17"


def test_exact_t_source_and_no_old_day_fill(monkeypatch):
    monkeypatch.setattr(s.base, "calendar", lambda: ([date(2026, 9, 15), date(2026, 9, 16), date(2026, 9, 17)], "c"))
    original = {"available": True, "core": 2}
    values = {T: {"date": T, "intraday_return_pct": 0.1, "close_location": -0.2, "annualized_spread_pct": -5.0}}
    market = s.extend(original, T, U, values)
    assert market["futures_curve"]["date"] == T
    assert s.original_row({"y": 1, "market": market}) == {"y": 1, "market": original}
    assert not s.extend(original, "2026-09-15", T, values)["futures_curve"]["available"]
    with pytest.raises(ValueError, match="ADJACENCY_INVALID"):
        s.extend(original, "2026-09-15", U, values)


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
def test_fixed_capture_window(at, expected):
    assert s.in_window(T, U, datetime.fromisoformat(at)) is expected


@pytest.fixture
def fake_capture(monkeypatch, tmp_path):
    clock = [datetime.fromisoformat("2026-09-17T08:14:00+08:00")]
    calls, status, late = [], [200], [False]
    rows = [
        ["IF2609.CFX", "20260916", 4410, 4460, 4400, 4440, 100, 200],
        ["IF2610.CFX", "20260916", 4400, 4460, 4380, 4430, 50, 300],
    ]

    class Response:
        def __init__(self):
            self.status_code = status[0]

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return None

        def iter_bytes(self):
            if late[0]:
                clock[0] = datetime.fromisoformat("2026-09-17T08:30:00+08:00")
            yield json.dumps({"code": 0, "data": {"fields": s.parser.FIELDS, "items": rows}}).encode()

    class Client:
        def __init__(self, **_):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return None

        def stream(self, method, url, json):
            assert method == "POST" and url == "https://api.tushare.pro"
            assert json["api_name"] == "fut_daily" and json["params"] == {"exchange": "CFFEX", "trade_date": "20260916"}
            calls.append(json["api_name"])
            return Response()

    meta = [
        {
            "ts_code": code,
            "exchange": "CFFEX",
            "fut_code": "IF",
            "multiplier": 300,
            "quote_unit": "指数点",
            "list_date": "20260101",
            "delist_date": expiry,
        }
        for code, expiry in [
            ("IF2609.CFX", "20260918"),
            ("IF2610.CFX", "20261016"),
            ("IF2612.CFX", "20261218"),
            ("IF2703.CFX", "20270319"),
        ]
    ]
    monkeypatch.setattr(s.curve, "samples", lambda: ({}, {"basic": meta}))
    monkeypatch.setattr(s, "live_root", lambda: tmp_path)
    monkeypatch.setattr(s.base, "now", lambda: clock[0])
    monkeypatch.setattr(
        s,
        "get_settings",
        lambda: SimpleNamespace(
            tushare_api_url="https://api.tushare.pro",
            tushare_token=SimpleNamespace(get_secret_value=lambda: "fake-private-token"),
        ),
    )
    monkeypatch.setattr(s.httpx, "Client", Client)
    return clock, calls, status, rows, tmp_path, late


def test_single_capture_and_immutable_reuse(fake_capture):
    _, calls, _, _, folder, _ = fake_capture
    value = s.capture(T, U, "plan")
    assert value["available"] and list(value["snapshot"]["points"]) == [T]
    assert s.load_live(T, U, "plan") == value
    assert s.capture(T, U, "plan") == value and len(calls) == 1
    assert all(b"fake-private-token" not in p.read_bytes() for p in folder.rglob("*.json"))


def test_provider_failure_is_explicit_missing_and_not_retried(fake_capture):
    _, calls, status, *_ = fake_capture
    status[0] = 403
    value = s.capture(T, U, "plan")
    assert not value["available"] and value["errors"]
    s.capture(T, U, "plan")
    assert len(calls) == 1


def test_missing_far_contract_is_not_zero_curve(fake_capture):
    _, calls, _, rows, *_ = fake_capture
    rows.pop()
    value = s.capture(T, U, "plan")
    assert len(calls) == 1 and not value["available"] and value["snapshot"]["points"] == {}


def test_interrupted_reserved_request_not_retried(fake_capture):
    clock, calls, _, _, folder, _ = fake_capture
    s.base.save(
        folder / U / "request.json",
        {"at": clock[0].isoformat(), "query": s.query(T), "plan_hash": "plan", "max_retries": 0},
    )
    value = s.capture(T, U, "plan")
    assert not calls and not value["available"] and value["errors"] == ["RESERVED_REQUEST_NOT_RETRIED"]


def test_previous_day_does_not_start_request(fake_capture):
    clock, calls, *_ = fake_capture
    clock[0] = datetime.fromisoformat("2026-09-16T08:14:00+08:00")
    assert s.capture(T, U, "plan") is None and not calls


def test_late_actual_response_never_becomes_forecast_source(fake_capture):
    _, calls, _, _, folder, late = fake_capture
    late[0] = True
    assert s.capture(T, U, "plan") is None and len(calls) == 1
    assert (folder / U / "response.json").exists() and not (folder / U / "snapshot.json").exists()


def test_changed_raw_is_rejected(fake_capture):
    *_, folder, _ = fake_capture
    s.capture(T, U, "plan")
    path = folder / U / "raw.json"
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="LIVE_RAW_CHANGED"):
        s.load_live(T, U, "plan")


def test_duplicate_concrete_contract_rejected(fake_capture):
    _, _, _, rows, *_ = fake_capture
    rows.append(list(rows[0]))
    value = s.capture(T, U, "plan")
    assert not value["available"]


def test_continuous_alias_cannot_replace_far_contract(fake_capture):
    _, _, _, rows, *_ = fake_capture
    rows[1][0] = "IFL1.CFX"
    assert not s.capture(T, U, "plan")["available"]
