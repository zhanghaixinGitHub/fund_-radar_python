"""独立SPX来源、时槽限流和市场输入；不创建NAV父预测即可运行。"""

import json
from datetime import datetime

import pytest
from app.services import direction_1d_sprint as b
from app.services import direction_1d_sprint_market_only_data as d


@pytest.fixture
def ready(tmp_path, monkeypatch):
    calendar, sessions = b.calendar(), d.overnight.sessions()
    monkeypatch.setattr(b, "ROOT", tmp_path)
    monkeypatch.setattr(b, "calendar", lambda: calendar)
    monkeypatch.setattr(d.overnight, "sessions", lambda: sessions)
    clock = [datetime(2026, 9, 16, 7, 10, tzinfo=b.ZONE)]
    monkeypatch.setattr(b, "now", lambda: clock[0])
    b.save(tmp_path / "protocol.json", {"deadline_at": "2026-09-17T12:11:38+08:00"})
    monkeypatch.setattr(d.overnight, "source", lambda: None)
    calls = []
    response = json.dumps(
        {
            "code": 0,
            "data": {
                "fields": ["ts_code", "trade_date", "close", "pre_close", "pct_chg"],
                "items": [["SPX", "20260914", 6000, 6000, 0], ["SPX", "20260915", 6060, 6000, 1]],
            },
        }
    )
    monkeypatch.setattr(d.overnight, "fetch_spx", lambda *args: calls.append(1) or response)
    return clock, calls


def test_independent_capture_needs_no_nav_parent_and_reuses_one_response(ready):
    _, calls = ready
    result = d.capture_spx(b.now(), "2026-09-15", "2026-09-16")
    assert result[1]["origin"] == "MARKET_ONLY"
    assert d.capture_spx(b.now(), "2026-09-15", "2026-09-16") == result
    assert calls == [1] and not (b.ROOT / "forward").exists()


def test_old_slot_reservation_prevents_duplicate_request(ready):
    _, calls = ready
    b.save(d.overnight.root() / "live/2026-09-16/0700-reserved.json", {"at": b.now().isoformat()})
    assert d.capture_spx(b.now(), "2026-09-15", "2026-09-16") is None
    assert calls == []


def test_valid_old_spx_response_is_shared_without_extra_fetch(ready):
    _, calls = ready
    aligned = d.overnight.alignment("2026-09-15", "2026-09-16")
    body = d.overnight.fetch_spx(None, None)
    calls.clear()
    folder = d.overnight.root() / "live/2026-09-16"
    b.save(folder / "0700-reserved.json", {"at": b.now().isoformat(), "plan": aligned})
    b.save(
        folder / "0700-response.json",
        {
            "received_at": b.now().isoformat(),
            "plan": aligned,
            "response": json.loads(body),
            "parsed": d.overnight.validate_response(body, aligned["required_us_dates"]),
        },
    )
    assert d.capture_spx(b.now(), "2026-09-15", "2026-09-16")[1]["origin"] == "ROUND3"
    assert calls == []


@pytest.mark.parametrize("fault", ["bytes", "parsed", "late", "before_close"])
def test_raw_reparse_and_time_boundaries_reject_mutation(ready, fault):
    d.capture_spx(b.now(), "2026-09-15", "2026-09-16")
    folder = d.root() / "2026-09-16/spx"
    value = b.read(folder / "0700-response.json")
    if fault == "bytes":
        (folder / "0700-response.bin").write_bytes(b"{}")
    elif fault == "parsed":
        value["parsed"]["status"] = "INCOMPLETE"
    else:
        value["received_at"] = f"2026-09-16T{'08:30:00' if fault == 'late' else '03:00:00'}+08:00"
    b.save(folder / "0700-response.json", value, replace=True)
    with pytest.raises(ValueError):
        d.spx_observation("2026-09-16", "2026-09-15", "MARKET_ONLY", "0700")


def test_deadline_stops_source_calls_even_inside_morning_slot(ready):
    clock, calls = ready
    b.save(b.ROOT / "protocol.json", {"deadline_at": clock[0].isoformat()}, replace=True)
    assert d.capture(clock[0]) is None
    assert d.capture_spx(clock[0], "2026-09-15", "2026-09-16") is None
    assert calls == []


def test_spx_missing_response_not_retried_in_same_slot(ready, monkeypatch):
    _, calls = ready
    monkeypatch.setattr(
        d.overnight, "fetch_spx", lambda *args: calls.append(1) or '{"code":0,"data":{"fields":[],"items":[]}}'
    )
    assert d.capture_spx(b.now(), "2026-09-15", "2026-09-16") is None
    assert d.capture_spx(b.now(), "2026-09-15", "2026-09-16") is None
    assert calls == [1]


def test_feature_builder_does_not_read_or_depend_on_fund_nav(ready, monkeypatch):
    monkeypatch.setattr(d.etfs, "features", lambda t, u, rows: [2, 0, 0, 0, 1])
    monkeypatch.setattr(d.cnya, "features", lambda t, u, rows: [3, 0, 1])
    parsed = d.capture_spx(b.now(), "2026-09-15", "2026-09-16")[0]["parsed"]["rows"]
    first = d.feature_values("2026-09-15", "2026-09-16", parsed, {}, {})
    b.save(b.ROOT / "history.json", {"funds": [{"rows": [{"nav": "bad", "ann_date": "2099-01-01"}]}]})
    assert d.feature_values("2026-09-15", "2026-09-16", parsed, {}, {}) == first
    assert first["features"] == pytest.approx([1, 2, 3])


def test_complete_market_input_binds_all_sources_without_history(ready, monkeypatch):
    _, calls = ready
    e = {"at": b.now().isoformat(), "base": "2026-09-15", "target": "2026-09-16", "rows": {"TEST": "ETF"}}
    c = e | {"rows": {"TEST": "CNYA"}}
    monkeypatch.setattr(d.etfs, "capture", lambda at: e)
    monkeypatch.setattr(d.etfs, "load", lambda target: e)
    monkeypatch.setattr(d.cnya, "capture", lambda at: c)
    monkeypatch.setattr(d.cnya, "load", lambda target: c)
    monkeypatch.setattr(d.etfs, "features", lambda t, u, rows: [2, 0, 0, 0, 1])
    monkeypatch.setattr(d.cnya, "features", lambda t, u, rows: [3, 0, 1])
    value = d.capture(b.now())
    assert value["market"]["features"] == pytest.approx([1, 2, 3])
    assert value["fund_nav_used_as_predictor"] is False and calls == [1]
    assert not (b.ROOT / "history.json").exists()
    e["rows"]["CHANGED"] = 1
    with pytest.raises(ValueError, match="LIVE_INPUT_CHANGED"):
        d.load("2026-09-16")
