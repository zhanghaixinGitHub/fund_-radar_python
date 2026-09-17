"""不联网验证一次采集、真实字节绑定、截止时间与明确不可用状态。"""

from datetime import datetime

import pytest
from app.services import direction_1d_sprint_china_health_live as s


@pytest.fixture
def fake_source(monkeypatch, tmp_path):
    now = [datetime.fromisoformat("2026-09-17T08:14:00+08:00")]
    calls = []
    monkeypatch.setattr(s, "root", lambda: tmp_path)
    monkeypatch.setattr(s.b, "now", lambda: now[0])

    def fetch(symbol):
        calls.append(symbol)
        return b"synthetic-quote-for-test", {}

    monkeypatch.setattr(s.api, "fetch_etf", fetch)
    points = {
        "2026-09-16": {
            "open": 10.0,
            "high": 11.0,
            "low": 9.0,
            "close": 10.1,
            "volume": 10,
            "available": True,
            "unavailable_reason": None,
        }
    }
    monkeypatch.setattr(s.api, "parse", lambda *a: points)
    return now, calls, tmp_path


def test_one_capture_and_replay_never_requests_twice(fake_source):
    _, calls, _ = fake_source
    value = s.capture("2026-09-16", "2026-09-17", "plan")
    assert value["available"] and len(calls) == 1
    assert s.capture("2026-09-16", "2026-09-17", "plan") == value
    assert len(calls) == 1


def test_wrong_plan_cannot_relabel_old_capture(fake_source):
    s.capture("2026-09-16", "2026-09-17", "plan")
    with pytest.raises(ValueError, match="REQUEST_CHANGED"):
        s.load_live("2026-09-16", "2026-09-17", "other")


def test_changed_raw_bytes_rejected(fake_source):
    _, _, folder = fake_source
    s.capture("2026-09-16", "2026-09-17", "plan")
    (folder / "2026-09-17/raw.bin").write_bytes(b"changed")
    with pytest.raises(ValueError, match="RAW_CHANGED"):
        s.load_live("2026-09-16", "2026-09-17", "plan")


def test_failed_request_is_unavailable_without_retry(fake_source, monkeypatch):
    calls = []

    def failed(symbol):
        calls.append(symbol)
        raise ValueError("synthetic network failure")

    monkeypatch.setattr(s.api, "fetch_etf", failed)
    value = s.capture("2026-09-16", "2026-09-17", "plan")
    assert not value["available"] and value["error"] == "ValueError"
    assert s.capture("2026-09-16", "2026-09-17", "plan") == value and len(calls) == 1


@pytest.mark.parametrize("stamp", ["2026-09-17T07:59:59+08:00", "2026-09-17T08:30:00+08:00"])
def test_outside_window_does_not_request(fake_source, stamp):
    now, calls, _ = fake_source
    now[0] = datetime.fromisoformat(stamp)
    assert s.capture("2026-09-16", "2026-09-17", "plan") is None and calls == []


def test_no_new_requests_after_sprint(fake_source):
    now, calls, _ = fake_source
    now[0] = datetime.fromisoformat("2026-09-18T08:14:00+08:00")
    assert s.capture("2026-09-17", "2026-09-18", "plan") is None and calls == []


def test_late_response_cannot_be_used_as_timely_snapshot(fake_source, monkeypatch):
    now, _, _ = fake_source

    def late(symbol):
        now[0] = datetime.fromisoformat("2026-09-17T08:30:01+08:00")
        return b"late-quote", {}

    monkeypatch.setattr(s.api, "fetch_etf", late)
    with pytest.raises(ValueError, match="RECEIPT_OR_TIME_CHANGED"):
        s.capture("2026-09-16", "2026-09-17", "plan")


def test_missing_session_is_preserved_as_unavailable(fake_source, monkeypatch):
    monkeypatch.setattr(s.api, "parse", lambda *a: {})
    value = s.capture("2026-09-16", "2026-09-17", "plan")
    assert not value["available"] and value["feature"]["reason"] == "MISSING_US_SESSION"
