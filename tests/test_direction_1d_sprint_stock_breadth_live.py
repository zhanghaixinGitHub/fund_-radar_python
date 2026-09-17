"""不联网验证真实响应绑定、只请求一次及缺失回退，不生成研究准确率。"""

from datetime import datetime

import pytest
from app.services import direction_1d_sprint_stock_breadth_live as s


@pytest.fixture
def fake_source(monkeypatch, tmp_path):
    now = [datetime.fromisoformat("2026-09-17T08:14:00+08:00")]
    calls = []
    monkeypatch.setattr(s, "root", lambda: tmp_path)
    monkeypatch.setattr(s.b, "now", lambda: now[0])

    def fetch(t):
        calls.append(t)
        return b"synthetic-daily-for-test", 200

    monkeypatch.setattr(s, "fetch", fetch)
    value = {
        "date": "2026-09-16",
        "available": True,
        "scope": "PROVIDER_RETURNED_SH_SZ_DAILY_QUOTES",
        "breadth": -0.2,
        "median_pct": -0.5,
        "iqr_pct": 2.5,
    }
    monkeypatch.setattr(s.parser, "parse", lambda *a: value)
    return now, calls, tmp_path


def test_one_capture_and_replay_never_requests_twice(fake_source):
    _, calls, _ = fake_source
    value = s.capture("2026-09-16", "2026-09-17", "plan")
    assert value["available"] and calls == ["2026-09-16"]
    assert s.capture("2026-09-16", "2026-09-17", "plan") == value
    assert calls == ["2026-09-16"]


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

    def failed(t):
        calls.append(t)
        raise ValueError("synthetic failure with no logged text")

    monkeypatch.setattr(s, "fetch", failed)
    value = s.capture("2026-09-16", "2026-09-17", "plan")
    assert not value["available"] and value["error"] == "ValueError"
    assert value["received_at"] is None
    assert s.capture("2026-09-16", "2026-09-17", "plan") == value and len(calls) == 1


@pytest.mark.parametrize(
    "stamp", ["2026-09-17T07:59:59+08:00", "2026-09-17T08:30:00+08:00", "2026-09-18T08:14:00+08:00"]
)
def test_outside_window_does_not_request(fake_source, stamp):
    now, calls, _ = fake_source
    now[0] = datetime.fromisoformat(stamp)
    assert s.capture("2026-09-16", "2026-09-17", "plan") is None and calls == []


def test_wrong_previous_day_never_requests(fake_source):
    _, calls, _ = fake_source
    assert s.capture("2026-09-15", "2026-09-17", "plan") is None and calls == []


def test_late_response_cannot_be_used_as_timely_snapshot(fake_source, monkeypatch):
    now, _, _ = fake_source

    def late(t):
        now[0] = datetime.fromisoformat("2026-09-17T08:30:01+08:00")
        return b"late-quote", 200

    monkeypatch.setattr(s, "fetch", late)
    with pytest.raises(ValueError, match="RECEIPT_OR_TIME_CHANGED"):
        s.capture("2026-09-16", "2026-09-17", "plan")


def test_invalid_provider_data_remains_missing_and_preserves_raw(fake_source, monkeypatch):
    _, calls, folder = fake_source

    def malformed(*args):
        raise ValueError("STOCK_BREADTH_MISSING_DATE")

    monkeypatch.setattr(s.parser, "parse", malformed)
    value = s.capture("2026-09-16", "2026-09-17", "plan")
    assert not value["available"] and value["points"] == {}
    assert (folder / "2026-09-17/raw.bin").is_file()
    assert s.capture("2026-09-16", "2026-09-17", "plan") == value and len(calls) == 1


@pytest.mark.parametrize("orphan", [False, True])
def test_interrupted_reserved_request_never_retries_or_uses_orphan(fake_source, orphan):
    now, calls, folder = fake_source
    r = folder / "2026-09-17"
    request = {
        "at": now[0].isoformat(),
        "t": "2026-09-16",
        "u": "2026-09-17",
        "plan_hash": "plan",
        "url": s.URL,
        "query": s.query("2026-09-16"),
        "attempt": 1,
    }
    s.b.save(r / "request.json", request)
    if orphan:
        (r / "raw.bin").write_bytes(b"orphaned-quote-with-no-receipt")
    value = s.capture("2026-09-16", "2026-09-17", "plan")
    assert not value["available"] and value["received_at"] is None and calls == []
    assert value["error"] == "REQUEST_RESERVED_WITHOUT_COMPLETION_RECEIPT"
    assert s.capture("2026-09-16", "2026-09-17", "plan") == value and calls == []


def test_non200_raw_is_kept_but_not_model_input(fake_source, monkeypatch):
    monkeypatch.setattr(s, "fetch", lambda t: (b"provider-failure", 403))
    value = s.capture("2026-09-16", "2026-09-17", "plan")
    assert value["error"] == "HTTP_403" and value["points"] == {} and not value["available"]


def test_tampered_snapshot_is_reconstructed_from_raw(fake_source):
    _, _, folder = fake_source
    value = s.capture("2026-09-16", "2026-09-17", "plan")
    value["points"]["2026-09-16"]["breadth"] = 0.7
    s.b.save(folder / "2026-09-17/snapshot.json", value, replace=True)
    with pytest.raises(ValueError, match="SNAPSHOT_CHANGED"):
        s.load_live("2026-09-16", "2026-09-17", "plan")
