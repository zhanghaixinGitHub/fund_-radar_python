"""不联网验证真实响应绑定、只请求一次及缺失回退，不生成研究准确率。"""

from datetime import datetime

import pytest
from app.services import direction_1d_sprint_credit_pair_live as s


def capture(t, u, ph):
    return s.capture_leg(t, u, ph, "HYG")


def load(t, u, ph):
    return s.load_leg(t, u, ph, "HYG")


@pytest.fixture
def fake_source(monkeypatch, tmp_path):
    now = [datetime.fromisoformat("2026-09-17T08:14:00+08:00")]
    calls = []
    monkeypatch.setattr(s, "root", lambda: tmp_path)
    monkeypatch.setattr(s.b, "now", lambda: now[0])

    def fetch(t, symbol):
        calls.append((t, symbol))
        return b"synthetic-daily-for-test", 200

    monkeypatch.setattr(s, "fetch", fetch)
    points = {
        "2026-09-16": {
            "open": 28.2,
            "high": 28.4,
            "low": 28.1,
            "close": 28.3,
            "volume": 2000,
            "available": True,
            "unavailable_reason": None,
        }
    }
    monkeypatch.setattr(s.parser, "parse", lambda *a: points)
    return now, calls, tmp_path


def test_one_capture_and_replay_never_requests_twice(fake_source):
    _, calls, _ = fake_source
    value = capture("2026-09-16", "2026-09-17", "plan")
    assert value["available"] and calls == [("2026-09-16", "HYG")]
    assert capture("2026-09-16", "2026-09-17", "plan") == value
    assert calls == [("2026-09-16", "HYG")]


def test_wrong_plan_cannot_relabel_old_capture(fake_source):
    capture("2026-09-16", "2026-09-17", "plan")
    with pytest.raises(ValueError, match="REQUEST_CHANGED"):
        load("2026-09-16", "2026-09-17", "other")


def test_changed_raw_bytes_rejected(fake_source):
    _, _, folder = fake_source
    capture("2026-09-16", "2026-09-17", "plan")
    (folder / "2026-09-17/HYG/raw.bin").write_bytes(b"changed")
    with pytest.raises(ValueError, match="RAW_CHANGED"):
        load("2026-09-16", "2026-09-17", "plan")


def test_failed_request_is_unavailable_without_retry(fake_source, monkeypatch):
    calls = []

    def failed(t, symbol):
        calls.append((t, symbol))
        raise ValueError("synthetic failure with no logged text")

    monkeypatch.setattr(s, "fetch", failed)
    value = capture("2026-09-16", "2026-09-17", "plan")
    assert not value["available"] and value["error"] == "ValueError"
    assert value["received_at"] is None
    assert capture("2026-09-16", "2026-09-17", "plan") == value and len(calls) == 1


@pytest.mark.parametrize(
    "stamp", ["2026-09-17T07:59:59+08:00", "2026-09-17T08:30:00+08:00", "2026-09-18T08:14:00+08:00"]
)
def test_outside_window_does_not_request(fake_source, stamp):
    now, calls, _ = fake_source
    now[0] = datetime.fromisoformat(stamp)
    assert capture("2026-09-16", "2026-09-17", "plan") is None and calls == []


def test_wrong_previous_day_never_requests(fake_source):
    _, calls, _ = fake_source
    assert capture("2026-09-15", "2026-09-17", "plan") is None and calls == []


def test_late_response_cannot_be_used_as_timely_snapshot(fake_source, monkeypatch):
    now, _, _ = fake_source

    def late(t, symbol):
        now[0] = datetime.fromisoformat("2026-09-17T08:30:01+08:00")
        return b"late-quote", 200

    monkeypatch.setattr(s, "fetch", late)
    with pytest.raises(ValueError, match="RECEIPT_OR_TIME_CHANGED"):
        capture("2026-09-16", "2026-09-17", "plan")


def test_invalid_provider_data_remains_missing_and_preserves_raw(fake_source, monkeypatch):
    _, calls, folder = fake_source

    def malformed(*args):
        raise ValueError("US_ETF_FUTURE_ROW")

    monkeypatch.setattr(s.parser, "parse", malformed)
    value = capture("2026-09-16", "2026-09-17", "plan")
    assert not value["available"] and value["points"] == {}
    assert (folder / "2026-09-17/HYG/raw.bin").is_file()
    assert capture("2026-09-16", "2026-09-17", "plan") == value and len(calls) == 1


@pytest.mark.parametrize("orphan", [False, True])
def test_interrupted_reserved_request_never_retries_or_uses_orphan(fake_source, orphan):
    now, calls, folder = fake_source
    r = folder / "2026-09-17/HYG"
    request = {
        "at": now[0].isoformat(),
        "t": "2026-09-16",
        "u": "2026-09-17",
        "plan_hash": "plan",
        "url": s.parser.url_for("HYG"),
        "query": s.query("2026-09-16", "HYG"),
        "attempt": 1,
    }
    s.b.save(r / "request.json", request)
    if orphan:
        (r / "raw.bin").write_bytes(b"orphaned-quote-with-no-receipt")
    value = capture("2026-09-16", "2026-09-17", "plan")
    assert not value["available"] and value["received_at"] is None and calls == []
    assert value["error"] == "REQUEST_RESERVED_WITHOUT_COMPLETION_RECEIPT"
    assert capture("2026-09-16", "2026-09-17", "plan") == value and calls == []


def test_non200_raw_is_kept_but_not_model_input(fake_source, monkeypatch):
    monkeypatch.setattr(s, "fetch", lambda t, symbol: (b"provider-failure", 403))
    value = capture("2026-09-16", "2026-09-17", "plan")
    assert value["error"] == "HTTP_403" and value["points"] == {} and not value["available"]


def test_tampered_snapshot_is_reconstructed_from_raw(fake_source):
    _, _, folder = fake_source
    value = capture("2026-09-16", "2026-09-17", "plan")
    value["points"]["2026-09-16"]["close"] = 27.7
    s.b.save(folder / "2026-09-17/HYG/snapshot.json", value, replace=True)
    with pytest.raises(ValueError, match="SNAPSHOT_CHANGED"):
        load("2026-09-16", "2026-09-17", "plan")


def test_pair_captures_each_symbol_once_and_binds_both(fake_source):
    _, calls, _ = fake_source
    result = s.capture("2026-09-16", "2026-09-17", "plan")
    assert result["available"] and result["reserved_request_slots"] == 2
    assert result["new_source_requests"] == 2 and set(result["points"]) == {"HYG", "IEF"}
    assert calls == [("2026-09-16", "HYG"), ("2026-09-16", "IEF")]
    assert s.capture("2026-09-16", "2026-09-17", "plan") == result
    assert len(calls) == 2


def test_one_failed_leg_does_not_become_complete_pair_or_retry(fake_source, monkeypatch):
    _, calls, _ = fake_source

    def fetch(t, symbol):
        calls.append((t, symbol))
        if symbol == "IEF":
            raise TimeoutError("synthetic source timeout")
        return b"good-hyg", 200

    monkeypatch.setattr(s, "fetch", fetch)
    result = s.capture("2026-09-16", "2026-09-17", "plan")
    assert not result["available"] and result["points"]["IEF"] == {}
    assert result["points"]["HYG"] and result["received_at_by_symbol"]["IEF"] is None
    assert result["new_source_requests"] is None
    assert s.capture("2026-09-16", "2026-09-17", "plan") == result and len(calls) == 2


def test_pair_rejects_changed_leg_even_if_pair_envelope_is_rehashed(fake_source):
    _, _, folder = fake_source
    result = s.capture("2026-09-16", "2026-09-17", "plan")
    result["points"]["HYG"]["2026-09-16"]["close"] = 23
    s.b.save(folder / "2026-09-17/snapshot.json", result, replace=True)
    with pytest.raises(ValueError, match="PAIR_SNAPSHOT_CHANGED"):
        s.load_live("2026-09-16", "2026-09-17", "plan")


def test_unapproved_symbol_rejected_before_filesystem_or_network(fake_source):
    _, calls, folder = fake_source
    for action in [s.capture_leg, s.load_leg]:
        with pytest.raises(ValueError, match="SYMBOL_INVALID"):
            action("2026-09-16", "2026-09-17", "plan", "../../UUP")
    assert not list(folder.iterdir()) and calls == []
