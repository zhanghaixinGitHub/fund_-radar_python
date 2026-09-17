"""不联网验证真实响应绑定、只请求一次及缺失回退，不生成研究准确率。"""

import hashlib
import json
from datetime import datetime

import pytest
from app.services import direction_1d_sprint_stock_moneyflow_live_v2 as s


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
    row = {k: 10 for k in s.parser.FIELDS}
    row.update(ts_code="600001.SH", trade_date="20260916", buy_lg_amount=30, net_mf_amount=-7)
    raw = json.dumps(
        {"code": 0, "data": {"fields": s.parser.FIELDS, "items": [[row[k] for k in s.parser.FIELDS]]}}
    ).encode()
    value = s.parser.parse(raw, "2026-09-16", minimum_included=1)
    monkeypatch.setattr(s.parser, "parse", lambda *a: value)
    monkeypatch.setattr(s, "quote_context", lambda *a: {"source_at": now[0].isoformat()})
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
        raise ValueError("MONEYFLOW_MISSING_DATE")

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
    value["points"]["2026-09-16"]["large_imbalance"] = 0.7
    s.b.save(folder / "2026-09-17/snapshot.json", value, replace=True)
    with pytest.raises(ValueError, match="SNAPSHOT_CHANGED"):
        s.load_live("2026-09-16", "2026-09-17", "plan")


def test_missing_actual_quotes_cannot_use_historical_quotes(fake_source, monkeypatch):
    def missing(*args):
        raise FileNotFoundError("actual quote response missing")

    monkeypatch.setattr(s, "quote_context", missing)
    value = s.capture("2026-09-16", "2026-09-17", "plan")
    assert not value["available"] and value["points"] == {} and value["quote_context"] is None


def test_quote_snapshot_must_exist_before_moneyflow_snapshot(fake_source, monkeypatch):
    monkeypatch.setattr(s, "quote_context", lambda *a: {"source_at": "2026-09-17T08:15:00+08:00"})
    with pytest.raises(ValueError, match="QUOTE_SNAPSHOT_LATE"):
        s.capture("2026-09-16", "2026-09-17", "plan")


@pytest.fixture
def actual_quotes(monkeypatch, tmp_path):
    # 固定100只等额报价，分别验证股票数量覆盖与成交金额覆盖，避免一个比例掩盖另一个。
    monkeypatch.setattr(s.b, "ROOT", tmp_path)
    monkeypatch.setattr(s.quote_live, "root", lambda: tmp_path / "quotes")
    s.b.save(tmp_path / "round-109/plan.json", {"quote_plan": "fixed"})
    folder = tmp_path / "quotes/2026-09-17"
    codes = [f"{600000 + i}.SH" for i in range(100)]
    source = {"available": True, "at": "2026-09-17T08:01:00+08:00"}

    def save_quote(amounts):
        raw = json.dumps({"data": {"items": [[k, v] for k, v in zip(codes, amounts, strict=True)]}}).encode()
        receipt = {"raw": {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}}
        s.b.save(folder / "response.json", receipt, replace=True)
        (folder / "raw.bin").write_bytes(raw)
        source["receipt_hash"] = s.b.digest(receipt)

    save_quote([1.0] * 100)
    monkeypatch.setattr(s.quote_live, "load_live", lambda *a: source)
    return codes, save_quote, folder, source


def test_exact_95_percent_counts_and_amounts_accepted(actual_quotes):
    codes, _, _, _ = actual_quotes
    flow = json.dumps({"data": {"items": [[k] for k in codes[:95]]}}).encode()
    result = s.quote_context("2026-09-16", "2026-09-17", flow)
    assert result["count_coverage"] == result["amount_coverage"] == 0.95
    assert result["missing_positive_quote_codes"] == codes[95:]


@pytest.mark.parametrize("count,missing_amount", [(94, 1.0), (95, 100.0)])
def test_count_or_amount_under_95_percent_rejected(actual_quotes, count, missing_amount):
    codes, save_quote, _, _ = actual_quotes
    save_quote([1.0] * 95 + [missing_amount] * 5)
    flow = json.dumps({"data": {"items": [[k] for k in codes[:count]]}}).encode()
    with pytest.raises(ValueError, match="COVERAGE_INSUFFICIENT"):
        s.quote_context("2026-09-16", "2026-09-17", flow)


def test_actual_quote_bytes_must_match_verified_receipt(actual_quotes):
    _, _, folder, _ = actual_quotes
    (folder / "raw.bin").write_bytes(b"changed quotes")
    with pytest.raises(ValueError, match="ACTUAL_QUOTE_RAW_CHANGED"):
        s.quote_context("2026-09-16", "2026-09-17", b"{}")


def test_missing_quote_source_stops_coverage_check(actual_quotes):
    _, _, _, source = actual_quotes
    source["available"] = False
    with pytest.raises(ValueError, match="ACTUAL_QUOTE_SOURCE_UNAVAILABLE"):
        s.quote_context("2026-09-16", "2026-09-17", b"{}")
