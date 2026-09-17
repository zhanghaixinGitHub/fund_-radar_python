"""真实接收链使用合成HTTP正文；验证期限、具体合约、故障与防重复请求。"""

import json
from datetime import datetime

import pytest
from app.services import direction_1d_sprint_futures_member_live as live

T, U = "2026-09-16", "2026-09-17"


@pytest.fixture
def setup(monkeypatch, tmp_path):
    base = live.base
    monkeypatch.setattr(base, "ROOT", tmp_path)
    monkeypatch.setattr(base, "now", lambda: datetime.fromisoformat("2026-09-17T08:01:00+08:00"))
    plan = {"candidate": "synthetic-fixed"}
    base.save(tmp_path / "round-120/plan.json", plan)
    source = {
        "plan_hash": "quote-plan",
        "source_hash": "actual-receipt",
        "source_at": "2026-09-17T08:00:00+08:00",
        "available": True,
        "main_contract": "IF2609.CFX",
        "total_open_interest": 5000,
        "quote_raw_sha256": "raw",
    }
    monkeypatch.setattr(live, "quote_context", lambda *_: dict(source))
    monkeypatch.setattr(live.data.base, "calendar", lambda: ([datetime.fromisoformat(d).date() for d in [T, U]], "cal"))
    rows = [["20260916", "IF2609", f"member{i:02d}", 100, 0, 100, 2, 90, -1, "CFFEX"] for i in range(20)]
    raw = json.dumps({"code": 0, "data": {"fields": list(live.parser.FIELDS), "items": rows}}).encode()
    calls = []

    def fetch(query):
        calls.append(query)
        return raw, 200

    monkeypatch.setattr(live, "fetch", fetch)
    return base.digest(plan), source, calls


def test_success_only_one_request_and_reconstruction(setup):
    ph, source, calls = setup
    saved = live.capture(T, U, ph)
    assert saved["available"] and saved["points"][T]["main_contract"] == source["main_contract"]
    assert saved["points"][T]["ranked_long_contracts"] == 2000
    assert saved["reserved_request_slots"] == 1 and saved["new_source_requests"] == 1
    assert calls == [live.query(T, source["main_contract"])]
    assert live.capture(T, U, ph) == saved and len(calls) == 1


def test_actual_parent_missing_does_not_request_or_guess(setup):
    ph, source, calls = setup
    source["available"] = False
    saved = live.capture(T, U, ph)
    assert not saved["available"] and saved["points"] == {}
    assert saved["reserved_request_slots"] == saved["new_source_requests"] == 0 and not calls


@pytest.mark.parametrize(
    "instant",
    [
        "2026-09-17T07:59:59+08:00",
        "2026-09-17T08:29:26+08:00",
        "2026-09-17T08:30:00+08:00",
        "2026-09-16T08:01:00+08:00",
    ],
)
def test_outside_or_insufficient_time_makes_no_request(setup, monkeypatch, instant):
    ph, _, calls = setup
    monkeypatch.setattr(live.base, "now", lambda: datetime.fromisoformat(instant))
    assert live.capture(T, U, ph) is None and not calls


def test_network_failure_is_recorded_and_not_retried(setup, monkeypatch):
    ph, _, calls = setup

    def fail(query):
        calls.append(query)
        raise TimeoutError("no supplier text retained")

    monkeypatch.setattr(live, "fetch", fail)
    saved = live.capture(T, U, ph)
    assert not saved["available"] and saved["error"] == "TimeoutError"
    assert live.capture(T, U, ph) == saved and len(calls) == 1


def test_interrupted_reserved_request_does_not_retry(setup):
    ph, source, calls = setup
    request = {
        "at": live.base.now().isoformat(),
        "t": T,
        "u": U,
        "plan_hash": ph,
        "query": live.query(T, source["main_contract"]),
        "url": live.URL,
        "quote_context_hash": live.base.digest(source),
        "attempt": 1,
    }
    live.base.save(live.root() / U / "request.json", request)
    saved = live.capture(T, U, ph)
    assert not saved["available"] and saved["error"] == "REQUEST_RESERVED_WITHOUT_RECEIPT" and not calls


@pytest.mark.parametrize("changed", ["raw", "contract", "source_hash", "late_receipt", "plan"])
def test_tampering_and_late_evidence_fail_closed(setup, changed):
    ph, source, _ = setup
    live.capture(T, U, ph)
    folder = live.root() / U
    if changed == "raw":
        (folder / "raw.bin").write_bytes(b"changed")
    elif changed in ("contract", "source_hash"):
        source["main_contract" if changed == "contract" else "source_hash"] = (
            "IF2612.CFX" if changed == "contract" else "changed"
        )
    elif changed == "late_receipt":
        receipt = live.base.read(folder / "response.json")
        receipt["at"] = "2026-09-17T08:30:00+08:00"
        live.base.save(folder / "response.json", receipt, replace=True)
    else:
        ph = "different"
    with pytest.raises(ValueError):
        live.load_live(T, U, ph)


def test_ranked_positions_cannot_exceed_actual_total(setup):
    ph, source, _ = setup
    source["total_open_interest"] = 1000
    saved = live.capture(T, U, ph)
    assert not saved["available"] and saved["points"] == {}
    assert "LIVE_RANKED_TOTAL_EXCEEDS_OI" in saved["error"]


def test_timezone_and_contract_rejected():
    with pytest.raises(ValueError, match="TIMEZONE"):
        live.in_window(T, U, datetime(2026, 9, 17, 8))
    with pytest.raises(ValueError, match="CONTRACT"):
        live.query(T, "IF.CFX")
