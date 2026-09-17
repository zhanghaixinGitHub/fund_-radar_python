"""核对新独立入口的日期、来源预算、原答案和成熟标签边界。"""

from copy import deepcopy
from datetime import datetime

import pytest
from app.services import direction_1d_sprint_core_forward as s


def stamp(value):
    return datetime.fromisoformat(value + "+08:00")


@pytest.fixture
def forecast(monkeypatch):
    fund = {"code": "001021", "family": "A", "group": "equity"}
    ctx = {"plan": {"test": 1}, "results": {"round-82": {"at": "2026-09-16T07:00:00+08:00", "model_sha256": "model"}}}
    choices = {"UP": {"prediction": 1}}
    source = {"at": "2026-09-17T08:15:00+08:00", "base": "2026-09-16", "target": "2026-09-17", "market": {}}
    value = fund | {
        "at": "2026-09-17T08:16:00+08:00",
        "t": source["base"],
        "u": source["target"],
        "input_hash": s.base.digest(source),
        "plan_hash": s.base.digest(ctx["plan"]),
        "model_hashes": {"round-82": "model"},
        "answers": choices,
        "status": "MODEL_NOT_RELEASED",
        "contract": s.CONTRACT,
    }
    receipt = {"readback_at": "2026-09-17T08:16:01+08:00", "forecast_hash": s.base.digest(value), "status": "VERIFIED"}
    monkeypatch.setattr(s.data, "scope", lambda: [fund])
    monkeypatch.setattr(s, "answers", lambda *_: choices)
    monkeypatch.setattr(s.evidence, "sprint_end", lambda: stamp("2026-09-17T12:11:38"))
    return value, receipt, source, ctx


def test_valid_early_forecast(forecast):
    assert s.validate(*forecast) == forecast[0]


@pytest.mark.parametrize(
    "key,value",
    [("readback_at", "2026-09-17T08:30:00+08:00"), ("status", "LATE_OR_INVALID"), ("forecast_hash", "changed")],
)
def test_invalid_receipt(forecast, key, value):
    forecast[1][key] = value
    with pytest.raises(ValueError):
        s.validate(*forecast)


@pytest.mark.parametrize(
    "field,value",
    [
        ("code", "999999"),
        ("family", "other"),
        ("u", "2026-09-18"),
        ("t", "2026-09-15"),
        ("input_hash", "x"),
        ("plan_hash", "x"),
        ("contract", "old"),
        ("status", "RELEASED"),
        ("answers", {"UP": {"prediction": 0}}),
        ("model_hashes", {"round-82": "new"}),
    ],
)
def test_forecast_tamper_not_hidden_by_new_outer_hash(forecast, field, value):
    forecast[0][field] = value
    forecast[1]["forecast_hash"] = s.base.digest(forecast[0])
    with pytest.raises(ValueError):
        s.validate(*forecast)


def test_late_model_rejected(forecast):
    forecast[3]["results"]["round-82"]["at"] = "2026-09-17T08:17:00+08:00"
    with pytest.raises(ValueError, match="MODEL_CREATED_LATE"):
        s.validate(*forecast)


def test_fxi_reuses_success_without_http(monkeypatch, tmp_path):
    monkeypatch.setattr(s.data.etfs, "root", lambda: tmp_path)
    path = tmp_path / "2026-09-17/FXI/raw/0730.json"
    path.parent.mkdir(parents=True)
    path.touch()
    monkeypatch.setattr(s.data.etfs, "load_piece", lambda *args: ({}, {"ok": 1}))
    monkeypatch.setattr(s.data.etfs, "fetch_etf", lambda *_: pytest.fail("不应重复请求"))
    assert s.capture_fxi("2026-09-16", "2026-09-17", stamp("2026-09-17T08:14:00"))["slot"] == "0730"


def test_reserved_slot_is_not_retried(monkeypatch, tmp_path):
    monkeypatch.setattr(s.data.etfs, "root", lambda: tmp_path)
    path = tmp_path / "2026-09-17/FXI/requests/0800.json"
    path.parent.mkdir(parents=True)
    path.touch()
    monkeypatch.setattr(s.base, "now", lambda: stamp("2026-09-17T08:14:00"))
    monkeypatch.setattr(s.data, "ended", lambda _: False)
    monkeypatch.setattr(s.data.etfs, "fetch_etf", lambda *_: pytest.fail("已预约槽不可重试"))
    assert s.capture_fxi("2026-09-16", "2026-09-17", s.base.now()) is None


@pytest.mark.parametrize("time", ["06:59:59", "08:30:00", "18:00:00"])
def test_no_fxi_request_outside_window(monkeypatch, tmp_path, time):
    monkeypatch.setattr(s.data.etfs, "root", lambda: tmp_path)
    monkeypatch.setattr(s.base, "now", lambda: stamp("2026-09-17T" + time))
    monkeypatch.setattr(s.data.etfs, "fetch_etf", lambda *_: pytest.fail("窗口外不可请求"))
    assert s.capture_fxi("2026-09-16", "2026-09-17", s.base.now()) is None


def test_missing_spx_does_not_block_other_sources(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(s, "root", lambda: tmp_path)
    monkeypatch.setattr(s.base, "now", lambda: stamp("2026-09-17T07:14:00"))
    monkeypatch.setattr(s.base, "calendar", lambda: (["2026-09-16", "2026-09-17"], "hash"))
    monkeypatch.setattr(s.data, "ended", lambda _: False)
    monkeypatch.setattr(s.data, "capture_spx", lambda *_: calls.append("spx"))
    monkeypatch.setattr(s, "capture_fxi", lambda *_: calls.append("fxi") or {"ok": 1})
    monkeypatch.setattr(s.data.cnya, "capture", lambda *_: calls.append("cnya") or {"ok": 1})
    assert s.capture({}) is None
    assert calls == ["spx", "fxi", "cnya"]


def test_raw_label_maturity_and_revisions(monkeypatch, tmp_path):
    monkeypatch.setattr(s, "root", lambda: tmp_path / "core")
    monkeypatch.setattr(s.base, "ROOT", tmp_path)
    f = {"code": "001021", "t": "2026-09-16", "u": "2026-09-17"}
    snap = {
        "at": "2026-09-17T18:00:00+08:00",
        "expires_at": "2026-09-20T00:00:00+08:00",
        "funds": [
            {
                "fund_code": "001021",
                "rows": [
                    {
                        "date": "2026-09-16",
                        "ann_date": "2026-09-16",
                        "received_at": "2026-09-17T18:00:01+08:00",
                        "nav": "1",
                    },
                    {
                        "date": "2026-09-17",
                        "ann_date": "2026-09-17",
                        "received_at": "2026-09-17T18:00:01+08:00",
                        "nav": "1.01",
                    },
                ],
            }
        ],
    }
    first = tmp_path / "observations/20260917T180002.json"
    s.base.save(first, snap)
    s.observe([f], stamp("2026-09-17T17:59:59"))
    outcome = s.root() / "outcomes/2026-09-17/001021.json"
    assert not outcome.exists()
    s.observe([f], stamp("2026-09-17T18:01:00"))
    saved = s.base.read(outcome)
    assert saved["y"] == 1
    revision = deepcopy(snap)
    revision["at"] = "2026-09-17T18:02:00+08:00"
    revision["funds"][0]["rows"][1]["nav"] = "0.99"
    second = tmp_path / "observations/20260917T180203.json"
    s.base.save(second, revision)
    s.observe([f], stamp("2026-09-17T18:03:00"))
    assert s.base.read(outcome) == saved
    revisions = list((s.root() / "revisions/2026-09-17/001021").glob("*.json"))
    assert len(revisions) == 1 and s.base.read(revisions[0])["y"] == 0
    assert s.evidence.validated_outcome(f, saved, {}, stamp("2026-09-17T18:03:00"))["y"] == 1
