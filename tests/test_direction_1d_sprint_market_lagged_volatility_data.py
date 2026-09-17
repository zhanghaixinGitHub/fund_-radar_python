"""历史尺度的时间隔离，以及未来实际原文、上下文和回读时点约束。"""

import hashlib
from copy import deepcopy
from datetime import datetime

import numpy as np
import pytest
from app.services import direction_1d_sprint as b
from app.services import direction_1d_sprint_market_lagged_volatility_data as d

from test_direction_1d_sprint_market_fxi_interval import paired
from test_direction_1d_sprint_market_only_forward import ready as parent_ready  # noqa: F401


def past():
    return [{"t": f"2025-01-{i:02d}", "u": f"2025-01-{i + 1:02d}", "market": paired()} for i in range(1, 21)]


def test_constant_scale_floor_and_sign_preservation():
    rows = past()
    result = d.context(paired(), "2025-02-01", rows)
    assert result["scale"] == [1e-6] * 3
    np.testing.assert_array_equal(np.array(result["normalized_features"]) >= 0, np.array(paired()["features"]) >= 0)


def test_current_value_cannot_affect_past_scale():
    rows = past()
    for i, row in enumerate(rows):
        row["market"]["features"] = [i / 10, i % 5, (i % 3) - 1]
    a = d.context(paired(), "2025-02-01", rows)
    changed = paired() | {"features": [123, -456, 789]}
    z = d.context(changed, "2025-02-01", rows)
    assert a["scale"] == z["scale"] and a["market_context_hash"] == z["market_context_hash"]
    np.testing.assert_allclose(a["scale"], np.std([r["market"]["features"] for r in rows], axis=0))
    with pytest.raises(ValueError, match="CURRENT_OR_DUPLICATE"):
        d.context(paired(), rows[-1]["u"], rows)
    assert d.context(paired(), "2025-02-01", rows[:19])["available"] is False


@pytest.fixture
def raw_ready(parent_ready, monkeypatch):  # noqa: F811
    """仅原字节解码器用替身；日历、所有市场计算、摘要、日期选择和保存回读走真实逻辑。"""
    clock, source, _ = parent_ready
    t, u = source["base"], source["target"]
    all_us = [day for day in d.parent.overnight.sessions() if "2026-05-01" <= day <= "2026-09-15"]
    spx = {day: {"close": 1000.0 + i, "pre_close": 999.0 + i} for i, day in enumerate(all_us)}
    e = {
        s: {
            day: {
                "open": 100.0 + i,
                "close": 100.2 + i,
                "high": 102.0 + i,
                "low": 99.0 + i,
                "volume": 100,
                "available": True,
                "unavailable_reason": None,
            }
            for i, day in enumerate(all_us)
        }
        for s in d.parent.etfs.SYMBOLS
    }
    cn = {
        day: {"nav_per_share": 100.0 + (i % 3), "non_fv_nav": 100.0, "available": True, "unavailable_reason": None}
        for i, day in enumerate(all_us)
    }
    needed = d.parent.etfs.required(t, u)
    etf = {
        "at": "2026-09-16T07:01:00+08:00",
        "base": t,
        "target": u,
        "rows": {s: {day: e[s][day] for day in needed} for s in e},
        "raw_refs": {s: {"slot": "0700"} for s in e},
    }
    cnya = {
        "at": etf["at"],
        "base": t,
        "target": u,
        "rows": {day: cn[day] for day in needed},
        "raw_ref": {"slot": "0700"},
    }
    source.update(
        {
            "etf_hash": b.digest(etf),
            "cnya_hash": b.digest(cnya),
            "spx_ref": {"origin": "MARKET_ONLY", "slot": "0700"},
            "market": d.parent.feature_values(t, u, spx, e, cn),
        }
    )
    b.save(d.parent.root() / u / "input.json", source)
    history = {"rows": spx, "expires_at": "2026-09-18T00:00:00+08:00"}
    b.save(d.parent.overnight.root() / "spx.json", history)
    b.save(b.ROOT / "nav-independent-market-feasibility-v1/plan.json", {"source_hashes": {"spx": b.digest(history)}})
    observed = {"received_at": "2026-09-16T07:00:00+08:00", "parsed": {"rows": {day: spx[day] for day in needed}}}
    monkeypatch.setattr(d.parent, "spx_observation", lambda *args: (observed, {"hash": b.digest(observed)}))
    metas = {}
    for symbol in e:
        raw = symbol.encode()
        meta = {
            "body_sha256": hashlib.sha256(raw).hexdigest(),
            "bytes": len(raw),
            "received_at": "2026-09-16T07:00:00+08:00",
        }
        path = d.parent.etfs.root() / u / symbol / "raw/0700.bin"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
        metas[symbol] = meta
    raw = b"CNYA_ISSUER_HISTORY"
    path = d.parent.cnya.root() / u / "raw/0700.xls"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    b.save(
        path.with_suffix(".json"),
        {"body_sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw), "received_at": "2026-09-16T07:00:00+08:00"},
    )
    monkeypatch.setattr(d.parent.etfs, "load", lambda *args: etf)
    monkeypatch.setattr(d.parent.cnya, "load", lambda *args: cnya)
    monkeypatch.setattr(
        d.parent.etfs, "load_piece", lambda target, base, symbol, slot: (etf["rows"][symbol], metas[symbol])
    )
    monkeypatch.setattr(
        d.parent.etfs,
        "parse",
        lambda raw, symbol, end: e[symbol] if raw == symbol.encode() else pytest.fail("wrong bytes"),
    )
    monkeypatch.setattr(
        d.parent.cnya, "parse", lambda raw: cn if raw == b"CNYA_ISSUER_HISTORY" else pytest.fail("wrong issuer bytes")
    )
    monkeypatch.setattr(d.parent.etfs, "fetch_etf", lambda *a: pytest.fail("no source request"))
    monkeypatch.setattr(d.parent.cnya, "fetch_cnya", lambda *a: pytest.fail("no source request"))
    monkeypatch.setattr(d.parent.overnight, "fetch_spx", lambda *a: pytest.fail("no source request"))
    return clock, source, spx, e, cn, observed


def test_full_raw_history_builds_twenty_past_contexts_and_frozen_receipt(raw_ready):
    clock, source, *_ = raw_ready
    value = d.capture(clock[0])
    assert len(value["context_rows"]) == 20
    assert max(r["u"] for r in value["context_rows"]) < source["target"]
    path = d.root() / source["target"] / "input.json"
    raw = path.read_bytes()
    assert d.capture(clock[0]) == value and path.read_bytes() == raw
    assert d.live_market(source)["volatility_input_hash"] == b.digest(value)
    assert (
        d.context(value["market"]["interval_market"], source["target"], value["context_rows"])
        == value["market"]["volatility_context"]
    )


@pytest.mark.parametrize(
    "fault", ["etf_raw", "cnya_raw", "context_row", "late_receipt", "future_response", "overlap_price"]
)
def test_source_revision_context_rewrite_and_late_time_are_rejected(raw_ready, fault):
    clock, source, _, _, _, observed = raw_ready
    value = d.capture(clock[0])
    u = source["target"]
    if fault == "etf_raw":
        (d.parent.etfs.root() / u / "FXI/raw/0700.bin").write_bytes(b"CHANGED")
    elif fault == "cnya_raw":
        (d.parent.cnya.root() / u / "raw/0700.xls").write_bytes(b"CHANGED")
    elif fault == "context_row":
        value["context_rows"][-1]["market"]["features"][0] += 1
        b.save(d.root() / u / "input.json", value, replace=True)
        p = d.root() / u / "receipt.json"
        b.save(p, b.read(p) | {"input_hash": b.digest(value)}, replace=True)
    elif fault == "late_receipt":
        p = d.root() / u / "receipt.json"
        b.save(p, b.read(p) | {"readback_at": "2026-09-16T08:30:00+08:00"}, replace=True)
    elif fault == "future_response":
        observed["received_at"] = "2026-09-16T07:20:00+08:00"
    else:
        observed["parsed"]["rows"] = deepcopy(observed["parsed"]["rows"])
        observed["parsed"]["rows"][sorted(observed["parsed"]["rows"])[0]]["close"] += 1
    with pytest.raises(ValueError):
        d.load(u)


def test_trimmed_only_source_waits_without_forecast_or_query(raw_ready):
    clock, source, _, e, cn, _ = raw_ready
    for symbol in e:
        for day in list(e[symbol])[:-2]:
            del e[symbol][day]
    for day in list(cn)[:-2]:
        del cn[day]
    assert d.capture(clock[0]) is None
    assert not (d.root() / source["target"] / "input.json").exists()
    assert d.capture(datetime(2026, 9, 16, 8, 30, tzinfo=b.ZONE)) is None


def test_current_day_extremes_do_not_enter_past_context(raw_ready):
    _, source, spx, e, cn, _ = raw_ready
    before, _ = d.past_vectors(source["target"], spx, e, cn)
    last = max(cn)
    cn[last]["nav_per_share"] = 10000
    after, _ = d.past_vectors(source["target"], spx, e, cn)
    assert before == after


def test_first_source_readback_crossing_deadline_is_not_verified(raw_ready, monkeypatch):
    clock, source, *_ = raw_ready
    original = d.rebuild
    calls = []

    def delayed(*args):
        result = original(*args)
        calls.append(1)
        if len(calls) == 2:
            clock[0] = datetime(2026, 9, 16, 8, 30, tzinfo=b.ZONE)
        return result

    monkeypatch.setattr(d, "rebuild", delayed)
    with pytest.raises(ValueError, match="INPUT_TIME_INVALID"):
        d.capture(clock[0])
    assert b.read(d.root() / source["target"] / "receipt.json")["status"] == "LATE_OR_INVALID"
