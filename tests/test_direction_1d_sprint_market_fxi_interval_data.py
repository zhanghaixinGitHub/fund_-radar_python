"""多日区间、原始价格完整性，以及父输入裁剪后的原始字节重读。"""

import hashlib

import pytest
from app.services import direction_1d_sprint as b
from app.services import direction_1d_sprint_market_fxi_interval_data as d

from test_direction_1d_sprint_market_only import market
from test_direction_1d_sprint_market_only_forward import ready as parent_ready  # noqa: F401


def points(t, u):
    days = d.parent.overnight.alignment(t, u)["new_us_dates"]
    result = {
        day: {
            "open": 100.0 + i,
            "close": 101.0 + i,
            "high": 103.0 + i,
            "low": 99.0 + i,
            "volume": 100,
            "available": True,
            "unavailable_reason": None,
        }
        for i, day in enumerate(days)
    }
    old = 100 * (result[days[-1]]["close"] / result[days[-1]]["open"] - 1)
    return result, market((0.1, old, 0.2))


def test_single_session_is_exactly_original_latest_intraday():
    source, original = points("2026-09-15", "2026-09-16")
    assert d.extend(original, "2026-09-15", "2026-09-16", source) == original


def test_long_holiday_uses_first_open_not_latest_open():
    source, original = points("2025-09-30", "2025-10-09")
    value = d.extend(original, "2025-09-30", "2025-10-09", source)
    assert len(source) == 7 and value["features"][1] == pytest.approx(7)
    assert original["features"][1] < 1


@pytest.mark.parametrize("fault", ["missing_middle", "bad_flag", "bad_range", "no_volume"])
def test_middle_session_integrity_is_required(fault):
    source, original = points("2025-09-30", "2025-10-09")
    middle = sorted(source)[3]
    if fault == "missing_middle":
        del source[middle]
    elif fault == "bad_flag":
        source[middle]["available"] = False
    elif fault == "bad_range":
        source[middle]["high"] = 1
    else:
        source[middle]["volume"] = 0
    with pytest.raises(ValueError):
        d.extend(original, "2025-09-30", "2025-10-09", source)


def test_original_fallback_does_not_require_unused_interval_prices():
    original = market((0, 0, 0), False)
    assert d.extend(original, "2026-09-15", "2026-09-16", {}) == original


def test_live_reparse_reads_all_days_from_validated_raw_bytes(parent_ready, monkeypatch):  # noqa: F811
    t, u = "2025-09-30", "2025-10-09"
    full, original = points(t, u)
    raw = b"ACTUAL_BYTES_TEST_DOUBLE"
    meta = {
        "body_sha256": hashlib.sha256(raw).hexdigest(),
        "bytes": len(raw),
        "received_at": "2025-10-09T06:00:00+08:00",
    }
    trimmed = {k: full[k] for k in sorted(full)[-2:]}
    etf = {"base": t, "target": u, "rows": {"FXI": trimmed}, "raw_refs": {"FXI": {"slot": "0700"}}}
    source = {"base": t, "target": u, "market": original, "etf_hash": b.digest(etf)}
    path = d.parent.etfs.root() / u / "FXI/raw/0700.bin"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    monkeypatch.setattr(d.parent, "load", lambda target: source)
    monkeypatch.setattr(d.parent.etfs, "load", lambda target: etf)
    monkeypatch.setattr(d.parent.etfs, "load_piece", lambda *args: (trimmed, meta))
    monkeypatch.setattr(
        d.parent.etfs,
        "parse",
        lambda body, symbol, end: (
            full if body == raw and symbol == "FXI" and end == "2025-10-08" else pytest.fail("wrong raw source")
        ),
    )
    value = d.live_market(source)
    assert value["features"][1] == pytest.approx(7) and value["original_market"] == original
    assert len(value["interval_us_dates"]) == 7
    path.write_bytes(b"CHANGED")
    with pytest.raises(ValueError, match="LIVE_RAW_BYTES_CHANGED"):
        d.live_market(source)
