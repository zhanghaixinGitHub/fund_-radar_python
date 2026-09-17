"""美元相关ETF信号的隔夜跳空隔离、异常会话与截止边界，不允许修造价格。"""

import math

import pytest
from app.services import direction_1d_sprint_dollar_etf_data as d


def quote(opening=10.0, close=11.0, volume=100.0):
    return {
        "open": opening,
        "high": max(opening, close),
        "low": min(opening, close),
        "close": close,
        "volume": volume,
        "available": volume > 0,
        "unavailable_reason": None if volume > 0 else "OHLC_RANGE_OR_NO_TRADE",
    }


def align(monkeypatch, days):
    monkeypatch.setattr(d.overnight, "alignment", lambda t, u: {"new_us_dates": days})


def test_multi_session_excludes_overnight_split_and_dividend_gaps(monkeypatch):
    align(monkeypatch, ["2025-01-02", "2025-01-03"])
    points = {"2025-01-02": quote(20, 22), "2025-01-03": quote(10, 11)}
    value = d.features("T", "U", points)
    assert value["available"] and value["intraday_log_pct"] == pytest.approx(100 * 2 * math.log(1.1))
    assert value["intraday_log_pct"] > 0  # 跨日拆股价差22→10不能进入输入。


def test_same_day_scaling_does_not_change_signal(monkeypatch):
    align(monkeypatch, ["2025-01-02"])
    a = d.features("T", "U", {"2025-01-02": quote(10, 11)})
    z = d.features("T", "U", {"2025-01-02": quote(20, 22)})
    assert a == z


def test_missing_middle_session_not_skipped(monkeypatch):
    align(monkeypatch, ["2025-01-02", "2025-01-03", "2025-01-06"])
    value = d.features("T", "U", {"2025-01-02": quote(), "2025-01-06": quote()})
    assert value["available"] is False and value["reason"] == "MISSING_US_SESSION"


@pytest.mark.parametrize("kind", ["zero_volume", "bad_range"])
def test_invalid_source_row_is_masked(monkeypatch, kind):
    align(monkeypatch, ["2025-01-02"])
    q = quote(volume=0) if kind == "zero_volume" else quote()
    if kind == "bad_range":
        q.update(high=10.5, available=False, unavailable_reason="OHLC_RANGE_OR_NO_TRADE")
    assert not d.features("T", "U", {"2025-01-02": q})["available"]


def test_quality_flag_tamper_rejected(monkeypatch):
    align(monkeypatch, ["2025-01-02"])
    q = quote(volume=0) | {"available": True, "unavailable_reason": None}
    with pytest.raises(ValueError, match="QUALITY_FLAG_CHANGED"):
        d.features("T", "U", {"2025-01-02": q})


def test_no_new_session_not_stale_price(monkeypatch):
    align(monkeypatch, [])
    assert d.features("T", "U", {})["reason"] == "NO_NEW_US_SESSION"


def test_only_new_closed_session_is_used():
    # 美股9月15日收盘在中国9月16日凌晨，9月16日美股收盘尚在16日08:30之后。
    assert d.features("2026-09-15", "2026-09-16", {"2026-09-15": quote(), "2026-09-16": quote(10, 1)})[
        "new_us_dates"
    ] == ["2026-09-15"]


def test_nonadjacent_china_target_rejected():
    with pytest.raises(ValueError):
        d.features("2026-09-14", "2026-09-16", {})


def test_clip_is_fixed(monkeypatch):
    align(monkeypatch, ["2025-01-02"])
    assert d.features("T", "U", {"2025-01-02": quote(1, 2)})["intraday_log_pct"] == 20
    assert d.features("T", "U", {"2025-01-02": quote(2, 1)})["intraday_log_pct"] == -20
