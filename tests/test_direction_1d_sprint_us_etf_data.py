"""ETF时点、价格质量和真实采集回读边界；不依赖网络或额外安装。"""

from datetime import datetime

import pytest
from app.integrations import sina_sprint_etf as client
from app.services import direction_1d_sprint as b
from app.services import direction_1d_sprint_us_etf_data as d


def quote(open_price=100.0, close=100.0):
    return {
        "open": open_price,
        "close": close,
        "low": min(open_price, close) - 1,
        "high": max(open_price, close) + 1,
        "volume": 100,
        "available": True,
        "unavailable_reason": None,
    }


def points(t="2026-09-14", u="2026-09-15"):
    days = d.required(t, u)
    return {
        "FXI": {days[0]: quote(100, 100), days[1]: quote(105, 110)},
        "CNYA": {days[0]: quote(50, 50), days[1]: quote(55, 50)},
    }


def test_us_session_move_is_distinct_from_close_to_close():
    assert d.features("2026-09-14", "2026-09-15", points()) == pytest.approx([100 * 5 / 105, 10, -100 * 5 / 55, 0, 1])
    with pytest.raises(ValueError, match="NOT_ADJACENT"):
        d.features("2026-09-14", "2026-09-16", points())


def test_no_new_us_close_uses_old_hk_route():
    # 2025年1月9日美国临时休市，不能把1月8日美国价格说成1月9日中国收盘后的新信息。
    assert d.required("2025-01-09", "2025-01-10")[-1] == "2025-01-08"
    assert d.features("2025-01-09", "2025-01-10", points("2025-01-09", "2025-01-10")) == [0] * 5


def test_invalid_ohlc_is_preserved_and_routes_to_control():
    p = points()
    row = p["CNYA"]["2026-09-14"]
    row.update(close=60, available=False, unavailable_reason="OHLC_RANGE_OR_NO_TRADE")
    assert d.features("2026-09-14", "2026-09-15", p) == [0] * 5
    assert row["close"] == 60
    row["available"] = True
    with pytest.raises(ValueError, match="QUALITY_FLAG_CHANGED"):
        d.features("2026-09-14", "2026-09-15", p)


def test_missing_latest_price_is_not_silently_forward_filled():
    p = points()
    del p["CNYA"]["2026-09-14"]
    with pytest.raises(ValueError, match="DATE_MISSING"):
        d.features("2026-09-14", "2026-09-15", p)


@pytest.mark.parametrize(
    "raw",
    [b"", b'var KLC_K2_FXI="abc";process.exit(0);', b'var KLC_K2_CNYA="abc";', b"x" * 200001],
    ids=["empty", "statement-injection", "wrong-symbol", "oversized"],
)
def test_data_response_is_never_executed(raw, monkeypatch):
    monkeypatch.setattr(client.subprocess, "run", lambda *a, **k: pytest.fail("decoder must not start"))
    with pytest.raises(ValueError):
        client.decode(raw, "FXI")


def test_row_quality_and_futures_are_checked_after_decoding(monkeypatch):
    row = {
        "date": "2023-12-13T00:00:00.000Z",
        "open": 26.48,
        "high": 26.71,
        "low": 26.37,
        "close": 26.7129,
        "volume": 53912,
    }
    monkeypatch.setattr(client, "decode", lambda *a: [row])
    result = client.parse(b"ignored", "CNYA", "2026-09-14")
    assert result["2023-12-13"]["available"] is False
    assert result["2023-12-13"]["close"] == 26.7129
    with pytest.raises(ValueError, match="FUTURE_ROW"):
        client.parse(b"ignored", "CNYA", "2023-12-12")
    monkeypatch.setattr(client, "decode", lambda *a: [row, row])
    with pytest.raises(ValueError, match="DUPLICATE"):
        client.parse(b"ignored", "CNYA")


@pytest.fixture
def live(monkeypatch, tmp_path):
    monkeypatch.setattr(b, "ROOT", tmp_path)
    monkeypatch.setattr(b, "now", lambda: datetime(2026, 9, 15, 7, 5, tzinfo=b.ZONE))
    monkeypatch.setattr(d, "fetch_etf", lambda symbol: (symbol.encode(), {}))
    monkeypatch.setattr(d, "parse", lambda raw, symbol, end: points()[symbol])


def test_live_capture_is_once_per_slot_and_rechecks_original_bytes(live, monkeypatch):
    calls = []
    monkeypatch.setattr(d, "fetch_etf", lambda symbol: (calls.append(symbol) or symbol.encode(), {}))
    value = d.capture(b.now())
    assert calls == ["FXI", "CNYA"] and value["source"] == d.SOURCE
    d.capture(b.now())
    assert len(calls) == 2
    path = d.root() / "2026-09-15/FXI/raw/0700.bin"
    path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="RAW_OR_TIME_CHANGED"):
        d.load("2026-09-15")


def test_slow_or_interrupted_fetch_does_not_create_early_input(live, monkeypatch):
    def slow(symbol):
        monkeypatch.setattr(b, "now", lambda: datetime(2026, 9, 15, 8, 30, tzinfo=b.ZONE))
        return symbol.encode(), {}

    monkeypatch.setattr(d, "fetch_etf", slow)
    assert d.capture(datetime(2026, 9, 15, 7, 5, tzinfo=b.ZONE)) is None
    assert not (d.root() / "2026-09-15/input.json").exists()


def test_rehashed_parsed_value_cannot_replace_original_response(live):
    d.capture(b.now())
    path = d.root() / "2026-09-15/input.json"
    value = b.read(path)
    value["rows"]["CNYA"]["2026-09-14"]["close"] = 52
    b.save(path, value, replace=True)
    with pytest.raises(ValueError, match="PARSED_INPUT_CHANGED"):
        d.load("2026-09-15")


@pytest.mark.parametrize("hour,minute", [(6, 59), (8, 30), (20, 0)])
def test_closed_window_never_requests_prices(live, monkeypatch, hour, minute):
    monkeypatch.setattr(d, "fetch_etf", lambda *a: pytest.fail("closed-window network call"))
    assert d.capture(datetime(2026, 9, 15, hour, minute, tzinfo=b.ZONE)) is None
