"""美元相关ETF来源的独立白名单、包络安全、原始质量标记和时间边界。"""

import pytest
from app.integrations import sina_sprint_dollar_etf as s


def sample(day="2025-01-02", close=10.0):
    return {"date": day + "T00:00:00.000Z", "open": 10.0, "high": 11.0, "low": 9.0, "close": close, "volume": 100.0}


def test_known_dollar_symbols_and_no_arbitrary_endpoint():
    assert all(s.url_for(n).endswith("/" + n) for n in s.SYMBOLS)
    with pytest.raises(ValueError, match="SYMBOL_INVALID"):
        s.url_for("FXI")
    with pytest.raises(ValueError, match="SYMBOL_INVALID"):
        s.url_for("../../other")


@pytest.mark.parametrize("raw", [b'var KLC_K2_UUP="abc";evil();', b'var KLC_K2_FXI="abc";', b"<html>error</html>"])
def test_invalid_envelope_never_reaches_node(raw, monkeypatch):
    monkeypatch.setattr(s.subprocess, "run", lambda *a, **k: pytest.fail("must reject before execution"))
    with pytest.raises(ValueError, match="ENVELOPE_INVALID"):
        s.decode(raw, "UUP")


def test_original_ohlc_not_repaired(monkeypatch):
    monkeypatch.setattr(s, "decode", lambda raw, symbol: [sample(close=11.001)])
    row = s.parse(b"raw", "UUP")["2025-01-02"]
    assert row["close"] == 11.001 and row["available"] is False


def test_duplicate_or_after_cutoff_rejected(monkeypatch):
    monkeypatch.setattr(s, "decode", lambda raw, symbol: [sample(), sample()])
    with pytest.raises(ValueError, match="DUPLICATE"):
        s.parse(b"raw", "UUP")
    monkeypatch.setattr(s, "decode", lambda raw, symbol: [sample("2026-09-15")])
    with pytest.raises(ValueError, match="FUTURE_ROW"):
        s.parse(b"raw", "UUP", "2026-09-14")
