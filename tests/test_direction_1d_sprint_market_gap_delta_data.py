"""两项来源特征的数学/时间口径和原始输入绑定，不访问供应商。"""

import pytest
from app.services import direction_1d_sprint as b
from app.services import direction_1d_sprint_market_gap_delta_data as d


def install(monkeypatch, session=10.0, daily=21.0, flag=True):
    monkeypatch.setattr(d.parent.etfs, "features", lambda *args: [session, daily, 0, 0, float(flag)])
    monkeypatch.setattr(d.parent.cnya, "features", lambda *args: [5.0, 3.0, 1.0])
    return {
        "features": [0.5, session, 5.0],
        "available": flag,
        "etf_available": flag,
        "cnya_available": True,
        "us_sessions": 1.0,
    }


def test_opening_gap_uses_compounding_not_return_subtraction(monkeypatch):
    market = install(monkeypatch)
    value = d.extend(market, "2026-09-15", "2026-09-16", {}, {})
    assert value["features"] == pytest.approx([0.5, 10, 5, 10, 3])
    assert value["features"][3] != 21 - 10
    assert value["features"][:3] == market["features"]


def test_original_price_gap_keeps_ex_dividend_effect_and_has_no_cash_reinvestment(monkeypatch):
    market = install(monkeypatch, session=0.0, daily=-2.0)
    value = d.extend(market, "2026-09-15", "2026-09-16", {}, {})
    assert value["features"][3] == pytest.approx(-2.0)


def test_unavailable_market_does_not_invent_extra_features(monkeypatch):
    market = install(monkeypatch, session=0.0, daily=0.0, flag=False)
    result = d.extend(market, "2026-09-15", "2026-09-16", {}, {})
    assert result["features"][-2:] == [0.0, 0.0] and result["available"] is False


@pytest.mark.parametrize("fault", ["old_value", "old_flag", "zero_price"])
def test_invalid_or_different_parent_input_rejected(monkeypatch, fault):
    market = install(monkeypatch, session=-100.0 if fault == "zero_price" else 10.0)
    if fault == "old_value":
        market["features"][1] = 2.0
    if fault == "old_flag":
        market["available"] = False
    with pytest.raises(ValueError):
        d.extend(market, "2026-09-15", "2026-09-16", {}, {})


def test_live_extension_reuses_bound_original_sources_and_rejects_changed_source(monkeypatch):
    market = install(monkeypatch)
    etf = {"base": "2026-09-15", "rows": {"synthetic": "ETF"}}
    cn = {"base": "2026-09-15", "rows": {"synthetic": "CNYA"}}
    source = {
        "base": "2026-09-15",
        "target": "2026-09-16",
        "market": market,
        "etf_hash": b.digest(etf),
        "cnya_hash": b.digest(cn),
    }
    monkeypatch.setattr(d.parent, "load", lambda target: source)
    monkeypatch.setattr(d.parent.etfs, "load", lambda target: etf)
    monkeypatch.setattr(d.parent.cnya, "load", lambda target: cn)
    result = d.live_market(source)
    assert result["features"] == pytest.approx([0.5, 10, 5, 10, 3])
    etf["rows"]["tampered"] = True
    with pytest.raises(ValueError, match="LIVE_SOURCE_CHANGED"):
        d.live_market(source)


def test_training_projection_preserves_label_and_maturity():
    old = {"code": "A", "u": "2025-01-02", "mature": "2025-01-09", "y": 0, "market": {"features": [1, 2, 3]}}
    added = old | {"market3": old["market"], "market": {"features": [1, 2, 3, 4, 5]}}
    assert d.original_row(added) == old
