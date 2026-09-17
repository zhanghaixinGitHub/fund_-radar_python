"""双来源同日比值、联合缺失及发行方交易收盘身份，不联网、不训练真实基金数据。"""

import json
import math
from copy import deepcopy
from html import escape

import pytest
from app.services import direction_1d_sprint_credit_pair_etf_data as s


def quote(opening=100.0, close=101.0):
    return {
        "open": opening,
        "close": close,
        "high": max(opening, close),
        "low": min(opening, close),
        "volume": 100,
        "available": True,
        "unavailable_reason": None,
    }


@pytest.fixture
def pair():
    return {"HYG": {"2026-09-16": quote()}, "IEF": {"2026-09-16": quote(100, 99)}}


def test_two_price_signals_are_not_pure_yield_or_close_to_close_returns(pair):
    pair["HYG"]["2026-09-15"] = quote(1, 1)
    value = s.features("2026-09-16", "2026-09-17", pair)
    assert value["available"]
    assert value["relative_log_pct"] == pytest.approx(100 * (math.log(1.01) - math.log(0.99)))
    assert value["treasury_log_pct"] == pytest.approx(100 * math.log(0.99))


@pytest.mark.parametrize("symbol", ["HYG", "IEF"])
def test_missing_either_leg_is_missing_not_flat(pair, symbol):
    pair.pop(symbol)
    value = s.features("2026-09-16", "2026-09-17", pair)
    assert not value["available"] and value["reason"] == symbol + "_MISSING_US_SESSION"


def test_old_and_future_sessions_cannot_replace_required_session(pair):
    pair["IEF"] = {"2026-09-15": quote(), "2026-09-17": quote()}
    assert not s.features("2026-09-16", "2026-09-17", pair)["available"]


def test_relative_value_is_clipped_after_subtraction_not_before(pair):
    pair["HYG"]["2026-09-16"] = quote(100, 200)
    pair["IEF"]["2026-09-16"] = quote(100, 180)
    value = s.features("2026-09-16", "2026-09-17", pair)
    assert value["relative_log_pct"] == pytest.approx(100 * math.log(2 / 1.8))
    assert value["treasury_log_pct"] == 20


def test_same_day_split_factor_does_not_change_signals(pair):
    changed = deepcopy(pair)
    for k in ["open", "high", "low", "close"]:
        changed["HYG"]["2026-09-16"][k] *= 3
    assert s.features("2026-09-16", "2026-09-17", pair) == s.features("2026-09-16", "2026-09-17", changed)


def test_invalid_ohlc_not_repaired_or_discarded(pair):
    pair["HYG"]["2026-09-16"].update(high=100, available=False, unavailable_reason="OHLC_RANGE_OR_NO_TRADE")
    value = s.features("2026-09-16", "2026-09-17", pair)
    assert not value["available"] and value["reason"] == "HYG_INVALID_OHLC_OR_NO_TRADE"


def fields():
    return {
        "closingPrice": {
            "name": "closingPrice",
            "label": "Closing Price",
            "prefix": "$",
            "formattedValue": "78.38",
            "formattedAsOfDate": "Sep 15, 2026",
        },
        "cusip": {"formattedValue": "464288513"},
        "exchange": {"formattedValue": "NYSE Arca"},
        "consolidatedVolume": {"formattedValue": "42,245,566.00", "formattedAsOfDate": "Sep 15, 2026"},
    }


def html(points):
    return (
        '<walrus-render-on-client componentprops="'
        + escape(json.dumps({"dataPoints": points}), quote=True)
        + '"></walrus-render-on-client>'
    ).encode()


def test_issuer_uses_closing_price_own_date_not_other_header_dates():
    raw = b"NAV as of Sep 16, 2026" + html(fields())
    actual = s.issuer_observation(raw, "HYG")
    assert actual["date"] == "2026-09-15" and actual["close"] == 78.38


@pytest.mark.parametrize("kind", ["cusip", "nav", "volume_date", "duplicates"])
def test_inconsistent_or_nontrade_issuer_field_is_rejected(kind):
    points = fields()
    if kind == "cusip":
        points["cusip"]["formattedValue"] = "wrong"
    elif kind == "nav":
        points["closingPrice"]["label"] = "NAV"
    elif kind == "volume_date":
        points["consolidatedVolume"]["formattedAsOfDate"] = "Sep 14, 2026"
    raw = html(points)
    if kind == "duplicates":
        raw += raw
    with pytest.raises(ValueError):
        s.issuer_observation(raw, "HYG")
