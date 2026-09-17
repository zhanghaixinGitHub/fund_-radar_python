"""黄金官方价格不能被NAV、中点、错误日期段或其他证券身份替代。"""

import pytest
from app.services import direction_1d_sprint_gold_etf_data as d


def page():
    return (
        b'<meta name="ISIN" content="US78463V1070"/><meta name="ticker" content="GLD"/>'
        b'<section><h2>Fund Market Price <span class="date">as of Sep 15 2026</span></h2>'
        b'<table><tr><th>Bid/Ask Midpoint</th><td class="data">$394.11</td></tr>'
        b"<tr><th>Closing Price <span>Exchange closing price</span></th>"
        b'<td class="data">$394.15</td></tr></table></section>'
    )


def test_only_official_closing_price_and_its_date():
    assert d.issuer_observation(page()) == {
        "symbol": "GLD",
        "isin": "US78463V1070",
        "date": "2026-09-15",
        "close": 394.15,
        "kind": "EXCHANGE_CLOSING_PRICE",
    }


@pytest.mark.parametrize(
    "before,after",
    [
        (b"US78463V1070", b"US0000000000"),
        (b'content="GLD"', b'content="OTHER"'),
        (b"Closing Price", b"Net Asset Value"),
        (b"Sep 15 2026", b"not a date"),
        (b"Fund Market Price", b"Fund NAV"),
        (b"$394.15", b"$0"),
    ],
)
def test_wrong_identity_nav_or_missing_values_rejected(before, after):
    with pytest.raises(ValueError):
        d.issuer_observation(page().replace(before, after))


def test_duplicate_market_section_is_ambiguous():
    with pytest.raises(ValueError, match="AMBIGUOUS"):
        d.issuer_observation(page() + page())


def test_gold_feature_missing_source_keeps_explicit_fallback():
    value = d.features("2026-09-15", "2026-09-16", {})
    assert value["available"] is False and value["reason"] == "MISSING_US_SESSION"


def test_original_fund_question_fields_preserved():
    original = {"code": "001021", "y": 1, "market": {"available": True}}
    extended = original | {"market": original["market"] | {"gold_etf": {"available": False}}}
    assert d.original_row(extended) == original
