"""实际公开表格版式及对抗用例：金额、币种、份额、时段和日历不得互相替代。"""

import json
from copy import deepcopy
from pathlib import Path

import pytest
from app.services.fund_listed_product_contract import (
    POLICY,
    PRODUCTS,
    european_calendar,
    european_market_ready,
    parent_positions,
    verified_products,
)
from app.services.qdii_valuation_contract import verified_qdii_window


@pytest.fixture
def parent(monkeypatch):
    # 这里隔离资产负债表读取器；另有真实整份原件在本机适配验收中按原链路对账。
    monkeypatch.setattr(
        "app.services.fund_holdings_publication.balance",
        lambda *a, **k: {
            "fund_amount": "1266058491.46",
            "net_assets": "1346272887.59",
        },
    )
    return Path(__file__).with_name("fixtures").joinpath("listed_oil_parent_excerpt.txt").read_text("utf-8")


def test_real_cross_column_amounts_reconcile_and_currency_is_derived(parent):
    rows = parent_positions(parent)
    assert [r["amount"] for r in rows] == [
        "252904113.99",
        "251403765.01",
        "251149576.53",
        "247900303.23",
        "114030869.30",
        "75727504.29",
        "72942359.11",
    ]
    assert [r["currency"] for r in rows] == ["USD"] * 5 + ["JPY"] * 2


@pytest.mark.parametrize(
    "old,new",
    [
        ("252,904,11", "252,904,12"),
        ("18.79", "19.79"),
        ("1,117,388,628.", "1,117,388,629."),
        ("148,669,863.40", "148,669,862.40"),
        ("    7      Long", "    8      Long"),
        ("Simplex", "WrongIssuer"),
    ],
)
def test_wrong_amount_weight_currency_row_or_parent_name_is_not_filled(parent, old, new):
    assert old in parent
    with pytest.raises(ValueError):
        parent_positions(parent.replace(old, new))


def sample(parent):
    docs = {
        "report": {"kind": "REPORT", "codes": ["006476"], "body": parent},
        "prospectus": {
            "kind": "PROSPECTUS",
            "codes": ["006476"],
            "body": "本基金的估值日为本基金的开放日。上市流通的基金按估值日收盘价估值。"
            "本基金外币资产价值按人民币汇率中间价折算。",
        },
        "fx": {
            "kind": "FX_REFERENCE_DATA",
            "url": "https://www.chinamoney.com.cn/fx",
            "body": json.dumps(
                {
                    "rates": [
                        {"currency": c, "date": "2026-10-20", "cny_per_unit": v}
                        for c, v in [("USD", 6.7), ("JPY", 0.04)]
                    ],
                }
            ),
        },
    }
    positions = parent_positions(parent)
    m = {
        "fund_code": "006476",
        "companies": [],
        "market": {},
        "facts": {
            "identity-limit": {"required_in_analysis": True, "direction_eligible": False},
        },
        "calendar_evidence": {
            "policy": POLICY,
            "base_date": "2026-10-20",
            "target_date": "2026-10-21",
            "report_ref": "report",
            "prospectus_ref": "prospectus",
            "fx_ref": "fx",
            "listed_identity_limitation_fact": "identity-limit",
            "listed_fund_evidence": {},
            "calendar_refs": {k: "calendar-" + k for k in ["US", "JP", "UK", "CH"]},
            "market_hours_refs": {k: "hours-" + k for k in ["UK", "CH"]},
            "valuation_quotes": {
                "day": "本基金的估值日为本基金的开放日。",
                "close": "上市流通的基金按估值日收盘价估值。",
                "fx": "本基金外币资产价值按人民币汇率中间价折算。",
            },
        },
    }
    bounds = {
        "US": "2026-10-20T16:00:00-04:00",
        "JP": "2026-10-20T15:30:00+09:00",
        "UK": "2026-10-20T17:15:00+01:00",
        "CH": "2026-10-20T17:35:00+02:00",
    }
    for slot, (symbol, (name, currency, market, isin, asset_type)) in enumerate(PRODUCTS.items()):
        product_id, quote_id = "product-" + symbol, "quote-" + symbol
        if market == "US":
            url, body = "https://www.uscfinvestments.com/" + symbol, name + " NYSE Arca oil futures"
        elif market == "UK":
            url = "https://dataspanapi.wisdomtree.com/" + isin
            body = (
                name
                + "\nBase Currency USD\nCurrency Hedged No\nDebt security / ETC\nfutures\n"
                + f"LSE USD {symbol.split('.')[0]} test {isin}\n"
            )
        elif market == "CH":
            url = "https://swissfunddata.ch/prospectus.pdf"
            body = "C. – CMCI Oil SF ETF (USD) A-dis\nCH0109967858\nOILUSA"
            docs["listing-CH"] = {
                "kind": "PRODUCT_LISTING_INFO",
                "url": "https://www.six-group.com/listing.csv",
                "body": "ISIN;ValorSymbol;TradingBaseCurrency;Exchange\nCH0109967858;OILUSA;USD;XSWX\n",
            }
        else:
            url = "https://www.simplexasset.com/1671" if symbol == "1671.JP" else "https://nextfunds.jp/1699"
            body = name + " " + symbol.split(".")[0] + " 原油先物"
            docs["listing-" + symbol] = {
                "kind": "PRODUCT_LISTING_INFO",
                "url": "https://www.jpx.co.jp/etfs",
                "body": symbol.split(".")[0] + " " + name + " Futures Type",
            }
        docs[product_id] = {"kind": "FUND_PRODUCT_INFO", "url": url, "body": body}
        docs[quote_id] = {"kind": "MARKET_DATA", "available_at": "2026-10-21T04:30:00+08:00"}
        quote = {
            "date": "2026-10-20",
            "currency": currency,
            "valuation_market": market,
            "session": "REGULAR_CLOSE",
            "closed_at": bounds[market],
            "change_pct": 0.5,
            "source_ref": quote_id,
        }
        if market in {"UK", "CH"}:
            quote["close_timestamp_kind"] = "EXCHANGE_COMPLETION_BOUND"
        m["companies"].append(
            {"code": symbol, "asset_type": asset_type, "weight_pct": positions[slot]["weight_pct"], "quote": quote}
        )
        m["calendar_evidence"]["listed_fund_evidence"][symbol] = {
            "product_ref": product_id,
            "parent_quote": parent,
            "product_quote": body,
            **(
                {"listing_ref": "listing-CH"}
                if market == "CH"
                else {"listing_ref": "listing-" + symbol}
                if market == "JP"
                else {}
            ),
        }
    docs["calendar-US"] = {
        "kind": "EXCHANGE_CALENDAR",
        "url": "https://www.nyse.com/hours",
        "body": json.dumps(
            {
                "year": 2026,
                "closed_dates": [f"2026-01-{d:02}" for d in range(1, 11)],
                "early_close_dates": ["2026-11-27", "2026-12-24"],
                "source_text": "Core Trading Session 4:00 p.m.",
            }
        ),
    }
    docs["calendar-JP"] = {
        "kind": "EXCHANGE_CALENDAR",
        "url": "https://www.jpx.co.jp/calendar",
        "body": json.dumps(
            {
                "year": 2026,
                "closed_dates": [f"2026-01-{d:02}" for d in range(1, 21)],
                "hours_source_text": "3:30 pm",
                "hours_source_url": "https://www.jpx.co.jp/hours",
            }
        ),
    }
    docs["calendar-CH"] = {
        "kind": "EXCHANGE_CALENDAR",
        "url": "https://www.six-group.com/sheldon/trading/v1/2026/calendar.json",
        "body": json.dumps(
            {"status": "Ok", "itemList": [{"mic": "XSWX", "holidays": [{"date": 20260100 + d} for d in range(1, 11)]}]}
        ),
    }
    docs["calendar-UK"] = {
        "kind": "EXCHANGE_CALENDAR",
        "url": "https://www.londonstockexchange.com/business-days",
        "body": json.dumps(
            {
                "coverage_start": "2026-08-31",
                "coverage_end": "2026-12-31",
                "weekday_clause": "London Stock Exchange each weekday England & Wales",
                "exchange_rows": [
                    ["Monday 31 August 2026", "Holiday", "NON-trading day"],
                    ["Thursday 24 December 2026", "Half", "12:30 London time"],
                    ["Friday 25 December 2026", "Holiday", "NON-trading day"],
                    ["Monday 28 December 2026", "Holiday", "NON-trading day"],
                    ["Thursday 31 December 2026", "Half", "12:30 London time"],
                ],
            }
        ),
    }
    docs["hours-CH"] = {
        "kind": "EXCHANGE_TRADING_HOURS",
        "url": "https://www.six-group.com/hours",
        "body": "17:35 Run Auction and Close Exchange Traded Funds",
    }
    docs["hours-UK"] = {
        "kind": "EXCHANGE_TRADING_HOURS",
        "url": "https://docs.londonstockexchange.com/parameters.xlsx",
        "body": "17:15 End of TRADEcho manual reporting - On Exchange; closing auction extensions and random periods",
    }
    return (
        {"fund_name": "南方原油(QDII-LOF-FOF)-C"},
        m,
        docs,
        {"base_nav_date": "2026-10-20", "target_nav_date": "2026-10-21"},
    )


def test_complete_seven_products_can_reach_original_window_checks(parent):
    p, m, d, w = sample(parent)
    assert verified_products(m, d, m["calendar_evidence"])
    assert verified_qdii_window(p, m, d, w)


@pytest.mark.parametrize(
    "problem",
    [
        "drop",
        "currency",
        "issuer",
        "listing",
        "isin",
        "legal",
        "limitation",
        "old_policy",
        "intraday",
        "before_auction",
        "wrong_close",
    ],
)
def test_multimarket_products_do_not_accept_missing_or_mislabeled_inputs(parent, problem):
    p, m, d, w = deepcopy(sample(parent))
    if problem == "drop":
        m["companies"].pop()
    elif problem == "currency":
        m["companies"][4]["quote"]["currency"] = "CHF"
    elif problem == "issuer":
        d["product-BRNT.L"]["url"] = "https://example.com/JE00B78CGV99"
    elif problem == "listing":
        d["listing-CH"]["body"] = d["listing-CH"]["body"].replace(";USD;", ";CHF;")
    elif problem == "isin":
        d["product-CRUD.L"]["body"] = d["product-CRUD.L"]["body"].replace("GB00B15KXV33", "BAD")
    elif problem == "legal":
        m["companies"][0]["asset_type"] = "LISTED_FUND"
    elif problem == "limitation":
        m["facts"]["identity-limit"]["required_in_analysis"] = False
    elif problem == "old_policy":
        m["calendar_evidence"]["policy"] = "QDII_CNY_LISTED_FUND_SESSION_V1"
    elif problem == "intraday":
        m["companies"][0]["quote"]["session"] = "INTRADAY"
    elif problem == "before_auction":
        d["quote-OILUSA.SW"]["available_at"] = "2026-10-20T23:31:00+08:00"
    else:
        m["companies"][0]["quote"]["closed_at"] = "2026-10-20T16:30:00+01:00"
    assert not verified_qdii_window(p, m, d, w)


def test_uk_calendar_is_bounded_and_swiss_calendar_uses_market_not_currency(parent):
    _, m, d, w = sample(parent)
    closed, start, end = european_calendar("UK", d["calendar-UK"], 2026)
    assert start.isoformat() == "2026-08-31" and end.isoformat() == "2026-12-31"
    assert date_in("2026-12-24", closed)
    q = m["companies"][0]["quote"]
    assert not european_market_ready("UK", q, d, m["calendar_evidence"], "2026-08-20", w["target_nav_date"])
    v = json.loads(d["calendar-CH"]["body"])
    v["itemList"][0]["mic"] = None
    d["calendar-CH"]["body"] = json.dumps(v)
    with pytest.raises(ValueError):
        european_calendar("CH", d["calendar-CH"], 2026)


def date_in(value, dates):
    return any(day.isoformat() == value for day in dates)


def test_future_listed_calendar_stops_at_first_real_market_difference(parent):
    from datetime import date, timedelta

    from app.services.comprehensive_multi_prediction import _matched_future_sessions
    from app.services.direction_1d_protocol import digest
    from app.services.prediction_contract import PredictionFailure

    _, material, docs, window = sample(parent)
    material["window"] = window
    clause = "本基金开放日为上海、深圳及境外主要投资市场的共同交易日，主要市场包括纽约、伦敦和东京。"
    docs["prospectus"]["body"] += clause
    material["calendar_evidence"]["joint_open_days_quote"] = clause
    for d in docs.values():
        d["body_hash"] = digest(d.get("body", ""))
    jp = json.loads(docs["calendar-JP"]["body"])
    jp["closed_dates"].append("2026-11-03")
    docs["calendar-JP"]["body"] = json.dumps(jp)
    days = tuple(
        date(2026, 10, 20) + timedelta(days=i)
        for i in range(73)
        if (date(2026, 10, 20) + timedelta(days=i)).weekday() < 5
    )
    actual, end, _ = _matched_future_sessions(material, docs, days, date(2026, 12, 31), "cn")
    assert end == date(2026, 11, 2) and actual[-1] == date(2026, 11, 2)
    material["calendar_evidence"].pop("joint_open_days_quote")
    with pytest.raises(PredictionFailure, match="条款"):
        _matched_future_sessions(material, docs, days, date(2026, 12, 31), "cn")
