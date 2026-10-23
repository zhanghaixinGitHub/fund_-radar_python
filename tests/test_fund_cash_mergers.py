"""现金交割只改变普通报价适用性；原份额、其他行情及真实截止仍须核验。"""

import json
from copy import deepcopy
from datetime import datetime

import pytest
from app.services.direction_1d_protocol import digest
from app.services.fund_cash_mergers import verified_cash_merged_security
from app.services.fund_information_contracts import time_context
from app.services.qdii_valuation_contract import POLICY, verified_qdii_window


def sample(symbol="EA"):
    is_ea = symbol == "EA"
    code = symbol + ".US"
    cik, day, first = (
        ("712515", "August 4, 2026", "2026-08-05") if is_ea else ("1437107", "October 6, 2026", "2026-10-06")
    )
    quantity = "16614" if is_ea else "166017"
    quotes = [
        f"Common Stock, $0.01 par value\n{symbol}\nNASDAQ Global Select Market",
        f"On {day} (the Closing Date), "
        + (
            "Merger Sub merged with and into the Company."
            if is_ea
            else "SKYD completed the previously announced acquisition of WBD."
        ),
        "Each share of Company Common Stock was converted into the right to receive "
        + (
            "$210 per share in cash, without interest."
            if is_ea
            else "an amount in cash equal to $31.01666668, without interest."
        ),
        "Trading of the Company Common Stock on Nasdaq was halted after the close of trading on the Closing Date."
        if is_ea
        else "WBD requested that Nasdaq halt trading of WBD Common Stock on Nasdaq prior to the opening "
        "of trading on the Closing Date.",
        "Holders of Company Common Stock ceased to have any rights as shareholders, other than the right "
        "to receive the Merger Consideration.",
    ]
    position = f"159632 2026年中期报告 {symbol} US 美国 {int(quantity):,}股。"
    docs = {}
    for key, kind, body, codes, url in [
        (
            "sec",
            "ANNOUNCEMENT",
            "FORM 8-K\n" + "\n".join(quotes),
            [code],
            f"https://www.sec.gov/Archives/edgar/data/{cik}/000123456726000001/filing8k.htm",
        ),
        ("positions", "REPORT", position, ["159632"], "https://example.com/etf-report"),
        ("own", "REPORT", "014978本基金持有159632。", ["014978"], "https://example.com/own-report"),
    ]:
        docs[key] = {
            "id": key,
            "kind": kind,
            "body": body,
            "body_hash": digest(body),
            "codes": codes,
            "url": url,
            "published_date": "2026-08-31" if key != "sec" else "2026-10-06",
            "available_at": "2026-10-20T18:00:00+08:00",
        }
    item = {
        "kind": "US_COMPLETED_CASH_MERGER",
        "purpose": "REGULAR_TRADING_QUOTE_ONLY",
        "fund_code": "014978",
        "stock_code": code,
        "base_date": "2026-10-20",
        "target_date": "2026-10-21",
        "merger_date": "2026-08-04" if is_ea else "2026-10-06",
        "first_non_trading_date": first,
        "sec_ref": "sec",
        "report_ref": "positions",
        "report_fund_code": "159632",
        "fact_id": "cash-merger",
        "position_quote": position,
    }
    item.update(
        dict(
            zip(["identity_quote", "completion_quote", "cash_quote", "exit_quote", "rights_quote"], quotes, strict=True)
        )
    )
    company = {"code": code, "quote": None, "reported_quantity_shares": quantity, "weight_pct": 0.2}
    data = {
        "fund_code": "014978",
        "as_of": "2026-10-21T05:00:00+08:00",
        "report": {"endDate": "2026-06-30"},
        "exposure_chain": ["014978", "159632"],
        "window": {"base_nav_date": "2026-10-20", "target_nav_date": "2026-10-21"},
        "companies": [company],
        "market": {},
        "documents": list(docs.values()),
        "nav": [{"nav_date": "2026-10-20"}],
        "latest_nav_date": "2026-10-20",
        "market_quote_exceptions": {code: item},
        "facts": {
            "cash-merger": {
                "required_in_analysis": True,
                "direction_eligible": False,
                "source_refs": ["sec", "positions"],
                "quotes": quotes + [position],
            }
        },
    }
    return data, docs, company


@pytest.mark.parametrize("symbol,price", [("EA", "210"), ("WBD", "31.01666668")])
def test_real_cash_terms_do_not_create_a_quote_or_estimate_fund_cash(symbol, price):
    data, docs, company = sample(symbol)
    before = deepcopy(data)
    proof = verified_cash_merged_security(data, docs, company)
    assert proof["cash_consideration_per_share"] == price
    assert proof["cash_received_by_fund_proved"] is False
    assert proof["fund_current_quantity_or_fair_value_proved"] is False
    assert proof["replacement_quote_created"] is False
    assert data == before and company["quote"] is None and company["weight_pct"] == 0.2
    result = time_context(data, ["2026-10-19", "2026-10-20", "2026-10-21", "2026-10-22"])
    assert result["baseline_complete"]
    assert result["stocks"][company["code"]]["status"] == "READY_DOCUMENTED_NON_TRADING"


@pytest.mark.parametrize(
    "problem",
    [
        "other_fund",
        "other_stock",
        "wrong_window",
        "changed_date",
        "future_base",
        "ordinary_stop",
        "bond_form25",
        "wrong_host",
        "wrong_cik",
        "changed_hash",
        "future_received",
        "future_published",
        "same_day_without_time",
        "missing_quote",
        "wrong_position",
        "wrong_quantity",
        "wrong_target_etf",
        "unread",
        "direction",
        "fake_current_quote",
        "listed_fund",
        "post_merger_report",
    ],
)
def test_missing_or_conflicting_cash_merger_evidence_is_not_an_exception(problem):
    data, docs, company = sample()
    item = data["market_quote_exceptions"]["EA.US"]
    if problem == "other_fund":
        item["fund_code"] = "018156"
    elif problem == "other_stock":
        item["stock_code"] = "WBD.US"
    elif problem == "wrong_window":
        item["target_date"] = "2026-10-22"
    elif problem == "changed_date":
        item["first_non_trading_date"] = "2026-08-04"
    elif problem == "future_base":
        data["window"]["base_nav_date"] = item["base_date"] = "2026-08-01"
    elif problem == "ordinary_stop":
        item["exit_quote"] = "ordinary suspension"
    elif problem == "bond_form25":
        docs["sec"]["body"] = "FORM 25 4.302% Senior Notes due 2030"
    elif problem == "wrong_host":
        docs["sec"]["url"] = docs["sec"]["url"].replace("www.sec.gov", "www.sec.gov.evil.test")
    elif problem == "wrong_cik":
        docs["sec"]["url"] = docs["sec"]["url"].replace("712515", "1437107")
    elif problem == "changed_hash":
        docs["sec"]["body_hash"] = "changed"
    elif problem == "future_received":
        docs["sec"]["available_at"] = "2026-10-21T06:00:00+08:00"
    elif problem == "future_published":
        docs["sec"]["published_date"] = "2026-10-22"
    elif problem == "same_day_without_time":
        docs["sec"]["published_date"] = "2026-10-21"
    elif problem == "missing_quote":
        item["cash_quote"] = ""
    elif problem == "wrong_position":
        item["position_quote"] = "159632 OTHER US 16,614股"
    elif problem == "wrong_quantity":
        company["reported_quantity_shares"] = "99999"
    elif problem == "wrong_target_etf":
        data["exposure_chain"] = ["014978", "159633"]
    elif problem == "unread":
        data["facts"]["cash-merger"]["required_in_analysis"] = False
    elif problem == "direction":
        data["facts"]["cash-merger"]["direction_eligible"] = True
    elif problem == "fake_current_quote":
        company["quote"] = {"date": "2026-10-20", "change_pct": 0}
    elif problem == "listed_fund":
        company["asset_type"] = "LISTED_FUND"
    else:
        data["report"]["endDate"] = "2026-09-30"
    assert verified_cash_merged_security(data, docs, company) is None


def qdii_sample():
    data, docs, company = sample()
    clauses = {"day": "估值日为相关交易日。", "close": "股票按收盘价估值。", "fx": "估值采用人民币汇率中间价。"}
    docs["prospectus"] = {
        "kind": "PROSPECTUS",
        "codes": ["014978"],
        "body": "".join(clauses.values()) + "目标 ETF 按净值估值。",
    }
    docs["fx"] = {
        "kind": "FX_REFERENCE_DATA",
        "url": "https://www.safe.gov.cn/fx",
        "body": json.dumps({"rates": [{"currency": "USD", "date": "2026-10-20", "cny_per_unit": "6.72"}]}),
    }
    docs["prices"] = {"kind": "MARKET_DATA", "available_at": "2026-10-21T04:30:00+08:00"}
    docs["calendar"] = {
        "kind": "EXCHANGE_CALENDAR",
        "url": "https://www.nyse.com/trade/hours-calendars",
        "body": json.dumps(
            {
                "year": 2026,
                "closed_dates": [f"2026-01-{n:02}" for n in range(1, 11)],
                "early_close_dates": ["2026-11-27", "2026-12-24"],
                "source_text": "Core Trading Session: 4:00 p.m.",
            }
        ),
    }
    quote = {
        "date": "2026-10-20",
        "currency": "USD",
        "valuation_market": "US",
        "session": "REGULAR_CLOSE",
        "closed_at": "2026-10-20T16:00:00-04:00",
        "change_pct": 1.0,
        "source_ref": "prices",
    }
    data["companies"].append({"code": "TPL.US", "quote": quote})
    data["calendar_evidence"] = {
        "policy": POLICY,
        "base_date": "2026-10-20",
        "target_date": "2026-10-21",
        "prospectus_ref": "prospectus",
        "report_ref": "own",
        "target_report_ref": "positions",
        "fx_ref": "fx",
        "calendar_refs": {"US": "calendar"},
        "valuation_quotes": clauses,
        "target_etf_valuation_quote": "目标 ETF 按净值估值。",
    }
    return {"fund_name": "华安纳斯达克100ETF联接(QDII)C"}, data, docs, data["window"]


def test_qdii_and_time_checks_use_same_exception_without_skipping_other_data():
    p, m, d, w = qdii_sample()
    at = datetime.fromisoformat(m["as_of"])
    assert verified_qdii_window(p, m, d, w, as_of=at)
    assert not verified_qdii_window(p, m, d, w)  # 缺真实截止不能核验例外。
    m["companies"][1]["quote"]["session"] = "INTRADAY"
    assert not verified_qdii_window(p, m, d, w, as_of=at)
    m["companies"][1]["quote"]["session"] = "REGULAR_CLOSE"
    d["fx"]["body"] = json.dumps({"rates": []})
    assert not verified_qdii_window(p, m, d, w, as_of=at)


def test_documented_merger_never_clears_missing_nav_or_other_stock():
    data, _, _ = sample()
    sessions = ["2026-10-19", "2026-10-20", "2026-10-21", "2026-10-22"]
    data["nav"] = [{"nav_date": "2026-10-19"}]
    data["latest_nav_date"] = "2026-10-19"
    assert not time_context(data, sessions)["baseline_complete"]
    data["nav"] = [{"nav_date": "2026-10-20"}]
    data["latest_nav_date"] = "2026-10-20"
    data["companies"].append({"code": "AAPL.US", "quote": None})
    assert not time_context(data, sessions)["baseline_complete"]
