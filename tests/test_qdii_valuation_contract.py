"""跨时区窗口的风险用例；盘中、次日行情和代理汇率都不能成为已核验收盘资料。"""

import json
from copy import deepcopy

import pytest
from app.services.qdii_valuation_contract import LISTED_FOF_POLICY, POLICY, verified_qdii_window


def sample():
    profile = {"fund_name": "境外行业股票(QDII)-C-CNY"}
    prospectus = "估值日为相关交易日。股票按收盘价估值。估值采用人民币汇率中间价。"
    calendar = {
        "year": 2026,
        "closed_dates": [f"2026-01-{x:02}" for x in range(1, 11)],
        "early_close_dates": ["2026-11-27", "2026-12-24"],
        "source_text": "Core Trading Session: 9:30 a.m. to 4:00 p.m. ET",
    }
    docs = {
        "report": {"kind": "REPORT", "codes": ["018853"]},
        "prospectus": {"kind": "PROSPECTUS", "codes": ["018853"], "body": prospectus},
        "fx": {
            "kind": "FX_REFERENCE_DATA",
            "url": "https://www.safe.gov.cn/fx",
            "body": json.dumps({"rates": [{"currency": "USD", "date": "2026-10-14", "cny_per_unit": "6.7249"}]}),
        },
        "quotes": {"kind": "MARKET_DATA", "available_at": "2026-10-15T04:40:00+08:00"},
        "calendar": {
            "kind": "EXCHANGE_CALENDAR",
            "url": "https://www.nyse.com/trade/hours-calendars",
            "body": json.dumps(calendar),
        },
    }
    quote = {
        "date": "2026-10-14",
        "currency": "USD",
        "valuation_market": "US",
        "session": "REGULAR_CLOSE",
        "closed_at": "2026-10-14T16:00:00-04:00",
        "change_pct": -1,
        "source_ref": "quotes",
    }
    material = {
        "fund_code": "018853",
        "companies": [{"code": "TPL.US", "quote": quote}],
        "market": {},
        "calendar_evidence": {
            "policy": POLICY,
            "base_date": "2026-10-14",
            "target_date": "2026-10-15",
            "report_ref": "report",
            "prospectus_ref": "prospectus",
            "fx_ref": "fx",
            "calendar_refs": {"US": "calendar"},
            "valuation_quotes": {
                "day": "估值日为相关交易日。",
                "close": "股票按收盘价估值。",
                "fx": "估值采用人民币汇率中间价。",
            },
        },
    }
    return profile, material, docs, {"base_nav_date": "2026-10-14", "target_nav_date": "2026-10-15"}


def test_completed_us_baseline_is_available_before_chinese_target_day():
    assert verified_qdii_window(*sample())


def test_listed_fof_requires_actual_fund_valuation_and_issuer_lookthrough():
    p, m, d, w = deepcopy(sample())
    p["fund_name"] = "原油基金(QDII-FOF)-C"
    e = m["calendar_evidence"]
    e["policy"] = LISTED_FOF_POLICY
    close = "上市流通的基金按估值日收盘价估值。"
    d["prospectus"]["body"] += close
    e["valuation_quotes"]["close"] = close
    d["report"]["body"] = "母基金实际持有美国原油基金。"
    product = "USO issues shares on NYSE Arca. The fund invests in oil futures."
    d["product"] = {"kind": "FUND_PRODUCT_INFO", "url": "https://www.uscfinvestments.com/uso", "body": product}
    m["companies"][0].update(code="USO.US", asset_type="LISTED_FUND")
    e["listed_fund_evidence"] = {
        "USO.US": {"product_ref": "product", "product_quote": product, "parent_quote": d["report"]["body"]}
    }
    assert verified_qdii_window(p, m, d, w)
    m["companies"][0]["asset_type"] = "UNLISTED_FUND"
    assert not verified_qdii_window(p, m, d, w)
    m["companies"][0]["asset_type"] = "LISTED_FUND"
    d["product"]["url"] = "https://example.com/uso"
    assert not verified_qdii_window(p, m, d, w)


def test_japan_closing_session_and_national_holiday_are_checked():
    p, m, d, w = deepcopy(sample())
    q = m["companies"][0]["quote"]
    q.update(valuation_market="JP", currency="JPY", closed_at="2026-10-14T15:30:00+09:00")
    d["fx"]["body"] = json.dumps({"rates": [{"currency": "JPY", "date": "2026-10-14", "cny_per_unit": "0.042412"}]})
    c = {
        "year": 2026,
        "closed_dates": [f"2026-01-{day:02}" for day in range(1, 21)],
        "hours_source_text": "afternoon session is from 12:30 pm to 3:30 pm",
        "hours_source_url": "https://www.jpx.co.jp/english/equities/trading/domestic/01.html",
    }
    d["calendar"].update(url="https://www.jpx.co.jp/calendar", body=json.dumps(c))
    m["calendar_evidence"]["calendar_refs"] = {"JP": "calendar"}
    assert verified_qdii_window(p, m, d, w)
    q["closed_at"] = "2026-10-14T15:00:00+09:00"
    assert not verified_qdii_window(p, m, d, w)
    q["closed_at"] = "2026-10-14T15:30:00+09:00"
    c["closed_dates"].append("2026-10-15")
    d["calendar"]["body"] = json.dumps(c)
    assert not verified_qdii_window(p, m, d, w)


@pytest.mark.parametrize(
    "problem",
    [
        "intraday",
        "target_close",
        "before_close",
        "ecb",
        "missing_currency",
        "wrong_share",
        "wrong_fund",
        "fof",
        "holiday",
        "invented_clause",
        "unknown_market",
    ],
)
def test_foreign_data_cannot_be_mislabeled_as_verified_baseline(problem):
    p, m, d, w = deepcopy(sample())
    q = m["companies"][0]["quote"]
    if problem == "intraday":
        q["session"] = "INTRADAY"
    elif problem == "target_close":
        q["date"] = "2026-10-15"
    elif problem == "before_close":
        d["quotes"]["available_at"] = "2026-10-14T22:55:00+08:00"
    elif problem == "ecb":
        d["fx"]["url"] = "https://www.ecb.europa.eu/rates"
    elif problem == "missing_currency":
        q["currency"] = "VND"
    elif problem == "wrong_share":
        p["fund_name"] += "美元"
    elif problem == "wrong_fund":
        d["prospectus"]["codes"] = ["000001"]
    elif problem == "fof":
        p["fund_name"] += "FOF"
    elif problem == "holiday":
        c = json.loads(d["calendar"]["body"])
        c["closed_dates"].append("2026-10-15")
        d["calendar"]["body"] = json.dumps(c)
    elif problem == "invented_clause":
        m["calendar_evidence"]["valuation_quotes"]["fx"] = "本基金使用中间价虚构条款"
    else:
        q["valuation_market"] = "JP"
    assert not verified_qdii_window(p, m, d, w)
