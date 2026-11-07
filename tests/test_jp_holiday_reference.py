"""日本休市参考必须绑定本份额条款，不能掩盖净值、真实日期及来源缺口。"""

import json
from copy import deepcopy

import pytest
from app.services.fund_information_contracts import time_context
from app.services.fund_prediction_followthrough import recovery_actions
from app.services.fund_valuation_sessions import verified_jp_holiday_reference
from app.services.qdii_valuation_contract import verified_qdii_window


def sample():
    clauses = {
        "close": "交易所上市流通的权益类证券按收盘价估值；估值日无交易的，以最近交易日收盘价估值。",
        "fx": "外币资产价值估值汇率采用中国人民银行的人民币汇率中间价。",
    }
    docs = {
        "r": {"id": "r", "kind": "REPORT", "codes": ["019449"]},
        "p": {"id": "p", "kind": "PROSPECTUS", "codes": ["019449"], "body": "估值日\n" + "\n".join(clauses.values())},
        "cal": {
            "id": "cal",
            "kind": "EXCHANGE_CALENDAR",
            "url": "https://www.jpx.co.jp/holidays",
            "body": json.dumps(
                {
                    "year": 2026,
                    "closed_dates": [*(f"2026-01-{i:02d}" for i in range(1, 21)), "2026-11-03"],
                    "hours_source_text": "3:30 pm",
                    "hours_source_url": "https://www.jpx.co.jp/hours",
                }
            ),
        },
        "fx": {
            "id": "fx",
            "kind": "FX_REFERENCE_DATA",
            "url": "https://www.safe.gov.cn/fx",
            "body": json.dumps({"rates": [{"date": "2026-11-03", "currency": "JPY", "cny_per_unit": "0.043"}]}),
        },
        "q": {"id": "q", "kind": "MARKET_DATA", "available_at": "2026-11-03T01:00:00+08:00"},
    }
    q = {
        "date": "2026-11-02",
        "close": 100,
        "change_pct": -1.1,
        "session": "REGULAR_CLOSE",
        "currency": "JPY",
        "valuation_market": "JP",
        "closed_at": "2026-11-02T15:30:00+09:00",
        "source_ref": "q",
    }
    w = {"base_nav_date": "2026-11-03", "target_nav_date": "2026-11-04"}
    m = {
        "fund_code": "019449",
        "companies": [{"code": "8035.JP", "quote": q, "financials": ["original"]}],
        "market": {"TOPIX": deepcopy(q)},
        "as_of": "2026-11-04T01:00:00+08:00",
        "window": w,
        "calendar_evidence": {
            "policy": "QDII_CNY_VERIFIED_SESSION_V1",
            "base_date": w["base_nav_date"],
            "target_date": w["target_nav_date"],
            "report_ref": "r",
            "prospectus_ref": "p",
            "fx_ref": "fx",
            "calendar_refs": {"JP": "cal"},
            "holiday_valuation_quotes": clauses,
            "valuation_quotes": {"day": "估值日", **clauses},
        },
    }
    return m, docs, w


def test_jp_reference_preserves_date_and_requires_current_nav():
    m, d, w = sample()
    before = deepcopy(m)
    assert (
        verified_jp_holiday_reference(m, d, "8035.JP", m["companies"][0]["quote"], w)["reference_date"] == "2026-11-02"
    )
    assert verified_qdii_window({"fund_name": "日本股票QDII人民币C"}, m, d, w)
    m.update(documents=list(d.values()), nav=[{"nav_date": "2026-11-03"}], latest_nav_date="2026-11-03")
    timing = time_context(m, ["2026-11-02", "2026-11-03", "2026-11-04"])
    assert timing["baseline_complete"] and timing["markets"]["TOPIX"]["status"] == "READY_LAST_SESSION"
    actions = recovery_actions(
        "019449",
        {"holdings": [{"stockCode": "8035.JP"}], "endDate": "2026-09-30"},
        {**m, "report": {"endDate": "2026-09-30"}},
        {},
        {},
        "2026-11-03",
    )
    assert not any(a["action"] == "SUPPLEMENT_HOLDING_MATERIALS" for a in actions)
    m.update(nav=[{"nav_date": "2026-11-02"}], latest_nav_date="2026-11-02")
    assert not time_context(m, ["2026-11-02", "2026-11-03", "2026-11-04"])["baseline_complete"]
    assert before["companies"] == m["companies"]


@pytest.mark.parametrize(
    "problem",
    [
        "wrong_owner",
        "older",
        "open_day",
        "target_closed",
        "fx_date",
        "proxy_fx",
        "intraday",
        "nan",
        "future",
        "wrong_hours",
        "missing_clause",
        "wrong_market",
    ],
)
def test_jp_incomplete_evidence_stays_blocked(problem):
    m, d, w = sample()
    q = m["companies"][0]["quote"]
    if problem == "wrong_owner":
        d["p"]["codes"] = ["000001"]
    elif problem == "older":
        q["date"] = "2026-10-30"
    elif problem == "open_day":
        c = json.loads(d["cal"]["body"])
        c["closed_dates"].remove("2026-11-03")
        d["cal"]["body"] = json.dumps(c)
    elif problem == "target_closed":
        c = json.loads(d["cal"]["body"])
        c["closed_dates"].append("2026-11-04")
        d["cal"]["body"] = json.dumps(c)
    elif problem == "fx_date":
        d["fx"]["body"] = d["fx"]["body"].replace("2026-11-03", "2026-11-02")
    elif problem == "proxy_fx":
        d["fx"]["url"] = "https://example.com"
    elif problem == "intraday":
        q["closed_at"] = "2026-11-02T14:00:00+09:00"
    elif problem == "nan":
        q["close"] = "NaN"
    elif problem == "future":
        m["as_of"] = "2026-11-02T01:00:00+08:00"
    elif problem == "wrong_hours":
        d["cal"]["body"] = d["cal"]["body"].replace("3:30 pm", "3:00 pm")
    elif problem == "missing_clause":
        m["calendar_evidence"].pop("holiday_valuation_quotes")
    else:
        q["valuation_market"] = "HK"
    assert verified_jp_holiday_reference(m, d, "8035.JP", q, w) is None
