"""香港休市参考必须有本基金条款和真实来源；不能将旧行情改日或绕过净值缺口。"""

import json
from copy import deepcopy

import pytest
from app.services.comprehensive_calendar import verified_domestic_window
from app.services.fund_information_contracts import time_context
from app.services.fund_valuation_sessions import verified_hk_holiday_reference


def sample():
    holidays = "\n".join([*(f"{day} January 2026" for day in range(1, 16)), "19 October 2026"])
    clauses = {
        "close": "估值日无交易的，采用最近交易日的市价（收盘价）估值；重大变化时确定公允价值。",
        "fx": "港币资产估值汇率采用估值日中国人民银行公布的人民币汇率中间价。",
    }
    docs = {
        "report": {"id": "report", "kind": "REPORT", "codes": ["018125"], "body": "港股通持有00179德昌电机。"},
        "own": {"id": "own", "kind": "PROSPECTUS", "codes": ["018125"], "body": "\n".join(clauses.values())},
        "calendar": {
            "id": "calendar",
            "kind": "EXCHANGE_CALENDAR",
            "url": "https://www.hkex.com.hk/calendar.pdf",
            "body": "Holiday Schedule for Year 2026\nSaturdays, Sundays\n" + holidays,
        },
        "fx": {
            "id": "fx",
            "kind": "FX_REFERENCE_DATA",
            "url": "https://www.safe.gov.cn/fx",
            "body": json.dumps({"rates": [{"date": "2026-10-19", "currency": "HKD", "cny_per_unit": "0.86"}]}),
        },
        "hours": {
            "id": "hours",
            "kind": "EXCHANGE_TRADING_HOURS",
            "url": "https://www.hkex.com.hk/CAS",
            "body": "Random Closing 16:08 – 16:10",
        },
        "quote": {"id": "quote", "kind": "MARKET_DATA", "available_at": "2026-10-17T10:00:00+08:00"},
    }
    q = {
        "date": "2026-10-16",
        "close": "13.22",
        "change_pct": 0.993,
        "valuation_market": "HK",
        "currency": "HKD",
        "session": "REGULAR_CLOSE",
        "closed_at": "2026-10-16T16:10:00+08:00",
        "source_ref": "quote",
    }
    w = {"base_nav_date": "2026-10-19", "target_nav_date": "2026-10-20"}
    m = {
        "fund_code": "018125",
        "companies": [{"code": "00179.HK", "quote": q}],
        "market": {},
        "calendar_evidence": {
            "policy": "CN_NAV_HK_CONNECT_WINDOW_V1",
            "base_date": w["base_nav_date"],
            "target_date": w["target_nav_date"],
            "report_ref": "report",
            "calendar_ref": "calendar",
            "prospectus_ref": "own",
            "fx_ref": "fx",
            "market_hours_ref": "hours",
            "holiday_valuation_quotes": clauses,
        },
    }
    return m, docs, w


def test_holiday_reference_keeps_actual_quote_date_and_nonzero_return():
    m, d, w = sample()
    before = deepcopy(m)
    assert (
        verified_hk_holiday_reference(m, d, "00179.HK", m["companies"][0]["quote"], w)["reference_date"] == "2026-10-16"
    )
    profile = {"fund_name": "境内制造股票C", "fund_type": "STOCK", "benchmark": "港股通"}
    assert verified_domestic_window(profile, m, d, w)
    data = {
        **m,
        "as_of": "2026-10-20T11:00:00+08:00",
        "window": w,
        "documents": list(d.values()),
        "nav": [{"nav_date": "2026-10-19"}],
        "latest_nav_date": "2026-10-19",
    }
    timing = time_context(data, ["2026-10-16", "2026-10-19", "2026-10-20"])
    assert timing["baseline_complete"] and timing["stocks"]["00179.HK"]["status"] == "READY_LAST_SESSION"
    assert timing["stocks"]["00179.HK"]["date"] == "2026-10-16" and m == before
    data.update(nav=[{"nav_date": "2026-10-16"}], latest_nav_date="2026-10-16")
    assert not time_context(data, ["2026-10-16", "2026-10-19", "2026-10-20"])["baseline_complete"]


@pytest.mark.parametrize(
    "problem",
    [
        "wrong_owner",
        "older_quote",
        "wrong_fx_date",
        "proxy_fx",
        "target_closed",
        "missing_clause",
        "intraday",
        "wrong_hours_source",
        "nan_price",
        "future_source",
    ],
)
def test_incomplete_holiday_evidence_cannot_unlock_baseline(problem):
    m, d, w = sample()
    q = m["companies"][0]["quote"]
    if problem == "wrong_owner":
        d["own"]["codes"] = ["000001"]
    elif problem == "older_quote":
        q["date"] = "2026-10-15"
    elif problem == "wrong_fx_date":
        d["fx"]["body"] = d["fx"]["body"].replace("2026-10-19", "2026-10-16")
    elif problem == "proxy_fx":
        d["fx"]["url"] = "https://example.com/fx"
    elif problem == "target_closed":
        d["calendar"]["body"] += "\n20 October 2026"
    elif problem == "missing_clause":
        m["calendar_evidence"].pop("holiday_valuation_quotes")
    elif problem == "intraday":
        q["closed_at"] = "2026-10-16T15:00:00+08:00"
    elif problem == "wrong_hours_source":
        d["hours"]["url"] = "https://example.com/hours"
    elif problem == "nan_price":
        q["close"] = "NaN"
    else:
        m["as_of"] = "2026-10-17T09:00:00+08:00"
    assert verified_hk_holiday_reference(m, d, "00179.HK", q, w) is None
