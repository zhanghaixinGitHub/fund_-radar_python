"""退市原代码的成交价不适用须有完整证据；不补假价格、不豁免其他资料。"""

from copy import deepcopy

import pytest
from app.services.direction_1d_protocol import digest
from app.services.fund_information_contracts import time_context


def sample():
    q1 = "信达证券（证券代码：601059）因被吸收合并，本所决定自2026年10月19日起终止其股票在本所上市交易。"
    q2 = (
        "证券代码601059，换股实施股权登记日收市后登记在册股份按协议转换，"
        "实施结果、股份变动暨新增股份上市将另行刊登公告。"
    )
    q3 = "006730 本基金中期报告：601059 信达证券，持仓日期为六月末。"
    documents = []
    for key, kind, body, codes, url in [
        (
            "exchange",
            "ANNOUNCEMENT",
            q1,
            ["601059.SH"],
            "https://www.sse.com.cn/disclosure/announcement/listing/stock/c/notice.shtml",
        ),
        ("issuer", "ANNOUNCEMENT", q2, ["601059.SH"], "https://static.cninfo.com.cn/finalpage/notice.PDF"),
        ("report", "REPORT", q3, ["006730"], "https://example.com/report"),
    ]:
        documents.append(
            {
                "id": key,
                "kind": kind,
                "body": body,
                "body_hash": digest(body),
                "codes": codes,
                "url": url,
                "published_date": "2026-10-16",
                "available_at": "2026-10-20T19:00:00+08:00",
            }
        )
    item = {
        "purpose": "REGULAR_TRADING_QUOTE_ONLY",
        "fund_code": "006730",
        "stock_code": "601059.SH",
        "base_date": "2026-10-20",
        "target_date": "2026-10-21",
        "effective_date": "2026-10-19",
        "exchange_ref": "exchange",
        "issuer_ref": "issuer",
        "report_ref": "report",
        "exchange_quote": q1,
        "issuer_quote": q2,
        "position_quote": q3,
        "fact_id": "cessation",
    }
    data = {
        "fund_code": "006730",
        "as_of": "2026-10-20T20:00:00+08:00",
        "window": {"base_nav_date": "2026-10-20", "target_nav_date": "2026-10-21"},
        "nav": [{"nav_date": "2026-10-20"}],
        "latest_nav_date": "2026-10-20",
        "companies": [{"code": "601059.SH", "name": "信达证券", "quote": None}],
        "market": {"000905.SH": {"date": "2026-10-20", "change_pct": 0.6}},
        "documents": documents,
        "market_quote_exceptions": {"601059.SH": item},
        "facts": {
            "cessation": {
                "required_in_analysis": True,
                "direction_eligible": False,
                "source_refs": ["exchange", "issuer", "report"],
                "quotes": [q1, q2, q3],
            }
        },
    }
    return data


def status(data):
    return time_context(data, ["2026-10-19", "2026-10-20", "2026-10-21", "2026-10-22"])


def test_terminal_security_has_no_invented_quote_or_fund_valuation():
    data = sample()
    original = deepcopy(data)
    result = status(data)
    assert data == original and data["companies"][0]["quote"] is None
    assert result["baseline_complete"] is True
    row = result["stocks"]["601059.SH"]
    assert row["status"] == "READY_DOCUMENTED_NON_TRADING"
    assert row["date"] is None and row["non_trading_evidence"]["replacement_quote_created"] is False
    assert row["non_trading_evidence"]["fund_current_quantity_or_fair_value_proved"] is False


@pytest.mark.parametrize(
    "problem",
    [
        "other_fund",
        "other_stock",
        "next_window",
        "future_effective",
        "wrong_host",
        "future_original",
        "future_publication",
        "changed_hash",
        "no_fact",
        "not_required",
        "direction_eligible",
        "wrong_quote",
        "fake_current_quote",
        "ordinary_suspension",
    ],
)
def test_incomplete_or_conflicting_applicability_stays_blocked(problem):
    data = sample()
    item = data["market_quote_exceptions"]["601059.SH"]
    if problem == "other_fund":
        item["fund_code"] = "007832"
    elif problem == "other_stock":
        item["stock_code"] = "601058.SH"
    elif problem == "next_window":
        item["target_date"] = "2026-10-22"
    elif problem == "future_effective":
        item["effective_date"] = "2026-10-21"
    elif problem == "wrong_host":
        data["documents"][0]["url"] = "https://www.sse.com.cn.evil.test/disclosure/announcement/listing/stock/notice"
    elif problem == "future_original":
        data["documents"][0]["available_at"] = "2026-10-20T21:00:00+08:00"
    elif problem == "future_publication":
        data["documents"][0]["published_date"] = "2026-10-21"
    elif problem == "changed_hash":
        data["documents"][0]["body_hash"] = "changed"
    elif problem == "no_fact":
        data["facts"].clear()
    elif problem == "not_required":
        data["facts"]["cessation"]["required_in_analysis"] = False
    elif problem == "direction_eligible":
        data["facts"]["cessation"]["direction_eligible"] = True
    elif problem == "wrong_quote":
        item["exchange_quote"] = "改写的上市状态"
    elif problem == "fake_current_quote":
        data["companies"][0]["quote"] = {"date": "2026-10-20", "change_pct": 0}
    else:
        data["documents"][0]["body"] = "因普通停牌没有行情"
        data["documents"][0]["body_hash"] = digest(data["documents"][0]["body"])
    assert status(data)["baseline_complete"] is False


@pytest.mark.parametrize("other_gap", ["nav", "ordinary_security", "index"])
def test_documented_cessation_does_not_clear_any_other_baseline_gap(other_gap):
    data = sample()
    if other_gap == "nav":
        data["nav"][0]["nav_date"] = data["latest_nav_date"] = "2026-10-19"
    elif other_gap == "ordinary_security":
        data["companies"].append({"code": "600000.SH", "name": "其他股票", "quote": None})
    else:
        data["market"]["000905.SH"]["date"] = "2026-10-19"
    assert status(data)["baseline_complete"] is False


def test_quote_exception_for_a_security_outside_the_fund_is_rejected():
    data = sample()
    data["market_quote_exceptions"]["600000.SH"] = {}
    with pytest.raises(ValueError, match="QUOTE_EXCEPTION_SCOPE_INVALID"):
        status(data)


def test_an_ordinary_missing_quote_has_no_implicit_exemption():
    data = sample()
    data.pop("market_quote_exceptions")
    assert status(data)["baseline_complete"] is False
