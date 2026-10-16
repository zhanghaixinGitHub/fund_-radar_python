"""境外资料适配的关键失败边界，不写入真实评级或把人工数据作为基金事实。"""

import calendar
from copy import deepcopy
from datetime import date

import pytest
from app.services.fund_rating_public_japan import disclosures, history_values, issuer_history, metrics, positions


@pytest.fixture
def own_history():
    dates = [
        date(m // 12, m % 12 + 1, calendar.monthrange(m // 12, m % 12 + 1)[1])
        for m in range(2023 * 12 + 8, 2026 * 12 + 9)
    ]
    rows = [
        {
            "product_id": "public-id",
            "nav_date": day.strftime("%Y%m%d"),
            "relate_price": "1.2",
            "cumulative_net": "1.2",
            "adjust_nav": "1.2",
        }
        for day in dates
    ]
    detail = {
        "code": 0,
        "data": {
            "product_info": {
                "product_id": "public-id",
                "product_code": "019449",
                "product_legal_name": "摩根日本精选股票型证券投资基金(QDII)",
                "money_type": "0",
                "found_date": "2023-09-08",
            }
        },
    }
    nav = {
        "code": 0,
        "data": {
            "product_nav": {
                "public-id": {"pageNum": 1, "pages": 1, "pageSize": len(rows), "total": len(rows), "list": rows}
            }
        },
    }
    provider = {
        "fund_code": "019449",
        "source": {"enabled": True, "authorization_verified_at": "2026-01-01"},
        "history": {
            "rows": [
                {
                    "nav_date": day.isoformat(),
                    "unit_nav": "1.2",
                    "adjusted_nav": "1.2",
                    "content_hash": "raw-source-hash",
                }
                for day in dates
            ]
        },
    }
    return provider, detail, nav


def test_own_valuation_dates_do_not_add_domestic_holiday_rows(own_history):
    """额外的第三方非估值日期不会被补进管理人完整列表。"""
    provider, detail, nav = own_history
    provider["history"]["rows"].append(
        {"nav_date": "2024-01-02", "unit_nav": "1.2", "adjusted_nav": "1.2", "content_hash": "licensed-extra-row"}
    )
    value = issuer_history(provider, detail, nav, date(2026, 9, 30))
    assert "2024-01-02" not in value["dates"]
    assert len(value["month_ends"]) == 37


@pytest.mark.parametrize("problem", ["truncated", "wrong_share", "conflict", "missing", "dividend"])
def test_reject_incomplete_or_conflicting_issuer_history(own_history, problem):
    provider, detail, nav = deepcopy(own_history)
    envelope = nav["data"]["product_nav"]["public-id"]
    if problem == "truncated":
        envelope["total"] += 1
    elif problem == "wrong_share":
        detail["data"]["product_info"]["product_code"] = "007280"
    elif problem == "conflict":
        envelope["list"][3]["relate_price"] = "1.21"
    elif problem == "missing":
        provider["history"]["rows"].pop(4)
    else:
        envelope["list"][4]["cumulative_net"] = "1.3"
    with pytest.raises(ValueError, match="RATING_JAPAN_"):
        issuer_history(provider, detail, nav, date(2026, 9, 30))


def test_month_without_issuer_valuation_is_not_interpolated(own_history):
    provider, detail, nav = deepcopy(own_history)
    envelope = nav["data"]["product_nav"]["public-id"]
    envelope["list"].pop(4)
    envelope["total"] -= 1
    with pytest.raises(ValueError, match="MONTH_END_MISSING"):
        issuer_history(provider, detail, nav, date(2026, 9, 30))


def test_stock_and_industry_totals_must_both_match():
    doc = [
        """7.3 期末按行业
工业 200.00 20.00
合计 200.00 20.00
注：以上分类采用全球行业分类标准(GICS)。
7.4 期末按公允价值
1 FIRST - 8035 日本交 日本 10 100.00 10.00
2 SECOND - 285A 易所 日本 20 100.00 10.00
7.5 报告期内权益
"""
    ]
    stocks, industries = positions(doc, "200.00")
    assert {s["stock_code"] for s in stocks} == {"8035.JP", "285A.JP"}
    assert industries[0]["value_cny"] == "200.00"
    with pytest.raises(ValueError, match="COMPLETE_STOCK_SUM"):
        positions(doc, "201.00")
    with pytest.raises(ValueError, match="COMPLETE_INDUSTRY_SUM"):
        positions([doc[0].replace("工业 200.00", "工业 199.00")], "200.00")


def test_missing_or_adverse_disclosure_is_never_zero_filled():
    with pytest.raises(ValueError):
        disclosures(["本报告不存在其他重大事项。"])


def test_empty_calendar_has_a_controlled_validation_error():
    with pytest.raises(ValueError, match="VALUATION_DATES_INVALID"):
        history_values({"dates": [], "month_ends": [], "rows": []}, date(2026, 9, 30))


@pytest.mark.parametrize("field,value", [("family", "INDEX"), ("currency", "USD")])
def test_same_rules_cannot_be_applied_to_a_different_asset_scope(field, value):
    from datetime import UTC, datetime

    bundle = {
        "fund_code": "019449",
        "source_profile": "CIFM_JAPAN_CNY_OWN_VALUATIONS_V1",
        "basis": "PUBLIC_DISCLOSURE_GLOBAL_EQUITY_CNY_V1",
        "family": "ACTIVE_EQUITY",
        "currency": "CNY",
        "as_of_date": "2026-09-30",
        field: value,
    }
    with pytest.raises(ValueError, match="RATING_JAPAN_SCOPE"):
        metrics(bundle, date(2026, 9, 30), datetime.now(UTC))
