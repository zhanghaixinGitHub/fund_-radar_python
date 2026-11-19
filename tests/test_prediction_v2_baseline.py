"""个股缺报价不等于基金净值缺基准；关键输入仍必须真实齐备。"""

from copy import deepcopy
from datetime import date, datetime

import pytest
from app.repositories import direction_1d as repo
from app.services.fund_prediction_nav_identity import available_prediction_nav, previous_qdii_nav_date
from app.services.prediction_v2_baseline import baseline_ready


def source():
    return {
        "window": {"base_nav_date": "2026-11-13"},
        "nav": [{"nav_date": "2026-11-13"}],
        "market": {"ETF": {"date": "2026-11-13", "change_pct": -1.2}},
        "time_context": {
            "baseline_complete": False,
            "baseline_date": "2026-11-13",
            "latest_nav_date": "2026-11-13",
            "nav_status": "READY",
            "stocks": {"HALTED": {"date": "2026-10-30", "status": "STALE"}},
            "markets": {"ETF": {"date": "2026-11-13", "status": "READY"}},
        },
    }


def test_partial_stock_gap_preserved_without_blocking_identified_baseline():
    data = source()
    original = deepcopy(data)
    assert baseline_ready(data)
    assert data == original


@pytest.mark.parametrize("gap", ["nav", "market", "value", "empty", "identity"])
def test_critical_baseline_gap_remains_blocked(gap):
    data = source()
    if gap == "nav":
        data["nav"][0]["nav_date"] = "2026-11-12"
    elif gap == "market":
        data["market"]["ETF"]["date"] = "2026-11-12"
    elif gap == "value":
        data["market"]["ETF"]["change_pct"] = None
    elif gap == "empty":
        data["market"] = data["time_context"]["markets"] = {}
    else:
        data["time_context"]["markets"]["OTHER"] = {"status": "READY"}
    assert not baseline_ready(data)


@pytest.mark.parametrize("problem", [None, "domestic", "older", "exact_arrived", "valuation", "market"])
def test_qdii_reference_uses_one_formal_prior_nav_with_current_market(problem):
    data = source()
    base, prior = "2026-11-13", "2026-11-12"
    nav = {"nav_date": prior, "unit_nav": "1.23", "ann_date": base, "source_id": "verified"}
    data.update(fund_name="指数基金(QDII)", nav=[nav], reference_nav_date=prior)
    data["time_context"].update(nav_status="READY_PREVIOUS_SESSION", latest_nav_date=prior)
    if problem == "domestic":
        data["fund_name"] = "境内基金"
    elif problem == "older":
        nav["nav_date"] = "2026-11-11"
    elif problem == "exact_arrived":
        data["nav"].append({**nav, "nav_date": base})
    elif problem == "valuation":
        data["valuation_status"] = {"ready": False}
    elif problem == "market":
        data["market"]["ETF"]["date"] = prior
    assert baseline_ready(data) is (problem is None)
    if problem in {"domestic", "older", "exact_arrived"}:
        assert previous_qdii_nav_date(data["nav"], base, data) is None


def test_available_nav_prefers_exact_when_published_and_skips_weekend(monkeypatch):
    at = datetime.fromisoformat("2026-11-17T09:40:00+08:00")
    base, prior = date(2026, 11, 16), date(2026, 11, 13)
    nav = {"nav_date": prior, "unit_nav": "1.23", "ann_date": base, "updated_at": at}
    rows = [nav, {**nav, "nav_date": base, "ann_date": date(2026, 11, 18)}]
    monkeypatch.setattr(
        repo, "navs", lambda c, code, source, start, end: [row for row in rows if start <= row["nav_date"] <= end]
    )
    qdii = {"fund_code": "000001", "fund_name": "指数基金(QDII)"}
    date_used, prior_identity = available_prediction_nav(None, qdii, "provider", base, at)
    assert date_used == str(prior)
    assert available_prediction_nav(None, {**qdii, "fund_name": "境内"}, "provider", base, at)[0] is None
    rows[1]["ann_date"] = at.date()
    date_used, exact_identity = available_prediction_nav(None, qdii, "provider", base, at)
    assert date_used == str(base)
    assert exact_identity != prior_identity
