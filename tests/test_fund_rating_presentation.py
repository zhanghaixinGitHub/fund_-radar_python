"""评分说明只展示同批次正式事实；研究目录、失效分和未知值不混入页面。"""
from copy import deepcopy
from types import SimpleNamespace

import pytest
from app.services.fund_rating_presentation import enrich_published_detail


def fixture(status="RATED"):
    item = {"status": status, "dataDates": {"holdings": "2026-06-30"}, "dimensions": [
        {"key": "liquidity", "name": "规模与流动性", "explanation": "披露持仓的承接压力较大。", "metrics": []},
    ]}
    result = SimpleNamespace(fund_code="002112", status="RATED", score="64.124959252060",
                             dimension_values={"liquidity": "52.424242424242"})
    snapshot = {"members": [{"fund_code": "002112", "bundle": {
        "basis": "PUBLIC_REVIEWED_ACTIVE_QUALITY_V2", "quality_metrics": {"observations": {
            "known_payment": {"kind": "payment", "cash_principal": "120", "all_book_liabilities": "80"},
        }},
    }}]}
    return item, result, snapshot


def test_numeric_scores_and_money_come_from_same_published_result_without_mutation():
    item, result, snapshot = fixture()
    originals = deepcopy((item, snapshot))
    actual = enrich_published_detail(item, result, snapshot)
    assert actual["score"] == "64.12" and actual["dimensions"][0]["score"] == "52.42"
    metrics = {m["name"]: m for m in actual["dimensions"][0]["metrics"]}
    assert metrics["现金对账面负债覆盖"]["value"] == "1.5000"
    assert metrics["账面负债"]["period"] == "2026-06-30"
    assert "未来大额赎回" in actual["dimensions"][0]["explanation"]
    assert (item, snapshot) == originals


@pytest.mark.parametrize("status", ["STALE", "WITHDRAWN", "NOT_RATED", "UNSUPPORTED"])
def test_invalid_or_missing_current_rating_never_exposes_old_numeric_score(status):
    item, result, snapshot = fixture(status)
    actual = enrich_published_detail(item, result, snapshot)
    assert actual["score"] is None and actual["dimensions"][0]["score"] is None
    assert not actual["dimensions"][0]["metrics"]


def test_zero_liability_has_no_infinite_coverage_or_future_safety_claim():
    item, result, snapshot = fixture()
    snapshot["members"][0]["bundle"]["quality_metrics"]["observations"]["known_payment"]["all_book_liabilities"] = "0"
    actual = enrich_published_detail(item, result, snapshot)
    metrics = {m["name"]: m["value"] for m in actual["dimensions"][0]["metrics"]}
    assert metrics["账面负债"] == "0.00" and "现金对账面负债覆盖" not in metrics


def test_other_fund_or_unreviewed_bundle_cannot_supply_payment_facts():
    item, result, snapshot = fixture()
    snapshot["members"][0]["fund_code"] = "008888"
    assert not enrich_published_detail(item, result, snapshot)["dimensions"][0]["metrics"]


@pytest.mark.parametrize("value", [None, "NaN", "Infinity", True, "-1", "101"])
def test_missing_or_invalid_score_does_not_become_zero(value):
    item, result, snapshot = fixture()
    result.score = value
    result.dimension_values["liquidity"] = value
    actual = enrich_published_detail(item, result, snapshot)
    assert actual["score"] is None and actual["dimensions"][0]["score"] is None
