"""验证全年预告与期间、比较列、未来信息和未知口径的边界。"""

from copy import deepcopy

import pytest
from app.services.fund_earnings_annual_forecast_v1 import compare_verified_claims, review_annual_forecast


def sample():
    spec = {
        "year": 2018,
        "source_title": "2018年第三季度报告正文",
        "period_start": "2018-01-01",
        "period_end": "2018-12-31",
        "published_date": "2018-10-30",
        "values": ["580,000", "620,000"],
        "yoy_percent": ["-3.41", "3.25"],
        "comparative_value": "600,470.68",
        "page": 1,
    }
    text = """四、对2018年度经营业绩的预计
2018年度预计的经营业绩情况：归属于上市公司股东的净利润为正值且不属于扭亏为盈的情形
2018年度归属于上市公司股东的净利润变动幅度 -3.41% 至 3.25%
2018年度归属于上市公司股东的净利润变动区间（万元） 580,000 至 620,000
2017年度归属于上市公司股东的净利润（万元） 600,470.68"""
    return spec, [text]


def test_keeps_annual_range_signed_yoy_and_unknown_basis():
    spec, pages = sample()
    fact = review_annual_forecast(spec, pages)
    assert fact["yoy_percent"] == ["-3.41", "3.25"]
    assert fact["amount_values_in_source_unit"] == ["580000", "620000"]
    assert fact["currency"] is None and not fact["accounting_standard_verified"]
    assert fact["available_at"] == "2018-10-31T08:00:00+08:00"
    assert not fact["comparative_is_independent_earlier_disclosure"]


@pytest.mark.parametrize(
    "key,value", [("period_end", "2018-09-30"), ("source_title", "2018年年度报告"), ("published_date", "2019-02-27")]
)
def test_rejects_wrong_period_stage_or_date(key, value):
    spec, pages = sample()
    spec[key] = value
    with pytest.raises(ValueError):
        review_annual_forecast(spec, pages)


@pytest.mark.parametrize(
    "original,wrong", [("（万元）", "（元）"), ("-3.41%", "3.41%"), ("600,470.68", "600,470.69"), ("预计", "实际")]
)
def test_source_unit_sign_comparative_and_forecast_words_must_match(original, wrong):
    spec, pages = sample()
    with pytest.raises(ValueError):
        review_annual_forecast(spec, [pages[0].replace(original, wrong)])


def claims():
    a = {
        "issuer": "002027",
        "period_start": "2018-01-01",
        "period_end": "2018-12-31",
        "metric": "NET_PROFIT_ATTRIBUTABLE_TO_PARENT",
        "basis": "CONSOLIDATED_CAS",
        "currency": "CNY",
        "accounting_standard_verified": True,
        "accounting_basis_anchor": {"page": 1, "text": "合成准则证据"},
        "consolidation_scope_verified": True,
        "available_at": "2018-10-31T08:00:00+08:00",
        "document_id": "a",
        "kind": "FORECAST",
        "revision_issues": [],
        "source_identity_verified": True,
        "money": [{"cny": "100"}, {"cny": "200"}],
    }
    b = deepcopy(a)
    b.update(
        document_id="b", kind="PRELIMINARY_RESULT", available_at="2019-02-28T08:00:00+08:00", money=[{"cny": "150"}]
    )
    return a, b


def test_both_unknown_standards_do_not_enable_comparison():
    a, b = claims()
    for c in (a, b):
        c.update(basis="CONSOLIDATED_STANDARD_UNSPECIFIED", accounting_standard_verified=False)
    with pytest.raises(ValueError, match="BASIS_NOT_VERIFIED"):
        compare_verified_claims(a, b, b["available_at"])


@pytest.mark.parametrize(
    "key,value",
    [
        ("accounting_basis_anchor", None),
        ("consolidation_scope_verified", False),
        ("currency", None),
        ("period_end", "2018-09-30"),
    ],
)
def test_incomplete_or_different_basis_is_blocked(key, value):
    a, b = claims()
    a[key] = value
    with pytest.raises(ValueError, match="BASIS_NOT_VERIFIED"):
        compare_verified_claims(a, b, b["available_at"])


def test_verified_basis_keeps_original_public_time_gate():
    a, b = claims()
    with pytest.raises(ValueError, match="NOT_AVAILABLE"):
        compare_verified_claims(a, b, "2019-02-28T07:59:59+08:00")
    result = compare_verified_claims(a, b, b["available_at"])
    assert result["relation"] == "OVERLAPS_PREVIOUS_RANGE"
    assert not result["is_market_consensus_surprise"] and not result["training_ready"]
