"""日常预测准入的回归检查；全部使用档案样例，不训练模型或写真实预测。"""

import pytest
from app.services.direction_1d_data import classify


@pytest.fixture
def profile():
    return {
        "fund_type": "MIXED",
        "fund_master_id": "family",
        "master_name": "product",
        "profile_hash": "hash",
        "source_code": "TUSHARE_PRO_FUND",
        "status": "ACTIVE",
        "benchmark": "存款利率",
    }


@pytest.mark.parametrize("benchmark", [None, "", "存款利率", "三年期定期存款税后利率+2%", "沪深300指数95%"])
@pytest.mark.parametrize(("fund_type", "group"), [("STOCK", "CN_EQUITY"), ("MIXED", "CN_MIXED"), ("BOND", "CN_BOND")])
def test_prediction_uses_verified_asset_type_without_benchmark_allowlist(profile, benchmark, fund_type, group):
    result = classify({**profile, "fund_type": fund_type, "benchmark": benchmark}, prediction=True)
    assert result["group_id"] == group
    assert result["classification_reason"] is None
    assert result["classification_details"] == []
    assert result["group_evidence"]["rule"] == "PROFILE_ASSET_TYPE_V2"


def test_frozen_research_keeps_original_rule(profile):
    result = classify(profile)
    assert result["group_id"] is None
    assert result["classification_reason"] == "GROUP_UNVERIFIED"
    assert result["group_evidence"]["rule"] == "DOMESTIC_BENCHMARK_ASSET_TYPE_V1"
    assert "classification_details" not in result


@pytest.mark.parametrize(
    ("patch", "detail"),
    [
        ({"benchmark": "沪深300指数80%+恒生指数20%"}, "CROSS_MARKET_MODEL_REQUIRED"),
        ({"fund_name": "海外股票QDII"}, "CROSS_MARKET_MODEL_REQUIRED"),
        ({"fund_name": "黄金ETF联接"}, "COMMODITY_MODEL_REQUIRED"),
        ({"source_fund_type": "FOF"}, "FOF_MODEL_REQUIRED"),
        ({"source_fund_type": "货币基金"}, "MONEY_MARKET_TARGET_REQUIRED"),
        ({"fund_name": "REIT"}, "REIT_MODEL_REQUIRED"),
        ({"fund_type": "UNKNOWN"}, "ASSET_MODEL_UNSUPPORTED"),
    ],
)
def test_unsupported_models_remain_blocked_with_explanation(profile, patch, detail):
    result = classify({**profile, **patch}, prediction=True)
    assert result["group_id"] is None
    assert result["classification_reason"] == "SPECIAL_POLICY_REQUIRED"
    assert detail in result["classification_details"]


@pytest.mark.parametrize(
    ("patch", "detail"),
    [
        ({"profile_hash": None}, "PROFILE_SOURCE_UNVERIFIED"),
        ({"source_code": "UNVERIFIED"}, "PROFILE_SOURCE_UNVERIFIED"),
        ({"status": "UNKNOWN"}, "FUND_NOT_ACTIVE"),
        ({"status": "TERMINATED"}, "FUND_NOT_ACTIVE"),
        ({"fund_master_id": None}, "PRODUCT_IDENTITY_UNVERIFIED"),
        ({"master_name": None}, "PRODUCT_IDENTITY_UNVERIFIED"),
    ],
)
def test_missing_identity_is_not_treated_as_supported(profile, patch, detail):
    result = classify({**profile, **patch}, prediction=True)
    assert result["group_id"] is None
    assert result["classification_reason"] == "GROUP_UNVERIFIED"
    assert detail in result["classification_details"]


def test_multiple_special_exposures_are_all_explained(profile):
    result = classify({**profile, "fund_name": "原油QDII-FOF"}, prediction=True)
    assert result["classification_details"] == [
        "CROSS_MARKET_MODEL_REQUIRED",
        "COMMODITY_MODEL_REQUIRED",
        "FOF_MODEL_REQUIRED",
    ]
    assert result["cross_border"] is True
    assert classify({**profile, "fund_name": "黄金ETF"}, prediction=True)["cross_border"] is False
