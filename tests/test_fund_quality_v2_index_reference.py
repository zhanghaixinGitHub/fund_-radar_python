"""指数原件重建及股票切片边界；临时测试行不作为真实产品或校准样本。"""

import copy

import pytest
from app.services.fund_quality_v2_index_reference import equity_slice_distance, rebuild_index_reference


def originals():
    identity = {
        "code": 0,
        "data": {"fields": ["ts_code", "name", "publisher"], "items": [["TEST_INDEX", "测试指数", "测试发布机构"]]},
    }
    weights = {
        "code": 0,
        "data": {
            "fields": ["index_code", "con_code", "trade_date", "weight"],
            "items": [["TEST_INDEX", "600001.SH", "20260630", 60], ["TEST_INDEX", "000001.SZ", "20260630", 40]],
        },
    }
    return identity, weights


def test_matching_concentrated_stock_slice_is_not_penalized_for_its_own_target_concentration():
    metadata, weights = originals()
    ref = rebuild_index_reference(metadata, weights, index_code="TEST_INDEX", observed_on="2026-06-30")
    actual = [
        {"stock_code": code, "market_value_cny": str(amount)}
        for code, amount in [("600001.SH", 600), ("000001.SZ", 400)]
    ]
    result = equity_slice_distance(actual, ref, expected_stock_amount="1000")
    assert result["half_l1_distance"] == "0.0"
    assert result["actual_stock_hhi"] == result["target_stock_hhi"] == "0.52"
    assert result["status"] == "EQUITY_SECURITY_SLICE_DIAGNOSTIC_NOT_V2_METRIC"


@pytest.mark.parametrize("problem", ["identity", "date", "duplicate", "incomplete", "negative", "missing"])
def test_bad_reference_cannot_pass_via_success_code(problem):
    metadata, weights = copy.deepcopy(originals())
    if problem == "identity":
        metadata["data"]["items"][0][0] = "ANOTHER_INDEX"
    elif problem == "date":
        weights["data"]["items"][0][2] = "20260930"
    elif problem == "duplicate":
        weights["data"]["items"][1][1] = "600001.SH"
    elif problem == "incomplete":
        weights["data"]["items"].pop()
    elif problem == "negative":
        weights["data"]["items"][0][3] = -60
    else:
        weights["data"]["fields"].remove("weight")
    with pytest.raises(ValueError):
        rebuild_index_reference(metadata, weights, index_code="TEST_INDEX", observed_on="2026-06-30")
