"""成对曲线身份、缺失和冲突边界；以下算例不能用于真实产品校准。"""

import json

import pytest
from app.services.fund_quality_v2_paired_history import paired_history


def inputs():
    days = [f"{2023 + month // 12:04}-{month % 12 + 1:02}-28" for month in range(37)]
    curve = {"dates": days, "month_ends": days, "adjusted_nav": ["1"] * 37}
    rows = [{"productCode": "123456", "navDate": day, "benchmarkRate": "0", "syNavRate": "0"} for day in days]
    body = {"code": 0, "data": rows}
    ref = {
        "url": "https://www.fullgoal.com.cn/ws-business-server/fund/getIncomeList?productCode=123456",
        "params": {"productCode": "123456"}, "sha256": "TEST_ONLY",
    }
    return curve, body, ref


def run(curve, body, ref):
    return paired_history(curve, json.dumps(body).encode(), ref, fund_code="123456")


def test_actual_zero_benchmark_is_valid_but_missing_value_is_rejected():
    curve, body, ref = inputs()
    assert run(curve, body, ref)["benchmark_index"] == ["1"] * 37
    body["data"][12]["benchmarkRate"] = None
    with pytest.raises(ValueError, match="DATES_MISSING"):
        run(curve, body, ref)


@pytest.mark.parametrize(
    "failure", ["foreign_product", "foreign_query", "duplicate_day", "missing_day", "nav_conflict"]
)
def test_real_alignment_and_product_identity_cannot_be_replaced_by_success_code(failure):
    curve, body, ref = inputs()
    if failure == "foreign_product":
        body["data"][4]["productCode"] = "654321"
    elif failure == "foreign_query":
        ref["url"] = ref["url"].replace("123456", "654321")
    elif failure == "duplicate_day":
        body["data"].append(dict(body["data"][0]))
    elif failure == "missing_day":
        body["data"].pop(12)
    else:
        body["data"][12]["syNavRate"] = "20"
    with pytest.raises(ValueError):
        run(curve, body, ref)


def test_wj_requires_actual_product_request_and_income_type():
    curve, body, ref = inputs()
    ref.update(
        url="https://www.wjasset.com/common-web/ets40/yieldListQueryAPI",
        params={"fundCode": "123456", "incomeType": "8"}, format="ISSUER_DAILY_BENCHMARK_WJ",
    )
    body = {"code": "ETS-5BP0000", "results": [
        {"tradeDate": day.replace("-", ""), "benchmarkRatio": "0", "returnRatio": "0"}
        for day in curve["dates"]
    ]}
    assert run(curve, body, ref)["dates"] == curve["dates"]
    ref["params"]["incomeType"] = "1"
    with pytest.raises(ValueError, match="WJ_PRODUCT_REQUEST"):
        run(curve, body, ref)
