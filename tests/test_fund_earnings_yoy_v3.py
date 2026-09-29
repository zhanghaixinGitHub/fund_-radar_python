"""同比原文数字与相邻金额不能因去空白合并；旧版本保持可复现。"""

import pytest
from app.services.fund_earnings_history_v2 import review_yoy as previous_review
from app.services.fund_earnings_yoy_v3 import review_yoy


def spec(quote, value):
    return {
        "quote": quote,
        "values": [value],
        "page": 1,
        "comparison": "YEAR_ON_YEAR",
        "unit": "PERCENT",
        "metric": "PARENT_NET_PROFIT",
        "basis": "CONSOLIDATED_ATTRIBUTABLE_TO_PARENT",
        "period_start": "2021-01-01",
        "period_end": "2021-03-31",
        "base_state": "POSITIVE",
        "direction_word": "原文有符号数值",
    }


def test_original_money_cells_cannot_absorb_the_percent_cell():
    quote = "归属于上市公司股东的净利润（元）286,855,117.10 41,504,117.44 591.15%"
    value = spec(quote, "591.15")
    with pytest.raises(ValueError, match="VALUE_NOT_IN_ANCHOR"):
        previous_review(value, [quote])
    assert review_yoy(value, [quote])["reported_values"] == ["591.15"]


@pytest.mark.parametrize("quote,value", [("金额 100.00 -15.25%", "-15.25"), ("金额 100.00 15.25 %", "15.25")])
def test_signed_percent_and_space_before_unit_are_preserved(quote, value):
    assert review_yoy(spec(quote, value), [quote])["reported_values"] == [value]


def test_digit_suffix_cannot_match_inside_another_percent():
    quote = "金额 100.00 591.15%"
    with pytest.raises(ValueError, match="VALUE_NOT_IN_ANCHOR"):
        review_yoy(spec(quote, "91.15"), [quote])


def test_percentage_points_stay_outside_yoy_percentage_scope():
    quote = "同比增长15.25%，增加3.00个百分点"
    with pytest.raises(ValueError, match="PERCENTAGE_POINT"):
        review_yoy(spec(quote, "15.25"), [quote])
