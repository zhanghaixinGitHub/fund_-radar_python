"""境外持仓必须完整对账，不能把证券代码误认成行号或只取前十名。"""
from copy import deepcopy

import pytest
from app.services.fund_rating_public_global import global_policy, positions
from app.services.fund_rating_public_market import market_policy


def report():
    return [
        "7.3 期末按行业分类的权益投资组合\n通讯 10.00 10.00\n科技 90.00 90.00\n"
        "合计 100.00 100.00\n注：本基金采用彭博行业分类标准。\n"
        "7.4 期末按公允价值排序的所有权益投资明细\n"
        "1 Marvell Technology Inc 美满 MRVL US 美国 美国 3 60.00 60.00\n"
        "2 CATL 宁德\n03750 HK 香港联\n合交易\n所 中国\n香港 2 40.00 40.00\n"
        "7.5 报告期内权益投资组合的重大变动\n"
    ]


def test_full_foreign_table_and_hong_kong_code_are_not_confused_with_row_number():
    stocks, industries = positions(report(), "100")
    assert [s["stock_code"] for s in stocks] == ["MRVL.US", "03750.HK"]
    assert len(industries) == 2


@pytest.mark.parametrize("changed", [
    "row_removed", "wrong_stock_sum", "wrong_industry_sum", "missing_taxonomy", "row_number_gap",
])
def test_incomplete_foreign_table_is_not_scored(changed):
    pages = deepcopy(report())
    if changed == "row_removed":
        pages[0] = pages[0].replace("1 Marvell Technology Inc 美满 MRVL US 美国 美国 3 60.00 60.00\n", "")
    elif changed == "wrong_stock_sum":
        pages[0] = pages[0].replace("3 60.00", "3 59.00")
    elif changed == "wrong_industry_sum":
        pages[0] = pages[0].replace("科技 90.00", "科技 89.00")
    elif changed == "missing_taxonomy":
        pages[0] = pages[0].replace("采用彭博行业分类标准", "分类未说明")
    else:
        pages[0] = pages[0].replace("2 CATL", "3 CATL")
    with pytest.raises(ValueError):
        positions(pages, "100")


def test_global_rule_keeps_all_eight_dimensions_and_frozen_weights():
    old, new = market_policy(), global_policy()
    for key in ("weights", "anchors", "terms"):
        assert old[key] == new[key]
    assert old["basis"] != new["basis"]
