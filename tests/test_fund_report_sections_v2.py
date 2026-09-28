"""独立章节解析的回归与拒绝边界，不读取真实标签、不联网。"""

import pytest
from app.integrations.dbfund_reports import parse_text as parse_v1
from app.integrations.fund_report_sections_v2 import normalize_pages, parse_text


def pages():
    """小型合成季报，行业金额与披露股票合计均为 100 元。"""
    return [
        "德邦鑫星价值 002112\n报告送出日期2021年4月21日\n"
        "5.1期末基金资产组合情况\n权益投资 100.00 50.00\n"
        "5.2报告期末按行业分类的股票投资组合\nC 制造业 100.00 50.00\n合计 100.00 50.00\n"
        "5.3报告期末按公允价值占基金资产净值比例大小排序的股票投资明细\n"
        "1 600000 浦发银行 10 100.00 50.00\n"
        "5.4报告期末按债券品种分类的债券投资组合\n"
    ]


def test_no_space_sections_work_and_business_values_match_v1():
    with pytest.raises(ValueError, match="HOLDINGS_SECTION_MISSING"):
        parse_v1(pages(), "德邦鑫星价值2021年第1季度报告")
    actual = parse_text(pages(), "德邦鑫星价值2021年第1季度报告")
    normalized, edits = normalize_pages(pages())
    expected = parse_v1(normalized, "德邦鑫星价值2021年第1季度报告")
    assert {k: v for k, v in actual.items() if k not in {"parser_version", "section_normalization"}} == {
        k: v for k, v in expected.items() if k != "parser_version"
    }
    assert actual["holding_count"] == 1 and len(edits) == 4
    assert normalize_pages(normalized) == (normalized, [])


@pytest.mark.parametrize(
    "before,after,error",
    [
        ("1 600000", "2 600000", "RANK_OR_DUPLICATE"),
        ("C 制造业 100.00", "C 制造业 99.00", "INDUSTRY_TOTAL_MISMATCH"),
        ("100.00 50.00\n5.4", "100.00 60.00\n5.4", "PARTIAL_WEIGHT_EXCEEDS_STOCK"),
        ("100.00 50.00\n5.4", "100.00 -\n5.4", "HOLDINGS_PARSE_EMPTY"),
        ("002112", "999999", "FUND_IDENTITY_MISMATCH"),
        ("5.4报告期末按债券品种分类的债券投资组合", "", "HOLDINGS_SECTION_MISSING"),
    ],
)
def test_bad_identity_totals_weights_or_truncated_table_still_fail(before, after, error):
    with pytest.raises(ValueError, match=error):
        parse_text([pages()[0].replace(before, after)], "德邦鑫星价值2021年第1季度报告")


def test_subsection_numbers_and_unrelated_numeric_rows_are_untouched():
    original = ["5.3.1报告期末股票投资明细\n5.3这是收益数值\n1 600000 名称 5.3 99.99\n"]
    assert normalize_pages(original) == (original, [])


@pytest.mark.parametrize("value", [[], [""], [None]])
def test_empty_or_nontext_pages_fail(value):
    with pytest.raises(ValueError, match="EMPTY_OR_INVALID_PAGES"):
        normalize_pages(value)
