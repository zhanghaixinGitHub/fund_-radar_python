"""章号视图只适配原文结构；虚构样例不用于真实校准。"""

from datetime import date

import pytest
from app.services.fund_quality_v2_restrictions import (
    full_asset_restriction_scope,
    restricted_numeric_view,
    restricted_securities,
)
from app.services.fund_quality_v2_stock_book import (
    _balance_parser_view,
    _stock_parser_view,
    annual_parser_view,
    full_stock_book,
)


def test_annual_view_changes_references_but_no_amounts_codes_or_original_pages():
    original = [
        "7.1资产负债表\n货币资金 7.4.7.1 7.40 80.00\n7.2利润表\n"
        "7.4.12.1因认购新发股票\n8.3期末按公允价值排序的股票投资明细\n"
        "1 600001 测试 100 7.40 1.23\n8.4报告期内股票投资组合的重大变动"
    ]
    result, edits = annual_parser_view(original)
    assert "货币资金 6.4.7.1 7.40 80.00" in result[0]
    assert "6.1 资产负债表" in result[0] and "6.4.12.1 因认购" in result[0]
    assert "7.3 期末" in result[0] and "1 600001 测试 100 7.40 1.23" in result[0]
    assert original[0].startswith("7.1资产负债表") and edits[0]["before"] == "7.1资产负债表"


def test_missing_pair_of_annual_titles_is_not_guessed():
    original = ["7.1 资产负债表\n8.3 股票投资内容不完整"]
    assert annual_parser_view(original) == (original, [])


def test_original_report_period_must_match_requested_year():
    with pytest.raises(ValueError, match="PERIOD_MISMATCH"):
        full_stock_book(["本报告期自2024年1月1日起至2024年12月31日止。"], report_end=date(2025, 12, 31))


def test_numberless_original_titles_can_be_internal_markers_without_creating_values():
    original = [
        "资产负债表\n货币资金 7.4.7.1 1.00 2.00\n利润表\n"
        "期末按公允价值占基金资产净值比例大小排序的股票投资明细\n"
        "1 600001 测试 100 100.00 1.00\n报告期内股票投资组合的重大变动"
    ]
    view, edits = annual_parser_view(original)
    assert "6.1 资产负债表" in view[0] and "7.3 期末" in view[0]
    assert "100 100.00 1.00" in view[0] and edits


def test_visible_amount_tail_is_restored_and_annotation_reference_never_used_as_amount():
    view, edits = _balance_parser_view(["6.1 资产负债表\n交易性金融资产 1,031.2 9,876.54\n 3\n6.2 利润表"])
    assert "1,031.23 9,876.54" in view[0] and len(edits) == 1
    original = ["6.1 资产负债表\n货币资金 6.4.7.1 12.34\n 5\n6.2 利润表"]
    assert _balance_parser_view(original) == (original, [])


def test_stock_only_scope_does_not_claim_pledged_bonds_are_zero_or_pass_all_asset_gate():
    source = """6.4.12.1 因认购新发证券
    本基金未持有因认购新发证券而流通受限证券。
    6.4.12.2 期末持有的暂时停牌等流通受限股票
    本基金未持有暂时停牌等流通受限股票。
    6.4.12.3 期末债券正回购
    6.4.12.3.1 银行间市场债券正回购
    无从事银行间市场债券正回购交易形成的卖出回购证券款余额。
    6.4.12.3.2 交易所市场债券正回购
    债券回购卖出回购证券款余额116,003,965.38元。
    6.4.12.4 期末参与转融通
    本基金无参与转融通证券出借业务的证券。
    6.4.13 金融工具风险
    """
    result = restricted_securities([source], [], "200000000", scope="STOCKS_ONLY")
    assert result["restricted_amount"] == "0" and "NOT_ALL_ASSETS" in result["scope"]
    assert "116,003,965.38" in result["excluded_pledged_bonds_excerpt"]
    with pytest.raises(ValueError, match="PLEDGED_BONDS"):
        restricted_securities([source], [], "200000000")


def test_joined_cost_and_fair_amounts_only_gain_a_separator():
    source = ["301656动力 12.48 25.46 7,395 92,289.60188,276.70-\n301657未核 12.34"]
    result, edits = _stock_parser_view(source, [{"stock_code": "301656.SZ"}])
    assert "301656 动力" in result[0] and "92,289.60 188,276.70-" in result[0]
    assert "301657未核" in result[0] and len(edits) == 1


@pytest.mark.parametrize("reference", ["附注", "附注“期末本基金持有的流通受限证券”"])
def test_explicit_unnumbered_note_reference_is_scope_but_general_risk_policy_is_not(reference):
    quote = f"除在{reference}中列示的部分基金资产流通暂时受限制外(如有)，其余均能及时变现。"
    assert full_asset_restriction_scope(quote)[0] == quote.rstrip("。")
    assert full_asset_restriction_scope("本基金监控7个工作日可变现资产，未发生重大流动性事件。") is None


@pytest.mark.parametrize(
    "reference,others",
    [
        ("“期末本基金持有的流通受限证券”", "本基金本报告期末的其他资产"),
        ("期末本基金持有的流通受限证券", "其余"),
    ],
)
def test_explicit_report_chapter_reference_still_requires_other_assets_scope(reference, others):
    quote = f"除本报告{reference}章节中所列示券种流通暂时受限制不能自由转让外，{others}均能及时变现。"
    assert full_asset_restriction_scope(quote)
    assert full_asset_restriction_scope("部分基金资产流通暂时受限制不能自由转让的情况参见附注6.4.12。") is None


def test_visible_comma_group_quantity_tail_is_not_truncated_to_last_digit():
    source = "6.85 5.31\n2,919,70\n8\n19,999,999.8\n0\n15,503,649.4\n8\n-"
    actual = restricted_numeric_view(source)
    assert "2,919,708" in actual and "19,999,999.80" in actual and "15,503,649.48" in actual
    assert actual.split() == ["6.85", "5.31", "2,919,708", "19,999,999.80", "15,503,649.48", "-"]
    invalid = "数量2,919,7\n8\n"
    with pytest.raises(ValueError, match="QUANTITY_WRAP_AMBIGUOUS"):
        restricted_numeric_view(invalid)
