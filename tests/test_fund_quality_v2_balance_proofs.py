"""精确账目约束的关键反例；合成分量不作为真实产品或校准资料。"""

import pytest
from app.services.fund_quality_v2_balance_proofs import identify_nonnegative_parts, strict_balance_proof


def test_zero_requires_exact_complete_nonnegative_sum_and_keeps_original_unknown():
    result = identify_nonnegative_parts("12.34", {"known": "12.34", "unknown1": None, "unknown2": None})
    assert result["derived_parts"] == {"unknown1": "0.00", "unknown2": "0.00"}
    assert result["explicit_parts"]["unknown1"] is None
    assert result["point_values"]["unknown1"] == "0.00"


def test_positive_residual_is_not_distributed_between_unknown_components():
    result = identify_nonnegative_parts("12.35", {"known": "12.34", "unknown1": None, "unknown2": None})
    assert result["state"] == "MULTIPLE_NONNEGATIVE_SOLUTIONS"
    assert result["derived_parts"] == {}
    assert result["point_values"]["unknown1"] is None
    assert result["remaining_amount"] == "0.01"


def test_one_unknown_is_unique_but_never_called_a_reported_cell():
    result = identify_nonnegative_parts("12.35", {"known": "12.34", "unknown": None})
    assert result["derived_parts"] == {"unknown": "0.01"}
    assert result["explicit_parts"]["unknown"] is None
    assert result["derivation"] == "SINGLE_UNKNOWN_BY_EXACT_SUM"


@pytest.mark.parametrize("total,parts", [
    ("12.33", {"known": "12.34", "unknown": None}),
    ("12.35", {"known": "12.34"}),
    ("12.34", {"known": "12.35", "negative": "-0.01"}),
    ("-1.00", {"known": None}),
])
def test_negative_or_even_one_cent_unexplained_difference_rejects_zero_claim(total, parts):
    with pytest.raises(ValueError):
        identify_nonnegative_parts(total, parts)


def balance_report():
    return ["""本报告期自2025年1月1日起至12月31日止。
6.1 资产负债表
单位：人民币元
本期末 上年度末
2025年12月31日 2024年12月31日
资产：
货币资金 700.00 500.00
结算备付金 0.00 -
存出保证金 - -
交易性金融资产 300.00 400.00
其中：股票投资 300.00 400.00
基金投资 - -
债券投资 - -
资产支持证券投资 - -
贵金属投资 - -
其他投资 - -
衍生金融资产 - -
买入返售金融资产 - -
应收清算款 - -
应收股利 - -
应收申购款 - -
递延所得税资产 - -
其他资产 - -
资产总计 1,000.00 900.00
负债和净资产：
负债合计 100.00 200.00
净资产合计 900.00 700.00
6.2 利润表
"""]


def prove(pages):
    return strict_balance_proof(pages, observed_on="2025-12-31", source_hash="a" * 64)


def test_complete_current_balance_proves_zero_but_keeps_printed_dash_unassigned():
    result = prove(balance_report())
    assert result["point_book"]["bond_amount"] == "0.00"
    assert result["original_rows"]["债券投资"]["value"] is None
    assert result["original_rows"]["结算备付金"]["state"] == "EXPLICIT_AMOUNT"
    assert "结算备付金" not in result["asset_equation"]["derived_parts"]
    assert result["trading_equation"]["derivation"] == "ZERO_UNIQUELY_IDENTIFIED_BY_COMPLETE_NONNEGATIVE_SUM"


def test_missing_asset_row_is_not_inferred_as_zero_from_a_total():
    pages = [balance_report()[0].replace("存出保证金 - -\n", "")]
    with pytest.raises(ValueError, match="ROW_MISSING"):
        prove(pages)


@pytest.mark.parametrize("row", ["应收补贴款 - -", "应收补贴款 0.00 0.00"])
def test_unreviewed_asset_category_cannot_be_ignored_even_if_it_looks_zero(row):
    pages = [balance_report()[0].replace("资产总计", row + "\n资产总计")]
    with pytest.raises(ValueError, match="UNREVIEWED_ASSET_ROW"):
        prove(pages)


def test_known_total_with_positive_ambiguous_trading_residual_returns_no_point_book():
    pages = [balance_report()[0].replace("股票投资 300.00", "股票投资 290.00")]
    result = prove(pages)
    assert result["state"] == "PARTIAL_BALANCE_NO_POINT_BOOK"
    assert result["point_book"] is None
    assert result["trading_equation"]["derived_parts"] == {}


def test_prior_date_or_other_unit_cannot_be_used_as_current_amounts():
    pages = [balance_report()[0].replace("2025年12月31日 2024年12月31日", "2024年12月31日 2023年12月31日")]
    with pytest.raises(ValueError, match="CURRENT_CNY"):
        prove(pages)


def test_visible_label_wrapping_does_not_change_currency_values_or_original_dash():
    pages = [balance_report()[0].replace("资产支持证券投资", "资产支持证券\n投资")]
    result = prove(pages)
    assert result["original_rows"]["资产支持证券投资"]["value"] is None
    assert "\n" in result["original_rows"]["资产支持证券投资"]["original_row"]
    assert result["point_book"]["stock_amount"] == "300.00"


def test_truncated_current_cell_cannot_be_replaced_with_comparison_cell():
    pages = [balance_report()[0].replace("货币资金 700.00 500.00", "货币资金 700.0\n0 500.00")]
    with pytest.raises(ValueError, match="ROW_MISSING"):
        prove(pages)


def test_same_child_label_belongs_to_actual_trading_or_debt_parent_not_first_match():
    pages = [balance_report()[0].replace("货币资金 700.00", "货币资金 650.00").replace(
        "应收清算款", "债权投资 50.00 -\n其中：债券投资 50.00 -\n"
        "资产支持证券投资 - -\n其他投资 - -\n应收清算款")]
    result = prove(pages)
    assert result["point_book"]["bond_amount"] == "0.00"
    assert result["point_book"]["assets"]["债权投资"] == "50.00"
    assert result["debt_equation"]["point_values"]["其中:债券投资"] == "50.00"
    assert result["original_rows"]["债权投资/资产支持证券投资"]["value"] is None


def test_known_label_suffix_after_both_actual_cells_does_not_change_the_cells():
    pages = [balance_report()[0].replace("资产支持证券投资 - -", "资产支持证 - -\n券投资")]
    result = prove(pages)
    row = result["original_rows"]["资产支持证券投资"]
    assert row["label_prefix_before_cells"] == "资产支持证"
    assert row["label_suffix_after_cells"] == "券投资"
    assert row["value"] is None and "- -\n券投资" in row["original_row"]
    assert result["point_book"]["stock_amount"] == "300.00"
