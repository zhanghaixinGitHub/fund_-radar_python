"""保证金工具范围的反例测试；合成文字仅核工程边界，不进入真实校准。"""

import pytest
from app.services.fund_quality_v2_cash_constraints import scoped_shortfall_bounds
from app.services.fund_quality_v2_margin_evidence import margin_absence_evidence

CLAUSE = ("本基金的投资组合比例为：每个交易日日终，在扣除{tools}需缴纳的交易保证金后，"
          "应当保持不低于基金资产净值5%的现金或到期日在一年以内的政府债券，"
          "其中现金不包括结算备付金、存出保证金和应收申购款等。")


def scoped(tools, statements):
    return scoped_shortfall_bounds(report_text=CLAUSE.format(tools=tools) + statements,
        cash_lower="90", cash_upper="100", government_upper="40", net_assets="1000")


def test_index_absence_does_not_cover_treasury_or_options():
    result = scoped("股指期货、国债期货和股票期权", "本基金本报告期末未投资股指期货。")
    assert result["available_amount_lower_bound"] is None
    assert result["shortfall_ratio_upper"] is None
    assert result["margin_scope_evidence"]["missing_absence_instruments"] == ["STOCK_OPTIONS", "TREASURY_FUTURES"]


def test_compound_explicit_absence_covers_only_the_named_requirement():
    result = scoped("股指期货及国债期货", "本基金本报告期内未持有国债期货和股指期货合约。")
    assert result["identified_shortfall_ratio"] == "0"
    assert result["margin_scope_evidence"]["missing_absence_instruments"] == []
    assert result["margin_scope_evidence"]["absence_originals_by_instrument"]["TREASURY_FUTURES"]


def test_separate_sentences_can_cover_several_instruments():
    result = scoped("股票期权和国债期货", "本基金本报告期末未投资股票期权。本基金本报告期内未持有国债期货。")
    assert result["identified_shortfall_ratio"] == "0"


def test_all_required_margin_is_not_exhausted_by_three_named_absences():
    result = scoped("全部", "本基金本报告期未投资股指期货、国债期货和股票期权。")
    assert result["shortfall_ratio_upper"] is None
    assert result["margin_scope_evidence"]["required_instruments"] is None
    assert result["margin_scope_evidence"]["missing_absence_instruments"] is None


def test_unqualified_required_margin_is_not_zero_from_named_absences():
    result = scoped("", "本基金本报告期未投资股指期货、国债期货和股票期权。")
    assert result["shortfall_ratio_upper"] is None
    assert result["margin_scope_evidence"]["requirement_scope"] == "UNQUALIFIED_REQUIRED_TRADING_MARGIN"


@pytest.mark.parametrize("statement", ["本基金本报告期未交易股指期货。", "本基金未来不投资股指期货。",
    "本基金本报告期末未投资股指期货策略的相关研究。", "衍生金融资产 - -", "股指期货保证金 0.00"])
def test_nonabsence_or_incomplete_noun_does_not_imply_zero(statement):
    result = scoped("股指期货", statement)
    assert result["margin_scope_evidence"]["absence_originals_by_instrument"] == {}
    assert result["shortfall_ratio_upper"] is None


def test_source_conflict_for_any_instrument_is_explicit():
    with pytest.raises(ValueError, match="SOURCE_CONFLICT"):
        scoped("股指期货和国债期货", "本基金本报告期内未投资国债期货。本基金本报告期末持有国债期货。")


def test_missing_cash_definition_does_not_enter_arithmetic():
    text = CLAUSE.format(tools="股指期货").split("，其中")[0]
    with pytest.raises(ValueError, match="BASIS_INCOMPLETE"):
        scoped_shortfall_bounds(report_text=text, cash_lower="90", cash_upper="100",
                                government_upper="40", net_assets="1000")


@pytest.mark.parametrize("scope,required", [("ALL_REQUIRED_TRADING_MARGIN", ["INDEX_FUTURES"]),
    ("EXPLICIT_NAMED_INSTRUMENTS", []), ("EXPLICIT_NAMED_INSTRUMENTS", ["UNKNOWN"]),
    ("EXPLICIT_NAMED_INSTRUMENTS", ["INDEX_FUTURES", "INDEX_FUTURES"]),
    ("DEDUCTION_NOT_SPECIFIED_IN_MATCHED_CLAUSE", None)])
def test_caller_cannot_substitute_an_unproven_instrument_set(scope, required):
    with pytest.raises(ValueError):
        margin_absence_evidence("", requirement_scope=scope, required_instruments=required)
