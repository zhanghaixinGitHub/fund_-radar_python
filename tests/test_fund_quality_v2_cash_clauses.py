"""现金条款的范围反例；合成文本只做工程验证，不作为真实校准资料。"""

import pytest
from app.services.fund_quality_v2_cash_clauses import cash_basket_clause, cash_clause_page_view, cash_report_clause

EXCLUSIONS = "，其中现金不包括结算备付金、存出保证金、应收申购款等。"


def clause(instruments="股指期货合约", tail=EXCLUSIONS):
    return ("本基金每个交易日日终在扣除" + instruments + "需缴纳的交易保证金后，"
            "保持不低于基金资产净值5%的现金或到期日在一年以内的政府债券" + tail)


@pytest.mark.parametrize("instruments,expected", [
    ("股指期货合约", ["INDEX_FUTURES"]),
    ("股指期货合约、国债期货合约", ["INDEX_FUTURES", "TREASURY_FUTURES"]),
    ("股指期货、股票期权合约", ["INDEX_FUTURES", "STOCK_OPTIONS"]),
    ("股指期货、国债期货和股票期权合约", ["INDEX_FUTURES", "STOCK_OPTIONS", "TREASURY_FUTURES"]),
    ("国债期货和股指期货合约", ["INDEX_FUTURES", "TREASURY_FUTURES"]),
])
def test_named_margin_types_are_not_reduced_to_index_futures(instruments, expected):
    value = cash_basket_clause(clause(instruments))
    assert value["state"] == "REPORTED_CLAUSE_BASIS_IDENTIFIED"
    assert value["required_margin_instruments"] == expected
    assert value["margin_deduction_original"] == instruments


def test_all_required_margin_is_not_identified_by_three_named_types():
    value = cash_basket_clause(clause("全部"))
    assert value["margin_requirement_scope"] == "ALL_REQUIRED_TRADING_MARGIN"
    assert value["required_margin_instruments"] is None


@pytest.mark.parametrize("tail", ["。", "，其中现金不包括结算备付金、19存出保证金、应收申购款等。"])
def test_missing_definition_or_unverified_page_number_retains_exact_gap(tail):
    value = cash_basket_clause(clause(tail=tail))
    assert value["state"] == "REPORTED_CLAUSE_ONLY_BASIS_INCOMPLETE"
    assert "CASH_EXCLUSION_DEFINITION_NOT_IDENTIFIED" in value["missing_basis"]
    assert value["following_unresolved_fragment"]


def test_reported_floor_without_deduction_cannot_prove_zero_margin():
    text = "本基金保持不低于基金资产净值5%的现金或者到期日在一年以内的政府债券" + EXCLUSIONS
    value = cash_basket_clause(text)
    assert value["reported_floor_ratio"] == "0.05"
    assert "MARGIN_DEDUCTION_SCOPE_NOT_EXPLICIT" in value["missing_basis"]
    assert value["margin_requirement_scope"] == "DEDUCTION_NOT_SPECIFIED_IN_MATCHED_CLAUSE"


def test_same_report_inconsistent_named_scope_requires_version_review():
    with pytest.raises(ValueError, match="MARGIN_SCOPE_CONFLICT"):
        cash_basket_clause(clause() + clause("股指期货合约、国债期货合约"))


def test_reordering_same_instruments_does_not_create_a_scope_conflict():
    value = cash_basket_clause(clause("股指期货合约、国债期货合约") + clause("国债期货和股指期货合约"))
    assert value["required_margin_instruments"] == ["INDEX_FUTURES", "TREASURY_FUTURES"]


def test_link_fund_subject_evidence_is_preserved_instead_of_a_verification_flag():
    value = cash_basket_clause("本基金投资于目标ETF的比例不低于基金资产净值90%；" + clause()[3:])
    assert "投资于目标ETF" in value["subject_evidence_normalized"]
    with pytest.raises(ValueError, match="OWN_SUBJECT"):
        cash_basket_clause("本基金所投资目标ETF的投资范围：" + clause())


def test_zero_floor_is_not_turned_into_ordinary_positive_floor_calculation():
    value = cash_basket_clause(clause().replace("5%", "0.0%"))
    assert "ZERO_FLOOR_CLAUSE_REQUIRES_SEPARATE_APPLICABILITY_REVIEW" in value["missing_basis"]


def test_floor_conflict_is_not_hidden_by_one_complete_looking_clause():
    with pytest.raises(ValueError, match="FLOOR_CONFLICT"):
        cash_basket_clause(clause() + clause().replace("5%", "10%"))


@pytest.mark.parametrize("word", ["不含", "不包括"])
def test_inline_cash_exclusions_do_not_disappear_inside_parentheses(word):
    text = ("本基金每个交易日日终，在扣除股指期货和国债期货合约需缴纳的交易保证金后，"
            f"保持现金（{word}结算备付金、存出保证金、应收申购款等）或者到期日在一年以内的政府债券"
            "投资比例合计不低于基金资产净值的5%。")
    result = cash_basket_clause(text)
    assert result["state"] == "REPORTED_CLAUSE_BASIS_IDENTIFIED"
    assert result["cash_exclusion_placement"] == "INLINE_PARENTHETICAL"
    assert result["required_margin_instruments"] == ["INDEX_FUTURES", "TREASURY_FUTURES"]


def test_explicit_deduction_without_daily_phrase_does_not_fabricate_daily_timing():
    result = cash_basket_clause(clause().replace("每个交易日日终", ""))
    assert result["state"] == "REPORTED_CLAUSE_BASIS_IDENTIFIED"
    assert result["timing_in_matched_clause"] == "NO_DAILY_TIME_PHRASE_IN_MATCHED_CLAUSE"


def test_unqualified_margin_deduction_is_distinct_from_a_missing_deduction():
    result = cash_basket_clause(clause("").replace("每个", "任何"))
    assert result["margin_requirement_scope"] == "UNQUALIFIED_REQUIRED_TRADING_MARGIN"
    assert result["required_margin_instruments"] is None


def test_free_cash_or_invested_bonds_and_own_definition_wording():
    text = ("本基金每个交易日日终在扣除股指期货合约需缴纳的保证金以后，基金保留的现金或投资于"
            "到期日在一年以内的政府债券的比例合计不低于基金资产净值的5%，"
            "本基金所指的现金不包括结算备付金、存出保证金、应收申购款等。")
    result = cash_basket_clause(text)
    assert result["state"] == "REPORTED_CLAUSE_BASIS_IDENTIFIED"
    assert result["margin_deduction_original"] == "股指期货合约"


def test_tight_own_etf_subject_chain_may_end_with_a_full_stop():
    text = "本基金投资于目标ETF的比例不低于基金资产净值90%。" + clause()[3:]
    result = cash_basket_clause(text)
    assert result["state"] == "REPORTED_CLAUSE_BASIS_IDENTIFIED"
    with pytest.raises(ValueError, match="OWN_SUBJECT"):
        cash_basket_clause("本基金所投资目标ETF的投资范围。" + clause())


def test_only_verified_header_and_three_page_sequence_can_rejoin_exclusions():
    title = "测试基金2026年中期报告\n"
    pages = [title + "1\n" + clause(tail="，其中现金不包括结算备付金、"),
             title + "2\n存出保证金、应收申购款等。", title + "3\n其他说明"]
    result = cash_report_clause(pages)
    assert result["state"] == "REPORTED_CLAUSE_BASIS_IDENTIFIED"
    removed = result["report_page_view"]["removed_headers"]
    assert all(any(r["page"] == number and r["removed_header"].strip() == str(number) for r in removed)
               for number in (1, 2, 3))
    assert "2存出保证金" not in result["original_clause_normalized"]


def test_one_page_transcript_does_not_invent_physical_pages_or_remove_an_inline_number():
    text = "测试基金2026年中期报告\n" + clause(tail=",其中现金不包括结算备付金、21存出保证金、应收申购款等。")
    result = cash_report_clause([text])
    assert "CASH_EXCLUSION_DEFINITION_NOT_IDENTIFIED" in result["missing_basis"]
    assert "21存出保证金" in result["following_unresolved_fragment"]


def test_unidentified_text_or_table_number_at_page_end_is_not_a_verified_page_number():
    pages = ["1\n正文", "2\n正文", "3\n正文"]
    view, proof = cash_clause_page_view(pages)
    assert not proof["removed_headers"] and view.startswith("1")
    title = "测试基金2026年中期报告\n"
    view, _ = cash_clause_page_view([title + "正文\n1", title + "正文\n2", title + "正文\n3"])
    assert all(str(number) in view for number in (1, 2, 3))
