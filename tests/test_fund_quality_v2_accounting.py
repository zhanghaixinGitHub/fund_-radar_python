"""账面观察的合成反例只测试拒收边界，不能作为真实校准或来源证明。"""

from datetime import date

import pytest
from app.services.fund_quality_v2_accounting import accounting_observations


def original(chapter=7):
    return [f"""测试股票基金123456 2025年年度报告
{chapter}.1 资产负债表
单位：人民币元
本期末2025年12月31日 上年度末2024年12月31日
货币资金 {chapter}.4.7.1 100.00 80.00
资产总计 200.00 180.00
负债合计 20.00 10.00
净资产合计 180.00 170.00
{chapter}.2 利润表
8.4 期末权益投资明细
{chapter}.4.7.1 货币资金
单位：人民币元
本期末2025年12月31日 上年度末2024年12月31日
活期存款 80.00 50.00
等于：本金 79.00 49.00
加：应计利息 1.00 1.00
定期存款 - -
{chapter}.4.7.2 交易性金融资产
"""]


@pytest.mark.parametrize("chapter", [5, 6, 7])
def test_financial_facts_do_not_depend_on_domestic_stock_chapter(chapter):
    pages = original(chapter)
    result = accounting_observations(pages, date(2025, 12, 31))
    assert len(result["observations"]) == 7 and len(result["reconciliations"]) == 2
    assert result["observations"]["money_funds"]["value"] == "100.00"
    assert result["observations"]["demand_deposits_total"]["value"] == "80.00"
    assert "8.4 期末权益" in pages[0] and "cash_target_deviation" not in result["observations"]


def test_current_cent_suffix_requires_exact_column_and_adjacent_total_rows():
    text = original()[0].replace("资产总计 200.00", "资产总计 200.56")
    block = ("未分配利润 100.00 90.00\n净资产合计     180.5   170.00\n"
             "                   6\n负债和净资产总计 200.56 180.00")
    text = text.replace("净资产合计 180.00 170.00", block)
    result = accounting_observations([text], date(2025, 12, 31))
    amount = result["observations"]["net_assets"]
    assert amount["value"] == "180.56" and "\n                   6" in amount["original_row"]
    assert amount["comparison_cell_not_used"] == "170.00"
    measure = amount["monetary_continuation_evidence"]["matching_column_measures"][0]
    assert measure["column_measure"] == "EAST_ASIAN_DISPLAY_WIDTH"
    for invalid in (text.replace("                   6", "                  6"),
                    text.replace("                   6", "                   66"),
                    text.replace("负债和净资产总计 200.56", "其他正文 200.56")):
        assert "net_assets" not in accounting_observations([invalid], date(2025, 12, 31))["observations"]


def unnumbered_balance():
    return original()[0].replace("7.1 资产负债表", "年度财务报告\n资产负债表").replace("7.2 利润表", "利润表")


def test_issuer_unnumbered_statement_keeps_original_heading_without_invented_chapter():
    result = accounting_observations([unnumbered_balance()], date(2025, 12, 31))
    assert len(result["observations"]) == 7
    assert result["financial_chapter"] is None and result["original_balance_heading"] == "资产负债表"
    assert result["balance_heading_pages"] == [1]


@pytest.mark.parametrize("mutation", ["no_parent", "contents_only", "no_stop", "duplicate"])
def test_unnumbered_statement_needs_financial_parent_unique_body_and_full_stop(mutation):
    text = unnumbered_balance()
    if mutation == "no_parent":
        text = text.replace("年度财务报告\n", "")
    elif mutation == "contents_only":
        text = text.replace("\n资产负债表\n", "\n资产负债表 ........ 23\n")
    elif mutation == "no_stop":
        text = text.replace("\n利润表\n", "\n利润表 ........ 24\n")
    else:
        text += unnumbered_balance()
    with pytest.raises(ValueError, match="BALANCE_DATE_CNY_UNIT_OR_UNIQUE_BODY"):
        accounting_observations([text], date(2025, 12, 31))


def test_two_column_parenthesized_note_dates_keep_explicit_comparison_and_original_header():
    prefix, note = original()[0].split("\n7.4.7.1 货币资金\n", 1)
    note = note.replace("本期末2025年12月31日 上年度末2024年12月31日",
                        "项目 本期末\n（2025年12月31日）\n上年度末\n（2024年12月31日）\n36")
    result = accounting_observations([prefix + "\n7.4.7.1 货币资金\n" + note], date(2025, 12, 31))
    assert result["observations"]["demand_deposits_total"]["comparison_cell_not_used"] == "50.00"
    assert "（2025年12月31日）" in result["cash_note_date_header"]
    assert len(result["observations"]) == 7
    invalid = note.replace("2025年12月31日", "2023年12月31日")
    rejected = accounting_observations([prefix + "\n7.4.7.1 货币资金\n" + invalid], date(2025, 12, 31))
    assert "demand_deposits_total" not in rejected["observations"]


@pytest.mark.parametrize("mutation", ["date", "unit", "duplicate"])
def test_date_currency_and_body_ambiguity_cannot_be_overridden(mutation):
    pages = original()
    if mutation == "date":
        pages[0] = pages[0].replace("本期末2025年12月31日", "本期末2024年12月31日")
    elif mutation == "unit":
        pages[0] = pages[0].replace("人民币元", "人民币万元")
    else:
        pages *= 2
    with pytest.raises(ValueError, match="BALANCE_DATE_CNY_UNIT_OR_UNIQUE_BODY"):
        accounting_observations(pages, date(2025, 12, 31))


def test_current_dash_is_not_zero_but_comparison_dash_does_not_erase_explicit_current_amount():
    pages = original()
    pages[0] = pages[0].replace("20.00 10.00", "- 10.00").replace("100.00 80.00", "100.00 -")
    result = accounting_observations(pages, date(2025, 12, 31))
    assert "total_liabilities" not in result["observations"]
    assert result["unresolved"]["total_liabilities"]["reason"] == "ACCOUNTING_CURRENT_DASH_NOT_ASSIGNED_ZERO"
    assert result["observations"]["money_funds"]["value"] == "100.00"
    assert len(result["reconciliations"]) == 1


def test_conservation_cannot_be_used_to_fill_missing_row():
    pages = original()
    pages[0] = pages[0].replace("负债合计 20.00 10.00\n", "")
    result = accounting_observations(pages, date(2025, 12, 31))
    assert "total_liabilities" not in result["observations"]
    assert result["unresolved"]["total_liabilities"]["reason"] == "ACCOUNTING_ROW_MISSING_OR_AMBIGUOUS"


@pytest.mark.parametrize("before,after", [("200.00 180.00", "300.00 180.00"), ("79.00 49.00", "78.00 49.00")])
def test_conflicting_explicit_amounts_reject_observation_bundle(before, after):
    with pytest.raises(ValueError, match="EXPLICIT_AMOUNTS_DO_NOT_RECONCILE"):
        accounting_observations([original()[0].replace(before, after)], date(2025, 12, 31))


def test_current_date_accepts_explicit_leading_zero_without_touching_original_digits():
    pages = [original()[0].replace("12月31日", "06月30日")]
    result = accounting_observations(pages, date(2025, 6, 30))
    assert result["observations"]["money_funds"]["observed_on"] == "2025-06-30"
    assert "06月30日" in pages[0]


def test_row_major_headers_still_bind_current_before_comparison():
    pages = [original()[0].replace("本期末2025年12月31日 上年度末2024年12月31日",
                                  "本期末 上年度末\n2025年12月31日 2024年12月31日")]
    result = accounting_observations(pages, date(2025, 12, 31))
    assert len(result["observations"]) == 7 and result["observations"]["money_funds"]["value"] == "100.00"
    for invalid in (pages[0].replace("本期末 上年度末", "上年度末 本期末"),
                    pages[0].replace("2025年12月31日 2024年12月31日", "2024年12月31日 2025年12月31日")):
        with pytest.raises(ValueError, match="BALANCE_DATE_CNY_UNIT_OR_UNIQUE_BODY"):
            accounting_observations([invalid], date(2025, 12, 31))


def single_note():
    prefix, note = original()[0].split("\n7.4.7.1 货币资金\n", 1)
    note = note.replace("本期末2025年12月31日 上年度末2024年12月31日", "本期末\n2025年12月31日")
    for before, after in (("80.00 50.00", "80.00"), ("79.00 49.00", "79.00"),
                          ("1.00 1.00", "1.00"), ("定期存款 - -", "定期存款 -")):
        note = note.replace(before, after)
    return prefix + "\n7.4.7.1 货币资金\n" + note


def test_explicit_single_current_column_is_not_a_broken_comparison_column():
    result = accounting_observations([single_note()], date(2025, 12, 31))
    assert len(result["observations"]) == 7
    row = result["observations"]["demand_deposits_principal"]
    assert row["declared_columns"] == 1 and row["comparison_cell_not_used"] is None
    assert row["value"] == "79.00" and "cash_target_deviation" not in result["observations"]


def test_single_current_note_date_in_parentheses_is_explicit_not_inherited():
    text = single_note().replace("本期末\n2025年12月31日", "本期末（2025年12月31日）")
    result = accounting_observations([text], date(2025, 12, 31))
    assert len(result["observations"]) == 7
    invalid = text.replace("本期末（2025年12月31日）", "本期末（2024年12月31日）")
    assert "demand_deposits_principal" not in accounting_observations([invalid], date(2025, 12, 31))["observations"]


def test_cash_note_number_follows_explicit_balance_reference():
    text = single_note().replace("7.4.7.1", "7.4.6.1").replace("7.4.7.2", "7.4.6.2")
    result = accounting_observations([text], date(2025, 12, 31))
    assert result["cash_note_reference"] == "7.4.6.1" and len(result["observations"]) == 7
    mismatched = text.replace("\n7.4.6.1 货币资金\n", "\n7.4.7.1 货币资金\n")
    rejected = accounting_observations([mismatched], date(2025, 12, 31))
    assert len(rejected["observations"]) == 4
    gap = rejected["unresolved"]["demand_deposits_detail"]
    assert gap["reason"] == "ACCOUNTING_BALANCE_NOTE_REFERENCE_CONFLICT"
    assert gap["conflicting_original_references"][0]["balance_reference"] == "7.4.6.1"
    assert gap["conflicting_original_references"][0]["note_reference"] == "7.4.7.1"


@pytest.mark.parametrize("mutation", ["previous_label", "wrong_date", "wrong_unit", "extra_values"])
def test_single_cash_note_needs_explicit_date_currency_and_actual_single_cells(mutation):
    prefix, note = single_note().split("\n7.4.7.1 货币资金\n", 1)
    if mutation == "previous_label":
        note = note.replace("本期末", "本期末 上年度末")
    elif mutation == "wrong_date":
        note = note.replace("2025年12月31日", "2024年12月31日")
    elif mutation == "wrong_unit":
        note = note.replace("人民币元", "美元")
    else:
        note = note.replace("等于：本金 79.00", "等于：本金 79.00 49.00")
    result = accounting_observations([prefix + "\n7.4.7.1 货币资金\n" + note], date(2025, 12, 31))
    assert "demand_deposits_principal" not in result["observations"]
    assert "money_funds" in result["observations"] and result["unresolved"]


def test_note_reference_conflict_requires_full_note_total_not_demand_subtotal():
    text = single_note().replace("\n7.4.7.1 货币资金\n", "\n5.4.7.1 货币资金\n")
    text = text.replace("7.4.7.2 交易性金融资产", "5.4.7.2 交易性金融资产")
    text = text.replace("定期存款 -", "定期存款 -\n其他存款 20.00\n合计 100.00")
    result = accounting_observations([text], date(2025, 12, 31))
    assert len(result["observations"]) == 7
    assert result["observations"]["demand_deposits_total"]["value"] == "80.00"
    proof = result["cash_note_binding"]
    assert proof["balance_reference"] == "7.4.7.1" and proof["note_reference"] == "5.4.7.1"
    assert proof["explicit_note_total"]["value"] == "100.00"
    for invalid in (text.replace("合计 100.00", "合计 80.00"), text.replace("合计 100.00", "合计 -"),
                    text.replace("合计 100.00", "合计 100.00\n合计 100.00"),
                    text.replace("其他存款 20.00\n合计 100.00", "其他存款 20.00")):
        rejected = accounting_observations([invalid], date(2025, 12, 31))
        assert len(rejected["observations"]) == 4 and rejected["cash_note_binding"] is None
        gap = rejected["unresolved"]["demand_deposits_detail"]
        assert gap["reason"] == "ACCOUNTING_BALANCE_NOTE_REFERENCE_CONFLICT"


def test_explicit_credit_loss_is_subtracted_and_literal_zero_is_an_observation():
    for credit_loss, principal in (("1.00", "80.00"), ("0.00", "79.00")):
        text = single_note().replace("等于：本金 79.00", "等于：本金 " + principal)
        text = text.replace("定期存款 -", "减：坏账准备 " + credit_loss + "\n定期存款 -")
        result = accounting_observations([text], date(2025, 12, 31))
        assert result["observations"]["demand_deposits_credit_loss"]["value"] == credit_loss
        assert result["reconciliations"][-1]["explicit_deduction"] == "demand_deposits_credit_loss"


@pytest.mark.parametrize("total", ["80.00", "76.00"])
def test_credit_loss_dash_remains_unknown_even_if_other_reported_amounts_match(total):
    text = single_note().replace("活期存款 80.00", "活期存款 " + total)
    text = text.replace("定期存款 -", "减：坏账准备 -\n定期存款 -")
    result = accounting_observations([text], date(2025, 12, 31))
    assert "demand_deposits_credit_loss" not in result["observations"]
    assert result["unresolved"]["demand_deposits_credit_loss"]["reason"] == "ACCOUNTING_CURRENT_DASH_NOT_ASSIGNED_ZERO"
    state = result["reconciliations"][-1]["state"]
    assert state == ("CREDIT_LOSS_UNRESOLVED_NO_IMPUTATION" if total == "76.00"
                     else "DISCLOSED_AMOUNTS_MATCH_CREDIT_LOSS_DASH_STILL_UNASSIGNED")


def test_explicit_credit_loss_conflict_is_not_masked_by_unknown_cash_semantics():
    text = single_note().replace("定期存款 -", "减：坏账准备 2.00\n定期存款 -")
    with pytest.raises(ValueError, match="EXPLICIT_AMOUNTS_DO_NOT_RECONCILE"):
        accounting_observations([text], date(2025, 12, 31))


@pytest.mark.parametrize("stub", [False, True])
def test_actual_column_date_fragments_admit_current_amounts_without_guessing_period(stub):
    header = "\n".join(["".ljust(30) + "本期末".ljust(26) + "上年度末",
                        *(["资 产 附注号"] if stub else []),
                        "".ljust(30) + "2025年12月31".ljust(26) + "2024年12月",
                        "".ljust(30) + "日".ljust(26) + "31日"])
    text = original()[0].replace("本期末2025年12月31日 上年度末2024年12月31日", header)
    result = accounting_observations([text], date(2025, 12, 31))
    assert result["observations"]["money_funds"]["value"] == "100.00"
    assert result["observations"]["demand_deposits_principal"]["value"] == "79.00"
    proof = result["balance_date_column_evidence"]
    assert proof["current_date"] == "2025-12-31" and proof["comparison_date"] == "2024-12-31"
    assert "2024年12月" in proof["original_header"] and proof["fragments"]


def test_single_cash_note_header_can_contain_only_an_explicit_project_stub_line():
    text = single_note().replace("本期末\n2025年12月31日", "本期末\n 项目 \n2025年12月31日")
    result = accounting_observations([text], date(2025, 12, 31))
    assert len(result["observations"]) == 7
    assert "项目" in result["cash_note_date_header"]
    assert result["observations"]["demand_deposits_principal"]["declared_columns"] == 1
    for invalid in (text.replace(" 项目 ", " 上年度末 "), text.replace(" 项目 ", " 日期待确认 ")):
        assert "demand_deposits_principal" not in accounting_observations([invalid], date(2025, 12, 31))["observations"]


def test_complete_statement_headers_are_not_silently_cut_in_the_middle_of_a_repeat():
    before, after = original()[0].split("7.2 利润表", 1)
    repeated = "本期末2025年12月31日 上年度末2024年12月31日"
    result = accounting_observations([before + "\n" * 1800 + repeated + "\n7.2 利润表" + after], date(2025, 12, 31))
    assert result["balance_date_column_evidence"]["concordant_header_count"] == 2
    assert len(result["observations"]) == 7
