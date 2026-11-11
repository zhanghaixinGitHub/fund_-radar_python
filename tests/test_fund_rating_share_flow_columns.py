"""份额变动表的并列折行仍须唯一识别A/C，不由金额或份额代码猜列。"""

import pytest
from app.services.fund_rating_direct_index import _flows


def report(heading, closing="90.00 240.00"):
    return [
        f"""开放式基金份额变动
单位：份
{heading}
基金合同生效日 100.00 200.00
本报告期期初基金份额总额 100.00 200.00
本报告期基金总申购份额 10.00 50.00
减：本报告期基金总赎回份额 20.00 10.00
本报告期基金拆分变动份额 - -
本报告期期末基金份额总额 {closing}
重大事件揭示
"""
    ]


@pytest.mark.parametrize(
    "heading",
    [
        "项目 隔离指数联接A 隔离指数联接C",
        "隔离指数ETF发  隔离指数ETF发\n项目\n起联接A 起联接C",
    ],
)
def test_verified_original_header_forms_keep_actual_share_flows(heading):
    rows, proof = _flows(report(heading))
    assert [r["share"] for r in rows] == ["A", "C"]
    assert [r["redeemed"] for r in rows] == ["20.00", "10.00"]
    assert len(proof) == 2


@pytest.mark.parametrize(
    "heading",
    [
        "隔离指数ETF发  另一个指数ETF发\n项目\n起联接A 起联接C",
        "隔离指数ETF发  隔离指数ETF发\n项目\n起联接C 起联接A",
        "隔离指数ETF发  隔离指数ETF发\n项目\n起联接A 另一个后缀C",
        "隔离指数ETF发  隔离指数ETF发\n项目\n起联接A 起联接C 起联接D",
    ],
)
def test_wrong_identity_reversed_columns_or_extra_share_is_rejected(heading):
    with pytest.raises(ValueError, match="SHARE_FLOW_COLUMNS"):
        _flows(report(heading))


def test_wrapped_headers_do_not_relax_share_balance_conservation():
    with pytest.raises(ValueError):
        _flows(report("隔离指数ETF发 隔离指数ETF发\n项目\n起联接A 起联接C", closing="91.00 240.00"))


@pytest.mark.parametrize("heading", ["项目 隔离指数C 隔离指数A", "隔离指数隔离指数CA"])
def test_explicit_reverse_adapter_keeps_actual_c_then_a_columns(heading):
    rows, _ = _flows(report(heading), allow_reversed=True)

    assert [r["share"] for r in rows] == ["A", "C"]
    assert [r["redeemed"] for r in rows] == ["10.00", "20.00"]


def test_detached_a_under_first_column_is_not_mistaken_for_reverse_c_a():
    rows, _ = _flows(report("项目    隔离指数    隔离指数 C\n        A"), allow_reversed=True)
    assert [r["share"] for r in rows] == ["A", "C"]
    assert [r["redeemed"] for r in rows] == ["20.00", "10.00"]


def test_detached_a_outside_first_column_is_not_guessed_by_amounts():
    with pytest.raises(ValueError, match="WRAPPED_CLASS_ALIGNMENT"):
        _flows(report("项目    隔离指数    隔离指数 C\n                      A"), allow_reversed=True)
    with pytest.raises(ValueError, match="SHARE_FLOW_COLUMNS"):
        _flows(report("项目    其他指数    隔离指数 C\n        A"), allow_reversed=True)


def test_interleaved_row_labels_keep_original_amounts():
    text = """开放式基金份额变动
单位：份
项目 隔离指数联接A 隔离指数联接C
基金合同生效日 100.00 200.00
本报告期期初基金 100.00 200.00 份额总额
本报告期基金总申 10.00 50.00 购份额
减:本报告期基金 20.00 10.00 总赎回份额
本报告期基金拆分 - - 变动份额
本报告期期末基金 90.00 240.00 份额总额
重大事件揭示
"""
    rows, _ = _flows([text], allow_interleaved=True)

    assert [r["redeemed"] for r in rows] == ["20.00", "10.00"]


def test_interleaved_row_labels_support_amounts_inside_share_words():
    text = """开放式基金份额变动
单位：份
项目 隔离指数A 隔离指数C
基金合同生效日 100.00 200.00
本报告期期初基金份 100.00 200.00 额总额
本报告期基金总申购 10.00 50.00 份额
减:本报告期基金总 20.00 10.00 赎回份额
本报告期基金拆分变 - - 动份额
本报告期期末基金份 90.00 240.00 额总额
重大事件揭示
"""
    rows, _ = _flows([text], allow_interleaved=True)

    assert [r["redeemed"] for r in rows] == ["20.00", "10.00"]


def test_three_share_report_checks_all_classes_and_alternate_section_end():
    text = """开放式基金份额变动
单位：份
项目 隔离指数A 隔离指数C 隔离指数Y
基金合同生效日 100.00 200.00 50.00
本报告期期初基金份额总额 100.00 200.00 50.00
本报告期基金总申购份额 10.00 50.00 5.00
减：本报告期基金总赎回份额 20.00 10.00 5.00
本报告期基金拆分变动份额 - - -
本报告期期末基金份额总额 90.00 240.00 50.00
基金管理人运用固有资金投资本基金情况
"""
    rows, _ = _flows([text], additional_shares=("Y",))

    assert [r["share"] for r in rows] == ["A", "C", "Y"]
    assert [r["redeemed"] for r in rows] == ["20.00", "10.00", "5.00"]


@pytest.mark.parametrize(
    "heading",
    [
        "易方达沪深300精选增 易方达沪深300精选增 易方达沪深300精选增强A 强C 强Y",
        "国泰海通中证500 国泰海通中证500 国泰海通中 指数增强A 指数增强C 证500指数增强Y",
    ],
)
def test_three_share_wrapped_header_preserves_repeated_prefixes_and_suffixes(heading):
    text = f"""开放式基金份额变动
单位：份
项目 {heading}
基金合同生效日 100.00 200.00 50.00
本报告期期初基金份额总额 100.00 200.00 50.00
本报告期基金总申购份额 10.00 50.00 5.00
减：本报告期基金总赎回份额 20.00 10.00 5.00
本报告期基金拆分变动份额 - - -
本报告期期末基金份额总额 90.00 240.00 50.00
基金管理人运用固有资金投资本基金情况
"""
    rows, _ = _flows([text], additional_shares=("Y",))

    assert [r["share"] for r in rows] == ["A", "C", "Y"]
    assert [r["redeemed"] for r in rows] == ["20.00", "10.00", "5.00"]


def test_three_share_e_report_checks_all_classes():
    text = """开放式基金份额变动
单位：份
项目 隔离指数A 隔离指数C 隔离指数E
基金合同生效日 100.00 200.00 50.00
本报告期期初基金份额总额 100.00 200.00 50.00
本报告期基金总申购份额 10.00 50.00 5.00
减：本报告期基金总赎回份额 20.00 10.00 5.00
本报告期基金拆分变动份额 - - -
本报告期期末基金份额总额 90.00 240.00 50.00
基金管理人运用固有资金投资本基金情况
"""
    rows, _ = _flows([text], additional_shares=("E",))

    assert [r["share"] for r in rows] == ["A", "C", "E"]
    assert [r["redeemed"] for r in rows] == ["20.00", "10.00", "5.00"]


def test_three_share_decimal_wraps_rejoin_only_printed_digits_and_reconcile():
    text = """开放式基金份额变动
单位：份
项目 隔离指数A 隔离指数C 隔离指数Y
基金合同生效日 100.00 200.00 50.00
本报告期期初基金份额总额  1,691,449,234.8  1,020,482,300.5  762,988.81
5  0
本报告期基金总申购份额  812,320,517.50  1,435,997,059.9  2,503,361.7
9  2
减：本报告期基金总赎回份额  960,886,354.21  1,260,178,097.5  684,805.78
8
本报告期基金拆分变动份额 - - -
本报告期期末基金份额总额  1,542,883,398.1  1,196,301,262.9  2,581,544.7
4  1  5
基金管理人运用固有资金投资本基金情况
"""
    rows, _ = _flows([text], additional_shares=("Y",))

    assert [r["opening"] for r in rows] == ["1691449234.85", "1020482300.50", "762988.81"]
    assert [r["redeemed"] for r in rows] == ["960886354.21", "1260178097.58", "684805.78"]


def test_four_share_lof_wrapped_header_checks_common_security_and_class_order():
    text = """开放式基金份额变动
单位：份
项目 南方中证 南方中证 南方中证 南方中证
500ETF联接 500ETF联接 500ETF联接 500ETF联接Y
(LOF)A (LOF)C (LOF)I
基金合同生效日 100.00 200.00 50.00 80.00
本报告期期初  3,485,811,266.3  278,114,470.86  137,621,193.57  5,194,834.05
基金份额总额  3
本报告期基金  700,105,392.47  630,604,142.61  214,099,714.36  22,432,201.23
总申购份额
减：本报告期基  1,152,466,403.1  561,160,288.63  199,820,275.11  9,988,885.95
金总赎回份额  0
本报告期基金  -  -  -  -
拆分变动份额
本报告期期末  3,033,450,255.7  347,558,324.84  151,900,632.82  17,638,149.33
基金份额总额  0
基金管理人运用固有资金投资本基金情况
"""
    rows, _ = _flows([text], additional_shares=("Y", "I"), allow_interleaved=True)

    assert [r["share"] for r in rows] == ["A", "C", "Y", "I"]
    assert [r["redeemed"] for r in rows] == ["1152466403.10", "561160288.63", "199820275.11", "9988885.95"]


def test_report_page_header_before_share_flow_unit_is_not_fund_identity():
    text = report("项目 隔离指数A 隔离指数C")[0].replace("单位：份", "隔离基金2026年中期报告第54页共61页\n单位：份", 1)
    rows, _ = _flows([text])

    assert [r["redeemed"] for r in rows] == ["20.00", "10.00"]


def test_split_change_note_after_label_accepts_reported_dashes():
    text = report("项目 隔离指数A 隔离指数C")[0].replace(
        "本报告期基金拆分变动份额 - -", '本报告期基金拆分变动份额(份额减少以"-"填列) - -'
    )
    rows, _ = _flows([text], allow_period_variant=True)

    assert [r["redeemed"] for r in rows] == ["20.00", "10.00"]


def test_period_variant_adapter_also_accepts_standard_report_labels():
    rows, _ = _flows(report("项目 隔离指数A 隔离指数C"), allow_period_variant=True)

    assert [r["redeemed"] for r in rows] == ["20.00", "10.00"]


def test_period_variant_without_contract_date_and_three_shares():
    text = """开放式基金份额变动
单位：份
项目 隔离指数A 隔离指数C 隔离指数Y
报告期期初基金份额总额 100.00 200.00 50.00
报告期期间基金总申购份额 10.00 50.00 5.00
减:报告期期间基金总赎回份额 20.00 10.00 5.00
报告期期间基金拆分变动份额 - - -
报告期期末基金份额总额 90.00 240.00 50.00
基金管理人运用固有资金投资本基金情况
"""
    rows, _ = _flows([text], additional_shares=("Y",), allow_period_variant=True)

    assert [r["share"] for r in rows] == ["A", "C", "Y"]
    assert [r["redeemed"] for r in rows] == ["20.00", "10.00", "5.00"]
