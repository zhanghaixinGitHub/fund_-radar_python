"""第二层审计必须保留原文边界，不能借“格式修复”放过条件和数值改写。"""

import pytest
from scripts.local_fund_semantic_audit import independent_issues, whitespace_anchor


def test_only_layout_whitespace_can_be_resolved_to_original_offsets():
    body = "前文。\r\n不超过 18,138,687 股，\r\n且须股东会审议。后文。"
    text = "不超过18,138,687股，且须股东会审议。"
    anchor = whitespace_anchor(body, text)
    assert anchor["source_text"] == body[anchor["start"] : anchor["end"]]
    assert anchor["raw_text"] == text
    assert anchor["match"] == "WHITESPACE_ONLY"


@pytest.mark.parametrize(
    "claim",
    [
        "18,138,687股，且须股东会审议。",
        "不超过18,138,687股，须股东会审议。",
        "不超过181,386,870股，且须股东会审议。",
        "不超过18,138,687股；且须股东会审议。",
    ],
)
def test_literal_fragment_matching_does_not_prove_condition_completeness(claim):
    # 去掉开头界限词后的短子串在原文中仍真实存在，因此定位本身无法证明
    # 条件完整性。调用方另查解释、条件；其他三个改字/改数/拼接须严格拒绝。
    source = "不超过 18,138,687 股，且须股东会审议。"
    if claim.startswith("18,"):
        assert whitespace_anchor(source, claim) is not None
    else:
        assert whitespace_anchor(source, claim) is None


def test_ambiguous_repeated_text_cannot_choose_an_occurrence():
    assert whitespace_anchor("不计利息。另一事项不计利息。", "不计利息。") is None


def test_amounts_cannot_be_asserted_when_quotes_cover_only_meeting_procedures():
    assert "AMOUNT_WITHOUT_QUOTED_SOURCE" in independent_issues(
        "中小股东持股1.8139亿股。", "会议通知时间符合规定。", "中小股东持股18,138,687股。"
    )


def test_target_requirement_variant_is_rejected_but_full_vesting_is_allowed():
    body = "目标值（Am），触发值（An）。An≤A<Am时X=A/Am；A≥Am时X=100%。"
    assert "TARGET_REQUIRED_FOR_ANY_VESTING" in independent_issues("需满足增长率目标方可归属。", body, body)
    assert "TARGET_REQUIRED_FOR_ANY_VESTING" not in independent_issues("达到目标值方可全额归属。", body, body)


def test_future_purchase_conditions_are_distinct_from_current_non_purchase():
    body = "截至公告披露日，员工持股计划尚未购买公司股份。"
    assert "EXPLICIT_NON_PURCHASE_LOST" not in independent_issues("未提及未来购买条件或机制。", body, body)
    assert "EXPLICIT_NON_PURCHASE_LOST" in independent_issues("未提及实际购买股份情况。", body, body)


def test_positive_profit_condition_does_not_use_the_annual_average_basis():
    body = "可分配利润为正时可以分红。累计现金分红不低于三年年均可分配利润的30%。"
    assert "ANNUAL_AVERAGE_BASIS_MISSING" not in independent_issues("现金分红条件包含可分配利润为正值。", body, body)
    assert "ANNUAL_AVERAGE_BASIS_MISSING" in independent_issues("累计现金分红不低于三年可分配利润的30%。", body, body)


def test_filing_value_disclaimer_cannot_cancel_a_false_truth_confirmation():
    body = "备案通知书不表明证监会对材料真实性、准确性、完整性作出保证或者认定。"
    wrong = "备案仅确认备案信息的真实性、准确性、完整性，不表明对投资价值作出保证。"
    assert "FILING_TRUTH_CONFIRMATION_UNSUPPORTED" in independent_issues(wrong, body, body)
    assert "FILING_TRUTH_CONFIRMATION_UNSUPPORTED" not in independent_issues(
        "备案不代表对材料真实性的确认。", body, body
    )


def test_rmb_amount_cannot_take_a_usd_suffix():
    assert "CURRENCY_UNIT_COMBINATION_UNSUPPORTED" in independent_issues(
        "债券本金39.6亿元美元。", "本金人民币3,960,000,000元，以美元结算。", ""
    )
