"""真实信息、材料真实性等变体及正确否定、发行人自证的原文边界。"""

import pytest
from scripts.local_fund_filing_guard import filing_confirmation_issues

SOURCE = "本通知书不表明对提交备案材料的真实性、准确性、完整性的保证或认定。"


@pytest.mark.parametrize(
    "statement",
    [
        "备案仅确认备案信息的真实性、准确性、完整性，不表明对投资价值的保证。",
        "公司收到证监会备案通知书，确认其境外发行上市备案信息真实、准确、完整。",
    ],
)
def test_affirmation_variants_are_flagged_even_with_other_disclaimer(statement):
    issues = filing_confirmation_issues(statement, SOURCE)
    assert issues and issues[0]["source_denials"][0]["raw"] in SOURCE


@pytest.mark.parametrize(
    "statement",
    [
        "备案不代表证监会确认信息真实、准确、完整。",
        "公司确认本公告真实，备案不代表对投资价值作出保证。",
    ],
)
def test_correct_denial_and_company_self_declaration_are_not_regulator_confirmation(statement):
    assert filing_confirmation_issues(statement, SOURCE) == []


def test_no_explicit_source_denial_means_rule_cannot_judge():
    assert filing_confirmation_issues("备案确认材料真实性。", "公司收到备案通知书。") == []
