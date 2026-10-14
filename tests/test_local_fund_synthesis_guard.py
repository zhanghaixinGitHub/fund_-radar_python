"""合法编号不能替代引用支持主体，历史日期不能替代预测目标日。"""

from scripts.local_fund_synthesis_guard import synthesis_issues


def test_cutoff_day_change_condition_is_not_a_target_day_condition():
    raw = {"change_conditions": ["若2026-10-09行情反弹，可改变当日判断。"]}
    assert (
        synthesis_issues(raw, {}, [], {"target_date": "2026-10-12"})[0]["code"]
        == "CHANGE_CONDITION_TARGET_DATE_MISMATCH"
    )
    assert synthesis_issues({"change_conditions": ["若目标日行情反弹。"]}, {}, [], {"target_date": "2026-10-12"}) == []


def test_a_valid_reference_to_a_different_company_does_not_support_the_claim():
    facts = {"FA": {"text": "甲公司净利润增长。"}, "FB": {"text": "乙公司净利润增长。"}}
    raw = {"reasons": [{"refs": ["FB"], "meaning": "甲公司净利润增长。"}]}
    issues = synthesis_issues(raw, facts, [{"name": "甲公司"}, {"name": "乙公司"}], {"target_date": "2026-10-12"})
    assert issues[0]["code"] == "REASON_COMPANY_NOT_SUPPORTED_BY_REFS"
    raw["reasons"][0]["refs"] = ["FA"]
    assert synthesis_issues(raw, facts, [{"name": "甲公司"}], {"target_date": "2026-10-12"}) == []
