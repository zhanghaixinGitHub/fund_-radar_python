"""持续运作披露的必要反例；合成报告只用于工程验证，不作为校准产品。"""

import pytest
from app.services.fund_quality_v2_operating_disclosure import operating_disclosure


def report(body, prefix=""):
    return [prefix + "\n本报告期自2024年1月1日起至12月31日止。\n"
            "4.8 报告期内管理人对本基金持有人数或基金资产净值预警情形的说明\n"
            + body + "\n§5 托管人报告\n5.1 其他内容"]


def parse(body, **kwargs):
    return operating_disclosure(report(body, **kwargs), observed_on="2024-12-31", source_hash="a" * 64)


def test_explicit_period_event_and_recovery_remain_separate():
    result = parse("本基金本报告期内连续二十个工作日以上出现基金资产净值低于五千万元的情形，"
                   "截至本报告期末，本基金基金资产净值已恢复至五千万元以上。")
    assert result["state"] == "REPORTED_ALERT_EXPLICITLY_RECOVERED"
    assert result["criteria_triggered"] == {"holders": None, "assets": True}
    assert result["period_end_recovery"]["assets"] == "EXPLICITLY_RECOVERED_AT_PERIOD_END"
    assert not result["score_or_grade_created"]


def test_no_recovery_statement_does_not_mean_still_below():
    result = parse("本报告期内，本基金连续二十个工作日出现基金资产净值低于五千万元的情形。")
    assert result["state"] == "REPORTED_ALERT_RECOVERY_UNKNOWN"
    assert result["period_end_recovery"]["assets"] is None


def test_explicit_still_below_differs_from_missing_recovery():
    result = parse("本报告期内，本基金连续二十个工作日出现基金资产净值低于五千万元的情形，"
                   "截至本报告期末，本基金基金资产净值仍低于五千万元。")
    assert result["state"] == "REPORTED_ALERT_EXPLICITLY_ONGOING"


@pytest.mark.parametrize("body", [
    "报告期内，本基金未出现连续二十个工作日基金份额持有人数量不满二百人或者基金资产净值低于五千万元的情形。",
    "本基金无连续二十个工作日出现基金份额持有人数量不满二百人或者基金资产净值低于五千万元情形。",
    "本报告期内未出现连续二十个工作日基金份额持有人数量不满二百人的情形。"
    "本报告期内未出现连续二十个工作日基金资产净值低于五千万的情形。",
])
def test_explicit_negative_has_both_scope_conditions(body):
    result = parse(body)
    assert result["state"] == "EXPLICIT_TWO_CRITERIA_NO_TRIGGER_REPORTED"
    assert result["criteria_triggered"] == {"holders": False, "assets": False}


@pytest.mark.parametrize("body", ["无。", "本报告期内本基金管理人无应说明预警信息。",
                                  "本基金本报告期无需要说明的相关情形。"])
def test_summary_absence_does_not_invent_threshold_verification(body):
    result = parse(body)
    assert result["report_alert_disclosed"] is False
    assert result["state"] == "SUMMARY_NO_APPLICABLE_ALERT_DISCLOSED"
    assert result["criteria_triggered"] == {"holders": None, "assets": None}


@pytest.mark.parametrize("body", [
    "如果连续二十个工作日基金资产净值低于五千万元应报告。",
    "期末基金资产净值为六亿元。", "-", "未发生其他重大事项。",
    "本报告期内，本基金未出现连续二十个工作日基金份额持有人数量不满二百人的情形。",
    "本报告期内，本基金连续二十个工作日出现基金资产净值低于五千万元的情形，预计下期恢复。",
])
def test_partial_or_hypothetical_information_cannot_become_complete_metric(body):
    assert parse(body)["state"] == "UNRESOLVED_DISCLOSURE"


def test_toc_or_other_chapter_none_cannot_replace_body():
    prefix = "4.8 报告期内管理人对本基金持有人数或基金资产净值预警情形的说明.........13\n"
    assert parse("本报告期内，本基金连续二十个工作日出现基金资产净值低于五千万元的情形。",
                 prefix=prefix)["report_alert_disclosed"] is True
    with pytest.raises(ValueError, match="BODY_SECTION"):
        operating_disclosure(["本报告期自2024年1月1日起至12月31日止。\n" + prefix +
                              "§5 托管人报告\n无。"], observed_on="2024-12-31", source_hash="a" * 64)


def test_recovery_of_assets_does_not_prove_recovery_of_holder_count():
    result = parse("本报告期内，本基金连续二十个工作日出现基金份额持有人数量不满二百人的情形，"
                   "截至本报告期末，本基金基金资产净值已恢复至五千万元以上。")
    assert result["state"] == "UNRESOLVED_DISCLOSURE"
    assert result["period_end_recovery"]["holders"] is None


def test_conflicting_period_assertions_reject_source():
    with pytest.raises(ValueError, match="CONFLICTING_TRIGGER"):
        parse("本报告期内，本基金未出现连续二十个工作日基金份额持有人数量不满二百人或者基金资产净值低于五千万元的情形。"
              "本报告期内，本基金连续二十个工作日出现基金资产净值低于五千万元的情形。")


def test_source_period_is_economic_end_not_acquisition_date():
    with pytest.raises(ValueError, match="SOURCE_OR_PERIOD"):
        operating_disclosure(report("无。"), observed_on="2025-04-30", source_hash="a" * 64)


def test_unverified_bare_integer_in_body_is_not_silently_removed_as_page_number():
    assert parse("无。\n50")["state"] == "UNRESOLVED_DISCLOSURE"


EXCEPTION = ("本基金为发起式基金，且截至本报告期末，本基金基金合同生效未满3年，"
             "暂不适用《公开募集证券投资基金运作管理办法》第四十一条第一款的规定。")


def test_explicit_reported_exception_preserves_unknown_trigger_and_contract_scope():
    result = parse(EXCEPTION, prefix="2.1 基金基本情况\n基金合同生效日 2023年5月8日\n2.2 基金产品说明\n")
    assert result["state"] == "REPORTED_EXCEPTION_BEFORE_THIRD_ANNIVERSARY"
    assert result["criteria_triggered"] == {"holders": None, "assets": None}
    assert result["report_alert_disclosed"] is None
    assert result["reported_applicability_exception"]["report_effective_date"] == "2023-05-08"


@pytest.mark.parametrize("prefix", ["", "基金合同生效日 2020年5月8日\n",
                                    "基金合同生效日 2023年5月8日\n基金合同生效日 2022年2月1日\n"])
def test_exception_missing_or_conflicting_effective_dates_cannot_be_default_absence(prefix):
    with pytest.raises(ValueError, match="EXCEPTION"):
        parse(EXCEPTION, prefix="2.1 基金基本情况\n" + prefix + "2.2 基金产品说明\n")


def test_link_product_exception_uses_own_effective_date_not_target_etf():
    prefix = ("2.1 基金基本情况\n基金主代码 123456\n基金合同生效日 2023年5月8日\n"
              "2.1.1 目标基金基本情况\n基金合同生效日 2008年9月1日\n2.2 基金产品说明\n")
    result = parse(EXCEPTION, prefix=prefix)
    assert result["reported_applicability_exception"]["report_effective_date"] == "2023-05-08"


def test_verified_title_then_physical_page_pair_restores_cross_page_statement():
    title = "测试基金2024年年度报告\n"
    pages = [title + "1\n本报告期自2024年1月1日起至12月31日止。\n"
             "4.8 基金持有人数或基金资产净值预警情形的说明\n"
             "本报告期内，本基金未出现连续二十个工作日基金份额持有人数量不满二百人或者基金资产净值",
             title + "2\n低于五千万元的情形。\n§5 托管人报告\n其他内容",
             title + "3\n其他内容"]
    result = operating_disclosure(pages, observed_on="2024-12-31", source_hash="a" * 64)
    assert result["state"] == "EXPLICIT_TWO_CRITERIA_NO_TRIGGER_REPORTED"
    assert any(row.get("position_rule") == "VERIFIED_REPORT_TITLE_THEN_PHYSICAL_PAGE_IN_CONSECUTIVE_RUN"
               for row in result["parser_view_trace"])


def test_unverified_single_page_title_number_pair_remains_unknown():
    pages = ["测试基金2024年年度报告\n1\n本报告期自2024年1月1日起至12月31日止。\n"
             "4.8 基金持有人数或基金资产净值预警情形的说明\n无。\n"
             "测试基金2024年年度报告\n2\n§5 托管人报告"]
    result = operating_disclosure(pages, observed_on="2024-12-31", source_hash="a" * 64)
    assert result["state"] == "UNRESOLVED_DISCLOSURE"


def test_unnumbered_trustee_title_closes_exact_original_warning_section():
    pages = report("本基金本报告期无需要说明的相关情形。")
    pages[0] = pages[0].replace("§5 托管人报告", "托管人报告")
    result = operating_disclosure(pages, observed_on="2024-12-31", source_hash="a" * 64)
    assert result["state"] == "SUMMARY_NO_APPLICABLE_ALERT_DISCLOSED"
    assert result["criteria_triggered"] == {"assets": None, "holders": None}


def test_short_title_and_unspecified_exception_do_not_infer_legal_basis_or_duration():
    pages = report("本基金为发起式基金，不适用本项说明。")
    pages[0] = pages[0].replace("报告期内管理人对本基金持有人数或基金资产净值预警情形的说明",
                               "报告期内基金持有人数或基金资产净值预警说明")
    result = operating_disclosure(pages, observed_on="2024-12-31", source_hash="a" * 64)
    assert result["state"] == "REPORTED_EXCEPTION_WITH_UNVERIFIED_RULE_BASIS"
    assert result["report_alert_disclosed"] is None
    assert result["criteria_triggered"] == {"assets": None, "holders": None}
    assert result["reported_applicability_exception"]["report_effective_date"] is None
    assert result["reported_applicability_exception"]["third_anniversary_bound"] is None


def test_unspecified_exception_with_extra_unsupported_claim_remains_unresolved():
    result = parse("本基金为发起式基金，不适用本项说明。所有风险已消除。")
    assert result["state"] == "UNRESOLVED_DISCLOSURE"
    assert result["reported_applicability_exception"] is None


def test_short_toc_title_cannot_replace_original_section():
    pages = ["本报告期自2024年1月1日起至12月31日止。\n"
             "4.9 报告期内基金持有人数或基金资产净值预警说明......12\n托管人报告\n无。"]
    with pytest.raises(ValueError, match="BODY_SECTION"):
        operating_disclosure(pages, observed_on="2024-12-31", source_hash="a" * 64)
