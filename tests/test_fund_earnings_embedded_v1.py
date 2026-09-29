"""覆盖相同栏目有预告、无预告、跨页和重复正文的真实边界。"""

import pytest
from app.services.fund_earnings_embedded_v1 import embedded_forecasts

ROW = {
    "secCode": "002709",
    "announcementId": "a",
    "published_date": "2021-04-20",
    "title_plain": "2021年第一季度报告全文",
}
TITLE = "六、对2021年1-6月经营业绩的预计"


def test_embedded_period_is_half_year_even_when_carrier_is_q1():
    value = embedded_forecasts(ROW, [TITLE + "√适用 □不适用 利润 65,000--75,000 万元"])[0]
    assert value["period_end"] == "2021-06-30"
    assert value["status"] == "APPLICABLE_REQUIRES_FACT_REVIEW"
    assert value["money"] is None and not value["training_ready"]


def test_not_applicable_does_not_mean_zero_or_no_other_disclosure():
    value = embedded_forecasts(ROW, [TITLE + "□适用 √不适用"])[0]
    assert value["status"] == "EXPLICIT_NOT_APPLICABLE_IN_THIS_SECTION"
    assert value["money"] is None and not value["absence_of_other_forecasts_proven"]


def test_next_section_checkbox_cannot_admit_forecast():
    value = embedded_forecasts(ROW, [TITLE + "七、日常经营重大合同√适用□不适用"])[0]
    assert value["status"] == "UNRESOLVED_CHECKBOX_OR_PAGE_CONTINUATION"


def test_continued_page_requires_review_instead_of_assuming_absence():
    value = embedded_forecasts(ROW, [TITLE, "√适用□不适用 净利润区间"])[0]
    assert value["status"] == "UNRESOLVED_CHECKBOX_OR_PAGE_CONTINUATION"


@pytest.mark.parametrize("check", ["适用 不适用", "√适用□不适用 □适用√不适用"])
def test_missing_or_conflicting_checks_are_unresolved(check):
    assert embedded_forecasts(ROW, [TITLE + check])[0]["status"].startswith("UNRESOLVED")


def test_duplicate_body_and_full_text_keep_ids_but_share_content_key():
    a = embedded_forecasts(ROW, [TITLE + "√适用□不适用"])[0]
    b = embedded_forecasts(
        {**ROW, "announcementId": "b", "title_plain": "2021年第一季度报告正文"}, [TITLE + "√适用□不适用"]
    )[0]
    c = embedded_forecasts({**ROW, "published_date": "2021-04-21"}, [TITLE + "√适用□不适用"])[0]
    assert a["document_id"] != b["document_id"] and a["content_group_key"] == b["content_group_key"]
    assert a["content_group_key"] != c["content_group_key"]


def test_invalid_month_range_is_rejected():
    with pytest.raises(ValueError, match="PERIOD_INVALID"):
        embedded_forecasts(ROW, ["对2021年7-6月经营业绩的预计√适用□不适用"])
