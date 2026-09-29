"""引用资料选择边界：重名、修订、错日和错主体不静默择一。"""

import pytest
from app.services.fund_earnings_targeted_v1 import select_referenced_quarter


def row(title="2018年第三季度报告全文", **changes):
    return {"title_plain": title, "secCode": "002027", "published_date": "2018-10-30", **changes}


def choose(rows):
    return select_referenced_quarter(rows, "002027", 2018, "2018-10-30")


def test_keeps_full_and_short_as_separate_documents():
    assert len(choose([row(), row("2018年第三季度报告正文")])) == 2


@pytest.mark.parametrize("modifier", ["更正", "修订", "补充", "更新", "取消"])
def test_revision_cannot_be_silently_dropped(modifier):
    with pytest.raises(ValueError, match="REVISION"):
        choose([row(), row(f"2018年第三季度报告{modifier}公告")])


def test_duplicate_original_cannot_be_selected_by_id():
    with pytest.raises(ValueError, match="NOT_UNIQUE"):
        choose([row(announcementId="1"), row(announcementId="2")])


def test_reference_date_cannot_backdate_catalog():
    with pytest.raises(ValueError, match="DATE_CONFLICT"):
        choose([row(published_date="2018-10-31")])


def test_other_issuer_cannot_supply_missing_report():
    with pytest.raises(ValueError, match="ISSUER"):
        choose([row(secCode="002120")])


def test_no_substitution_from_other_year_or_quarter():
    with pytest.raises(ValueError, match="NOT_FOUND"):
        choose([row("2019年第三季度报告全文"), row("2018年第一季度报告全文")])
