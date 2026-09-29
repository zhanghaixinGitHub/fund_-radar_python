"""年报参考原件必须同主体、同年度、明确摘要且唯一。"""

import pytest
from app.services.fund_earnings_reference_v2 import select_annual_summary


def row(title="2022年年度报告摘要", stock="688083", identity="one"):
    return {"title_plain": title, "secCode": stock, "announcementId": identity}


@pytest.mark.parametrize("title", ["2022年年度报告摘要", "2022年度报告摘要", "中望软件2022 年年度报告摘要"])
def test_explicit_same_year_summary(title):
    expected = row(title)
    assert select_annual_summary([expected], "688083", 2022) == expected


@pytest.mark.parametrize(
    "title",
    [
        "2021年年度报告摘要",
        "2022年年度报告",
        "2022年年度报告摘要（更新后）",
        "关于2022年年度报告摘要的更正公告",
        "2022年年度报告摘要取消公告",
        "2022年半年度报告摘要",
    ],
)
def test_other_year_carrier_or_revision_rejected(title):
    with pytest.raises(ValueError, match="NOT_UNIQUE"):
        select_annual_summary([row(title)], "688083", 2022)


def test_wrong_company_and_ambiguous_duplicates_rejected():
    for rows in ([row(stock="688023")], [row(), row(identity="two")], []):
        with pytest.raises(ValueError, match="NOT_UNIQUE"):
            select_annual_summary(rows, "688083", 2022)
