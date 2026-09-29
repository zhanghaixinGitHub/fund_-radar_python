"""拒绝跨日、错期、更正载体、相似公司名称及自证引用。"""

from copy import deepcopy

import pytest
from app.services.fund_earnings_same_day_identity_v1 import review_same_day_quarterly_identity

NAME = "浙江新和成股份有限公司"


def pair():
    row = {"secCode": "002001", "orgId": "gssz0002001", "published_date": "2019-10-24"}
    a = {
        "row": {**row, "announcementId": "a", "title_plain": "2019年第三季度报告全文"},
        "receipt": {"sha256": "a"},
        "revision_issues": [],
        "pages": [NAME + "\n2019年第三季度报告全文"],
    }
    b = deepcopy(a)
    b["row"].update(announcementId="b", title_plain="2019年第三季度报告正文")
    b["receipt"]["sha256"] = "b"
    b["pages"] = ["证券代码：002001\n" + NAME + "\n2019年第三季度报告正文"]
    return a, b


def test_identity_only_and_same_day_available_time():
    result = review_same_day_quarterly_identity(*pair(), NAME)
    assert result["passed"] and result["available_at"] == "2019-10-25T08:00:00+08:00"
    assert not any(
        result[k]
        for k in ("semantic_verified", "version_history_verified", "independent_disclosure_pair", "training_ready")
    )


@pytest.mark.parametrize(
    "change",
    [
        "day",
        "org",
        "code",
        "period",
        "summary",
        "revision_title",
        "revision_metadata",
        "id",
        "bytes",
        "unknown_date",
        "early_date",
        "wrong_body_code",
        "wrong_target_code",
        "similar_name",
        "cover_period",
    ],
)
def test_unproven_bridge_rejected(change):
    a, b = pair()
    if change == "day":
        b["row"]["published_date"] = "2019-10-25"
    elif change == "org":
        b["row"]["orgId"] = "another"
    elif change == "code":
        b["row"]["secCode"] = "000001"
    elif change in {"period", "summary", "revision_title"}:
        b["row"]["title_plain"] = {
            "period": "2019年第一季度报告正文",
            "summary": "2019年第三季度报告摘要",
            "revision_title": "2019年第三季度报告正文（更正后）",
        }[change]
    elif change == "revision_metadata":
        a["revision_issues"] = ["later updated bytes"]
    elif change == "id":
        b["row"]["announcementId"] = "a"
    elif change == "bytes":
        b["receipt"]["sha256"] = "a"
    elif change == "unknown_date":
        a["row"]["published_date"] = ""
    elif change == "early_date":
        a["row"]["published_date"] = b["row"]["published_date"] = "2019-09-30"
    elif change == "wrong_body_code":
        b["pages"][0] = b["pages"][0].replace("002001", "000001")
    elif change == "wrong_target_code":
        a["pages"][0] += "\n证券代码：000001"
    elif change == "similar_name":
        a["pages"][0] = a["pages"][0].replace(NAME, "另一家" + NAME)
    else:
        a["pages"][0] = a["pages"][0].replace("第三", "第一")
    with pytest.raises(ValueError):
        review_same_day_quarterly_identity(a, b, NAME)
