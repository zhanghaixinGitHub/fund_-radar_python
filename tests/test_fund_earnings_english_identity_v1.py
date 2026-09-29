"""译文身份核验的日期、主体、名称和报告期间均不得由相似文本替代。"""

from copy import deepcopy

import pytest
from app.services.fund_earnings_english_identity_v1 import review_english_identity


def annual():
    spec = dict(
        document_id="a",
        stock="000333",
        year=2016,
        mode="ANNUAL_PROFILE",
        chinese_name="美的集团股份有限公司",
        english_name="Midea Group Co., Ltd.",
        cover_name_quote="Midea Group Co., Ltd.",
        title_quote="The 2016 Annual Report",
        profile_page=2,
        code_page=2,
        code_quote="Stock code 000333",
        chinese_heading="Name of the Company in Chinese",
        english_heading="Name of the Company in English (if any)",
    )
    d = dict(
        row=dict(
            announcementId="a",
            secCode="000333",
            orgId="org333",
            published_date="2017-04-25",
            title_plain="2016年年度报告（英文版）",
        ),
        revision_issues=[],
        receipt={"sha256": "a"},
        pages=[
            "Midea Group Co., Ltd.\nThe 2016 Annual Report\n31 March 2017",
            "Stock code 000333\nName of the Company in Chinese 美的集团股份有限公司\n"
            "Name of the Company in English (if any) Midea Group Co., Ltd.",
        ],
    )
    return spec, d


def quarterly():
    s, d = annual()
    s = deepcopy(s)
    d = deepcopy(d)
    s.update(
        document_id="q", year=2017, mode="Q1_EARLIER_PROFILE", title_quote="Interim Report for the First Quarter 2017"
    )
    d["row"].update(announcementId="q", title_plain="2017年第一季度报告全文（英文版）", published_date="2017-05-03")
    d["pages"] = ["Midea Group Co., Ltd.\nInterim Report for the First Quarter 2017\nApril 2017"]
    d["receipt"]["sha256"] = "q"
    return s, d


def test_translation_preserves_later_public_date_and_identity_only():
    s, d = annual()
    p = review_english_identity(s, d)
    assert p["published_date"] == "2017-04-25" and p["available_at"] == "2017-04-26T08:00:00+08:00"
    assert not p["earlier_chinese_release_date_inherited"] and not p["translation_content_equivalence_verified"]
    q, t = quarterly()
    p = review_english_identity(q, t, d, s)
    assert p["passed"] and p["reference_proof"]["passed"] and not p["independent_earnings_event"]
    assert not p["training_ready"] and p["old_stop_entry_preserved"]


@pytest.mark.parametrize(
    "change",
    [
        "stock",
        "suffix_code",
        "wrong_cover_code",
        "legal_cn",
        "legal_en",
        "similar_cover",
        "missing_english_heading",
        "wrong_year",
        "summary",
        "updated",
        "early_public",
        "unknown_public",
        "revision",
        "wrong_document",
    ],
)
def test_intrinsic_identity_rejects_wrong_or_incomplete_proof(change):
    s, d = annual()
    if change == "stock":
        s["stock"] = "000001"
    elif change == "suffix_code":
        d["pages"][1] = d["pages"][1].replace("000333", "0003331")
    elif change == "wrong_cover_code":
        d["pages"][0] += "\nStock Code: 000001"
    elif change == "legal_cn":
        s["chinese_name"] = "另一家股份有限公司"
    elif change == "legal_en":
        s["english_name"] = "Another Co., Ltd."
    elif change == "similar_cover":
        d["pages"][0] = "Longer" + d["pages"][0]
    elif change == "missing_english_heading":
        d["pages"][1] = d["pages"][1].replace("Name of the Company in English (if any)", "Alias")
    elif change == "wrong_year":
        s["year"] = 2015
    elif change == "summary":
        d["row"]["title_plain"] = "2016年年度报告摘要（英文版）"
    elif change == "updated":
        d["row"]["title_plain"] += "（更正后）"
    elif change == "early_public":
        d["row"]["published_date"] = "2016-12-31"
    elif change == "unknown_public":
        d["row"]["published_date"] = ""
    elif change == "revision":
        d["revision_issues"] = ["later overwrite"]
    else:
        s["document_id"] = "b"
    with pytest.raises(ValueError):
        review_english_identity(s, d)


@pytest.mark.parametrize(
    "change",
    ["future", "org", "reference_code", "reference_cn", "reference_en", "self", "no_reference", "q3", "explicit_code"],
)
def test_quarterly_reference_does_not_bypass_intrinsic_identity(change):
    s, d = annual()
    q, t = quarterly()
    if change == "future":
        d["row"]["published_date"] = "2017-05-04"
    elif change == "org":
        d["row"]["orgId"] = "different"
    elif change == "reference_code":
        d["pages"][1] = d["pages"][1].replace("000333", "000001")
    elif change == "reference_cn":
        s["chinese_name"] = "另外一家股份有限公司"
    elif change == "reference_en":
        s["english_name"] = "Another Co., Ltd."
    elif change == "self":
        d["row"]["announcementId"] = "q"
    elif change == "no_reference":
        d = None
    elif change == "q3":
        t["row"]["title_plain"] = t["row"]["title_plain"].replace("第一", "第三")
    else:
        t["pages"][0] += "\nStock Code: 000001"
    with pytest.raises(ValueError):
        review_english_identity(q, t, d, s)
