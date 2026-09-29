"""用合成页验证身份补证边界，不把年份、其他证券或附件误当独立年报。"""

from copy import deepcopy

import pytest
from app.services.fund_earnings_identity_supplement_v2 import review_explicit_identity


def example(mode="ANNUAL_CHINESE_YEAR"):
    spec = dict(
        document_id="d",
        stock="601318",
        year=2019,
        legal_name="甲股份有限公司",
        short_name="甲",
        mode=mode,
        profile_page=2,
        title_page=1,
        period_page=3,
        annual_page=4,
    )
    doc = dict(
        row=dict(announcementId="d", secCode="601318", title_plain="2019年年度报告", published_date="2020-02-21"),
        pages=[
            "二零一九年年报",
            "二零一九年年报甲股份有限公司\n法定名称中文／英文全称甲股份有限公司\n证券简称及代码A股甲601318",
        ],
        receipt={"sha256": "fixed"},
        revision_issues=[],
    )
    if mode == "ANNUAL_BODY_PERIOD":
        doc["pages"] = [
            "",
            "公司的中文名称甲股份有限公司\n股票简称甲股票代码601318",
            "报告期指2019年1月1日—2019年12月31日",
            "公司董事会、监事会及董事、监事、高级管理人员保证年度报告内容的真实、准确、完整",
        ]
    return spec, doc


@pytest.mark.parametrize("mode", ["ANNUAL_CHINESE_YEAR", "ANNUAL_BODY_PERIOD"])
def test_valid_identity_preserves_date_and_excludes_semantic_admission(mode):
    s, d = example(mode)
    result = review_explicit_identity(s, d)
    assert result["passed"] and result["published_date"] == "2020-02-21"
    assert not result["training_ready"] and not result["semantic_verified"]
    assert result["old_stop_entry_preserved"] and not result["permission_to_redownload"]


@pytest.mark.parametrize(
    "field,value",
    [
        ("document_id", "other"),
        ("stock", "600036"),
        ("year", 2020),
        ("legal_name", "乙股份有限公司"),
        ("short_name", "乙"),
        ("mode", "UNKNOWN"),
    ],
)
def test_cannot_relabel_source(field, value):
    s, d = example()
    s[field] = value
    with pytest.raises(ValueError):
        review_explicit_identity(s, d)


@pytest.mark.parametrize(
    "problem", ["revision", "summary", "wrong_code", "code_tail", "duplicate", "wrong_year", "wrong_market"]
)
def test_reject_source_conflicts(problem):
    s, d = example()
    if problem == "revision":
        d["revision_issues"] = ["LATE_REVISION"]
    elif problem == "summary":
        d["row"]["title_plain"] += "摘要"
    elif problem == "wrong_code":
        d["pages"][1] = d["pages"][1].replace("601318", "601319")
    elif problem == "code_tail":
        d["pages"][1] = d["pages"][1].replace("601318", "6013180")
    elif problem == "duplicate":
        d["pages"][0] *= 2
    elif problem == "wrong_year":
        d["pages"][0] = "二零二零年年报"
    else:
        d["pages"][1] = d["pages"][1].replace("A股", "H股")
    with pytest.raises(ValueError):
        review_explicit_identity(s, d)


def audit_example():
    s, d = example()
    s["mode"] = "AUDIT_ATTACHMENT_SAME_DAY_ANNUAL"
    s["reference_legal_page"] = 1
    ref = deepcopy(d)
    ref["row"]["announcementId"] = "ref"
    ref["pages"] = ["证券代码：601318 甲股份有限公司 2019年年度报告"]
    d["row"]["title_plain"] = "2019年度报告审计报告（含经审计的财务报告及附注）"
    d["pages"] = ["甲股份有限公司\n财务报表及审计报告\n2019年12月31日止年度"]
    return s, d, ref


def test_attachment_is_support_only():
    s, d, r = audit_example()
    result = review_explicit_identity(s, d, r)
    assert "NOT_INDEPENDENT_EARNINGS_EVENT" in result["purpose"]


@pytest.mark.parametrize("change", ["date", "stock", "id", "revision", "title", "name", "missing"])
def test_attachment_needs_same_day_independent_annual(change):
    s, d, r = audit_example()
    if change == "date":
        r["row"]["published_date"] = "2020-02-22"
    elif change == "stock":
        r["row"]["secCode"] = "600036"
    elif change == "id":
        r["row"]["announcementId"] = "d"
    elif change == "revision":
        r["revision_issues"] = ["LATE"]
    elif change == "title":
        r["row"]["title_plain"] = "2018年年度报告"
    elif change == "name":
        r["pages"][0] = r["pages"][0].replace("甲股份有限公司", "乙股份有限公司")
    else:
        r = None
    with pytest.raises(ValueError):
        review_explicit_identity(s, d, r)


@pytest.mark.parametrize(
    "variant,catalog,body",
    [
        ("FORECAST_YEAR_WORD", "甲股份有限公司关于2019年业绩预告的公告", "甲股份有限公司关于2019年度业绩预告的公告"),
        ("INCREASE_YEAR_WORD", "甲2019年年度业绩预增公告", "甲股份有限公司2019年度业绩预增公告"),
        ("SUMMARY_PARENTHESES", "甲股份有限公司2019年年度报告(摘要)", "甲股份有限公司2019年年度报告摘要"),
    ],
)
@pytest.mark.parametrize("change", [None, "year", "kind", "code", "legal"])
def test_title_variants_keep_year_kind_and_issuer(variant, catalog, body, change):
    s, d = example()
    s.update(mode="EXPLICIT_CATALOG_TITLE_VARIANT", variant=variant)
    d["row"]["title_plain"] = catalog
    d["pages"] = ["证券代码：601318 " + body]
    if change == "year":
        d["pages"][0] = d["pages"][0].replace("2019", "2020")
    elif change == "kind":
        d["row"]["title_plain"] += "摘要" if variant != "SUMMARY_PARENTHESES" else "全文"
    elif change == "code":
        d["pages"][0] = d["pages"][0].replace("601318", "600036")
    elif change == "legal":
        d["pages"][0] = d["pages"][0].replace("甲股份有限公司", "乙股份有限公司")
    if change:
        with pytest.raises(ValueError):
            review_explicit_identity(s, d)
    else:
        assert review_explicit_identity(s, d)["passed"]
