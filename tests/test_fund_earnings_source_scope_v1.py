"""用途准入的反例：监管或薪酬资料不能偷换为利润事实，普通更正也不能误排除。"""

from copy import deepcopy

import pytest
from app.services.fund_earnings_source_scope_v1 import review_non_profit_sources


def sample(role="REGULATORY_SOLVENCY_REPORT"):
    title, quote = (
        ("中国人寿偿付能力季度报告摘要（2023年第三季度）", "保险公司偿付能力报告摘要")
        if role == "REGULATORY_SOLVENCY_REPORT"
        else ("2022年年度报告补充公告", "2022年度最终全部薪酬情况披露如下")
    )
    return (
        [{"document_id": "a", "role": role, "anchors": [[1, quote]]}],
        {"a": {"body_saved": True, "revision_issues": [], "pages": [quote],
               "row": {"title_plain": title, "published_date": "2023-12-16"},
               "receipt": {"sha256": "fixed"}}},
    )


@pytest.mark.parametrize("role", ["REGULATORY_SOLVENCY_REPORT", "REMUNERATION_SUPPLEMENT"])
def test_review_preserves_source_and_public_date(role):
    specs, docs = sample(role)
    before = deepcopy(docs)
    result = review_non_profit_sources(specs, docs, {"a"}, [])
    assert result[0]["excluded_from_profit_changes"]
    assert result[0]["published_date"] == "2023-12-16"
    assert docs == before


def test_excluded_source_cannot_supply_money_even_if_identity_passes():
    specs, docs = sample()
    with pytest.raises(ValueError, match="NON_PROFIT_SOURCE_USED"):
        review_non_profit_sources(specs, docs, {"a"}, [{"document_id": "a"}])


@pytest.mark.parametrize("verified", [set(), {"different"}])
def test_unverified_identity_rejected(verified):
    specs, docs = sample()
    with pytest.raises(ValueError, match="IDENTITY_NOT_VERIFIED"):
        review_non_profit_sources(specs, docs, verified, [])


@pytest.mark.parametrize("field,value", [("body_saved", False), ("revision_issues", ["late_revision"])])
def test_failed_or_revised_source_cannot_be_admitted(field, value):
    specs, docs = sample()
    docs["a"][field] = value
    with pytest.raises(ValueError, match="BODY_OR_DATE_INVALID"):
        review_non_profit_sources(specs, docs, {"a"}, [])


def test_ordinary_earnings_correction_cannot_be_excluded_as_salary():
    specs, docs = sample("REMUNERATION_SUPPLEMENT")
    docs["a"]["row"]["title_plain"] = "2022年年度报告更正公告"
    with pytest.raises(ValueError, match="TITLE_OR_ANCHORS_INVALID"):
        review_non_profit_sources(specs, docs, {"a"}, [])


def test_title_alone_cannot_prove_salary_scope():
    specs, docs = sample("REMUNERATION_SUPPLEMENT")
    docs["a"]["pages"] = ["2022年度净利润更正如下"]
    specs[0]["anchors"] = [[1, "2022年度净利润更正如下"]]
    with pytest.raises(ValueError, match="BODY_ROLE_NOT_ESTABLISHED"):
        review_non_profit_sources(specs, docs, {"a"}, [])


def test_missing_anchor_rejected():
    specs, docs = sample()
    docs["a"]["pages"] = ["另一份资料"]
    with pytest.raises(ValueError):
        review_non_profit_sources(specs, docs, {"a"}, [])


def test_duplicate_scope_rejected():
    specs, docs = sample()
    with pytest.raises(ValueError, match="DUPLICATE_SOURCE_SCOPE"):
        review_non_profit_sources(specs + specs, docs, {"a"}, [])


def test_unknown_role_rejected():
    specs, docs = sample()
    specs[0]["role"] = "ANY_SUPPLEMENT"
    with pytest.raises(ValueError, match="UNKNOWN_NON_PROFIT_SOURCE_ROLE"):
        review_non_profit_sources(specs, docs, {"a"}, [])
