"""身份补证不能跳过主体、报告类型、公开先后和原件版本检查。"""

import pytest
from app.services.fund_earnings_identity_v1 import profile_proof, supplement_identity, title_proof


def document(code="002709", published="2021-03-25"):
    return {
        "row": {"announcementId": "a", "secCode": code, "title_plain": "2020年年度报告", "published_date": published},
        "pages": [
            "广州天赐高新材料股份有限公司2020年年度报告",
            "目录",
            "股票代码002709\n公司的中文名称广州天赐高新材料股份有限公司",
        ],
        "receipt": {"sha256": "test"},
        "revision_issues": [],
    }


def target(published="2021-04-20"):
    d = document(published=published)
    d["row"]["title_plain"] = "2021年第一季度报告全文"
    d["pages"] = ["广州天赐高新材料股份有限公司2021年第一季度报告"]
    return d


def test_profile_and_quarterly_cover_chain():
    d = document()
    proof = profile_proof(d)
    assert proof["passed"]
    result = supplement_identity(target(), [{"document": d, "proof": proof}])
    assert result["passed"] and not result["version_history_verified"]


def test_future_reference_cannot_identify_earlier_document():
    d = document(published="2021-05-01")
    assert not supplement_identity(target(), [{"document": d, "proof": profile_proof(d)}])["passed"]


@pytest.mark.parametrize("change", ["code", "name", "date"])
def test_intrinsic_profile_requires_matching_code_name_and_dates(change):
    d = document()
    if change == "code":
        d["row"]["secCode"] = "000001"
    elif change == "name":
        d["pages"][0] = "另外一家股份有限公司2020年年度报告"
    else:
        d["revision_issues"] = ["later revision"]
    assert profile_proof(d) is None


def test_explicit_different_issuer_blocks_even_cover_name_matches():
    d = target()
    d["pages"][0] += "\n证券代码：000001\n"
    ref = document()
    assert not supplement_identity(d, [{"document": ref, "proof": profile_proof(ref)}])["passed"]


def test_summary_and_wrong_period_are_not_equivalent():
    d = target()
    assert not title_proof({**d["row"], "title_plain": "2021年第一季度报告摘要"}, d["pages"])["passed"]
    assert not title_proof({**d["row"], "title_plain": "2021年第二季度报告全文"}, d["pages"])["passed"]


def test_correction_attachment_identity_does_not_backdate_or_prove_history():
    d = target(published="2021-05-24")
    d["row"]["title_plain"] += "（更正前）"
    ref = document()
    result = supplement_identity(d, [{"document": ref, "proof": profile_proof(ref)}])
    assert result["passed"] and result["published_date"] == "2021-05-24"
    assert not result["version_history_verified"] and "更正前" in result["original_title"]
