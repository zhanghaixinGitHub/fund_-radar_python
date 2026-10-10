"""验证综合资料分析的证据边界，不调用文字服务或数据库。"""

from copy import deepcopy

import pytest
from scripts.fund_002112_information_review import validate_items


def example():
    doc = {
        "id": "company-1",
        "title": "股份回购进展",
        "kind": "ANNOUNCEMENT",
        "date": "2026-10-08",
        "url": "https://static.cninfo.com.cn/example.pdf",
        "codes": ["300308.SZ"],
        "sourceHash": "a" * 64,
        "bodyTruncated": False,
        "body": "截至本公告披露日，公司尚未实施本次回购计划。",
    }
    item = {
        "id": doc["id"],
        "assessment": "NEUTRAL",
        "stage": "拟议",
        "facts": [{"quote": doc["body"], "meaning": "有回购计划，但尚未执行。"}],
        "mechanism": "尚无已执行回购形成的股份需求。",
        "caveat": "后续执行情况仍需跟踪。",
    }
    return doc, item


def test_verified_quote_preserves_negative_and_neutral_is_valid():
    doc, item = example()
    result = validate_items({"items": [item]}, [doc])
    assert result[0]["assessment"] == "NEUTRAL"
    assert "尚未实施" in result[0]["facts"][0]["quote"]
    assert result[0]["sourceHash"] == doc["sourceHash"]


@pytest.mark.parametrize(
    "replacement", ["公司已经实施本次回购计划。", "截至本公告披露日，公司尚未实施本次回购计划，后续将上涨。"]
)
def test_invented_or_negation_changed_quote_is_rejected(replacement):
    doc, item = example()
    item["facts"][0]["quote"] = replacement
    with pytest.raises(ValueError, match="QUOTE_INVALID"):
        validate_items({"items": [item]}, [doc])


def test_layout_whitespace_is_allowed_without_rewriting_facts():
    doc, item = example()
    doc["body"] = "截至本公告披露日，公司\n尚未实施\n本次回购计划。"
    assert validate_items({"items": [item]}, [doc])[0]["facts"] == item["facts"]


def test_missing_or_duplicate_document_cannot_claim_full_coverage():
    doc, item = example()
    second = {**doc, "id": "company-2"}
    with pytest.raises(ValueError, match="IDS_INVALID"):
        validate_items({"items": [item]}, [doc, second])
    with pytest.raises(ValueError, match="IDS_INVALID"):
        validate_items({"items": [item, deepcopy(item)]}, [doc, second])
