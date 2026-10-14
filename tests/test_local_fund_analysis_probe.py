"""离线对照的资料覆盖、引用范围和上下文边界。"""

import json

import pytest
from app.schemas.fund_information_analysis import quote_catalog
from scripts.local_fund_analysis_probe import decode_local, document_requests, model_facts, request_payload


def doc(body):
    return {
        "id": "doc1",
        "title": "公告",
        "body": body,
        "date": "2026-01-01",
        "codes": ["000001.SZ"],
        "kind": "ANNOUNCEMENT",
        "body_scope": "FULL_TEXT",
        "body_truncated": False,
        "gold": "不可读的评审答案",
    }


@pytest.mark.parametrize(
    "body", ["公司拟投资人民币九千万元，尚需股东会审批。", "说明事项和保留完整条件。" * 2500], ids=["short", "long"]
)
def test_chunks_cover_every_original_character_and_use_original_ids(body):
    source = doc(body)
    chunks = document_requests(source)
    catalog = quote_catalog(body)
    assert [ref for chunk in chunks for ref in chunk["quote_ids"]] == list(catalog)
    assert "".join(body[chunk["start"] : chunk["end"]] for chunk in chunks) == body
    for chunk in chunks:
        request = chunk["request"]
        assert request["options"]["num_ctx"] <= 65536
        assert "不可读的评审答案" not in str(request)
        value = json.loads(request["messages"][1]["content"])
        assert value["quote_catalogs"]["doc1"] == {ref: catalog[ref] for ref in chunk["quote_ids"]}
        assert value["documents"][0]["body_truncated"] == chunk["split"]
    assert (len(chunks) > 1) == (len(body) > 20000)


def test_fact_projection_preserves_full_claim_and_limits():
    facts = {
        "F01": {
            "text": "不超过人民币3亿元。" * 1000,
            "category": "公告",
            "body_truncated": True,
            "direction_eligible": False,
            "required_in_analysis": True,
            "quote_receipts": ["审计索引"],
        }
    }
    projected = model_facts(facts)
    assert projected["F01"]["text"] == facts["F01"]["text"]
    assert projected["F01"]["body_truncated"] is True
    assert projected["F01"]["direction_eligible"] is False
    assert projected["F01"]["required_in_analysis"] is True


def test_oversize_input_and_truncated_response_are_not_success():
    with pytest.raises(ValueError, match="TOO_LARGE"):
        request_payload("规则", {"body": "原文" * 40000}, {"type": "object"})
    request = request_payload("规则", {"body": "原文"}, {"type": "object"})
    with pytest.raises(ValueError, match="INCOMPLETE"):
        decode_local({"done_reason": "length", "message": {"content": "{}"}}, request)
    with pytest.raises(ValueError, match="CONTEXT_EXHAUSTED"):
        decode_local({"done_reason": "stop", "prompt_eval_count": 16384, "message": {"content": "{}"}}, request)
