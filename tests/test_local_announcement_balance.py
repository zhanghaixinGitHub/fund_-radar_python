"""验证余额修复保留跨页原句，并拒绝缺限定词、分项冒充总额和范围不明的结果。"""

import json
from copy import deepcopy

import pytest
from scripts.local_announcement_balance import extract_balance
from scripts.local_announcement_document import text_document
from scripts.local_announcement_tasks import read_tasks

QUESTION = ["截至公告实际担保余额是多少？"]


def packet(text):
    return text_document(text, scope="FULL_TEXT")


def response(value):
    return {"done_reason": "stop", "message": {"content": json.dumps(value, ensure_ascii=False)}}


def test_cross_page_sentence_and_original_offsets():
    source = packet("占位。")
    source["pages"] = [
        {
            "page": 1,
            "blocks": [
                {
                    "id": "P1B",
                    "kind": "text",
                    "page": 1,
                    "text": "公司为甲担保8亿元。\n截至目前公司存续的对外担保余额不超过人民币18.72",
                }
            ],
            "tables": [],
        },
        {
            "page": 2,
            "blocks": [{"id": "P2B", "kind": "text", "page": 2, "text": "亿元，占净资产比例21%。\n其他事项。"}],
            "tables": [],
        },
    ]
    original = deepcopy(source)
    result = extract_balance("公告", source, QUESTION)
    selected = result["selection"]
    assert not selected["issues"]
    assert selected["candidates"][0]["text"] == "截至目前公司存续的对外担保余额不超过人民币18.72亿元，占净资产比例21%。"
    spans = selected["candidates"][0]["spans"]
    assert [s["id"] for s in spans] == ["P1B", "P2B"]
    assert source == original
    answer = result["answers"][0]
    assert answer["safe_answer"] == "不超过人民币18.72亿元"
    assert answer["source_slots"] == {"qualifier": "不超过", "currency": "人民币", "amount": "18.72", "unit": "亿元"}
    assert answer["answer_origin"] == "source_rule"
    assert not answer["model_called"]
    assert result["calls"] == 0 and result["attempts"] == []
    for span in spans:
        block = next(b for p in source["pages"] for b in p["blocks"] if b["id"] == span["id"])
        compact_text = "".join(block["text"].split())
        assert compact_text[span["compact_start"] : span["compact_end"]] == span["text"]


@pytest.mark.parametrize(
    "text, expected",
    [
        ("截至目前公司对外担保余额为人民币18.72亿元。", "人民币18.72亿元"),
        ("截至本公告日公司实际担保总余额约人民币18.72亿元。", "约人民币18.72亿元"),
        ("截至目前公司对外担保余额不超过2,350万美元。", "不超过2,350万美元"),
        ("截至目前公司对外担保余额为0元。", "0元"),
    ],
)
def test_distinct_bounds_currencies_units_and_zero(text, expected):
    result = extract_balance("公告", packet(text), QUESTION)
    assert not result["issues"]
    assert result["answers"][0]["safe_answer"] == expected


@pytest.mark.parametrize(
    "text",
    [
        "截至目前公司担保总额度不超过99亿元。",
        "截至目前公司为甲公司担保余额为2亿元。",
        "截至目前子公司担保余额为2亿元。",
        "截至2025年12月31日公司对外担保余额为7亿元。",
        "截至目前公司对外担保余额为2亿元。截至目前公司对外担保余额为3亿元。",
        "截至目前公司对外担保余额为2亿元",
        "截至目前公司预计对外担保余额为2亿元。",
        "截至目前公司对外担保余额不是2亿元。",
        "截至目前公司对外担保余额为2亿元至3亿元。",
        "截至目前公司对外担保余额为2亿元以下。",
        "截至目前公司对外担保余额为2亿元—3亿元。",
        "截至目前公司对外担保余额为2亿元/年。",
        "截至目前公司对外担保余额为2亿元，不包括子公司。",
    ],
)
def test_ambiguous_or_unsupported_source_never_calls_model(text):
    result = extract_balance("公告", packet(text), QUESTION)
    assert result["calls"] == 0
    assert result["answers"][0]["safe_answer"] is None
    assert result["selection"]["issues"]


def test_no_gold_used_and_invalid_packet_rejected():
    source = packet("截至目前公司对外担保余额不超过人民币18.72亿元。")
    source["gold"] = "完全不同的答案，不能被使用"
    result = extract_balance("公告", source, QUESTION)
    assert result["answers"][0]["safe_answer"] == "不超过人民币18.72亿元"
    source["issues"] = ["TEXT_LAYER_MISSING"]
    with pytest.raises(ValueError, match="提取"):
        extract_balance("公告", source, QUESTION)
    with pytest.raises(ValueError, match="单题"):
        extract_balance("公告", packet("原文。"), ["某子公司的余额是多少？"])


def test_tasks_opt_in_changes_only_balance_subquestion():
    source = packet("担保总额度调整至30亿元。截至目前公司对外担保余额不超过人民币18.72亿元。")
    questions = ["担保总额度调整后是多少，截至公告实际担保余额是多少，两者是否相同？"]
    baseline_requests, candidate_requests = [], []

    def capture(target):
        def transport(payload):
            target.append(payload)
            return response(
                {
                    "answers": [
                        {"question_number": 1, "answer": "无法确定", "status": "insufficient", "evidence_ids": ["E001"]}
                    ]
                }
            )

        return transport

    read_tasks("公告", source, questions, focus_details=True, transport=capture(baseline_requests))
    result = read_tasks(
        "公告", source, questions, focus_details=True, focus_balance=True, transport=capture(candidate_requests)
    )
    assert baseline_requests[0] == candidate_requests[0]
    assert baseline_requests[2] == candidate_requests[1]
    assert len(baseline_requests) == 3 and len(candidate_requests) == 2
    assert result["answers"][0]["parts"][1]["safe_answer"] == "不超过人民币18.72亿元"
    assert result["calls"] == 2
