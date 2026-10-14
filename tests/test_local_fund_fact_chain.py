"""验证年度绑定、完整输入、事实不可改写和解释不能自动发布的工程边界。"""

import json
from copy import deepcopy

import pytest
from scripts.local_fund_fact_chain import build_book, check_result, request_for


def document(body=None):
    return {
        "id": "test",
        "title": "公告",
        "body_scope": "FULL_TEXT",
        "body_truncated": False,
        "body": body
        or (
            "营业收入（A）触发值(An) 目标值(Am)\n"
            "第一个归属期2025年9亿元12亿元\n第二个归属期2026年12亿元16亿元\n"
            "各归属期对应公司层面归属比例为X\n若Am＞A≥An X=A/Am\n"
            "公司2025年营业收入为12.58亿元。"
        ),
    }


def correct_raw(book, explanation="原文条件仍需逐项核对。"):
    return {
        "error": None,
        "response": {
            "message": {
                "content": json.dumps(
                    {
                        "facts": [
                            {"id": f["id"], "data_json": json.dumps(f["data"], ensure_ascii=False)}
                            for f in book["facts"]
                            if f["required"]
                        ],
                        "explanation": explanation,
                        "limitations": [],
                    },
                    ensure_ascii=False,
                )
            }
        },
    }


def test_actual_year_binds_matching_threshold_not_next_year():
    book = build_book(document())
    relation = next(f for f in book["facts"] if f["kind"] == "actual_threshold_relation")
    assert relation["data"] == {
        "year": "2025",
        "actual": "12.58",
        "unit": "亿元",
        "trigger": "9",
        "target": "12",
        "actual_reaches_target": True,
    }
    assert relation["source_refs"][0]["table_rows"]


def test_ambiguous_same_year_pairs_are_not_selected_as_expected_answer():
    doc = document(document()["body"].replace("第二个归属期2026年", "第二个归属期2025年"))
    assert not any(f["kind"] == "actual_threshold_relation" for f in build_book(doc)["facts"])


def test_model_cannot_overwrite_source_fact_and_raw_is_not_repaired():
    doc = document()
    book = build_book(doc)
    raw = correct_raw(book)
    parsed = json.loads(raw["response"]["message"]["content"])
    target = next(f for f in parsed["facts"] if json.loads(f["data_json"]).get("actual") == "12.58")
    bad = json.loads(target["data_json"])
    bad["target"] = "16"
    target["data_json"] = json.dumps(bad)
    raw["response"]["message"]["content"] = json.dumps(parsed)
    before = deepcopy(raw)
    result = check_result(raw, doc, book)
    assert result["status"] == "BLOCKED"
    assert any(i.startswith("PROGRAM_FACT_MODIFIED") for i in result["issues"])
    assert raw == before
    assert next(f for f in result["safe_facts"] if f["kind"] == "actual_threshold_relation")["data"]["target"] == "12"


def test_missing_required_fact_and_generation_failure_cannot_pass():
    doc = document()
    book = build_book(doc)
    raw = correct_raw(book)
    raw["response"]["message"]["content"] = json.dumps({"facts": [], "explanation": "已核验", "limitations": []})
    raw["error"] = ["LOCAL_GENERATION_INCOMPLETE"]
    result = check_result(raw, doc, book)
    assert any(i.startswith("REQUIRED_FACTS_MISSING") for i in result["issues"])
    assert "LOCAL_GENERATION_INCOMPLETE" in result["issues"]


def test_explanation_check_rejects_cross_year_value_even_when_facts_copied():
    doc = document()
    book = build_book(doc)
    result = check_result(correct_raw(book, "公司2025年营业收入12.58亿元，对应目标值16亿元。"), doc, book)
    assert "ACTUAL_YEAR_THRESHOLD_CONFLICT" in result["issues"]


def test_good_copy_is_not_model_semantics_or_publish_permission():
    doc = document()
    result = check_result(correct_raw(build_book(doc)), doc, build_book(doc))
    assert result["status"] == "FACT_COPY_CHECKED_EXPLANATION_REVIEW_REQUIRED"
    assert result["model_explanation_adopted"] is False
    assert result["model_semantics_verified"] is False


def test_boolean_relation_cannot_be_replaced_by_integer_one():
    doc = document()
    book = build_book(doc)
    raw = correct_raw(book)
    parsed = json.loads(raw["response"]["message"]["content"])
    for fact in parsed["facts"]:
        data = json.loads(fact["data_json"])
        if "actual_reaches_target" in data:
            data["actual_reaches_target"] = 1
            fact["data_json"] = json.dumps(data)
    raw["response"]["message"]["content"] = json.dumps(parsed)
    assert any(i.startswith("PROGRAM_FACT_MODIFIED") for i in check_result(raw, doc, book)["issues"])


def test_partial_source_is_preserved_and_cannot_be_adopted():
    doc = document()
    doc["body_scope"] = "VERIFIED_EXCERPT"
    result = check_result(correct_raw(build_book(doc)), doc, build_book(doc))
    assert "PARTIAL_SOURCE" in result["issues"]
    assert result["source_partial"] is True


@pytest.mark.parametrize("value", [[], {"facts": "bad", "explanation": "bad", "limitations": []}])
def test_malformed_model_shape_is_blocked_without_crashing(value):
    doc = document()
    raw = {"response": {"message": {"content": json.dumps(value)}}}
    assert "MODEL_CONTRACT_INVALID" in check_result(raw, doc, build_book(doc))["issues"]


def test_complete_source_retained_and_budget_refusal_does_not_crop():
    class Budget:
        def count(self, text):
            return len(text)

    doc = document()
    request = request_for(doc, build_book(doc), None, Budget())
    assert doc["body"] in request["messages"][1]["content"]

    class Huge:
        def count(self, text):
            return 65536

    before = deepcopy(doc)
    with pytest.raises(ValueError, match="OVER_BUDGET"):
        request_for(doc, build_book(doc), None, Huge())
    assert doc == before
