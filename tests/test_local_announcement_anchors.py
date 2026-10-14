"""用不同数字、不同主体反例验证引文锚定，防止靠放宽检查通过旧题。"""

import json
from copy import deepcopy

import pytest
from scripts.local_announcement_anchors import resolve_quotes
from scripts.local_announcement_checks import final_checks
from scripts.local_announcement_document import text_document
from scripts.local_announcement_reader import inspect_answer
from scripts.local_announcement_structured_reader import read_document, structured_request
from scripts.local_announcement_tasks import plan_question, plan_tasks


def raw(snippets, answer="余额不超过24.6亿元。"):
    return {"answers": [{"question_number": 1, "evidence_quotes": snippets, "answer": answer, "status": "answered"}]}


def units(*texts):
    return [{"id": f"S{n}", "text": t, "kind": "text"} for n, t in enumerate(texts, 1)]


def test_exact_quote_locates_source_and_never_changes_model_response():
    packet = units("授信额度79亿元。", "本次实际担保余额不超过24.6亿元。")
    result = raw(["实际担保余额不超过24.6亿元。"])
    before = deepcopy(result)
    normalized, audits = resolve_quotes(result, packet)
    assert normalized["answers"][0]["evidence_ids"] == ["S2"]
    assert not audits[1]["anchor_issues"]
    assert result == before


@pytest.mark.parametrize("quote", ["实际担保余额不超过24.7亿元。", "实际担保余额为24.6亿元。"])
def test_quote_cannot_change_a_number_or_remove_bound_and_match_fuzzily(quote):
    result, audits = resolve_quotes(raw([quote]), units("实际担保余额不超过24.6亿元。"))
    assert result["answers"][0]["evidence_ids"] == []
    assert audits[1]["anchor_issues"] == ["QUOTE_NOT_FOUND"]


def test_ambiguous_duplicate_citation_and_unrelated_bare_number_are_refused():
    sentence = "本次担保余额不超过24.6亿元。"
    _, audits = resolve_quotes(raw([sentence]), units(sentence, "无关内容。", sentence))
    assert audits[1]["anchor_issues"] == ["QUOTE_LOCATION_AMBIGUOUS"]
    _, audits = resolve_quotes(raw(["24.6"]), units(sentence))
    assert audits[1]["anchor_issues"] == ["QUOTE_LENGTH_INVALID"]


def test_adjacent_cross_page_quote_keeps_both_original_units_and_skips_layout():
    source = units("实际担保余额不超过24.6", "亿元，包含对子公司的担保。")
    source.insert(1, {"id": "FOOTER", "text": "7", "kind": "page_layout"})
    normalized, audits = resolve_quotes(raw(["实际担保余额不超过24.6亿元"]), source)
    assert normalized["answers"][0]["evidence_ids"] == ["S1", "S2"]
    assert not audits[1]["anchor_issues"]


def test_quote_mode_is_one_call_checks_all_anchors_and_retains_uncertainty():
    packet = text_document("本次实际担保余额不超过24.6亿元。", scope="FULL_TEXT")
    packet["gold"] = "绝不发送的答案"
    calls = []

    def transport(payload):
        calls.append(payload)
        return {"done_reason": "stop", "message": {"content": json.dumps(raw(["本次实际担保余额不超过24.6亿元。"]))}}

    result = read_document("公告", packet, ["实际担保余额多少？"], citation_mode="quotes", transport=transport)
    assert len(calls) == 1
    assert "绝不发送" not in json.dumps(calls, ensure_ascii=False)
    answer = result["answers"][0]
    assert answer["resolved_evidence_ids"] == ["SB1"]
    assert answer["safe_answer"] == "余额不超过24.6亿元。"
    assert not answer["semantic_verified"]
    fields = calls[0]["format"]["properties"]["answers"]["items"]["properties"]
    assert list(fields) == ["question_number", "evidence_quotes", "answer", "status"]
    with pytest.raises(ValueError):
        structured_request("公告", packet, ["金额？"], citation_mode="invalid")


def test_a_valid_quote_never_rescues_another_invented_quote():
    packet = text_document("本次实际担保余额不超过24.6亿元。", scope="FULL_TEXT")
    content = json.dumps(raw(["本次实际担保余额不超过24.6亿元。", "董事会批准上限为九十九亿元。"]), ensure_ascii=False)
    result = read_document(
        "公告",
        packet,
        ["金额？"],
        citation_mode="quotes",
        transport=lambda _: {"done_reason": "stop", "message": {"content": content}},
    )
    assert "QUOTE_NOT_FOUND" in result["answers"][0]["issues"]
    assert result["answers"][0]["safe_answer"] is None


def test_boolean_yes_is_complete_only_for_a_single_polar_question():
    assert "ANSWER_UNFINISHED" not in inspect_answer("议案甲是否通过？", "是", ["议案甲通过。"])
    assert "ANSWER_UNFINISHED" in inspect_answer("金额是多少？", "是", ["金额80元。"])
    assert "ANSWER_UNFINISHED" in inspect_answer("议案是否通过，回避多少股？", "是", ["通过，回避0股。"])
    assert "ANSWER_UNFINISHED" in inspect_answer("议案是否通过？", "结果是", ["通过。"])


def test_candidate_progress_question_does_not_require_repeating_independence_conditions():
    source = "同意提名独立董事候选人，并将该议案提交董事会审议。"
    answer = "该审核意见只是同意提名，不证明已经选举。"
    assert "CANDIDATE_SCOPE_MISSING" not in inspect_answer("候选人是否已当选？", answer, [source])
    assert "CANDIDATE_SCOPE_MISSING" in inspect_answer("候选人任职资格的审核条件？", answer, [source])


def test_currency_scale_normalization_does_not_allow_wrong_unit_value_or_other_number_roles():
    def check(answer):
        return final_checks(
            "金额？", {"answer": answer, "quotes": ["本次增资7.2亿元。"], "issues": ["NUMBER_NOT_IN_EVIDENCE:72000"]}
        )

    assert not check("本次增资72000万元。")["issues"]
    for answer in ("本次增资72000元。", "本次增资72000万股。", "本次增资72000万美元。", "72000万元，期限72000天。"):
        assert check(answer)["issues"]


def test_cause_comma_omission_is_allowed_but_negation_is_preserved():
    source = "因工作内容调整，不再认定为核心人员。"
    good = {"answer": "因工作内容调整不再认定为核心人员。", "quotes": [source]}
    assert final_checks("为何调整？", good)["safe_answer"]
    bad = {"answer": "因工作内容调整仍认定为核心人员。", "quotes": [source]}
    assert final_checks("为何调整？", bad)["safe_answer"] is None


def test_display_prefix_and_quote_glyphs_require_literal_business_content_and_matching_id():
    source = units("本次“稳健型”产品认购208万元。", "本次“稳健型”产品认购209万元。")
    for n, u in enumerate(source, 1):
        u["citation_id"] = f"E{n:03d}"
    normalized, audit = resolve_quotes(raw(['[E001] 本次"稳健型"产品认购208万元。']), source)
    assert normalized["answers"][0]["evidence_ids"] == ["S1"]
    assert not audit[1]["anchor_issues"]
    for quote in ('[E002] 本次"稳健型"产品认购208万元。', '[E999] 本次"稳健型"产品认购208万元。'):
        normalized, audit = resolve_quotes(raw([quote]), source)
        assert not normalized["answers"][0]["evidence_ids"]
        assert audit[1]["anchor_issues"]


def test_focused_questions_preserve_subject_and_never_include_evaluation_values():
    q = "认购价格如何调整，触发调整的具体事项是什么？"
    legacy = plan_question(q)
    focused = plan_question(q, focus_details=True)
    assert legacy["questions"][0] == "本次认购价格调整前是多少？"
    assert "认购价格" in focused["questions"][0] and "数值计算式" in focused["questions"][0]
    assert not any(c.isdigit() for c in json.dumps(focused, ensure_ascii=False))
    question = "实施季度分配需同时满足哪些条件？"
    plans, tasks = plan_tasks([question, "是否披露实际金额？"], focus_details=True)
    assert plans[0]["rule"] == "conditions_with_exceptions"
    assert len(plans[0]["questions"]) == 2
    assert all("实施季度分配" in part for part in plans[0]["questions"])
    assert tasks[0]["bindings"] == [[2, 1]]
    assert [t["bindings"] for t in tasks[1:]] == [[[1, 1]], [[1, 2]]]
    limit = plan_question("融资额度调整后是多少，实际借款余额是多少，两者是否相同？", focus_details=True)
    assert "限定词" in limit["questions"][1]
    assert "仅说明区别" in limit["questions"][2]
