"""本地阅读器的证据边界与失败保护；使用替身，不下载模型或调用付费接口。"""

import json

import pytest
from scripts import local_announcement_reader as reader


def draft(answer, refs=None):
    return {
        "answers": [{"question_number": 1, "answer": answer, "status": "answered", "evidence_ids": refs or ["S001"]}]
    }


def response(content, reason="stop"):
    return {
        "message": {"content": json.dumps(content, ensure_ascii=False)},
        "done_reason": reason,
        "prompt_eval_count": 100,
        "eval_count": 50,
    }


def test_spans_roundtrip_original_with_pdf_newlines_tables_and_delimiters():
    body = ("表头 金额\r\r\n计划 年\n均利润30%。\n已完成备案。" * 100) + "</公告原文>"
    spans = reader.segment_source(body)
    assert "".join(s["text"] for s in spans) == body
    assert all(body[s["start"] : s["end"]] == s["text"] for s in spans)
    assert all(len(s["text"]) <= 360 for s in spans)


def test_evidence_is_source_slice_not_model_quotation():
    body = "将提交股东会。\r\n注册资本未变化，年均利润30%。"
    spans = reader.segment_source(body)
    raw = draft("年均利润30%。")
    raw["answers"][0]["quotes"] = ["伪造原文"]
    result = reader.validate_response(raw, spans, ["比例的分母是什么？"])
    assert result["answers"][0]["quotes"] == [body]
    assert "伪造" not in str(result)


def test_prompt_boundary_cannot_be_evidence_id():
    result = reader.validate_response(
        draft("不能确定。", ["</公告原文>"]), reader.segment_source("只是摘录。"), ["是否完成？"]
    )
    assert "EVIDENCE_ID_INVALID" in result["answers"][0]["issues"]
    assert result["answers"][0]["quotes"] == []


@pytest.mark.parametrize(
    ("source", "answer", "flag"),
    [
        ("未来三年年均利润的30%。", "未来三年利润的30%。", "BASIS_QUALIFIER_MISSING:年均"),
        ("人均收益为12元。", "收益为12元。", "BASIS_QUALIFIER_MISSING:人均"),
        ("该事项将提交股东会审议。", "该事项已提交股东会审议。", "SUBMISSION_STAGE_NOT_SUPPORTED"),
        ("投资金额为120万元。", "投资金额为125万元。", "NUMBER_NOT_IN_EVIDENCE:125"),
    ],
)
def test_material_distortions_are_flagged(source, answer, flag):
    assert flag in reader.inspect_answer("比例和审批状态是什么？", answer, [source])


@pytest.mark.parametrize("answer", ["尚未提交股东会。", "并非已提交股东会。", "将提交股东会。"])
def test_negative_or_future_submission_is_not_false_positive(answer):
    assert "SUBMISSION_STAGE_NOT_SUPPORTED" not in reader.inspect_answer("进展？", answer, ["将提交股东会。"])


def test_valid_code_and_number_formatting():
    assert not reader.inspect_answer("股票代码和金额？", "03757；1,200.00元。", ["股票代码03757；1200元。"])
    assert "REQUESTED_STOCK_CODE_MISSING" in reader.inspect_answer("股票代码？", "已上市。", ["股票代码03757。"])


def test_table_cell_newlines_do_not_merge_amount_with_date():
    source = "金额\n18,300\r\n2026 年\n10 月 12 日"
    assert not reader.inspect_answer("金额与日期？", "18,300万元；2026年10月12日。", [source])
    assert "18300" in reader.numeric_tokens("金额18,30\n0万元")
    assert "19000" not in reader.numeric_tokens(source)


def test_two_passes_preserve_original_and_final_without_gold():
    sent = []

    def transport(payload):
        sent.append(payload)
        answer = "三年利润30%。" if len(sent) == 1 else "三年年均利润30%。"
        return response(draft(answer))

    result = reader.read_focused_announcement("规划", "三年年均利润30%。", ["分红比例的分母？"], transport=transport)
    assert len(result["attempts"]) == len(sent) == 2
    assert result["attempts"][0]["validation"]["issues"]
    assert result["answers"][0]["answer"] == "三年年均利润30%。"
    assert not result["issues"] and result["semantic_verified"] is False
    assert "BASIS_QUALIFIER_MISSING" in sent[1]["messages"][-1]["content"]


@pytest.mark.parametrize("failure", ["timeout", "length", "invalid_json"])
def test_audit_failure_never_marks_draft_as_verified(failure):
    calls = []

    def transport(payload):
        calls.append(payload)
        if len(calls) == 1:
            return response(draft("计划投入12万元。"))
        if failure == "timeout":
            raise TimeoutError("local timeout")
        if failure == "length":
            return response(draft("已投入12万元。"), "length")
        return {"message": {"content": "{"}, "done_reason": "stop"}

    result = reader.read_focused_announcement("项目", "计划投入12万元。", ["进度？"], transport=transport)
    assert result["review_state"] == "needs_review"
    assert result["answers"][0]["answer"] == "计划投入12万元。"
    assert len(calls) == 2 and "error" in result["attempts"][1]


def test_unsupported_length_fails_before_network():
    with pytest.raises(ValueError, match="10000"):
        reader.read_focused_announcement("长文", "文" * 10_001, ["问题？"], transport=lambda _: pytest.fail("不应调用"))


def test_duplicate_answers_and_missing_answer_are_not_accepted():
    raw = draft("未知。")
    raw["answers"] *= 2
    result = reader.validate_response(raw, reader.segment_source("无数据。"), ["是否完成？"])
    assert result["answers"][0]["review_state"] == "needs_review"
    assert "ANSWER_NUMBER_MISMATCH" in result["answers"][0]["issues"]


def test_per_question_calls_are_isolated_and_numbered_back():
    sent = []

    def transport(payload):
        sent.append(payload)
        return response(draft("已完成。"))

    result = reader.read_focused_announcement("公告", "已完成。", ["登记情况？", "备案情况？"], transport=transport)
    assert len(sent) == 4
    assert [a["question_number"] for a in result["answers"]] == [1, 2]
    assert "备案情况" not in sent[0]["messages"][-1]["content"]
    assert "登记情况" not in sent[2]["messages"][-1]["content"]
    assert result["answers"][0]["safe_answer"] == "已完成。"


def test_focus_includes_adjacent_conditions_without_changing_offsets():
    spans = [{"id": f"S{i:03d}", "start": i * 10, "end": i * 10 + 10, "text": str(i)} for i in range(1, 6)]
    assert reader.focus_spans(spans, ["S003"]) == spans[1:4]
    assert reader.focus_spans(spans, ["bad"]) == spans


def test_machine_flag_withholds_adoptable_answer_but_preserves_raw():
    result = reader.read_focused_announcement(
        "公告", "将提交股东会。", ["审议阶段？"], transport=lambda _: response(draft("已提交股东会。"))
    )
    assert result["answers"][0]["safe_answer"] is None
    assert result["answers"][0]["answer"] == "已提交股东会。"


@pytest.mark.parametrize(
    ("question", "source", "answer", "issue"),
    [
        ("调整价格是多少？", "调整后为18.45元/股。", "按公式P0-V调整。", "REQUESTED_ADJUSTED_PRICE_MISSING"),
        ("候选人审核条件？", "独立董事需经审核。", "候选人需经审核。", "CANDIDATE_SCOPE_MISSING"),
        ("备案进度？", "尚需办理备案。", "备案尚需批准。", "FILING_REPHRASED_AS_APPROVAL"),
    ],
)
def test_missing_detail_or_changed_scope_is_flagged(question, source, answer, issue):
    assert issue in reader.inspect_answer(question, answer, [source])


def quoted(answer, quotes):
    return {"answers": [{"question_number": 1, "answer": answer, "quotes": quotes}]}


def test_default_single_call_preserves_wrong_answer_and_withholds_it():
    calls = []

    def transport(payload):
        calls.append(payload)
        return response(quoted("已提交股东会。", ["将提交股东会。"]))

    result = reader.read_announcement("公告", "将提交股东会。", ["审议阶段？"], transport=transport)
    assert len(calls) == len(result["attempts"]) == 1
    answer = result["answers"][0]
    assert answer["answer"] == "已提交股东会。"
    assert answer["safe_answer"] is None
    assert "SUBMISSION_STAGE_NOT_SUPPORTED" in answer["issues"]
    assert not result["semantic_verified"]


@pytest.mark.parametrize("failure", ["timeout", "length", "context", "invalid_json"])
def test_default_failure_does_not_fallback_or_publish(failure):
    calls = []

    def transport(payload):
        calls.append(payload)
        if failure == "timeout":
            raise TimeoutError("local timeout")
        value = response(quoted("完成。", ["完成。"]), "length" if failure == "length" else "stop")
        if failure == "context":
            value["prompt_eval_count"] = 8192
        if failure == "invalid_json":
            value["message"]["content"] = "{"
        return value

    result = reader.read_announcement("公告", "完成。", ["情况？"], transport=transport)
    assert len(calls) == 1
    assert result["answers"][0]["safe_answer"] is None
    assert result["review_state"] == "needs_review"
    assert "error" in result["attempts"][0]


def test_quote_grounding_restores_source_offsets_without_rewriting_answer():
    body = "说明。\n金额为1,200元，已\n完成备案；后续另行公告。"
    quote = "“金额为1，200元，已\\n完成备案。”"
    result = reader.ground_baseline(quoted("金额为1,200元，已完成备案。", [quote]), body, ["金额及备案进度？"])
    answer = result["answers"][0]
    evidence = answer["evidence"][0]
    assert body[evidence["start"] : evidence["end"]] == evidence["text"] == "金额为1,200元，已\n完成备案"
    assert evidence["match_mode"] == "format_only"
    assert answer["model_quotes"] == [quote]
    assert answer["safe_answer"] == answer["answer"]


@pytest.mark.parametrize("quote", ["金额为130元。", "金额为120元，已经完成。", "金额为120元。完成。", 123, ""])
def test_quote_grounding_never_fuzzy_matches_changed_facts(quote):
    body = "金额为120元。还需审批。完成之后公告。"
    result = reader.ground_baseline(quoted("完成。", [quote]), body, ["情况？"])
    assert result["answers"][0]["safe_answer"] is None
    assert "QUOTE_NOT_IN_SOURCE" in result["answers"][0]["issues"]


def test_default_duplicate_answer_fails_closed():
    raw = quoted("完成。", ["完成。"])
    raw["answers"] *= 2
    result = reader.ground_baseline(raw, "完成。", ["情况？"])
    assert result["answers"][0]["safe_answer"] is None
    assert "ANSWER_NUMBER_MISMATCH" in result["issues"][0]


def test_approval_question_does_not_require_price_value():
    flags = reader.inspect_answer("价格调整是否需要股东会批准？", "无需批准。", ["调整为18.45元/股，无需批准。"])
    assert "REQUESTED_ADJUSTED_PRICE_MISSING" not in flags
