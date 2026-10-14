"""防止表格错列、静默裁剪、摘录冒充全文及引用定位造假。"""

import json
from copy import deepcopy

import pytest
from scripts.local_announcement_document import evidence_units, extract_pdf, text_document
from scripts.local_announcement_reader import inspect_answer
from scripts.local_announcement_structured_reader import check_structured, read_document, structured_request


def parsed(text="拟增资100万元。", refs=None, status="answered"):
    return {"answers": [{"question_number": 1, "answer": text, "status": status, "evidence_ids": refs or ["SB1"]}]}


def test_single_call_keeps_all_source_and_does_not_send_gold():
    packet = text_document("拟增资100万元。尚需备案。", scope="FULL_TEXT")
    packet["gold"] = "不应进入模型的黄金答案"
    calls = []

    def transport(request):
        calls.append(request)
        return {"done_reason": "stop", "message": {"content": json.dumps(parsed())}}

    result = read_document("增资公告", packet, ["拟增资多少？"], transport=transport)
    assert len(calls) == 1
    assert "尚需备案" in calls[0]["messages"][1]["content"]
    assert "黄金答案" not in json.dumps(calls[0], ensure_ascii=False)
    assert result["answers"][0]["quotes"] == ["拟增资100万元。尚需备案。"]
    assert result["answers"][0]["safe_answer"] == "拟增资100万元。"
    assert not result["semantic_verified"]


def test_invalid_evidence_id_never_becomes_a_quote():
    units = evidence_units(text_document("拟增资100万元。"))
    result = check_structured(parsed(refs=["</原文资料>"]), units, ["金额？"], "FULL_TEXT")
    assert result["answers"][0]["safe_answer"] is None
    assert "EVIDENCE_ID_INVALID" in result["answers"][0]["issues"]


def test_display_brackets_are_accepted_only_around_exact_existing_id():
    units = evidence_units(text_document("拟增资100万元。"))
    raw = parsed(refs=["[SB1]"])
    before = deepcopy(raw)
    result = check_structured(raw, units, ["金额？"], "FULL_TEXT")
    assert result["answers"][0]["safe_answer"] == "拟增资100万元。"
    assert result["answers"][0]["model_evidence_ids"] == ["[SB1]"]
    assert raw == before
    result = check_structured(parsed(refs=["[SB2]"]), units, ["金额？"], "FULL_TEXT")
    assert result["answers"][0]["safe_answer"] is None


def test_chinese_prefix_does_not_hide_stock_code():
    assert "REQUESTED_STOCK_CODE_MISSING" not in inspect_answer("股票代码？", "股票代码为03757。", ["代码03757"])
    assert "REQUESTED_STOCK_CODE_MISSING" in inspect_answer("股票代码？", "股票代码为375。", ["代码03757"])


def test_numbers_are_normalized_inside_cell_without_concatenating_next_cell():
    unit = {
        "id": "P1T1R2",
        "kind": "table_row",
        "page": 1,
        "text": "18,30\n0\n2026年",
        "cells": [
            {"column_start": 1, "column_end": 1, "text": "18,30\n0"},
            {"column_start": 2, "column_end": 2, "text": "2026年"},
        ],
    }
    correct = check_structured(parsed("金额18,300。", [unit["id"]]), [unit], ["金额？"], "FULL_TEXT")
    assert correct["answers"][0]["safe_answer"]
    wrong = check_structured(parsed("金额18301。", [unit["id"]]), [unit], ["金额？"], "FULL_TEXT")
    assert wrong["answers"][0]["safe_answer"] is None

    joined = check_structured(parsed("金额183002026。", [unit["id"]]), [unit], ["金额？"], "FULL_TEXT")
    assert joined["answers"][0]["safe_answer"] is None


def test_recusal_value_must_match_its_column_even_if_other_footnote_has_the_number():
    def row(identifier, values):
        return {
            "id": identifier,
            "kind": "table_row",
            "page": 1,
            "text": "\n".join(values),
            "cells": [{"column_start": i, "column_end": i, "text": v} for i, v in enumerate(values, 1)],
        }

    units = [
        row("P1T1R1", ["议案", "反对", "回避"]),
        row("P1T1R2", ["现金管理", "17214534", "0"]),
        {"id": "P1B1", "kind": "text", "text": "其他议案回避408784683股。"},
    ]
    wrong = check_structured(
        parsed("通过；回避表决股数408784683股。", ["P1T1R2", "P1B1"]),
        units,
        ["现金管理回避表决股数多少？"],
        "FULL_TEXT",
    )
    assert "TABLE_COLUMN_VALUE_MISMATCH:回避" in wrong["answers"][0]["issues"]
    assert wrong["answers"][0]["safe_answer"] is None
    good = check_structured(parsed("回避表决股数0股。", ["P1T1R2"]), units, ["现金管理回避表决股数多少？"], "FULL_TEXT")
    assert good["answers"][0]["safe_answer"]


def test_excerpt_unknown_is_not_adopted_and_document_overclaim_is_flagged():
    units = evidence_units(text_document("本次解除质押100股。"))
    result = check_structured(parsed("公告未披露剩余质押数量。", status="insufficient"), units, ["剩余？"], "EXCERPT")
    answer = result["answers"][0]
    assert {"EXCERPT_SCOPE_OVERCLAIM", "SOURCE_INSUFFICIENT"} <= set(answer["issues"])
    assert answer["safe_answer"] is None


@pytest.mark.parametrize(
    "response",
    [
        {"done_reason": "length", "message": {"content": json.dumps(parsed())}},
        {"done_reason": "stop", "message": {"content": "不是JSON"}},
        {"done_reason": "stop", "prompt_eval_count": 32768, "message": {"content": json.dumps(parsed())}},
    ],
)
def test_incomplete_response_fails_closed(response):
    result = read_document("公告", text_document("拟增资100万元。"), ["金额？"], transport=lambda _: response)
    assert result["answers"][0]["safe_answer"] is None
    assert "MODEL_CALL_OR_FORMAT_FAILED" in result["issues"]


def test_invalid_or_over_budget_packet_is_rejected_before_call():
    packet = text_document("拟增资100万元。")
    packet["issues"] = ["P1:CHARACTER_COVERAGE_MISMATCH"]
    with pytest.raises(ValueError, match="提取"):
        structured_request("公告", packet, ["金额？"])
    packet = text_document("金额100万元。")
    packet["pages"] *= 2
    with pytest.raises(ValueError, match="重复"):
        structured_request("公告", packet, ["金额？"])
    packet = text_document("未披露。" * 2500)
    with pytest.raises(ValueError, match="预算"):
        structured_request("公告", packet, ["金额？"])


def test_table_location_numbers_are_not_fact_evidence():
    packet = text_document("单位：股。")
    packet["pages"][0]["tables"] = [
        {
            "bbox": [0, 100, 500, 150],
            "rows": [
                {
                    "id": "P1T1R17",
                    "kind": "table_row",
                    "page": 1,
                    "cells": [
                        {"column_start": 4, "column_end": 5, "text": "反对"},
                        {"column_start": 6, "column_end": 6, "text": "0"},
                    ],
                }
            ],
        }
    ]
    original = deepcopy(packet)
    units = evidence_units(packet)
    assert "C4-C5=反对" in units[1]["rendered"]
    result = check_structured(parsed("反对17股。", ["P1T1R17"]), units, ["反对多少股？"], "FULL_TEXT")
    assert result["answers"][0]["safe_answer"] is None
    assert "NUMBER_NOT_IN_EVIDENCE:17" in result["answers"][0]["issues"]
    assert original == packet


def test_pdf_table_headers_cells_scope_and_footnotes_are_preserved(tmp_path):
    pytest.importorskip("pdfplumber")
    canvas = pytest.importorskip("reportlab.pdfgen.canvas")
    path = tmp_path / "example.pdf"
    pdf = canvas.Canvas(str(path), pagesize=(400, 400))
    pdf.drawString(30, 365, "Unit: shares")
    for y in [200, 240, 280]:
        pdf.line(30, y, 360, y)
    for x in [30, 140, 250, 360]:
        pdf.line(x, 200, x, 280)
    for x, label in [(40, "Proposal"), (150, "Against"), (260, "Recusal")]:
        pdf.drawString(x, 255, label)
    for x, value in [(40, "Cash"), (150, "17214534"), (260, "0")]:
        pdf.drawString(x, 215, value)
    pdf.drawString(30, 170, "Note: proposed, not completed.")
    pdf.showPage()
    pdf.drawString(30, 300, "Other page.")
    pdf.save()
    packet = extract_pdf(path, selected_pages=[1])
    assert packet["scope"] == "EXCERPT" and packet["total_pages"] == 2
    assert not packet["issues"]
    units = evidence_units(packet)
    rows = [u for u in units if u["kind"] == "table_row"]
    assert "C2=Against" in rows[0]["rendered"]
    assert "C2=17214534【列名：Against】" in rows[1]["rendered"]
    assert "C3=0【列名：Recusal】" in rows[1]["rendered"]
    assert any("Note: proposed, not completed." in u["text"] for u in units)
