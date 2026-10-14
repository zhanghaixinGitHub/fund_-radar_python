"""引用映射、表头依赖与条件完整性；用合成反例检查不因放宽规则而放过错数。"""

from copy import deepcopy
from types import SimpleNamespace

import pytest
from scripts.local_announcement_checks import final_checks
from scripts.local_announcement_document import text_document
from scripts.local_announcement_layout import layout_blocks, running_page_numbers, table_context
from scripts.local_announcement_structured_reader import check_structured, structured_request


def response(text, refs):
    return {"answers": [{"question_number": 1, "answer": text, "status": "answered", "evidence_ids": refs}]}


def test_short_citations_are_whitelisted_and_mapped_without_changing_raw_response():
    packet = text_document("本次认购208万元；尚需履行备案。", scope="FULL_TEXT")
    payload, units = structured_request("公告", packet, ["金额多少？"])
    allowed = payload["format"]["properties"]["answers"]["items"]["properties"]["evidence_ids"]["items"]["enum"]
    assert allowed == ["E001"]
    assert "[E001]" in payload["messages"][1]["content"]
    raw = response("本次认购208万元。", ["E001"])
    previous = deepcopy(raw)
    answer = check_structured(raw, units, ["金额多少？"], "FULL_TEXT")["answers"][0]
    assert answer["evidence"][0]["id"] == "SB1"
    assert answer["model_evidence_ids"] == ["E001"]
    assert answer["safe_answer"]
    assert raw == previous
    invalid = check_structured(response("208万元。", ["E002"]), units, ["金额？"], "FULL_TEXT")
    assert "EVIDENCE_ID_INVALID" in invalid["answers"][0]["issues"]


def test_table_context_adds_only_actual_header_and_merged_cell_dependencies():
    header = {
        "id": "P1T1R1",
        "citation_id": "E001",
        "kind": "table_row",
        "text": "2024年度",
        "cells": [{"text": "2024年度", "column_start": 1, "column_end": 1}],
    }
    row = {
        "id": "P1T1R2",
        "citation_id": "E002",
        "kind": "table_row",
        "text": "收入\n208",
        "header_rows": [header["id"]],
        "cells": [
            {"text": "收入", "column_start": 1, "column_end": 1},
            {"text": "208", "column_start": 2, "column_end": 2},
        ],
    }
    unrelated = {"id": "SB2", "citation_id": "E003", "text": "另外一家公司的收入是909万元。", "kind": "text"}
    units = [header, row, unrelated]
    correct = check_structured(response("2024年度收入208万元。", ["E002"]), units, ["收入？"], "FULL_TEXT")["answers"][
        0
    ]
    assert correct["context_evidence_ids"] == [header["id"]]
    assert correct["safe_answer"]
    wrong = check_structured(response("2024年度收入909万元。", ["E002"]), units, ["收入？"], "FULL_TEXT")["answers"][0]
    assert "NUMBER_NOT_IN_EVIDENCE:909" in wrong["issues"]
    assert unrelated not in wrong["evidence"]
    assert wrong["safe_answer"] is None


def test_exception_is_scoped_to_same_predicate_and_does_not_rewrite_answer():
    source = "审计机构对年度财报出具标准无保留意见的审计报告（季度分配另按有关规定执行）；利润应为正值。"
    raw = {"answer": "利润为正值；年度财报具有标准无保留意见的审计报告。", "quotes": [source], "issues": []}
    result = final_checks("分配需满足哪些条件？", raw, source)
    assert any(i.startswith("PARENTHETICAL_EXCEPTION") for i in result["issues"])
    assert result["answer"] == raw["answer"]
    preserved = {**raw, "answer": raw["answer"] + "季度分配另按有关规定执行。"}
    assert not final_checks("分配需满足哪些条件？", preserved, source)["issues"]
    unrelated = {"answer": "利润为正值。", "quotes": [source], "issues": []}
    assert not final_checks("利润要求是什么？", unrelated, source)["issues"]


def test_page_sequence_requires_multiple_pages_and_matching_bottom_positions():
    def page(n, top=92):
        return SimpleNamespace(
            width=100, height=100, extract_text_lines=lambda: [{"text": str(n), "top": top, "x0": 49, "x1": 51}]
        )

    assert running_page_numbers([page(1)]) == set()
    assert running_page_numbers([page(1), page(2)]) == {1, 2}
    assert running_page_numbers([page(1), page(2, 60)]) == set()


def test_sentence_and_parenthetical_exception_are_not_cut_at_length_threshold():
    pytest.importorskip("pdfplumber")
    sentence = "担保总额由47亿元增至82亿元（其他情况另按规定执行）。"
    text = "序" * 270 + sentence + "后" * 280 + "续句。"
    chars = []
    for n, char in enumerate(text):
        x, y = n % 35 * 10, n // 35 * 15
        offset = 1.7 if char.isdigit() else 0
        chars.append(
            {
                "text": char,
                "x0": x,
                "x1": x + 10,
                "top": y + offset,
                "bottom": y + 12 + offset,
                "doctop": y + offset,
                "size": 12,
                "upright": True,
                "height": 12,
                "width": 10,
            }
        )
    blocks = layout_blocks(chars, "P1Z1", 1)
    assert len(blocks) == 2
    assert "".join(blocks[0]["text"].split()).endswith(sentence)
    assert "47" not in blocks[1]["text"] and "82" not in blocks[1]["text"]
    assert "".join("".join(b["text"].split()) for b in blocks) == text


def test_new_layout_still_accepts_old_v2_packets():
    packet = text_document("已认购208万元。", scope="FULL_TEXT")
    packet["version"] = "local-document-v2"
    _, units = structured_request("公告", packet, ["认购金额？"])
    assert check_structured(response("208万元。", ["E001"]), units, ["金额？"], "FULL_TEXT")["answers"][0][
        "safe_answer"
    ]


def test_financial_subtable_switches_header_after_company_profile_without_mutation():
    def cell(text, col, row, end_col=None, end_row=None):
        return {
            "text": text,
            "column_start": col,
            "column_end": end_col or col,
            "row_start": row,
            "row_end": end_row or row,
        }

    rows = [
        {"id": "R1", "cells": [cell("经营范围", 1, 1), cell("产品研发和销售" * 20, 2, 1, 4)]},
        {"id": "R2", "cells": [cell("股权结构", 1, 2), cell("甲公司持有100%", 2, 2, 4)]},
        {
            "id": "R3",
            "cells": [
                cell("财务数据（万元）", 1, 3, end_row=4),
                cell("项目", 2, 3),
                cell("2024年1-6月（未经审计）", 3, 3),
                cell("2023年度（经审计）", 4, 3),
            ],
        },
        {"id": "R4", "cells": [cell("净利润", 2, 4), cell("112.34", 3, 4), cell("289.56", 4, 4)]},
    ]
    original = deepcopy(rows)
    context = table_context(rows)
    assert not any(context["R2"]["column_labels"].values())
    assert context["R4"]["header_rows"] == ["R3"]
    assert context["R4"]["column_labels"][3] == "2024年1-6月（未经审计）"
    assert context["R4"]["column_labels"][4] == "2023年度（经审计）"
    assert rows == original
    # 数值行即使写着“项目”或含年度文字，也不能冒充第二个表头。
    rows.append({"id": "R5", "cells": [cell("项目", 2, 5), cell("2022年度", 3, 5), cell("9", 4, 5)]})
    assert table_context(rows)["R5"]["header_rows"] == ["R3"]
