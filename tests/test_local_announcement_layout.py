"""跨页数字、公式、合并格和拆题隔离的回归验证；不使用真实样本答案充当修复。"""

import json
from copy import deepcopy

import pytest
from scripts.local_announcement_checks import final_checks
from scripts.local_announcement_document import evidence_units, extract_pdf, text_document
from scripts.local_announcement_layout import table_context
from scripts.local_announcement_structured_reader import check_structured
from scripts.local_announcement_tasks import plan_question, read_tasks


def test_pdf_margin_number_and_formula_remain_separate(tmp_path):
    pytest.importorskip("pdfplumber")
    canvas = pytest.importorskip("reportlab.pdfgen.canvas")
    pdf = canvas.Canvas(str(tmp_path / "layout.pdf"), pagesize=(400, 400))
    for page_number in (1, 2):
        pdf.setFont("Helvetica", 12)
        pdf.drawString(30, 380, "Repeated announcement header")
        if page_number == 1:
            pdf.drawString(30, 320, "P")
            pdf.setFont("Helvetica", 6)
            pdf.drawString(38, 316, "2")
            pdf.setFont("Helvetica", 12)
            pdf.drawString(43, 320, "= X - Y")
            pdf.drawString(30, 35, "Balance no more than 27.48")
        else:
            pdf.drawString(30, 335, "million yuan, as disclosed.")
        pdf.drawString(198, 10, str(page_number))
        pdf.showPage()
    pdf.save()
    packet = extract_pdf(tmp_path / "layout.pdf")
    assert not packet["issues"]
    units = evidence_units(packet)
    body = [u for u in units if u.get("role") == "body"]
    assert any("P_{2}" in u["rendered"] for u in body)
    amount = next(u for u in body if "27.48" in u["text"])
    assert not amount["text"].endswith("\n1")
    assert len([u for u in units if u.get("role") == "footer"]) == 2
    assert len([u for u in units if u.get("role") == "header"]) == 2
    footer = next(u for u in units if u.get("role") == "footer")
    result = check_structured(
        {
            "answers": [
                {"question_number": 1, "answer": "金额1元", "status": "answered", "evidence_ids": [footer["id"]]}
            ]
        },
        units,
        ["金额？"],
        "FULL_TEXT",
    )
    assert "PAGE_LAYOUT_IS_NOT_BUSINESS_EVIDENCE" in result["answers"][0]["issues"]
    assert result["answers"][0]["safe_answer"] is None


def test_merged_cells_keep_source_and_do_not_fill_unmerged_blanks():
    def cell(text, col, start, end):
        return {"text": text, "column_start": col, "column_end": col, "row_start": start, "row_end": end}

    rows = [
        {"id": "R1", "cells": [cell("提案名称", 1, 1, 1), cell("回避", 2, 1, 1)]},
        {"id": "R2", "cells": [cell("事项甲", 1, 2, 3), cell("20", 2, 2, 2)]},
        {"id": "R3", "cells": [cell("0", 2, 3, 3)]},
        {"id": "R4", "cells": [cell("", 1, 4, 4), cell("7", 2, 4, 4)]},
    ]
    original = deepcopy(rows)
    context = table_context(rows)
    assert context["R3"]["context_cells"][0]["source_row_id"] == "R2"
    assert context["R3"]["column_labels"][2] == "回避"
    assert context["R4"]["context_cells"] == []
    assert rows == original


def test_bare_wrong_vote_number_is_checked():
    rows = [
        {
            "id": "P1T1R1",
            "kind": "table_row",
            "text": "回避",
            "cells": [{"column_start": 1, "column_end": 1, "text": "回避"}],
        },
        {
            "id": "P1T1R2",
            "kind": "table_row",
            "text": "0",
            "cells": [{"column_start": 1, "column_end": 1, "text": "0"}],
        },
    ]
    parsed = {"answers": [{"question_number": 1, "answer": "765股", "status": "answered", "evidence_ids": ["P1T1R2"]}]}
    checked = check_structured(parsed, rows, ["该议案的回避表决股数是多少？"], "FULL_TEXT")
    assert "TABLE_COLUMN_VALUE_MISMATCH:回避" in checked["answers"][0]["issues"]


def test_research_metric_does_not_inherit_other_metric_benchmark():
    raw = {
        "answer": "研发投入强度不低于8%。",
        "quotes": ["净资产收益率不低于12%，且不低于同行业平均值；研发投入强度不低于8%。"],
        "issues": [],
    }
    checked = final_checks("研发占比要求是什么？", raw)
    assert "BENCHMARK_CONDITION_MISSING" not in checked["issues"]
    raw["answer"] = "净资产收益率不低于12%。"
    assert "BENCHMARK_CONDITION_MISSING" in final_checks("净资产收益率要求是什么？", raw)["issues"]


def test_quote_boundaries_do_not_append_unrelated_digit():
    raw = {"answer": "实际余额27.48亿元。", "quotes": ["实际余额不超过人民币27.48", "1", "亿元。"], "issues": []}
    assert "BOUND_QUALIFIER_REQUIRES_REVIEW:27.48" in final_checks("实际余额是多少？", raw)["issues"]


def test_atomic_calls_are_independent_and_failures_are_not_rewritten():
    requests = []

    def transport(payload):
        requests.append(payload)
        if len(requests) == 1:
            return {"done_reason": "length", "message": {"content": "{"}}
        return {
            "done_reason": "stop",
            "message": {
                "content": json.dumps(
                    {
                        "answers": [
                            {"question_number": 1, "answer": "0股", "status": "answered", "evidence_ids": ["SB1"]}
                        ]
                    }
                )
            },
        }

    source = text_document("事项甲通过，回避0股。其他资料仍保留。", scope="FULL_TEXT")
    result = read_tasks("公告", source, ["事项甲是否通过，其回避表决股数是多少？"], transport=transport)
    assert len(requests) == 2
    assert all("其他资料仍保留" in r["messages"][1]["content"] for r in requests)
    assert result["answers"][0]["safe_answer"] is None
    assert "MODEL_CALL_OR_FORMAT_FAILED" in result["answers"][0]["parts"][0]["issues"]
    assert result["answers"][0]["parts"][1]["answer"] == "0股"
    assert "answers" not in requests[1]["messages"][1]["content"]
    assert plan_question("额度是多少，若发生特殊情况怎么办？")["rule"] == "unchanged"


def test_decomposition_does_not_introduce_reference_answers():
    plan = plan_question("哪些提案属于特别决议，哪些提案属于普通决议，各自通过门槛是什么？")
    assert len(plan["questions"]) == 4
    assert "三分之二" not in json.dumps(plan, ensure_ascii=False)
    assert not any(char.isdigit() for q in plan["questions"] for char in q)
    assert len(plan_question("哪些提案属于特别决议，哪些属于普通决议，各自通过门槛是什么？")["questions"]) == 4


def test_ordinary_questions_keep_context_and_single_question_schema_fixes_number():
    from scripts.local_announcement_structured_reader import structured_request
    from scripts.local_announcement_tasks import plan_tasks

    packet = text_document("甲产品为浮动收益。乙产品为固定收益。", scope="FULL_TEXT")
    _, tasks = plan_tasks(["甲产品金额是多少？", "该产品收益是否固定？", "事项乙是否通过，其回避表决股数是多少？"])
    assert tasks[0]["questions"] == ["甲产品金额是多少？", "该产品收益是否固定？"]
    assert tasks[0]["bindings"] == [[1, 1], [2, 1]]
    assert len(tasks) == 3
    assert tasks[1]["bindings"] == [[3, 1]]
    payload, _ = structured_request("公告", packet, tasks[1]["questions"])
    assert "该产品收益是否固定？" not in payload["messages"][1]["content"]
    assert payload["format"]["properties"]["answers"]["maxItems"] == 1
    assert payload["format"]["properties"]["answers"]["items"]["properties"]["question_number"]["enum"] == [1]


def test_repeated_header_switches_balance_sheet_dates_to_periods():
    rows = []
    for n, values in enumerate(
        [
            ["财务指标", "2024年12月31日", "2025年6月30日"],
            ["资产", "12", "20"],
            ["财务指标", "2024年度", "2025年1-6月"],
            ["收入", "6", "7"],
        ],
        1,
    ):
        rows.append(
            {
                "id": f"R{n}",
                "cells": [
                    {"text": value, "column_start": col, "column_end": col, "row_start": n, "row_end": n}
                    for col, value in enumerate(values, 1)
                ],
            }
        )
    contexts = table_context(rows)
    assert contexts["R2"]["column_labels"][2] == "2024年12月31日"
    assert contexts["R4"]["column_labels"][2] == "2024年度"
    assert contexts["R4"]["header_rows"] == ["R3"]


def test_nested_table_does_not_duplicate_its_characters(tmp_path):
    pytest.importorskip("pdfplumber")
    canvas = pytest.importorskip("reportlab.pdfgen.canvas")
    path = tmp_path / "nested.pdf"
    pdf = canvas.Canvas(str(path), pagesize=(400, 400))
    for x in [30, 250, 360]:
        pdf.line(x, 100, x, 340)
    for y in [100, 260, 340]:
        pdf.line(30, y, 360, y)
    pdf.drawString(40, 300, "Outer title")
    pdf.drawString(270, 300, "Outcome")
    for x in [40, 130, 230]:
        pdf.line(x, 130, x, 210)
    for y in [130, 170, 210]:
        pdf.line(40, y, 230, y)
    pdf.drawString(45, 185, "Grade")
    pdf.drawString(140, 185, "Factor")
    pdf.drawString(45, 145, "A")
    pdf.drawString(140, 145, "0.8")
    pdf.save()
    packet = extract_pdf(path)
    assert not packet["issues"]
    assert len(packet["pages"][0]["tables"]) == 2
    source = "".join(u["text"] for u in evidence_units(packet))
    assert source.count("0.8") == 1
