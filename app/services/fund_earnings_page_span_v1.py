"""限定原件一至两页连续表格核验；独立单位行与跨页表格均保留真实页锚点。"""

import re

from app.services.fund_earnings_evidence_v1 import review_claim
from app.services.fund_earnings_table_facts_v1 import CELL
from app.services.fund_information_history_v1 import normalize


def review_page_span_money_row(spec, pages):
    """核验人工冻结、目视确认的连续表格；不自动推断页间单位归属。

    page_segments 必须是一页连续范围，或相邻两页的完整页尾及页首。
    单位、表头和金额行各有真实页锚点；跨页布局仍须人工查看。独立的
    “元/万元/亿元”只接受同页表头中的完整单独一行，绝不从正文找字借用。
    选中单元格、负号、不适用、期间和原披露精度沿用严格验证，不填零。
    """
    segments = spec["page_segments"]
    numbers = [part["page"] for part in segments]
    if (
        len(segments) not in (1, 2)
        or any(isinstance(n, bool) or not isinstance(n, int) or not 1 <= n <= len(pages) for n in numbers)
        or numbers != list(range(numbers[0], numbers[0] + len(numbers)))
    ):
        raise ValueError("TABLE_PAGE_SPAN_INVALID")
    anchors = []
    for index, part in enumerate(segments):
        source, quote = normalize(pages[part["page"] - 1]), normalize(part["quote"])
        if not quote or source.count(quote) != 1:
            raise ValueError("TABLE_SEGMENT_NOT_UNIQUE")
        if len(segments) == 2 and (
            (index == 0 and not source.endswith(quote)) or (index == 1 and not source.startswith(quote))
        ):
            raise ValueError("TABLE_PAGE_BOUNDARY_GAP")
        anchors.append({"page": part["page"], "offset": source.index(quote), "text": quote})
    # 逻辑表格仅拼接已经逐页核验的相邻原文；下方复用字段验证后还原真实页锚点。
    combined = "\n".join(part["quote"] for part in segments)
    source = scope = normalize(combined)
    unit, header, row, label = (
        normalize(spec[key]) for key in ("unit_quote", "header_quote", "row_quote", "row_label")
    )
    if unit not in anchors[0]["text"] or row not in anchors[-1]["text"]:
        raise ValueError("TABLE_COMPONENT_PAGE_MISMATCH")
    if sum(header in anchor["text"] for anchor in anchors) != 1:
        raise ValueError("TABLE_HEADER_PAGE_AMBIGUOUS")
    if not scope or source.count(scope) != 1:
        raise ValueError("SCOPED_TABLE_ANCHOR_NOT_UNIQUE")
    if any(not value or scope.count(value) != 1 for value in (unit, header, row)):
        raise ValueError("SCOPED_TABLE_COMPONENT_NOT_UNIQUE")
    match = re.fullmatch(r"单位[：:](?:人民币)?(亿元|万元|元)(?:币种[：:]人民币)?", unit)
    standalone = spec.get("unit_form") == "STANDALONE_HEADER_LINE"
    if standalone:
        raw_lines = [line.strip() for line in combined.splitlines() if line.strip()]
        valid_unit = len(segments) == 1 and unit == spec["unit"] and unit in {"元", "万元", "亿元"}
        if not valid_unit or raw_lines[0] != unit or raw_lines.count(unit) != 1:
            raise ValueError("TABLE_STANDALONE_UNIT_INVALID")
    elif not match or match[1] != spec["unit"]:
        raise ValueError("SCOPED_TABLE_UNIT_MISMATCH")
    if not scope.startswith(unit):
        raise ValueError("SCOPED_TABLE_UNIT_MISMATCH")
    if len(re.findall(r"单位[：:]", scope)) != (0 if standalone else 1):
        raise ValueError("SCOPED_TABLE_CROSSES_UNIT_BOUNDARY")
    bare_units = re.findall(r"(?m)^\s*(?:亿元|万元|元)\s*$", combined)
    if len(bare_units) != (1 if standalone else 0):
        raise ValueError("TABLE_CROSSES_STANDALONE_UNIT_BOUNDARY")
    if not len(unit) <= scope.index(header) < scope.index(header) + len(header) <= scope.index(row):
        raise ValueError("SCOPED_TABLE_COMPONENT_ORDER")
    if not scope.endswith(row) or not label or not row.startswith(label):
        raise ValueError("SCOPED_TABLE_ROW_BOUNDARY")
    if re.search(r"[%％]|元|股", label.replace("股东", "")):
        raise ValueError("SCOPED_TABLE_ROW_UNIT_OVERRIDE")
    raw = spec["row_quote"]
    end = next((i for i in range(1, len(raw) + 1) if normalize(raw[:i]) == label), None)
    if end is None:
        raise ValueError("SCOPED_TABLE_ROW_LABEL_MISSING")
    tokens = raw[end:].split()
    cells = []
    for token in tokens:
        if token == "不适用":
            cells.append(None)
        elif CELL.fullmatch(token):
            cells.append(token.replace(",", "").replace("−", "-"))
        else:
            raise ValueError("SCOPED_TABLE_UNPARSED_CELL")
    expected = [None if value is None else value.replace(",", "").replace("−", "-") for value in spec["cells"]]
    columns = spec["columns"]
    if cells != expected or len(cells) != len(columns):
        raise ValueError("SCOPED_TABLE_CELL_SEQUENCE_MISMATCH")
    index = spec["selected_column"]
    if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(cells):
        raise ValueError("SCOPED_TABLE_COLUMN_OUT_OF_RANGE")
    column, value = columns[index], cells[index]
    if value is None or column["value_kind"] != "AMOUNT" or value.endswith("%"):
        raise ValueError("SCOPED_TABLE_SELECTED_CELL_NOT_MONEY")
    if not normalize(column["label"]) or normalize(column["label"]) not in header:
        raise ValueError("SCOPED_TABLE_COLUMN_LABEL_MISSING")
    if any(column[key] != spec[key] for key in ("period_start", "period_end", "basis")):
        raise ValueError("SCOPED_TABLE_PERIOD_OR_BASIS_MISMATCH")
    fact = review_claim({**spec, "page": 1, "quote": combined, "values": [value]}, [combined])
    # 不将逻辑表格的临时页号写成 PDF 页号；输出使用已核验的逐页定位。
    fact["page"] = numbers[-1]
    fact["anchor"] = {"page_span": numbers, "segments": anchors}
    fact["source_layout_requires_visual_review"] = True
    return {
        **fact,
        "column_proof": {
            "unit_declaration": unit,
            "header": header,
            "row": row,
            "all_cells": cells,
            "selected_index": index,
            "selected_column": column,
            "not_applicable_preserved": True,
            "column_layout_requires_visual_review": True,
        },
    }
