"""限定同页表头单位的金额核验；保留不适用列，禁止借用其他表格单位。"""

import re

from app.services.fund_earnings_evidence_v1 import review_claim
from app.services.fund_earnings_table_facts_v1 import CELL
from app.services.fund_information_history_v1 import normalize


def review_scoped_money_row(spec, pages):
    """核验人工冻结的表格范围、单位、完整列和金额；不推断跨页表格。

    scope_quote 必须是原件同一页上连续的单位声明至金额行末。NA 单元格
    仅接受明确写出的“不适用”，保持其列位置和 None 值，不能选为金额。
    列名、期间及合并/重述口径仍需原件目视核定；本函数不推断表格布局。
    """
    page = spec["page"]
    if not 1 <= page <= len(pages):
        raise ValueError("SCOPED_TABLE_PAGE_OUT_OF_RANGE")
    source, scope, unit, header, row, label = (
        normalize(value)
        for value in (
            pages[page - 1],
            spec["scope_quote"],
            spec["unit_quote"],
            spec["header_quote"],
            spec["row_quote"],
            spec["row_label"],
        )
    )
    if not scope or source.count(scope) != 1:
        raise ValueError("SCOPED_TABLE_ANCHOR_NOT_UNIQUE")
    if any(not value or scope.count(value) != 1 for value in (unit, header, row)):
        raise ValueError("SCOPED_TABLE_COMPONENT_NOT_UNIQUE")
    match = re.fullmatch(r"单位[：:](?:人民币)?(亿元|万元|元)(?:币种[：:]人民币)?", unit)
    if not match or match[1] != spec["unit"] or not scope.startswith(unit):
        raise ValueError("SCOPED_TABLE_UNIT_MISMATCH")
    if len(re.findall(r"单位[：:]", scope)) != 1:
        raise ValueError("SCOPED_TABLE_CROSSES_UNIT_BOUNDARY")
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
    # 使用原件连续片段满足金额和单位锚定，不拼接或伪造带单位的指标行。
    fact = review_claim({**spec, "quote": spec["scope_quote"], "values": [value]}, pages)
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
