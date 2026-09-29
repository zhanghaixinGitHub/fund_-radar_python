"""核验带行内单位的财报金额行，避免在多期间、多比较口径表格中取错列。

本模块不猜测表格布局。列名、期间和完整单元格序列由冻结的原件复核计划
提供；缺单元格、错行、百分比列、期间不匹配均拒绝，不填零或自动前移列。
"""

import re

from app.services.fund_earnings_evidence_v1 import review_claim
from app.services.fund_information_history_v1 import normalize

CELL = re.compile(r"[-−]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?%?")


def review_money_row(spec, pages):
    """按完整行和表头重放选定金额；只支持单位直接写在指标行名中的表格。

    spec 沿用金额事实字段，另含 row_label、row_quote、header_quote、cells、
    columns 和 selected_column。columns 逐列保留原表列名、期间、调整口径及
    金额/百分比类型；cells 保留全部数值，不能为了选择本期金额先删掉比较列。
    """
    page = spec["page"]
    if not 1 <= page <= len(pages):
        raise ValueError("TABLE_PAGE_OUT_OF_RANGE")
    text, row, header = (normalize(t) for t in (pages[page - 1], spec["row_quote"], spec["header_quote"]))
    label = normalize(spec["row_label"])
    if not header or text.count(header) != 1 or not row or text.count(row) != 1:
        raise ValueError("TABLE_ANCHOR_NOT_UNIQUE")
    if text.index(header) + len(header) > text.index(row):
        raise ValueError("TABLE_HEADER_NOT_BEFORE_ROW")
    if not label or not row.startswith(label) or spec["unit"] not in label:
        raise ValueError("TABLE_ROW_LABEL_OR_INLINE_UNIT_MISMATCH")
    # 保留原行空白后解析数字，不能把相邻单元格的数字粘成一个大数。
    raw = spec["row_quote"]
    end = next((i for i in range(1, len(raw) + 1) if normalize(raw[:i]) == label), None)
    if end is None:
        raise ValueError("TABLE_ROW_LABEL_PREFIX_MISSING")
    body = raw[end:]
    if normalize(CELL.sub("", body)):
        raise ValueError("TABLE_ROW_HAS_MISSING_OR_UNPARSED_CELLS")
    cells = [m[0].replace(",", "").replace("−", "-") for m in CELL.finditer(body)]
    expected = [v.replace(",", "").replace("−", "-") for v in spec["cells"]]
    columns = spec["columns"]
    if cells != expected or len(cells) != len(columns):
        raise ValueError("TABLE_CELL_SEQUENCE_MISMATCH")
    index = spec["selected_column"]
    if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(cells):
        raise ValueError("TABLE_SELECTED_COLUMN_OUT_OF_RANGE")
    column = columns[index]
    if column["value_kind"] != "AMOUNT" or cells[index].endswith("%"):
        raise ValueError("TABLE_PERCENT_IS_NOT_MONEY")
    if not normalize(column["label"]) or normalize(column["label"]) not in header:
        raise ValueError("TABLE_SELECTED_COLUMN_LABEL_MISSING")
    if any(column[k] != spec[k] for k in ("period_start", "period_end")):
        raise ValueError("TABLE_SELECTED_PERIOD_MISMATCH")
    if column["basis"] != spec["basis"]:
        raise ValueError("TABLE_SELECTED_BASIS_MISMATCH")
    fact = review_claim({**spec, "quote": raw, "values": [cells[index]]}, pages)
    return {
        **fact,
        "column_proof": {
            "header": header,
            "row": row,
            "all_cells": cells,
            "selected_index": index,
            "selected_column": column,
            "column_layout_requires_visual_review": True,
        },
    }
