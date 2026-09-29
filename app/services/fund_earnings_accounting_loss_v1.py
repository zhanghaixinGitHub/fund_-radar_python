"""明确净亏损行的会计括号金额核验；保留百万元原单位和全部比较列。"""

import re
from decimal import Decimal

from app.services.fund_earnings_evidence_v1 import review_claim
from app.services.fund_earnings_table_facts_v1 import CELL
from app.services.fund_information_history_v1 import explicit_money, normalize


def review_explicit_profit_range(spec, pages):
    """核验“盈利：金额单位-金额单位”，单连字符只在完整盈利区间内作分隔符。

    两端必须各带相同单位且为非负数；不接受夹带负号、混合单位或倒序。
    先严格解析完整原引文，再复用原件、期间、日期及第一端点验证；第二
    端点直接来自完整正则捕获，原引文不改写，也不伪造带空格的新原件。
    """
    unit = spec["unit"]
    if unit not in {"元", "万元", "百万元", "亿元"} or spec["kind"] != "FORECAST":
        raise ValueError("EXPLICIT_PROFIT_RANGE_SCOPE")
    number = r"(\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)"
    part = number + re.escape(unit)
    match = re.fullmatch(r"盈利[：:]" + part + r"[-–—至到~～]" + part, normalize(spec["quote"]))
    if not match:
        raise ValueError("EXPLICIT_PROFIT_RANGE_REQUIRED")
    captured = [v.replace(",", "") for v in match.groups()]
    values = [Decimal(v) for v in captured]
    wanted = [Decimal(str(v).replace(",", "")) for v in spec["values"]]
    if values != wanted or values != sorted(values):
        raise ValueError("PROFIT_ENDPOINTS_MISMATCH")
    fact = review_claim({**spec, "values": [captured[0]]}, pages)
    return {
        **fact,
        "values": captured,
        "money": [explicit_money(v, unit) for v in captured],
        "polarity": "PROFIT",
        "separator_role": "EXPLICIT_PROFIT_RANGE_DELIMITER",
    }


def review_parenthesized_loss_row(spec, pages):
    """只对同页、明示净亏损且选中金额带会计括号的原表生成负值。

    不能由同比下降或普通净利润行推断负号。括号必须完整且只包住非负
    数字；负号嵌套、注释、缺值标记均拒绝。原表金额、单位、括号和全部
    单元格仍保留，折算人民币不提高原披露精度。人工列映射须目视确认。
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
    match = re.fullmatch(r"单位[：:](?:人民币)?(亿元|百万元|万元|元)(?:币种[：:]人民币)?", unit)
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
    if "净亏损" not in label:
        raise ValueError("EXPLICIT_NET_LOSS_LABEL_REQUIRED")
    raw = spec["row_quote"]
    end = next((i for i in range(1, len(raw) + 1) if normalize(raw[:i]) == label), None)
    if end is None:
        raise ValueError("SCOPED_TABLE_ROW_LABEL_MISSING")
    tokens = raw[end:].split()
    cells = []
    for token in tokens:
        if token == "不适用":
            cells.append(None)
        elif re.fullmatch(r"\((?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?\)", token):
            cells.append("-" + token[1:-1].replace(",", ""))
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
    if not tokens[index].startswith("(") or not tokens[index].endswith(")"):
        raise ValueError("SELECTED_LOSS_MUST_HAVE_ACCOUNTING_PARENTHESES")
    if value is None or column["value_kind"] != "AMOUNT" or value.endswith("%"):
        raise ValueError("SCOPED_TABLE_SELECTED_CELL_NOT_MONEY")
    if not normalize(column["label"]) or normalize(column["label"]) not in header:
        raise ValueError("SCOPED_TABLE_COLUMN_LABEL_MISSING")
    if any(column[key] != spec[key] for key in ("period_start", "period_end", "basis")):
        raise ValueError("SCOPED_TABLE_PERIOD_OR_BASIS_MISMATCH")
    # 使用原件连续片段满足金额和单位锚定，不拼接或伪造带单位的指标行。
    magnitude = value[1:]
    fact = review_claim({**spec, "quote": spec["scope_quote"], "values": [magnitude]}, pages)
    reported = fact["money"]
    fact["reported_magnitudes"] = reported
    fact["money"] = [
        {
            **m,
            "value": str(-Decimal(m["value"])),
            "cny": str(-Decimal(m["cny"])),
            "sign_source": "PARENTHESES_AND_EXPLICIT_NET_LOSS_ROW",
        }
        for m in reported
    ]
    fact["polarity"] = "LOSS"
    fact["values"] = [value]
    return {
        **fact,
        "column_proof": {
            "unit_declaration": unit,
            "header": header,
            "row": row,
            "all_cells": cells,
            "original_cell_tokens": tokens,
            "selected_sign_proof": {"token": tokens[index], "row_label": label},
            "selected_index": index,
            "selected_column": column,
            "not_applicable_preserved": True,
            "column_layout_requires_visual_review": True,
        },
    }
