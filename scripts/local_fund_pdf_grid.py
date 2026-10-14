"""本地验证专用PDF列网格：保留空列、合并范围和真实表头，不改生产读取器。

依赖沿用local_announcement_document的逐字符覆盖检查。这里只依据几何与原文
恢复展示，不根据公司、错题或金额猜列义；无法确认的单位、表头明确留空。
"""

import re
from decimal import Decimal

from scripts.local_announcement_document import evidence_units, extract_pdf
from scripts.local_announcement_reader import compact

UNITS = r"亿元|万元|元|亿股|万股|股|%"


def column_grid(row: dict, columns: int) -> list[dict]:
    """每列都有位置；普通空格不继承邻列值，合并格只指向其实际来源。

    纵向合并格可沿已有几何范围回指原格，但不能当成新披露金额重复累加。
    多个单元格覆盖同列属于版面歧义，直接标记，不选一个看起来正确的值。
    """
    result = []
    cells = row["cells"] + row.get("context_cells", [])
    for column in range(1, columns + 1):
        candidates = [c for c in cells if c["column_start"] <= column <= c["column_end"]]
        unique = {tuple(c["bbox"]): c for c in candidates}
        if len(unique) > 1:
            result.append({"column": column, "status": "AMBIGUOUS_GEOMETRY", "text": None})
            continue
        if not unique:
            result.append({"column": column, "status": "EMPTY", "text": ""})
            continue
        cell = next(iter(unique.values()))
        merged = cell["column_start"] != cell["column_end"]
        result.append(
            {
                "column": column,
                "status": "MERGED" if merged else "CELL",
                "text": cell["text"] if not merged else None,
                "source_cell": cell,
                "source_row": cell.get("source_row_id", row["id"]),
                "merged_text": cell["text"] if merged else None,
            }
        )
    return result


def preceding_unit(page: dict, table: dict) -> dict | None:
    """仅接受同页、紧邻表格上方的明确单位声明；不从金额反猜单位。"""
    top = table["bbox"][1]
    candidates = []
    for block in page["blocks"]:
        box = block.get("bbox")
        if not box or not 0 <= top - box[3] <= 120:
            continue
        if any(box[3] < other["bbox"][3] <= top and other is not table for other in page["tables"]):
            continue
        matches = list(re.finditer(rf"单位[：:]\s*(人民币)?\s*({UNITS})", block["text"]))
        if matches:
            match = matches[-1]
            candidates.append(
                (
                    box[3],
                    {
                        "unit": match[2],
                        "currency": "CNY" if match[1] else "NOT_BOUND",
                        "source_id": block["id"],
                        "quote": match[0],
                    },
                )
            )
    return max(candidates, key=lambda item: item[0])[1] if candidates else None


def enrich_packet(packet: dict) -> dict:
    """生成按列可读文本和带表头的数值记录；不改变原文包或补算空合计。

    raw_quote是实际单元格文字；canonical_text是程序说明，两者严格分开。
    标题无单位、同列标签冲突或几何重叠时保留待核对，不给确定金额。
    """
    if packet["scope"] != "FULL_TEXT" or packet["issues"]:
        raise ValueError("PDF_SOURCE_INCOMPLETE_OR_INVALID")
    if not all(page["character_coverage_ok"] for page in packet["pages"]):
        raise ValueError("PDF_CHARACTER_COVERAGE_FAILED")
    units = evidence_units(packet)
    catalog = {unit["id"]: unit for unit in units}
    facts, rendered, issues = [], [], []
    for page in packet["pages"]:
        for table in page["tables"]:
            columns = max(c["column_end"] for row in table["rows"] for c in row["cells"])
            declared = preceding_unit(page, table)
            for original in table["rows"]:
                row = catalog[original["id"]]
                grid = column_grid(row, columns)
                total = any(compact(c["text"]) in ("合计", "总计") for c in row["cells"])
                row_label = " / ".join(compact(c["text"]) for c in row["cells"][:2])
                pieces = []
                for slot in grid:
                    column = slot["column"]
                    label = row.get("column_labels", {}).get(column, "")
                    if not label:
                        # 原读取器只给当前行的单列格命名。合计横跨前四列时，这两列
                        # 金额格不存在，但表头仍存在；按表头真实覆盖范围恢复空列名。
                        names = [
                            compact(c["text"])
                            for ref in row.get("header_rows", [])
                            for c in catalog[ref]["cells"]
                            if c["column_start"] <= column <= c["column_end"] and c["text"].strip()
                        ]
                        label = "/".join(dict.fromkeys(names))
                    status = slot["status"]
                    text = slot.get("text")
                    shown = text if text and text.strip() else "（空白，无金额）"
                    if status == "MERGED":
                        cell = slot["source_cell"]
                        shown = f"（原合并格C{cell['column_start']}-C{cell['column_end']}：{cell['text']}，未单列数值）"
                    if status == "AMBIGUOUS_GEOMETRY":
                        issues.append(row["id"] + ":AMBIGUOUS_GEOMETRY")
                    pieces.append(f"C{column}【{label or '表头未确认'}】={shown}")
                    if row["id"] in row.get("header_rows", []) or not label:
                        continue
                    unit_match = re.search(rf"[（(]({UNITS})[）)]", label)
                    unit = unit_match[1] if unit_match else (declared or {}).get("unit")
                    number = compact(text or "").replace(",", "")
                    numeric = status == "CELL" and re.fullmatch(r"[+-]?\d+(?:\.\d+)?", number)
                    if not total and not numeric:
                        continue
                    value = str(Decimal(number)) if numeric else None
                    label_clean = re.sub(rf"[（(]({UNITS})[）)]", "", label)
                    facts.append(
                        {
                            "row_id": row["id"],
                            "page": page["page"],
                            "column": column,
                            "label": label_clean,
                            "row_label": row_label,
                            "is_total": total,
                            "value": value,
                            "unit": unit,
                            "raw_quote": text or "",
                            "header_refs": row.get("header_rows", []),
                            "source_cell": slot.get("source_cell"),
                            "unit_source": declared if not unit_match else {"header": label},
                            "status": "BOUND_CELL"
                            if numeric and unit
                            else "NO_NUMERIC_SOURCE"
                            if not numeric
                            else "UNBOUND_UNIT",
                        }
                    )
                rendered.append(f"第{page['page']}页 {row['id']}：" + " | ".join(pieces))
    return {
        "version": "LOCAL_PDF_GRID_1",
        "source_sha256": packet["source_sha256"],
        "total_pages": packet["total_pages"],
        "source_character_coverage_ok": True,
        "facts": facts,
        "rendered": "\n".join(rendered),
        "issues": issues,
    }


def read_pdf(path) -> tuple[dict, dict]:
    """完整读取所有页并恢复列网格；任何丢字、无文字层或歧义均留在结果中。"""
    packet = extract_pdf(path)
    return packet, enrich_packet(packet)
