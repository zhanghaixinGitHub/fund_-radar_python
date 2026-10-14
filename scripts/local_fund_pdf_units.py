"""PDF 列单位的独立复核，保留首次网格，不把表顶金额单位套到序号等元数据。

新公告发现序号误配金额及带百分号的单元格漏解析后增加。原实验冻结文件不改；
复核版另存，不能把修复后结果冒充原首次模型已正确。
"""

import copy
import re
from decimal import Decimal

from scripts.local_announcement_document import evidence_units
from scripts.local_announcement_reader import compact
from scripts.local_fund_pdf_grid import column_grid, enrich_packet

META_LABEL = r"^(?:序号|编号|姓名|职务|项目名称|公司名称|成立日期|统一社会信用代码)$"


def checked_grid(packet: dict) -> dict:
    """逐个原格核对显式单位；混合单位声明不猜列，百分号按原格绑定。

    表顶单位仅可用于有金额/数量口径的列或纯年度金额列。多个单位声明且表头
    没有明确对应单位时，标记待核对，绝不选第一个单位。
    """
    base = enrich_packet(packet)
    result = copy.deepcopy(base)
    changes = []
    catalog = {row["id"]: row for row in evidence_units(packet)}
    cells = {}
    for page in packet["pages"]:
        for table in page["tables"]:
            columns = max(cell["column_end"] for row in table["rows"] for cell in row["cells"])
            for source in table["rows"]:
                row = catalog[source["id"]]
                for slot in column_grid(row, columns):
                    cells[(source["id"], slot["column"])] = (slot, row, page)
    seen = {(fact["row_id"], fact["column"]) for fact in result["facts"]}
    # 原格以 % 明示的数值，既不是空白，也不是默认万元；补回所有此类事实。
    for (row_id, column), (slot, row, page) in cells.items():
        text = compact(slot.get("text") or "")
        if (row_id, column) in seen or slot["status"] != "CELL" or not re.fullmatch(r"[+-]?\d+(?:\.\d+)?%", text):
            continue
        label = row.get("column_labels", {}).get(column, "")
        if not label or row_id in row.get("header_rows", []):
            continue
        result["facts"].append(
            {
                "row_id": row_id,
                "page": page["page"],
                "column": column,
                "label": label,
                "row_label": " / ".join(compact(cell["text"]) for cell in row["cells"][:2]),
                "is_total": any(compact(cell["text"]) in ("合计", "总计") for cell in row["cells"]),
                "value": None,
                "unit": None,
                "raw_quote": slot["text"],
                "header_refs": row.get("header_rows", []),
                "source_cell": slot["source_cell"],
                "unit_source": None,
                "status": "NO_NUMERIC_SOURCE",
            }
        )
    for fact in result["facts"]:
        old = {key: fact[key] for key in ("value", "unit", "status")}
        slot, row, page = cells[(fact["row_id"], fact["column"])]
        text = compact(slot.get("text") or "").replace(",", "")
        if re.fullmatch(META_LABEL, compact(fact["label"])):
            fact.update(unit=None, status="SOURCE_METADATA_NOT_AMOUNT")
        elif slot["status"] == "CELL" and re.fullmatch(r"[+-]?\d+(?:\.\d+)?%", text):
            fact.update(
                value=str(Decimal(text[:-1])),
                unit="%",
                status="BOUND_CELL",
                unit_source={"explicit_cell": slot["text"]},
            )
        elif fact.get("unit_source", {}):
            reference = fact["unit_source"]
            block = next((block for block in page["blocks"] if block["id"] == reference.get("source_id")), None)
            if block:
                declaration = next((line for line in block["text"].splitlines() if "单位" in line), "")
                units = re.findall(r"亿元|万元|元|亿股|万股|股|%", declaration)
                if len(set(units)) > 1:
                    fact.update(
                        unit=None, status="MIXED_UNITS_NOT_BOUND", unit_source={"full_declaration": declaration}
                    )
                elif fact["status"] == "BOUND_CELL" and not re.search(
                    r"金额|本金|收益|费用|总额|收入|利润|数量|额度|资本|资产|^20\d{2}年$", fact["label"]
                ):
                    fact.update(unit=None, status="COLUMN_DIMENSION_NOT_BOUND")
        new = {key: fact[key] for key in ("value", "unit", "status")}
        if old != new:
            changes.append(
                {
                    "row_id": fact["row_id"],
                    "column": fact["column"],
                    "label": fact["label"],
                    "raw_quote": fact["raw_quote"],
                    "first_grid": old,
                    "checked_grid": new,
                }
            )
    result.update(version="LOCAL_PDF_GRID_UNITS_2", first_grid_preserved=True, unit_corrections=changes)
    result["rendered"] += "\n单位独立核对（原网格仅作版面定位，以本核对结果为准）：\n" + "\n".join(
        f"{fact['row_id']} C{fact['column']} {fact['label']}：{fact['value']} {fact['unit']}，{fact['status']}"
        for fact in result["facts"]
    )
    return result


def rebind_book(book: dict, grid: dict) -> dict:
    """保留所有原事实编号和原句，将误绑定或未确认单位改为显式状态。

    未删除序号原始字符，也不把它变成金额。新增的百分比事实留在网格审计中，
    此函数只修复旧账本同编号事实，避免改变原引用或假称首次结果已经正确。
    """
    revised = copy.deepcopy(book)
    cells = {(fact["row_id"], fact["column"]): fact for fact in grid["facts"]}
    for fact in revised["facts"]:
        if fact["kind"] != "pdf_table_cell":
            continue
        matched = cells[(fact["data"]["row_id"], fact["data"]["column"])]
        for key in ("value", "unit", "status"):
            fact["data"][key] = matched[key]
        import json

        fact["canonical"] = json.dumps(fact["data"], ensure_ascii=False)
    revised.update(pdf_unit_review_version=grid["version"], first_book_preserved=True)
    return revised
