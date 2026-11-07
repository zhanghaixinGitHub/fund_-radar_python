"""发行方SpreadsheetML全文及逻辑单元格：保留原数字字符串，不计算或改写原件。"""
from __future__ import annotations

import json
import re
from xml.etree import ElementTree as ET

from bs4 import BeautifulSoup

from app.services import material_content as store
from app.services.material_text_pilot import _part

NS = "urn:schemas-microsoft-com:office:spreadsheet"
TAG = "{" + NS + "}"
CATALOG_ENGINE = "SPREADSHEETML_CATALOG_V1"
SHEET_ENGINE = "SPREADSHEETML_SHEET_V1"


def capture_spreadsheetml(body, source_hash, repo):
    """严格解析前仅在内存转义裸&；用独立DOM逐单元核对，原始XML全文保持不变。

    原发行方链接包含未转义的查询参数分隔符。仅转义XML五种实体之外的裸&，
    不修补其他损坏。数值、日期、公式、稀疏位置、合并关系、格式/单位均以文本保存。
    视觉节点不保存载荷，缺单元格值或解析器不一致继续保留缺口，不发布完成。
    """
    if "<!DOCTYPE" in body.upper() or "<!ENTITY" in body.upper():
        raise ValueError("SPREADSHEETML_XML_ENTITY_FORBIDDEN")
    corrected, count = re.subn(r"&(?!#\d+;|#x[0-9a-fA-F]+;|amp;|lt;|gt;|quot;|apos;)", "&amp;", body)
    root = ET.fromstring(corrected)
    if root.tag != TAG + "Workbook":
        raise ValueError("SPREADSHEETML_WORKBOOK_REQUIRED")
    pending = []

    def dump(node):
        name = node.tag.rsplit("}", 1)[-1]
        if re.search(r"image|picture|chart|drawing|object", name, re.I):
            pending.append({"reason": "SPREADSHEETML_VISUAL_REQUIRES_TEXT_REVIEW", "tag": name})
            return {"tag": node.tag, "text_extraction_required": True}
        return {"tag": node.tag, "attributes": dict(node.attrib), "text": node.text,
                "tail": node.tail, "children": [dump(child) for child in node]}

    native = ["".join(node.itertext()) for node in root.findall(".//" + TAG + "Data")]
    independent = [node.get_text() for node in BeautifulSoup(body, "html.parser").find_all("ss:data")]
    if native != independent:
        pending.append({"reason": "SPREADSHEETML_INDEPENDENT_CELL_TEXT_DISAGREEMENT"})
    references, rows_total, cells_total, tables_total = [], 0, 0, 0
    for sheet in root.findall(TAG + "Worksheet"):
        tables = []
        for table in sheet.findall(TAG + "Table"):
            rows, row_number = [], 0
            for row in table.findall(TAG + "Row"):
                row_number = int(row.get(TAG + "Index", row_number + 1))
                cells, column = [], 0
                for cell in row.findall(TAG + "Cell"):
                    column = int(cell.get(TAG + "Index", column + 1))
                    if row_number < 1 or column < 1:
                        raise ValueError("SPREADSHEETML_INVALID_CELL_POSITION")
                    data = cell.find(TAG + "Data")
                    if data is None:
                        pending.append({"reason": "SPREADSHEETML_CELL_VALUE_REVIEW_REQUIRED",
                                        "sheet": sheet.get(TAG + "Name"), "row": row_number, "column": column})
                    cells.append({"column": column, "attributes": dict(cell.attrib),
                                  "data_type": None if data is None else data.get(TAG + "Type"),
                                  "value": None if data is None else "".join(data.itertext()),
                                  "other_nodes": [dump(node) for node in cell if node.tag != TAG + "Data"]})
                    across = int(cell.get(TAG + "MergeAcross", "0"))
                    if across < 0:
                        raise ValueError("SPREADSHEETML_INVALID_MERGE")
                    column += across
                rows.append({"row": row_number, "attributes": dict(row.attrib), "cells": cells})
                cells_total += len(cells)
            rows_total += len(rows)
            tables.append({"attributes": dict(table.attrib), "rows": rows,
                           "other_nodes": [dump(node) for node in table if node.tag != TAG + "Row"]})
        tables_total += len(tables)
        references.append(_part(repo, source_hash, SHEET_ENGINE, len(references), {
            "source_hash": source_hash, "name": sheet.get(TAG + "Name"), "attributes": dict(sheet.attrib),
            "tables": tables, "other_nodes": [dump(node) for node in sheet if node.tag != TAG + "Table"]}))
    if not references:
        raise ValueError("SPREADSHEETML_NO_WORKSHEETS")
    metadata = [dump(node) for node in root if node.tag != TAG + "Worksheet"]
    header = {"source_hash": source_hash, "format": "SPREADSHEETML_TEXT_V1", "sheets": references,
              "tables": tables_total, "rows": rows_total, "cells": cells_total,
              "root_attributes": dict(root.attrib), "metadata": metadata,
              "raw_source_not_modified": True, "bare_ampersands_escaped_in_memory": count,
              "all_cell_data_independently_equal": native == independent, "pending": pending}
    _part(repo, source_hash, CATALOG_ENGINE, 0, header)
    return header


def read_spreadsheetml(source_hash, *, repo=None):
    """只读数据库重建逻辑工作表；逐片核摘要与来源，不访问Excel或外接盘文件。"""
    repo = repo or store.repository()
    catalog = repo.page(source_hash, CATALOG_ENGINE, 0)
    if catalog is None:
        raise ValueError("SPREADSHEETML_TEXT_NOT_CAPTURED")
    header = json.loads(catalog)
    if header["source_hash"] != source_hash:
        raise ValueError("SPREADSHEETML_SOURCE_CHANGED")
    sheets = []
    for reference in header["sheets"]:
        raw = repo.page(source_hash, reference["engine"], reference["index"])
        if raw is None or store.checksum(raw.encode()) != reference["hash"]:
            raise ValueError("SPREADSHEETML_SHEET_MISSING_OR_CHANGED")
        sheet = json.loads(raw)
        if sheet["source_hash"] != source_hash:
            raise ValueError("SPREADSHEETML_SOURCE_CHANGED")
        sheets.append(sheet)
    return {**header, "sheets": sheets}
