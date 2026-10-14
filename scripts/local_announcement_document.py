"""本地公告试验的可追溯输入：保留页码、表格单元格和摘录边界，不改生产采集。

PDF 提取需要可选依赖 pdfplumber；阅读已保存的 JSON 文档包只使用标准库。
运行：python -m scripts.local_announcement_document --pdf file.pdf --output packet.json
全文不会按问题裁剪，扫描页、字符遗漏或超过上下文预算时必须停止采用结果。
"""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from scripts.local_announcement_layout import (
    character_key,
    layout_blocks,
    margin_roles,
    repeated_headers,
    running_page_numbers,
    table_context,
)
from scripts.local_announcement_reader import compact, segment_source

DOCUMENT_VERSION = "local-document-v3"
SUPPORTED_VERSIONS = {"local-document-v1", "local-document-v2", DOCUMENT_VERSION}


def _inside(char: dict, box: tuple) -> bool:
    """按字符中心归属单元格，避免边框上的文字被两个单元格重复收取。"""
    x = (char["x0"] + char["x1"]) / 2
    y = (char["top"] + char["bottom"]) / 2
    return box[0] <= x < box[2] and box[1] <= y < box[3]


def _blocks(text: str, prefix: str, page: int | None) -> list[dict]:
    if not text.strip():
        return []
    return [
        {"id": f"{prefix}B{i}", "kind": "text", "page": page, **{k: s[k] for k in ("start", "end", "text")}}
        for i, s in enumerate(segment_source(text), 1)
    ]


def extract_pdf(path: Path, *, selected_pages: list[int] | None = None) -> dict:
    """完整提取所选页；页号从 1 开始，选择部分页时强制标为摘录。

    表格使用 PDF 真实单元格边界，不依据问题或答案猜列义。跨页表头不自动复制，
    合并格保存其覆盖列；空白格保持空白。每页核对非空白字符多重集，无法保持字符
    覆盖时记录问题，后续阅读会拒绝调用。此检查不能证明 PDF 自带文字层没有错误。
    """
    import pdfplumber  # 可选的本地 PDF 提取环境，模型读取阶段不需要此依赖。

    packet = {
        "version": DOCUMENT_VERSION,
        "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "source_path": str(path.resolve()),
        "scope": "FULL_TEXT",
        "pages": [],
        "issues": [],
    }
    with pdfplumber.open(path) as pdf:
        if not 1 <= len(pdf.pages) <= 30:
            raise ValueError("本地试验仅接受 1 至 30 页 PDF。")
        chosen = list(range(1, len(pdf.pages) + 1)) if selected_pages is None else selected_pages
        if not chosen or chosen != sorted(set(chosen)) or any(p < 1 or p > len(pdf.pages) for p in chosen):
            raise ValueError("页号必须唯一、递增且位于 PDF 范围内。")
        packet["total_pages"] = len(pdf.pages)
        packet["scope"] = "FULL_TEXT" if len(chosen) == len(pdf.pages) else "EXCERPT"
        headers = repeated_headers(pdf.pages)
        footer_pages = running_page_numbers(pdf.pages)
        for number in chosen:
            page = pdf.pages[number - 1]
            chars = page.chars
            roles = margin_roles(page, number, headers, footer_pages)
            page_data = {"page": number, "raw_text": page.extract_text() or "", "blocks": [], "tables": []}
            if not chars:
                packet["issues"].append(f"P{number}:NO_TEXT_LAYER")
            tables = [t for t in page.find_tables() if len(t.rows) >= 2 and len(t.rows[0].cells) >= 2]
            all_boxes = [box for table in tables for box in table.cells]
            # 嵌套表格会令父单元格与子单元格同时覆盖同一字符。按面积最小的实际
            # 单元格唯一分配字符，防止原文被重复输入；各格几何范围仍完整保留。
            owners = {}
            for char_index, char in enumerate(chars):
                candidates = [
                    (t_index, box) for t_index, table in enumerate(tables) for box in table.cells if _inside(char, box)
                ]
                if candidates:
                    owners[char_index] = min(
                        candidates,
                        key=lambda item: ((item[1][2] - item[1][0]) * (item[1][3] - item[1][1]), item[0], item[1]),
                    )
            # 每个单元格都从实际字符生成，避免 extractor 的隐式去重遗漏字符。
            for t_index, table in enumerate(tables, 1):
                xs = sorted({x for box in table.cells for x in (box[0], box[2])})
                ys = sorted({y for box in table.cells for y in (box[1], box[3])})
                rows = []
                for r_index, row in enumerate(table.rows, 1):
                    cells = []
                    for box in row.cells:
                        if box is None:
                            continue
                        from pdfplumber.utils import extract_text

                        cell_chars = [
                            char
                            for char_index, char in enumerate(chars)
                            if owners.get(char_index) == (t_index - 1, box)
                        ]
                        raw_text = extract_text(cell_chars) or ""
                        cells.append(
                            {
                                "column_start": xs.index(box[0]) + 1,
                                "column_end": xs.index(box[2]),
                                "row_start": ys.index(box[1]) + 1,
                                "row_end": ys.index(box[3]),
                                "bbox": list(box),
                                "text": raw_text,
                            }
                        )
                    rows.append(
                        {"id": f"P{number}T{t_index}R{r_index}", "kind": "table_row", "page": number, "cells": cells}
                    )
                page_data["tables"].append({"id": f"P{number}T{t_index}", "bbox": list(table.bbox), "rows": rows})
            # 非表格文字按表格的纵向边界分区，保留题注、单位、脚注与正文顺序。
            boundaries = sorted(
                {0.0, float(page.height), *(float(t.bbox[1]) for t in tables), *(float(t.bbox[3]) for t in tables)}
            )
            for band, (top, bottom) in enumerate(zip(boundaries, boundaries[1:], strict=False), 1):
                body_chars = [
                    c
                    for c in chars
                    if top <= (c["top"] + c["bottom"]) / 2 < bottom
                    and character_key(c) not in roles
                    and not any(_inside(c, box) for box in all_boxes)
                ]
                page_data["blocks"].extend(layout_blocks(body_chars, f"P{number}Z{band}", number))
            for role in ("header", "footer"):
                margin_chars = [
                    c
                    for c in chars
                    if roles.get(character_key(c)) == role and not any(_inside(c, box) for box in all_boxes)
                ]
                page_data["blocks"].extend(layout_blocks(margin_chars, f"P{number}{role.upper()}", number, role))
            raw_chars = Counter(compact("".join(c["text"] for c in chars)))
            recovered = Counter(
                compact(
                    "".join(b["text"] for b in page_data["blocks"])
                    + "".join(c["text"] for t in page_data["tables"] for r in t["rows"] for c in r["cells"])
                )
            )
            if recovered != raw_chars:
                packet["issues"].append(f"P{number}:CHARACTER_COVERAGE_MISMATCH")
            page_data["character_coverage_ok"] = recovered == raw_chars
            packet["pages"].append(page_data)
    return packet


def text_document(body: str, *, scope: str = "EXCERPT") -> dict:
    """无 PDF 页码的历史摘录仍可读，但不得伪造页号或冒充全文。"""
    if scope not in ("FULL_TEXT", "EXCERPT"):
        raise ValueError("资料范围必须显式指定 FULL_TEXT 或 EXCERPT。")
    return {
        "version": DOCUMENT_VERSION,
        "scope": scope,
        "source_sha256": hashlib.sha256(body.encode()).hexdigest(),
        "total_pages": None,
        "issues": [],
        "pages": [{"page": None, "raw_text": body, "blocks": _blocks(body, "S", None), "tables": []}],
    }


def evidence_units(packet: dict) -> list[dict]:
    """引用单位为正文片段或表格整行；整行含列号但不把列号当原文数字。"""
    units = []
    for page in packet["pages"]:
        ordered = [(b.get("band", [0])[0], i, b) for i, b in enumerate(page["blocks"])]
        for table in page["tables"]:
            context = table_context(table["rows"]) if packet["version"] != "local-document-v1" else {}
            for i, row in enumerate(table["rows"]):
                # raw_text 只保留 PDF 字符，rendered 才加入列号和分隔符。
                unit = {**row, "text": "\n".join(c["text"] for c in row["cells"])}
                unit.update(context.get(row["id"], {}))
                pieces = []
                for cell in row["cells"]:
                    start, end = cell["column_start"], cell["column_end"]
                    label = f"C{start}" if start == end else f"C{start}-C{end}"
                    if cell.get("row_end", 0) > cell.get("row_start", 0):
                        label += f"(跨R{cell['row_start']}-R{cell['row_end']})"
                    column_name = unit.get("column_labels", {}).get(start, "")
                    value = compact(cell["text"]) or "（空白）"
                    # 保留稳定列号用于审计，同时直接标出真实表头，避免模型自行心算列位置。
                    pieces.append(f"{label}={value}" + (f"【列名：{column_name}】" if column_name else ""))
                for inherited in unit.get("context_cells", []):
                    start = inherited["column_start"]
                    pieces.append(
                        f"合并格继承C{start}【{unit.get('column_labels', {}).get(start, '')}】="
                        f"{compact(inherited['text'])}（原格引用{inherited['source_row_id']}）"
                    )
                unit["rendered"] = " | ".join(pieces)
                ordered.append((table["bbox"][1], i, unit))
        units.extend(entry[2] for entry in sorted(ordered, key=lambda entry: entry[:2]))
    return units


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pages", type=int, nargs="+")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("输出已存在，请保留旧文档包并指定新文件。")
    packet = extract_pdf(args.pdf, selected_pages=args.pages)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(packet, stream, ensure_ascii=False, indent=2)
    print(json.dumps({"output": str(args.output), "issues": packet["issues"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
