"""固定训练池的报告方向覆盖核查；只读来源和人工判读，不训练模型。

报告方向不是基金名称，也不由历史预测效果反推。没有明确投资说明时保留未知。
所有结果仅写本次独立目录，原资料准入、训练输入和旧实验决定均不修改。
"""

import json
import re
from collections import Counter
from pathlib import Path

from bs4 import BeautifulSoup
from pypdf import PdfReader

from app.services.fund_002112_zero_fit_review import file_hash, read_json, save_once

ROOT = Path(__file__).resolve().parents[2] / ".local-runs/fund-exposure-002112"
OUT = ROOT / "style-training-coverage/20260928-v1"
FROZEN = ROOT / "peer-recovered-experiment/20260928-v1/frozen"
TAGS = {"MEDICINE_FOCUS", "AI_COMPUTE_FOCUS", "OTHER_EXPLICIT_FOCUS", "DIVERSIFIED_OR_MULTI", "UNSPECIFIED"}
HEAD = re.compile(r"4[.．]4(?:[.．]1)?报告期内基金的?投资策略(?:和|及)运作分析")
END = re.compile(r"4[.．](?:4[.．]2|5)(?:报告期|管理人)")


def compact(text):
    return re.sub(r"\s+", "", text)


def strategy_section(pages):
    """按明确章节边界抽取正文；忽略目录短项，保留原页码和字符偏移。"""
    parts = [compact(p["text"]) for p in pages]
    text = "".join(parts)
    candidates = []
    for match in HEAD.finditer(text):
        end = END.search(text, match.end())
        if end is None:
            continue
        body = text[match.end() : end.start()]
        # 网页转写的目录使用大量英文点线，长度可能超过正文下限。
        # 只有点线和页码的条目仍是目录；不能因目录重复而把真实正文判为缺失。
        if re.fullmatch(r"[.．·…\d]+", body):
            continue
        if len(body) < 80 or len(body) > 15000 or body.count("……") > 2:
            continue
        positions, offset = [], 0
        for page, part in zip(pages, parts, strict=True):
            if offset + len(part) > match.start() and offset < end.start():
                positions.append(page["page"])
            offset += len(part)
        candidates.append({"start": match.start(), "end": end.start(), "text": body, "pages": positions})
    if len(candidates) != 1:
        return {"status": "SECTION_NOT_UNAMBIGUOUS", "candidate_count": len(candidates), "text": "", "pages": []}
    return {"status": "EXTRACTED", **candidates[0]}


def verify_protocol():
    protocol = read_json(OUT / "protocol.json")
    if protocol["fit_budget"] != 0 or protocol["network_requests"] != 0:
        raise ValueError("ZERO_FIT_OFFLINE_SCOPE_CHANGED")
    for path, expected in protocol["source_files"].items():
        if file_hash(path) != expected:
            raise ValueError("SOURCE_CHANGED:" + path)
    return protocol


def extract():
    protocol = verify_protocol()
    records = []
    # PDF 只需要读到策略段终点；不把后续业绩表当作人工分类依据。
    for spec in sorted(protocol["report_manifest"], key=lambda r: (r["fund_code"], r["report_end"], r["report_type"])):
        path = Path(spec["raw_path"])
        if path.suffix.lower() == ".pdf":
            reader, pages = PdfReader(path), []
            section = None
            for i, page in enumerate(reader.pages):
                pages.append({"page": i + 1, "text": page.extract_text()})
                if i >= 3:
                    section = strategy_section(pages)
                    if section["status"] == "EXTRACTED":
                        break
            section = section or strategy_section(pages)
        elif path.suffix.lower() == ".json":
            value = json.loads(path.read_text(encoding="utf-8-sig"))
            section = strategy_section([{"page": None, "text": value["data"]["notice_content"]}])
        elif path.suffix.lower() == ".html":
            soup = BeautifulSoup(path.read_bytes(), "html.parser")
            body = soup.select_one("#sohu_content")
            section = strategy_section([{"page": None, "text": body.get_text("\n", strip=True) if body else ""}])
        else:
            raise ValueError("UNSUPPORTED_FROZEN_SOURCE")
        records.append({"source": spec, "section": section})
        if len(records) % 20 == 0:
            print(json.dumps({"reports_extracted": len(records)}, ensure_ascii=False), flush=True)
    save_once(OUT / "sections.json", records)
    save_once(
        OUT / "extraction-receipt.json",
        {
            "reports": len(records),
            "status": dict(Counter(r["section"]["status"] for r in records)),
            "code_sha256": file_hash(__file__),
            "protocol_sha256": file_hash(OUT / "protocol.json"),
            "sections_sha256": file_hash(OUT / "sections.json"),
            "new_fits": 0,
        },
    )
    return {"reports": len(records), "status": dict(Counter(r["section"]["status"] for r in records))}


def describe_rows(rows, tag_index):
    by_tag = {}
    for tag in sorted(TAGS):
        selected = [r for r in rows if tag_index[r["report_sha256"]]["tag"] == tag]
        by_tag[tag] = {
            "rows": len(selected),
            "unique_dates": len({r["target"] for r in selected}),
            "funds": dict(Counter(r["fund_code"] for r in selected)),
            "report_count": len({r["report_sha256"] for r in selected}),
            "first_target": min((r["target"] for r in selected), default=None),
            "last_target": max((r["target"] for r in selected), default=None),
            "classes": dict(Counter(r["actual_direction"] for r in selected)),
            "target_own_rows": sum(r["fund_code"] == "002112" for r in selected),
            "other_fund_rows": sum(r["fund_code"] != "002112" for r in selected),
        }
    return by_tag


def count():
    protocol = verify_protocol()
    reviewed = read_json(OUT / "reviewed-tags.json")
    receipt = read_json(OUT / "tags-freeze.json")
    # 修正抽取时保留初版与补充版，人工标签明确绑定最终判读所用的章节文件。
    if receipt["sections_file"] not in {"sections.json", "sections-reviewed.json"}:
        raise ValueError("UNREGISTERED_SECTION_FILE")
    section_path = OUT / receipt["sections_file"]
    sections = read_json(section_path)
    if receipt["tags_sha256"] != file_hash(OUT / "reviewed-tags.json") or receipt["sections_sha256"] != file_hash(
        section_path
    ):
        raise ValueError("REVIEWED_TAGS_CHANGED_AFTER_FREEZE")
    index = {r["report_sha256"]: r for r in reviewed}
    source_index = {r["source"]["report_sha256"]: r for r in sections}
    if len(index) != len(reviewed) or set(index) != set(source_index):
        raise ValueError("REPORT_CLASSIFICATION_COVERAGE_CHANGED")
    for h, item in index.items():
        if item["tag"] not in TAGS or not item["reason"]:
            raise ValueError("CLASSIFICATION_UNSUPPORTED")
        if item["evidence"] and item["evidence"] not in source_index[h]["section"]["text"]:
            raise ValueError("CLASSIFICATION_QUOTE_NOT_IN_SOURCE")
        if item["tag"] != "UNSPECIFIED" and not item["evidence"]:
            raise ValueError("CLASSIFICATION_REQUIRES_EXPLICIT_EVIDENCE")
    pools = {
        "final_5137": read_json(FROZEN / "inputs.json")["train"],
        "original_4419": read_json(FROZEN / "original-inputs.json")["train"],
    }
    final_index = {(r["fund_code"], r["target"]): r for r in pools["final_5137"]}
    for row in pools["original_4419"]:
        if final_index[(row["fund_code"], row["target"])] != row:
            raise ValueError("ORIGINAL_ROW_NOT_EXACTLY_PRESERVED")
    output, daily = {}, []
    for name, rows in pools.items():
        for row in rows:
            source = source_index[row["report_sha256"]]["source"]
            if source["fund_code"] != row["fund_code"] or not row["report_publication"] < row["target"] <= "2023-12-31":
                raise ValueError("ROW_SOURCE_TIME_CHANGED")
            if row["report_publication"] not in source["effective_publication_dates"]:
                raise ValueError("EFFECTIVE_PUBLICATION_CHANGED")
        folds = read_json(FROZEN / ("folds.json" if name == "final_5137" else "original-folds.json"))
        row_index = {(r["fund_code"], r["target"]): r for r in rows}
        output[name] = {
            "all": describe_rows(rows, index),
            "by_fund": {
                c: describe_rows([r for r in rows if r["fund_code"] == c], index) for c in protocol["scope"]["funds"]
            },
            "by_year": {
                year: describe_rows([r for r in rows if r["target"][:4] == year], index)
                for year in sorted({r["target"][:4] for r in rows})
            },
            "folds": {f["name"]: describe_rows([row_index[tuple(k)] for k in f["train_ids"]], index) for f in folds},
        }
        if name == "final_5137":
            daily = [
                {
                    "fund_code": r["fund_code"],
                    "target": r["target"],
                    "report_sha256": r["report_sha256"],
                    "report_publication": r["report_publication"],
                    "tag": index[r["report_sha256"]]["tag"],
                }
                for r in rows
            ]
    save_once(OUT / "coverage.json", output)
    save_once(OUT / "row-context.json", daily)
    return {name: value["all"] for name, value in output.items()}
