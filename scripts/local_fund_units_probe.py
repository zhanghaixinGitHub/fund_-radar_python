"""单位修复后的独立新公告验证：先冻结代码，三份完整 PDF 各调用一次。

前三批新公告已经暴露程序单位问题，不能继续称为未见样本。新选三份只用于
此阶段，不混入旧预测，不重发以前的失败请求。
"""

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from urllib.request import Request, urlopen

from scripts.local_announcement_document import extract_pdf
from scripts.local_fund_fact_chain import build_book, check_result, request_for
from scripts.local_fund_input_ablation import load_budget, once
from scripts.local_fund_pdf_units import checked_grid
from scripts.local_fund_repair_run import local_api, read, save, sha
from scripts.local_fund_validation_chain import verify_freeze

SOURCES = [
    ("现金管理", "2026-05-15", "1225305776"),
    ("股权激励", "2026-08-12", "1225468652"),
    ("分红规划", "2026-03-20", "1225019532"),
]


def prepare(root: Path, parent: Path, repair: Path):
    """先冻结本阶段代码和名单，再下载完整新 PDF；错误下载或读取不换题。"""
    verify_freeze(parent)
    if root.exists():
        raise ValueError("NEW_DIRECTORY_REQUIRED")
    root.mkdir(parents=True)
    code = Path(__file__).resolve()
    paths = [
        code,
        code.with_name("local_fund_pdf_units.py"),
        code.with_name("local_fund_fact_chain.py"),
        code.with_name("local_fund_pdf_grid.py"),
        code.parents[1] / "tests/test_local_fund_pdf_units.py",
    ]
    save(
        root / "code-freeze.json",
        {
            "frozen_at": datetime.now(UTC).isoformat(),
            "files": {str(p): sha(p) for p in paths},
            "new_full_pdf_bodies_not_used_for_rule_development": True,
        },
    )
    save(
        root / "selection.json",
        {
            "sources": SOURCES,
            "scope": "本地实验未使用过；不表示模型训练未见",
            "selection_rule": "前次官方搜索中每类尚未进入实验的首个截止前PDF；不根据模型回答选题",
        },
    )
    for path in paths:
        target = root / "source" / path.name
        target.parent.mkdir(exist_ok=True)
        target.write_bytes(path.read_bytes())
    sys.path.insert(0, str(repair / "pdf-dependencies"))
    budget = load_budget(repair)
    jobs, requests = [], {}
    history_urls = {read(p).get("url") for p in (parent / "documents").glob("*.json")}
    for i, (category, date, source_id) in enumerate(SOURCES, 1):
        job = f"new-unit-{i:02}"
        url = f"https://static.cninfo.com.cn/finalpage/{date}/{source_id}.PDF"
        if url in history_urls:
            raise ValueError("NEW_SOURCE_ALREADY_USED")
        path = root / "pdfs" / f"{job}.pdf"
        path.parent.mkdir(exist_ok=True)
        issues = []
        document = {
            "id": "HOLDOUT_UNIT:" + source_id,
            "title": category + "独立公告",
            "date": date,
            "available_at": date + "T23:59:59+08:00",
            "url": url,
            "body_scope": "FULL_TEXT",
            "body_truncated": False,
            "body": "",
        }
        try:
            with urlopen(Request(url, headers={"User-Agent": "Mozilla/5.0"}), timeout=60) as response:
                raw = response.read()
            if not raw.startswith(b"%PDF"):
                raise ValueError("NOT_PDF")
            with path.open("xb") as target:
                target.write(raw)
            packet = extract_pdf(path)
            document["body"] = "\n".join(page["raw_text"] for page in packet["pages"])
            grid = checked_grid(packet)
            save(root / "packets" / f"{job}.json", packet)
            save(root / "grids" / f"{job}.json", grid)
            book = build_book(document, grid)
            save(root / "books" / f"{job}.json", book)
            requests[job] = request_for(document, book, grid, budget)
        except Exception as error:
            issues.append(f"FIRST_SOURCE_FAILED_NO_REPLACEMENT:{type(error).__name__}:{error}")
        save(root / "documents" / f"{job}.json", document)
        jobs.append(
            {"id": job, "issues": issues, "ready": job in requests, "pdf_sha256": sha(path) if path.exists() else None}
        )
    save(root / "jobs.json", jobs)
    save(root / "requests.json", requests)
    save(
        root / "protocol.json",
        {
            "max_calls": 3,
            "retries": 0,
            "paid_calls": 0,
            "parent": str(parent.resolve()),
            "history_prediction_included": False,
            "production_switched": False,
            "model_digest": read(parent / "protocol.json")["model_digest"],
        },
    )
    save(
        root / "manifest.json",
        {
            "files": {str(p.resolve()): sha(p) for p in root.rglob("*") if p.is_file()},
            "code_files": read(root / "code-freeze.json")["files"],
        },
    )
    print(
        json.dumps({"prepared": len(jobs), "requests": len(requests), "issues": jobs}, ensure_ascii=False), flush=True
    )


def run(root: Path):
    """逐份第一次输出完整保存；比对的是程序事实，解释仍须原文独立核对。"""
    for filename, expected in read(root / "manifest.json")["files"].items():
        if sha(Path(filename)) != expected:
            raise ValueError("FROZEN_INPUT_CHANGED")
    for filename, expected in read(root / "manifest.json")["code_files"].items():
        if sha(Path(filename)) != expected:
            raise ValueError("FROZEN_CODE_CHANGED")
    requests = read(root / "requests.json")
    for job in read(root / "jobs.json"):
        target = root / "checks" / f"{job['id']}.json"
        if target.exists():
            continue
        if not job["ready"]:
            save(target, {**job, "status": "SOURCE_BLOCKED", "called": False})
            continue
        expected = read(root / "protocol.json")["model_digest"]
        actual = next(m for m in local_api("tags")["models"] if m["name"] == "qwen3.5:4b")
        if actual["digest"] != expected:
            raise ValueError("MODEL_CHANGED")
        raw = once(root, job["id"], requests[job["id"]])
        result = check_result(
            raw, read(root / "documents" / f"{job['id']}.json"), read(root / "books" / f"{job['id']}.json")
        )
        save(
            target,
            {
                **job,
                **result,
                "called": True,
                "elapsed_seconds": raw["elapsed_seconds"],
                "first_response_sha256": sha(root / "raw" / f"{job['id']}.json"),
            },
        )
        print(
            json.dumps({"job": job["id"], "status": result["status"], "issues": result["issues"]}, ensure_ascii=False),
            flush=True,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--repair", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    if not (args.output / "protocol.json").exists():
        prepare(args.output, args.parent, args.repair)
    if not args.prepare_only:
        run(args.output)
