"""本机完整验证驱动：历史54份、独立新样本、经理及综合诊断，均不发布。

只使用项目解释器和已有实验依赖。不重试、不静默删资料、不复用DeepSeek公告
提取结果。程序事实及原引用与模型首次草稿分开保存，未核验解释不进入安全输出。
"""

import argparse
import hashlib
import json
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path

from app.schemas.fund_information_analysis import Analysis, validate_analysis
from app.services.fund_information_analysis import SYNTHESIS_PROMPT

from scripts.local_fund_fact_chain import build_book, check_result, request_for
from scripts.local_fund_input_ablation import once
from scripts.local_fund_pdf_grid import read_pdf
from scripts.local_fund_repair_run import local_api, read, save, sha
from scripts.local_fund_synthesis_guard import synthesis_issues
from scripts.local_fund_token_budget import LocalTokenBudget

SOURCE_FILES = (
    "local_fund_pdf_grid.py",
    "local_fund_fact_chain.py",
    "local_fund_validation_chain.py",
    "local_fund_verified_facts.py",
    "local_fund_filing_guard.py",
    "local_fund_semantic_audit.py",
    "local_announcement_document.py",
    "local_announcement_layout.py",
    "local_fund_input_ablation.py",
)


def verify_freeze(root: Path) -> None:
    """代码在读取新样本正文之前冻结；准备和运行时再次核对。"""
    for filename, expected in read(root / "code-freeze.json")["files"].items():
        if sha(Path(filename)) != expected:
            raise ValueError("LOCAL_CODE_CHANGED_AFTER_HOLDOUT_FREEZE:" + filename)
    manifest = root / "manifest.json"
    if manifest.exists():
        for group in read(manifest).values():
            for filename, expected in group.items():
                if sha(Path(filename)) != expected:
                    raise ValueError("LOCAL_FROZEN_INPUT_CHANGED:" + filename)


def freeze_code(root: Path) -> None:
    """只新增冻结文件；已有实验和代码副本绝不覆盖。"""
    if root.exists():
        raise ValueError("LOCAL_NEW_DIRECTORY_REQUIRED")
    root.mkdir(parents=True)
    repo = Path(__file__).resolve().parents[1]
    paths = [repo / "scripts" / name for name in SOURCE_FILES]
    paths += [
        repo / "tests" / name
        for name in (
            "test_local_fund_pdf_grid.py",
            "test_local_fund_fact_chain.py",
            "test_local_fund_validation_chain.py",
        )
    ]
    save(
        root / "code-freeze.json",
        {
            "frozen_at": datetime.now(UTC).isoformat(),
            "files": {str(p): sha(p) for p in paths},
            "heldout_body_not_used_to_develop_rules": True,
        },
    )
    for source in paths:
        target = root / "source" / source.parent.name / source.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def compressed_evidence(books: dict) -> tuple[str, dict]:
    """按类型声明一次字段，逐行保存全部值和引句；无损往返，不裁剪条件。

    PDF坐标与计算过程保留在完整账本文件，模型综合仍取得每项数据和原始引句。
    本格式只消除重复JSON键名，不根据方向或难度筛选事实。
    """
    layouts, rows, registry = {}, [], {}
    for job, book in books.items():
        for fact in book["facts"]:
            kind = fact["kind"]
            columns = list(fact["data"])
            if kind in layouts and layouts[kind] != columns:
                raise ValueError("FACT_TYPE_LAYOUT_CHANGED")
            layouts[kind] = columns
            ref = job + ":" + fact["id"]
            values = [fact["data"][key] for key in columns]
            row = [ref, kind, values, fact["quote"], book["source_scope"], book["source_truncated"]]
            rows.append(row)
            registry[ref] = {
                "category": "公告",
                "fact_id": ref,
                "text": fact["canonical"] + "\n原文：" + fact["quote"],
                "relation": "来自冻结公告" + book["document_id"] + "；原句范围不扩大",
                "body_scope": book["source_scope"],
                "body_truncated": book["source_truncated"],
                "direction_eligible": False,
                "required_in_analysis": False,
                "identity_status": "PROGRAM_SOURCE_ONLY",
                "weight_policy": "NO_AUTOMATIC_PRICE_EFFECT",
            }
    value = {
        "type_columns": layouts,
        "row_columns": ["id", "kind", "values", "quote", "scope", "truncated"],
        "rows": rows,
    }
    encoded = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    decoded = json.loads(encoded)
    for row in decoded["rows"]:
        ref, kind, values, quote, _, _ = row
        data = dict(zip(decoded["type_columns"][kind], values, strict=True))
        if registry[ref]["text"] != json.dumps(data, ensure_ascii=False) + "\n原文：" + quote:
            raise ValueError("ALL_FACT_EVIDENCE_ROUNDTRIP_FAILED")
    return encoded, registry


def summary_request(data: dict, books: dict, timing: dict, budget) -> tuple[dict, dict, dict]:
    """全部历史账本和冻结的非公告程序事实综合；不混入新样本或较晚经理数据。"""
    encoded, registry = compressed_evidence(books)
    base = {key: value for key, value in data["facts"].items() if value["category"] != "公告"}
    registry.update(base)
    schema = Analysis.model_json_schema()
    schema["properties"].pop("evidence_units", None)
    schema["$defs"]["Reason"]["properties"]["refs"]["items"]["enum"] = list(registry)
    metadata = {key: data[key] for key in ("fund_code", "window", "as_of", "latest_nav_date", "inventory", "overflow")}
    metadata.update(time_context=timing, gaps=["公告解释未独立验收，程序账本不证明价格影响；原摘录/截断边界保留"])
    prompt = SYNTHESIS_PROMPT + (
        "\n这里只作诊断草稿。程序账本各类型的字段由type_columns说明，rows逐项完整包含数值、条件和原始引句。"
        "literal_quantity只验证原单位和换算，主体/期间未绑定，不得自行归给某公司某年。"
        "所有公告原句的条件、否定和范围必须保留，不能把原文事实改成确定的目标日价格影响。"
        "新公告样本和10月10日独立经理快照未纳入。不得输出已验证胜率或假称可以发布。"
    )
    messages = [
        {"role": "system", "content": prompt},
        {
            "role": "user",
            "content": "冻结时间与范围：\n"
            + json.dumps(metadata, ensure_ascii=False, separators=(",", ":"))
            + "\n原始非公告程序事实：\n"
            + json.dumps(base, ensure_ascii=False, separators=(",", ":"))
            + "\n全部54份公告程序账本：\n"
            + encoded,
        },
    ]
    tokens = sum(budget.count(m["content"]) for m in messages)
    upper = tokens + budget.count(json.dumps(schema, ensure_ascii=False)) + 1024 + 8192
    context = next((size for size in (32768, 65536) if upper < size), None)
    receipt = {
        "content_tokens_reconstructed": tokens,
        "upper_with_schema_template_output": upper,
        "all_history_documents": len(books),
        "all_fact_count": len(registry),
        "roundtrip_checked": True,
        "max_context": 65536,
        "native_capacity_not_inferred": True,
    }
    if context is None:
        return {}, registry, {**receipt, "status": "BUDGET_BLOCKED_NO_SOURCE_REMOVED"}
    request = {
        "model": "qwen3.5:4b",
        "stream": False,
        "think": False,
        "keep_alive": "10m",
        "format": schema,
        "messages": messages,
        "options": {"num_ctx": context, "num_predict": 8192, "seed": 42, "temperature": 0},
    }
    return request, registry, {**receipt, "status": "READY", "num_ctx": context}


def prepare(root: Path, repair: Path, previous: Path, new_sources: Path) -> None:
    """先源文件/版面/账本，后冻结全部请求；从不向模型传人工错题答案。"""
    verify_freeze(root)
    if (root / "protocol.json").exists():
        raise ValueError("PREPARED_EXPERIMENT_ALREADY_EXISTS")
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(repair / "pdf-dependencies"))
    sys.path.insert(0, str(repair / "dependencies"))
    budget = LocalTokenBudget(read(repair / "model-show.json")["model_info"])
    data = read(repair / "snapshot.json")
    cutoff = datetime.fromisoformat(data["as_of"])
    documents = [(f"doc-{i:02}", doc, "history") for i, doc in enumerate(data["documents"], 1)]
    fresh = read(new_sources)
    if len(fresh) != 6:
        raise ValueError("FROZEN_HOLDOUT_COUNT_CHANGED")
    sources = read(root / "pdf-map.json")
    for i, source in enumerate(fresh, 1):
        documents.append((f"new-{i:02}", source["document"], "holdout"))
        sources[source["document"]["id"]] = source["pdf_path"]
    runtime = {"tags": local_api("tags"), "version": local_api("version")}
    model = next(m for m in runtime["tags"]["models"] if m["name"] == "qwen3.5:4b")
    if model["digest"] != read(repair / "protocol.json")["model_digest"]:
        raise ValueError("LOCAL_MODEL_DIGEST_CHANGED")
    requests, jobs, books = {}, [], {}
    for job, original, group in documents:
        doc = dict(original)
        if datetime.fromisoformat(doc["available_at"]) > cutoff:
            raise ValueError("DOCUMENT_AFTER_HISTORY_CUTOFF")
        grid, packet, issues = None, None, []
        pdf_path = sources.get(doc["id"])
        if pdf_path:
            try:
                packet, grid = read_pdf(Path(pdf_path))
                save(root / "pdf-packets" / f"{job}.json", packet)
                save(root / "pdf-grids" / f"{job}.json", grid)
                if group == "holdout":
                    doc["body"] = "\n".join(page["raw_text"] for page in packet["pages"])
            except Exception as error:
                issues.append(f"PDF_SOURCE_FAILED:{type(error).__name__}:{error}")
        save(root / "documents" / f"{job}.json", doc)
        book = build_book(doc, grid)
        book["pdf_grid_issues"].extend(issues)
        save(root / "books" / f"{job}.json", book)
        if group == "history":
            books[job] = book
        try:
            if not doc["body"].strip():
                raise ValueError("EMPTY_SOURCE_NOT_SENT_TO_MODEL")
            request = request_for(doc, book, grid, budget)
            requests[job] = request
        except ValueError as error:
            issues.append(str(error))
        jobs.append(
            {
                "id": job,
                "kind": "document",
                "group": group,
                "source_issues": issues,
                "source_body_sha256": hashlib.sha256(doc["body"].encode()).hexdigest(),
                "request_ready": job in requests,
            }
        )
    # 重新执行现有经理程序，核对原始净值、日历和公开理由；不采用较晚数据预测旧目标日。
    from app.services.direction_1d_protocol import calendar, digest

    from scripts.local_fund_verified_facts import manager_facts, manager_text

    source = read(repair / "manager-source.json")
    original = read(Path(source["source_path"]))
    if original["hash"] != source["source_hash"] or digest(original["payload"]) != source["source_hash"]:
        raise ValueError("MANAGER_FROZEN_SOURCE_CHANGED")
    days, version = calendar()
    manager = manager_facts(
        source, original["payload"], tuple(str(d) for d in days if str(d) <= source["as_of"][:10]), version
    )
    save(root / "manager-facts.json", manager)
    save(
        root / "manager-safe-text.json",
        {"text": manager_text(manager), "origin": "PROGRAM_TEMPLATE", "historical_prediction_included": False},
    )
    manager_doc = {
        "id": "separate-manager",
        "title": "独立经理资料",
        "body_scope": "FULL_TEXT",
        "body_truncated": False,
        "body": json.dumps(source, ensure_ascii=False),
    }
    manager_book = {
        "facts": [
            {
                "id": f"M{i:02}",
                "kind": "manager_scope",
                "data": metric,
                "quote": "程序从冻结净值重算，非公告原文引句",
                "required": True,
                "canonical": json.dumps(metric, ensure_ascii=False),
            }
            for i, metric in enumerate(manager["metrics"], 1)
        ],
        "source_scope": "FULL_TEXT",
        "source_truncated": False,
        "pdf_grid_issues": [],
        "document_id": manager_doc["id"],
    }
    manager_book["facts"].insert(0, {
        "id": "M00", "kind": "manager_disclosure", "required": True,
        "data": {"reason": manager["reason_source"]["groups"]["value"], "transfer_status": manager["transfer_status"]},
        "quote": manager["reason_source"]["raw"] + "\n" + manager["transfer_source"]["raw"],
        "canonical": "公开离任理由及转任披露状态，不能将未披露改成确定发生"})
    save(root / "documents/manager.json", manager_doc)
    save(root / "books/manager.json", manager_book)
    requests["manager"] = request_for(manager_doc, manager_book, None, budget)
    jobs.append(
        {"id": "manager", "kind": "document", "group": "separate_manager", "source_issues": [], "request_ready": True}
    )
    request, registry, receipt = summary_request(data, books, read(repair / "time-context.json"), budget)
    save(root / "synthesis-facts.json", registry)
    save(root / "synthesis-budget.json", receipt)
    if request:
        requests["synthesis"] = request
    jobs.append(
        {
            "id": "synthesis",
            "kind": "synthesis",
            "group": "history",
            "source_issues": [],
            "request_ready": bool(request),
        }
    )
    save(root / "requests.json", requests)
    save(root / "jobs.json", jobs)
    save(root / "runtime-before.json", runtime)
    save(
        root / "protocol.json",
        {
            "version": "LOCAL_VALIDATION_CHAIN_5",
            "as_of": data["as_of"],
            "target": data["window"]["target_nav_date"],
            "model_digest": model["digest"],
            "max_calls": 62,
            "retries": 0,
            "paid_calls": 0,
            "history_documents": 54,
            "heldout_documents": 6,
            "manager_separate_as_of": source["as_of"],
            "later_manager_and_holdout_excluded_from_synthesis": True,
            "program_assisted_not_model_independent_comprehension": True,
            "partial_sources_preserved": True,
            "production_switched": False,
            "metadata_available_at_is_frozen_source_marker": True,
        },
    )
    protected = {
        str(p.resolve()): sha(p) for directory in (previous, repair) for p in directory.rglob("*") if p.is_file()
    }
    inputs = {
        str(p.resolve()): sha(p)
        for p in root.rglob("*.json")
        if not any(t in p.parts for t in ("raw", "checks", "progress"))
    }
    for path in sources.values():
        inputs[str(Path(path).resolve())] = sha(Path(path))
    save(root / "manifest.json", {"previous_runs": protected, "frozen_inputs": inputs})
    print(
        json.dumps(
            {"prepared_documents": 60, "prepared_requests": len(requests), "synthesis_budget": receipt},
            ensure_ascii=False,
        ),
        flush=True,
    )


def run(root: Path, repair: Path) -> None:
    """一次顺序执行，检查后仅输出程序事实；原草稿和失败永久保留。"""
    verify_freeze(root)
    requests = read(root / "requests.json")
    expected = read(root / "protocol.json")["model_digest"]
    all_results = []
    for job in read(root / "jobs.json"):
        name = job["id"]
        target = root / "checks" / f"{name}.json"
        if target.exists():
            all_results.append(read(target))
            continue
        if not job["request_ready"]:
            result = {**job, "status": "INPUT_OR_BUDGET_BLOCKED", "issues": job["source_issues"], "model_called": False}
        else:
            actual = next(m for m in local_api("tags")["models"] if m["name"] == "qwen3.5:4b")
            if actual["digest"] != expected:
                raise ValueError("LOCAL_MODEL_DIGEST_CHANGED")
            print(json.dumps({"stage": "start", "job": name, "group": job["group"]}), flush=True)
            raw = once(root, name, requests[name])
            if job["kind"] == "document":
                result = {
                    **job,
                    **check_result(
                        raw, read(root / "documents" / f"{name}.json"), read(root / "books" / f"{name}.json")
                    ),
                }
            else:
                facts = read(root / "synthesis-facts.json")
                timing = read(repair / "time-context.json")
                issues = list(raw["error"] or [])
                try:
                    parsed = json.loads(raw["response"]["message"]["content"])
                    validate_analysis(parsed, facts, timing)
                    issues.extend(
                        i["code"]
                        for i in synthesis_issues(parsed, facts, read(repair / "snapshot.json")["companies"], timing)
                    )
                except Exception as error:
                    issues.append(f"SYNTHESIS_CONTRACT:{type(error).__name__}:{error}")
                result = {
                    **job,
                    "status": "DIAGNOSTIC_ONLY",
                    "issues": issues,
                    "upstream_rejections": [
                        r["id"] for r in all_results if r["group"] == "history" and r["status"] == "BLOCKED"
                    ],
                    "independent_full_pipeline_passed": False,
                    "formal_output_adopted": False,
                }
            result.update(
                model_called=True,
                elapsed_seconds=raw["elapsed_seconds"],
                first_response_sha256=sha(root / "raw" / f"{name}.json"),
            )
        save(target, result)
        all_results.append(result)
        print(
            json.dumps(
                {"stage": "complete", "job": name, "status": result["status"], "issues": result.get("issues", [])},
                ensure_ascii=False,
            ),
            flush=True,
        )
    verify_freeze(root)
    summary = {
            "jobs": len(all_results),
            "calls": sum(r.get("model_called", False) for r in all_results),
            "retries": 0,
            "paid_calls": 0,
            "results": [
                {k: r.get(k) for k in ("id", "group", "status", "issues", "elapsed_seconds")} for r in all_results
            ],
            "explanation_auto_adopted": False,
            "production_switched": False,
        }
    if not (root / "summary.json").exists():
        save(root / "summary.json", summary)
    elif read(root / "summary.json") != summary:
        raise ValueError("FINAL_SUMMARY_CHANGED")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repair", type=Path)
    parser.add_argument("--previous", type=Path)
    parser.add_argument("--new-sources", type=Path)
    parser.add_argument("--freeze-code", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    if args.freeze_code:
        freeze_code(args.output)
        return
    if not (args.output / "protocol.json").exists():
        prepare(args.output, args.repair, args.previous, args.new_sources)
    if not args.prepare_only:
        run(args.output, args.repair)


if __name__ == "__main__":
    main()
