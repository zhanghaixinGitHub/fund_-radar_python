"""冻结错题的四组本机配置对照；不改提示词、不补答案、不重试或发布预测。

历史4096输出预算的回答仅作参照。本轮四组统一8192输出预算，以免将预算变化
归因于开启思考。每组每份完整资料恰好一次；未覆盖、未完成和复核告警分别记录。
"""

import argparse
import hashlib
import json
import sys
import time
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path

from scripts.local_announcement_reader import call_local
from scripts.local_announcement_structured_reader import check_structured as check_pdf
from scripts.local_announcement_structured_reader import structured_request as pdf_request
from scripts.local_fund_analysis_probe import decode_local
from scripts.local_fund_repair_run import local_api, read, save, sha
from scripts.local_fund_semantic_audit import audit_response
from scripts.local_fund_semantics import manager_issues
from scripts.local_fund_table_guard import table_claim_issues
from scripts.local_fund_token_budget import LocalTokenBudget

VERSION = "LOCAL_FUND_CONFIG_3"
OUTPUT_TOKENS = 8192
MAX_CONTEXT = 65536
CONFIGS = {
    "off_zero": {"think": False, "temperature": 0.0, "top_p": 0.95},
    "on_zero": {"think": True, "temperature": 0.0, "top_p": 0.95},
    "off_recommended": {"think": False, "temperature": 0.7, "top_p": 0.8},
    "on_recommended": {"think": True, "temperature": 1.0, "top_p": 0.95},
}
REFERENCES = [
    "https://huggingface.co/Qwen/Qwen3.5-4B#best-practices",
    "https://docs.ollama.com/capabilities/thinking",
]


def digest(value: dict) -> str:
    """请求身份只用于冻结校验，不代替项目历史输入哈希协议。"""
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def configured_request(original: dict, config: str, budget: LocalTokenBudget) -> dict:
    """四组正文和schema逐字相同；输出预算、上下文和其余采样项逐项固定。

    num_predict包含运行器计入的思考与回答生成预算；不足时保存length，不补发。
    官方建议是候选设置，不是金融资料的正确性保证；本实验限制输出为8192。
    """
    payload = deepcopy(original)
    selected = CONFIGS[config]
    payload["think"] = selected["think"]
    payload["keep_alive"] = "10m"
    upper = budget.content_tokens(payload) + budget.count(json.dumps(payload["format"])) + 1024 + OUTPUT_TOKENS
    context = next((size for size in (16384, 32768, MAX_CONTEXT) if upper < size), None)
    if context is None:
        raise ValueError(f"LOCAL_FROZEN_FULL_INPUT_OVER_BUDGET:{upper}")
    payload["options"] = {
        "num_ctx": context,
        "num_predict": OUTPUT_TOKENS,
        "seed": 42,
        "temperature": selected["temperature"],
        "top_p": selected["top_p"],
        "top_k": 20,
        "min_p": 0.0,
        "presence_penalty": 1.5,
        "repeat_penalty": 1.0,
    }
    return payload


def once(root: Path, name: str, request: dict, transport=call_local) -> dict:
    """先落开始凭证；已完成原样复用，开始后无响应则禁止再次发送。

    网络错误可能意味着服务端仍在运行，保存错误后停止整个队列。格式或生成
    长度失败有确定响应，可以继续其他预先冻结的题，但绝不重试此题。
    """
    path = root / "raw" / f"{name}.json"
    started = root / "raw" / f"{name}-started.json"
    if path.exists():
        value = read(path)
        if value["request"] != request:
            raise ValueError("LOCAL_FROZEN_REQUEST_CHANGED")
        if value.get("transport_error"):
            raise ValueError("LOCAL_TRANSPORT_OUTCOME_UNKNOWN")
        return value
    if started.exists():
        raise ValueError("LOCAL_PREVIOUS_CALL_OUTCOME_UNKNOWN")
    if len(list((root / "raw").glob("*-started.json"))) >= 64:
        raise ValueError("LOCAL_CONFIG_CALL_CAP_EXCEEDED")
    save(started, {"request_hash": digest(request), "started_at": datetime.now(UTC).isoformat()})
    value = {"request": request}
    tick = time.perf_counter()
    try:
        value["response"] = transport(request)
    except Exception as error:
        value["transport_error"] = f"{type(error).__name__}: {error}"
        value["elapsed_seconds"] = round(time.perf_counter() - tick, 3)
        save(path, value)
        raise ValueError("LOCAL_TRANSPORT_OUTCOME_UNKNOWN") from error
    try:
        value["parsed"] = decode_local(value["response"], request)
    except Exception as error:
        value["error"] = f"{type(error).__name__}: {error}"
    value["elapsed_seconds"] = round(time.perf_counter() - tick, 3)
    save(path, value)
    return value


def prepare(root: Path, repair: Path, original: Path) -> None:
    """以全部15条旧错题确定14份全文，不按新模型答案筛选资料。

    经理资料、PDF版面是另两项独立任务，不混入历史002112预测。人工核对意见
    单独存放，绝不写入四组请求；相同资料各组只改变冻结的配置项。
    """
    if root.exists():
        raise ValueError("LOCAL_NEW_EXPERIMENT_DIRECTORY_REQUIRED")
    reviews = []
    for path in sorted(original.glob("manual-findings-*.json")):
        data = read(path)
        reviews.extend(data if isinstance(data, list) else [data])
    jobs = sorted({row["job"].split("-part-")[0] for row in reviews})
    if len(reviews) != 15 or len(jobs) != 14:
        raise ValueError("LOCAL_FROZEN_CASE_COUNT_CHANGED")
    snapshot = read(repair / "snapshot.json")
    documents = {f"doc-{i:02}": doc for i, doc in enumerate(snapshot["documents"], 1)}
    cutoff = datetime.fromisoformat(snapshot["as_of"])
    tasks = []
    for job in jobs:
        doc = documents[job]
        if datetime.fromisoformat(doc["available_at"]) > cutoff:
            raise ValueError("LOCAL_DOCUMENT_AFTER_CUTOFF")
        old = read(repair / "raw" / f"{job}.json")
        original_request = old["request"]
        catalogs = json.loads(original_request["messages"][1]["content"])["quote_catalogs"]
        if "".join(v["text"] for v in catalogs[doc["id"]].values()) != doc["body"]:
            raise ValueError("LOCAL_COMPLETE_SOURCE_NOT_PRESERVED")
        tasks.append({"id": job, "kind": "announcement", "document": doc, "request": original_request})
    manager = read(repair / "manager-source.json")
    tasks.append(
        {
            "id": "manager",
            "kind": "manager",
            "source": manager,
            "request": read(repair / "raw" / "manager-only.json")["request"],
        }
    )
    pdf = read(repair / "pdf-layout-01" / "document.json")
    if pdf["scope"] != "FULL_TEXT" or pdf["issues"] or not all(p["character_coverage_ok"] for p in pdf["pages"]):
        raise ValueError("LOCAL_PDF_PAGE_COVERAGE_FAILED")
    task = read(repair / "pdf-layout-01" / "protocol.json")
    request, units = pdf_request(task["source_pdf"].split("\\")[-1], pdf, task["questions"], citation_mode="ids_first")
    # 标题从已冻结请求中保留，重建units只用于引用核对，不改变模型输入。
    request = read(repair / "pdf-layout-01" / "request.json")
    tasks.append(
        {
            "id": "pdf-table",
            "kind": "pdf",
            "units": units,
            "source": pdf,
            "questions": task["questions"],
            "request": request,
        }
    )
    tags = local_api("tags")
    model = next(m for m in tags["models"] if m["name"] == "qwen3.5:4b")
    expected = read(repair / "protocol.json")["model_digest"]
    if model["digest"] != expected:
        raise ValueError("LOCAL_MODEL_DIGEST_CHANGED")
    show = local_api("show", {"model": "qwen3.5:4b"})
    if show.get("thinking", {}).get("values") != [False, True]:
        raise ValueError("LOCAL_THINKING_MODES_NOT_CONFIRMED")
    root.mkdir(parents=True)
    save(root / "cases.json", reviews)
    save(root / "tasks.json", tasks)
    save(root / "runtime-before.json", {"tags": tags, "show": show, "version": local_api("version")})
    protocol = {
        "version": VERSION,
        "frozen_at": datetime.now(UTC).isoformat(),
        "as_of": snapshot["as_of"],
        "target": snapshot["window"]["target_nav_date"],
        "model": "qwen3.5:4b",
        "model_digest": expected,
        "configs": CONFIGS,
        "references": REFERENCES,
        "output_tokens": OUTPUT_TOKENS,
        "max_context": MAX_CONTEXT,
        "tasks": len(tasks),
        "announcement_documents": len(jobs),
        "known_cases": len(reviews),
        "max_calls": 64,
        "repeats_per_config_task": 1,
        "retries": 0,
        "paid_calls": 0,
        "production_writes": False,
        "manager_separate_as_of": manager["as_of"],
        "gold_answers_in_requests": False,
        "historical_output_budget": 4096,
        "new_groups_output_budget": OUTPUT_TOKENS,
        "limitations": [
            "单次采样不能证明稳定性",
            "8192是实验预算，非官方建议的完整输出预算",
            "机器告警未命中不等于语义正确",
            "14份是冻结错题集，不代表公告总体正确率",
            "完整54份资料预测暂不重跑；先判断配置是否改善冻结错误",
        ],
    }
    save(root / "protocol.json", protocol)
    protected = {
        str(p.resolve()): sha(p) for directory in (repair, original) for p in directory.rglob("*") if p.is_file()
    }
    inputs = {str(p.resolve()): sha(p) for p in root.glob("*.json")}
    sources = [
        Path(__file__).resolve(),
        *(
            Path(__file__).parent / name
            for name in (
                "local_fund_analysis_probe.py",
                "local_fund_repair_run.py",
                "local_fund_semantic_audit.py",
                "local_fund_semantics.py",
                "local_fund_table_guard.py",
                "local_fund_token_budget.py",
                "local_announcement_reader.py",
                "local_announcement_structured_reader.py",
                "local_announcement_document.py",
                "local_announcement_layout.py",
                "local_announcement_continuations.py",
            )
        ),
    ]
    for source in sources:
        target = root / "source" / "scripts" / source.name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
    save(root / "manifest.json", {"historical": protected, "inputs": inputs, "code": {str(p): sha(p) for p in sources}})


def verify(root: Path) -> None:
    """旧实验与本轮输入完整校验；源码修改需另建实验，不能续跑旧请求。"""
    for group in read(root / "manifest.json").values():
        for name, expected in group.items():
            if sha(Path(name)) != expected:
                raise ValueError("LOCAL_PROTECTED_FILE_CHANGED:" + name)


def check_task(task: dict, raw: dict) -> dict:
    """只输出独立检查结果；不以模型的VERIFIED字段或自评宣布语义通过。"""
    if task["kind"] == "announcement":
        return audit_response(raw, task["document"])
    if task["kind"] == "manager":
        return {"issues": manager_issues(raw, task["source"]), "semantic_verified": False}
    checked = check_pdf(raw, task["units"], task["questions"], "FULL_TEXT")
    table_issues = []
    for answer in checked["answers"]:
        table_issues.extend(table_claim_issues(answer["answer"], answer["evidence"], task["units"]))
    return {"issues": checked["issues"], "table_issues": table_issues, "checked": checked, "semantic_verified": False}


def run(root: Path, repair: Path) -> None:
    """按题交错运行四配置，避免四组完全按时间分批；队列顺序事先固定。"""
    verify(root)
    # 依赖沿用上轮实验目录，不安装到后端环境，也不在受保护目录生成pyc。
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(repair / "dependencies"))
    budget = LocalTokenBudget(read(repair / "model-show.json")["model_info"])
    tasks = read(root / "tasks.json")
    for task in tasks:
        for config in CONFIGS:
            name = f"{task['id']}--{config}"
            print(json.dumps({"stage": "start", "task": task["id"], "config": config}), flush=True)
            request = configured_request(task["request"], config, budget)
            record = once(root, name, request)
            target = root / "checks" / f"{name}.json"
            if not target.exists():
                result = {
                    "task": task["id"],
                    "kind": task["kind"],
                    "config": config,
                    "error": record.get("error"),
                    "elapsed_seconds": record["elapsed_seconds"],
                    "issues": [],
                    "semantic_verified": False,
                }
                response = record["response"]
                content_tokens = budget.content_tokens(request)
                observed = response.get("prompt_eval_count", 0)
                runtime_input_ok = content_tokens - 128 <= observed <= content_tokens + 1024
                result.update(
                    {
                        "prompt_tokens": response.get("prompt_eval_count"),
                        "output_tokens": response.get("eval_count"),
                        "done_reason": response.get("done_reason"),
                        "thinking_present": bool(response.get("message", {}).get("thinking")),
                        "content_characters": len(response.get("message", {}).get("content", "")),
                        "content_tokens_reconstructed": content_tokens,
                        "runtime_input_count_ok": runtime_input_ok,
                    }
                )
                if not runtime_input_ok:
                    result["issues"].append("LOCAL_RUNTIME_INPUT_COUNT_MISMATCH")
                if "parsed" in record:
                    checked = check_task(task, record["parsed"])
                    checked["issues"] = result["issues"] + checked["issues"]
                    result.update(checked)
                save(target, result)
            print(
                json.dumps(
                    {
                        "stage": "complete",
                        **{
                            k: read(target)[k]
                            for k in ("task", "config", "error", "elapsed_seconds", "done_reason", "thinking_present")
                        },
                    }
                ),
                flush=True,
            )
    verify(root)
    rows = [read(p) for p in sorted((root / "checks").glob("*.json"))]
    summary = {
        "version": VERSION,
        "calls": len(rows),
        "retries": 0,
        "paid_calls": 0,
        "semantic_verified": False,
        "configs": {},
    }
    for config in CONFIGS:
        group = [r for r in rows if r["config"] == config]
        summary["configs"][config] = {
            "responses": len(group),
            "generation_failures": sum(bool(r["error"]) for r in group),
            "flagged": sum(bool(r["issues"] or r.get("table_issues")) for r in group),
            "thinking_present": sum(r["thinking_present"] for r in group),
            "call_seconds": round(sum(r["elapsed_seconds"] for r in group), 3),
        }
    save(root / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repair", type=Path, required=True)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    if not (args.output / "protocol.json").exists():
        prepare(args.output, args.repair, args.original)
    if not args.prepare_only:
        run(args.output, args.repair)


if __name__ == "__main__":
    main()
