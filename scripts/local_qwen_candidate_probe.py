"""9B 候选的一次有限验收：同输入旧题、完整新公告、PDF补充及独立经理。

不修改正式服务，不使用付费接口；数字及引用不要求模型复制程序字段。
旧题请求仅改变模型名，其余逐字复用。新版PDF补充另列，不冒充模型单变量改进。
"""

import argparse
import hashlib
import json
import shutil
import sys
import time
import urllib.error
import urllib.request
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path

from scripts.local_announcement_document import extract_pdf
from scripts.local_fund_input_ablation import QUESTIONS, SYSTEM, request_hash
from scripts.local_fund_pdf_units import checked_grid
from scripts.local_fund_repair_run import local_api, read, save, sha
from scripts.local_fund_token_budget import LocalTokenBudget

MODEL = "qwen3.5:9b-q4_K_M"
MAX_CALLS = 20  # 19项基础验收；只有独立复核通过才允许另一次完整综合。
NEW_SOURCES = [
    {
        "date": "2026-04-08",
        "source_id": "1225082578",
        "category": "分红规划",
        "questions": [
            "现金分红需要同时满足哪些条件？年度与三年累计下限的比例和基数分别是什么？",
            "不同发展阶段与重大资金支出如何共同决定现金分红比例？重大资金支出的定义是什么？",
        ],
    },
    {
        "date": "2026-07-01",
        "source_id": "1225403353",
        "category": "员工持股进展",
        "questions": ["截至公告所述日期，持股计划已获哪些审批、实际买入股份进展如何？公告能否确认开户状态？"],
    },
    {
        "date": "2026-06-03",
        "source_id": "1225349605",
        "category": "境外发行备案",
        "questions": [
            "备案确认了什么，发行数量上限及原单位是什么？材料真实性和投资收益是否得到保证？",
            "发行上市还需要哪些审批？备案后的时间要求与后续手续是什么？",
        ],
    },
]


def verify(root: Path) -> None:
    """开跑及结束均核对冻结代码、原实验和原始输入；新回答绝不覆盖旧档。"""
    for filename, expected in read(root / "code-freeze.json")["files"].items():
        if sha(Path(filename)) != expected:
            raise ValueError("CODE_CHANGED_AFTER_FREEZE:" + filename)
    if (root / "manifest.json").exists():
        for group in read(root / "manifest.json").values():
            for filename, expected in group.items():
                if sha(Path(filename)) != expected:
                    raise ValueError("FROZEN_INPUT_CHANGED:" + filename)


def freeze(root: Path) -> None:
    """正文读取前固定代码、三个新公告名单和退出条件。"""
    if root.exists():
        raise ValueError("NEW_EXPERIMENT_DIRECTORY_REQUIRED")
    root.mkdir(parents=True)
    code = Path(__file__).resolve()
    files = [
        code,
        code.with_name("local_fund_pdf_units.py"),
        code.with_name("local_fund_pdf_grid.py"),
        code.with_name("local_announcement_document.py"),
        code.with_name("local_announcement_layout.py"),
        code.with_name("local_fund_input_ablation.py"),
        code.with_name("local_fund_token_budget.py"),
        code.parents[1] / "tests/test_local_qwen_candidate_probe.py",
    ]
    save(
        root / "code-freeze.json",
        {
            "frozen_at": datetime.now(UTC).isoformat(),
            "new_pdf_full_bodies_not_used_to_develop_probe": True,
            "files": {str(path): sha(path) for path in files},
        },
    )
    save(
        root / "selection.json",
        {
            "sources": NEW_SOURCES,
            "selection_rule": "三类固定；官方检索中此前本地实验未用的首个合格2026年PDF，依据来源和日期选，失败不换题",
            "unseen_scope": "本地实验未用，不表示预训练未见过；检索摘要不作为正文或正确答案输入",
        },
    )
    save(
        root / "acceptance-policy.json",
        {
            "max_calls": MAX_CALLS,
            "base_calls": 19,
            "retries": 0,
            "paid_calls": 0,
            "critical_semantic_errors_allowed": 0,
            "critical_condition_omissions_allowed": 0,
            "fabricated_supporting_quotes_allowed": 0,
            "model_self_grading_used": False,
            "required_independent_review": True,
            "synthesis_only_after_base_acceptance": True,
            "runtime_failure_is_not_semantic_failure": True,
            "production_switched": False,
        },
    )
    for path in files:
        target = root / "source" / path.parent.name / path.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)


def clone_old(request: dict) -> dict:
    """同一旧题仅替换模型名；提示词、正文、问题、参数和容量逐项不变。"""
    revised = deepcopy(request)
    revised["model"] = MODEL
    assert {k: v for k, v in revised.items() if k != "model"} == {k: v for k, v in request.items() if k != "model"}
    return revised


def natural_request(document: dict, questions: list[str], budget, reference: dict, supplement="") -> dict:
    """完整原文与问题直接给模型；资料不裁剪，表格补充不是人工正确答案。"""
    metadata = {key: document.get(key) for key in ("id", "title", "date", "body_scope", "body_truncated")}
    content = "公告信息：" + json.dumps(metadata, ensure_ascii=False, separators=(",", ":"))
    content += "\n请回答以下问题：\n" + "\n".join(f"{i}. {q}" for i, q in enumerate(questions, 1))
    content += "\n<公告资料>\n" + document["body"] + "\n</公告资料>"
    if supplement:
        content += "\n同源PDF列位置补充（程序据几何提取，并非原文逐字引句）：\n" + supplement
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": content}]
    upper = sum(budget.count(m["content"]) for m in messages) + 1024 + 8192
    context = next((size for size in (16384, 32768, 65536) if upper < size), None)
    if context is None:
        raise ValueError("COMPLETE_SOURCE_OVER_BUDGET_NO_TRUNCATION")
    options = deepcopy(reference["options"])
    options["num_ctx"] = context
    return {
        "model": MODEL,
        "stream": False,
        "think": False,
        "keep_alive": "10m",
        "messages": messages,
        "options": options,
    }


def prepare(root: Path, legacy: Path, chain: Path, repair: Path) -> None:
    """下载完整新PDF后冻结全部请求；旧人工评审文件不进入模型输入。"""
    verify(root)
    if (root / "protocol.json").exists():
        raise ValueError("EXPERIMENT_ALREADY_PREPARED")
    sys.path.insert(0, str(repair / "pdf-dependencies"))
    sys.path.insert(0, str(repair / "dependencies"))
    runtime = {"tags": local_api("tags"), "version": local_api("version")}
    model = next(m for m in runtime["tags"]["models"] if m["name"].lower() == MODEL.lower())
    show = local_api("show", {"model": MODEL, "verbose": True})
    save(root / "model-show.json", show)
    budget = LocalTokenBudget(show["model_info"])
    old = read(legacy / "requests.json")
    tasks = {task["id"]: task for task in read(legacy / "tasks.json")}
    jobs, requests = [], {}
    # 先用一份短的真实错题检验是否能加载，不额外发“你好”占用或掩盖第一次结果。
    order = ["doc-17"] + [name for name in QUESTIONS if name != "doc-17"]
    for name in order:
        original = old[name + "--full"]
        request = clone_old(original)
        if budget.content_tokens(request) + 1024 + request["options"]["num_predict"] >= request["options"]["num_ctx"]:
            raise ValueError("OLD_EXACT_REQUEST_OVER_RUNTIME_BUDGET:" + name)
        requests[name] = request
        save(root / "documents" / f"{name}.json", tasks[name]["document"])
        jobs.append(
            {
                "id": name,
                "group": "old_same_input",
                "ready": True,
                "legacy_request_hash": request_hash(original),
                "same_except_model": True,
                "body_sha256": hashlib.sha256(tasks[name]["document"]["body"].encode()).hexdigest(),
            }
        )
    if not (root / "source-acquisition.json").exists():
        fetch_sources(root, legacy, chain, repair)
    acquisition = {row["id"]: row for row in read(root / "source-acquisition.json")["results"]}
    for i, source in enumerate(NEW_SOURCES, 1):
        name = f"new-{i:02}"
        document = read(root / "documents" / f"{name}.json")
        issues = list(acquisition[name]["issues"])
        try:
            if not document["body"]:
                raise ValueError("FIRST_SOURCE_FAILED_NOT_SENT")
            requests[name] = natural_request(document, source["questions"], budget, old["doc-17--full"])
        except Exception as error:
            issues.append(f"SOURCE_REQUEST_NOT_READY:{type(error).__name__}:{error}")
        jobs.append({"id": name, "group": "unseen_full_text", "ready": name in requests, "issues": issues})
    # 表格恢复是不同输入的独立任务，不用它覆盖doc-32原纯文本第一次回答。
    packet = read(chain / "pdf-packets/doc-32.json")
    grid = checked_grid(packet)
    name = "pdf-layout"
    document = tasks["doc-32"]["document"]
    save(root / "documents" / f"{name}.json", document)
    save(root / "grids" / f"{name}.json", grid)
    requests[name] = natural_request(
        document, [q for _, q in QUESTIONS["doc-32"]], budget, old["doc-32--full"], grid["rendered"]
    )
    jobs.append({"id": name, "group": "pdf_supplement_different_input", "ready": True})
    manager = read(chain / "manager-facts.json")
    document = {
        "id": "manager-separate",
        "title": "独立经理资料",
        "date": manager["as_of"][:10],
        "body_scope": "FULL_TEXT",
        "body_truncated": False,
        "body": json.dumps(manager, ensure_ascii=False),
    }
    save(root / "documents/manager.json", document)
    requests["manager"] = natural_request(
        document,
        [
            "公开离任原因是什么？资料能否确认已经转任其他岗位？",
            "分别列出陆阳参与管理全期、共同管理子区间和独立管理区间的日期、回撤及历史正收益窗口占比，说明归属范围。",
            "这些统计能够证明个人超额能力或目标日预测胜率吗？有哪些证据限制？",
        ],
        budget,
        old["doc-17--full"],
    )
    jobs.append({"id": "manager", "group": "separate_manager", "ready": True, "historical_prediction_included": False})
    save(root / "requests.json", requests)
    save(root / "jobs.json", jobs)
    save(root / "runtime-before.json", runtime)
    save(
        root / "protocol.json",
        {
            "model": MODEL,
            "model_digest": model["digest"],
            "max_calls": MAX_CALLS,
            "retries": 0,
            "paid_calls": 0,
            "historical_cutoff": read(legacy / "protocol.json")["as_of"],
            "historical_target": "2026-10-12",
            "manager_as_of": manager["as_of"],
            "history_source_count": 14,
            "fresh_source_count": 3,
            "per_request_timeout_seconds": 900,
            "no_input_truncation": True,
            "production_switched": False,
        },
    )
    protected = {
        str(path.resolve()): sha(path) for folder in (legacy, chain) for path in folder.rglob("*") if path.is_file()
    }
    inputs = {str(path.resolve()): sha(path) for path in root.rglob("*") if path.is_file()}
    save(root / "manifest.json", {"previous_files": protected, "input_files": inputs})
    print(
        json.dumps({"prepared": len(jobs), "ready": len(requests), "digest": model["digest"]}, ensure_ascii=False),
        flush=True,
    )


def fetch_sources(root: Path, legacy: Path, chain: Path, repair: Path) -> None:
    """权重下载期间独立取得全文，每份新源至多下载一次，失败不换公告。"""
    verify(root)
    if (root / "source-acquisition.json").exists():
        print("FROZEN_SOURCES_ALREADY_ACQUIRED")
        return
    sys.path.insert(0, str(repair / "pdf-dependencies"))
    used_urls = {task["document"].get("url") for task in read(legacy / "tasks.json")}
    used_urls.update(read(path).get("url") for path in (chain / "documents").glob("*.json"))
    used_urls.update(read(path).get("url") for path in (chain / "units-holdout-01/documents").glob("*.json"))
    results = []
    for i, source in enumerate(NEW_SOURCES, 1):
        name = f"new-{i:02}"
        url = f"https://static.cninfo.com.cn/finalpage/{source['date']}/{source['source_id']}.PDF"
        if url in used_urls:
            raise ValueError("HELDOUT_ALREADY_USED")
        document = {
            "id": "HELDOUT_9B:" + source["source_id"],
            "title": source["category"],
            "date": source["date"],
            "url": url,
            "body": "",
            "body_scope": "FULL_TEXT",
            "body_truncated": False,
            "availability_note": "本轮下载，仅独立验证，未加入旧预测",
        }
        receipt = root / "acquisition" / f"{name}.json"
        if receipt.exists():
            results.append(read(receipt))
            continue
        started = root / "acquisition" / f"{name}-started.json"
        if started.exists():
            raise ValueError("PREVIOUS_SOURCE_DOWNLOAD_UNKNOWN_NO_RETRY")
        save(started, {"url": url, "started_at": datetime.now(UTC).isoformat()})
        issues, pages = [], 0
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=60) as response:
                pdf = response.read()
            if not pdf.startswith(b"%PDF"):
                raise ValueError("NOT_PDF")
            path = root / "new-pdfs" / f"{name}.pdf"
            path.parent.mkdir(exist_ok=True)
            with path.open("xb") as target:
                target.write(pdf)
            packet = extract_pdf(path)
            grid = checked_grid(packet)
            save(root / "packets" / f"{name}.json", packet)
            save(root / "grids" / f"{name}.json", grid)
            document["body"] = "\n".join(page["raw_text"] for page in packet["pages"])
            pages = packet["total_pages"]
        except Exception as error:
            issues.append(f"FIRST_SOURCE_FAILED_NO_REPLACEMENT:{type(error).__name__}:{error}")
        save(root / "documents" / f"{name}.json", document)
        result = {"id": name, "url": url, "pages": pages, "body_characters": len(document["body"]), "issues": issues}
        save(receipt, result)
        results.append(result)
    save(root / "source-acquisition.json", {"results": results, "retries": 0, "failed_sources_not_replaced": True})
    print(json.dumps(results, ensure_ascii=False), flush=True)


def local_chat(request: dict) -> dict:
    """只访问本机固定端点；为9B首轮固定900秒上限，没有自动重试。"""
    req = urllib.request.Request(
        "http://127.0.0.1:11434/api/chat",
        data=json.dumps(request, ensure_ascii=False).encode(),
        headers={"Content-Type": "application/json"},
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(req, timeout=900) as response:
        return json.load(response)


def once(root: Path, name: str, transport=local_chat) -> dict:
    """每个冻结请求只发一次；未知传输结果不续发，首次失败也永久保存。"""
    request = read(root / "requests.json")[name]
    target = root / "raw" / f"{name}.json"
    started = root / "raw" / f"{name}-started.json"
    if target.exists():
        value = read(target)
        if value["request"] != request or value.get("transport_unknown"):
            raise ValueError("FIRST_REQUEST_CHANGED_OR_OUTCOME_UNKNOWN")
        return value
    if started.exists():
        raise ValueError("FIRST_OUTCOME_UNKNOWN_NO_RETRY")
    if len(list((root / "raw").glob("*-started.json"))) >= read(root / "protocol.json")["max_calls"]:
        raise ValueError("CALL_CAP_EXCEEDED")
    save(started, {"request_hash": request_hash(request), "started_at": datetime.now(UTC).isoformat()})
    value, tick = {"request": request}, time.perf_counter()
    try:
        response = transport(request)
        value["response"] = response
        errors = []
        if response.get("model", "").lower() != MODEL.lower():
            errors.append("MODEL_IDENTITY_MISMATCH")
        if response.get("done_reason") != "stop" or not response.get("done"):
            errors.append("GENERATION_INCOMPLETE")
        if response.get("prompt_eval_count", 0) + response.get("eval_count", 0) >= request["options"]["num_ctx"]:
            errors.append("CONTEXT_EXHAUSTED")
        if not response.get("message", {}).get("content", "").strip():
            errors.append("CONTENT_EMPTY")
        value["errors"] = errors
    except urllib.error.HTTPError as error:
        value.update(
            server_error={"status": error.code, "body": error.read().decode(errors="replace")},
            errors=["SERVER_REJECTED_FIRST_REQUEST"],
        )
    except Exception as error:
        value.update(
            transport_unknown=True,
            transport_error=f"{type(error).__name__}:{error}",
            errors=["TRANSPORT_OUTCOME_UNKNOWN_NO_RETRY"],
        )
    value["elapsed_seconds"] = round(time.perf_counter() - tick, 3)
    save(target, value)
    if value.get("transport_unknown"):
        raise ValueError("TRANSPORT_OUTCOME_UNKNOWN_NO_RETRY")
    return value


def run(root: Path) -> None:
    """完成固定19项并停在人工验收门；错误不改变题目、不降低标准。"""
    verify(root)
    if (root / "runtime-stop.json").exists():
        print("FIRST_SERVER_REJECTION_RETAINED_NO_FURTHER_CALLS")
        return
    rows = []
    for job in read(root / "jobs.json"):
        name = job["id"]
        path = root / "checks" / f"{name}.json"
        if path.exists():
            rows.append(read(path))
            continue
        if not job["ready"]:
            result = {**job, "status": "SOURCE_FAILED", "model_called": False}
        else:
            print(json.dumps({"stage": "start", "job": name}), flush=True)
            model = next(m for m in local_api("tags")["models"] if m["name"].lower() == MODEL.lower())
            if model["digest"] != read(root / "protocol.json")["model_digest"]:
                raise ValueError("MODEL_DIGEST_CHANGED")
            raw = once(root, name)
            result = {
                **job,
                "status": "GENERATION_FAILED" if raw["errors"] else "INDEPENDENT_REVIEW_REQUIRED",
                "errors": raw["errors"],
                "model_called": True,
                "elapsed_seconds": raw["elapsed_seconds"],
                "first_response_sha256": sha(root / "raw" / f"{name}.json"),
                "semantic_passed": False,
                "first_response_preserved": True,
            }
            save(root / "runtime" / f"{name}.json", local_api("ps"))
        save(path, result)
        rows.append(result)
        print(
            json.dumps(
                {"stage": "complete", "job": name, "status": result["status"], "errors": result.get("errors", [])},
                ensure_ascii=False,
            ),
            flush=True,
        )
        if job["ready"] and raw.get("server_error"):
            # 服务器已明确拒绝加载，停止后续调用，不能把硬件失败归为读错。
            save(
                root / "runtime-stop.json",
                {"job": name, "reason": "FIRST_SERVER_REJECTION", "remaining_not_called": True},
            )
            break
    verify(root)
    summary = {
        "calls": sum(row["model_called"] for row in rows),
        "results": rows,
        "paid_calls": 0,
        "retries": 0,
        "base_acceptance_pending": True,
        "synthesis_not_run": True,
        "production_switched": False,
    }
    if not (root / "generation-summary.json").exists():
        save(root / "generation-summary.json", summary)
    elif read(root / "generation-summary.json") != summary:
        raise ValueError("FIRST_SUMMARY_CHANGED")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--legacy", type=Path)
    parser.add_argument("--chain", type=Path)
    parser.add_argument("--repair", type=Path)
    parser.add_argument("--freeze-code", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--fetch-sources", action="store_true")
    args = parser.parse_args()
    if args.freeze_code:
        freeze(args.output)
    elif args.fetch_sources:
        fetch_sources(args.output, args.legacy, args.chain, args.repair)
    else:
        if not (args.output / "protocol.json").exists():
            prepare(args.output, args.legacy, args.chain, args.repair)
        if not args.prepare_only:
            run(args.output)
