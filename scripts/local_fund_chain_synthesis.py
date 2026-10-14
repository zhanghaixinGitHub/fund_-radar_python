"""历史账本的无损分组综合，独立于已经冻结的新公告提取实验。

只去掉重复字段、重复常量与重复原句，每条事实都能还原。新样本和较晚经理资料
不读入；不修改首次超预算回执，不用删公告或加大窗口掩盖输入预算问题。
"""

import argparse
import json
import re
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from app.schemas.fund_information_analysis import Analysis, validate_analysis
from app.services.fund_information_analysis import SYNTHESIS_PROMPT

from scripts.local_fund_input_ablation import once
from scripts.local_fund_pdf_units import checked_grid, rebind_book
from scripts.local_fund_repair_run import local_api, read, save, sha
from scripts.local_fund_synthesis_guard import synthesis_issues
from scripts.local_fund_token_budget import LocalTokenBudget
from scripts.local_fund_validation_chain import verify_freeze

# 综合契约仅输出简短诊断，保留固定生成空间；全部证据仍在请求中。
# 该预算在第一次综合调用前冻结，不能用生成失败后的加量重试冒充通过。
SYNTHESIS_OUTPUT_TOKENS = 3840


def pack_books(books: dict) -> dict:
    """以公告/类型分组，相同字段值只声明一次；原句去重后使用整数引用。

    常量判定按 JSON 字面量比较，保留空值和布尔类型，避免 true 等同于 1。
    不根据真假、方向或难度筛选任何行。
    """
    documents = {}
    for job, book in books.items():
        by_kind, groups, quotes = {}, {}, []
        for fact in book["facts"]:
            by_kind.setdefault(fact["kind"], []).append(fact)
        for kind, facts in by_kind.items():
            fields = list(facts[0]["data"])
            if any(list(f["data"]) != fields for f in facts):
                raise ValueError("FACT_FIELDS_INCONSISTENT")
            constants = {
                key: facts[0]["data"][key]
                for key in fields
                if len({json.dumps(f["data"][key], sort_keys=True) for f in facts}) == 1
            }
            columns = [key for key in fields if key not in constants]
            rows = []
            for fact in facts:
                if fact["quote"] not in quotes:
                    quotes.append(fact["quote"])
                context_refs = []
                for context in fact.get("scope_context", []):
                    if context not in quotes:
                        quotes.append(context)
                    context_refs.append(quotes.index(context))
                rows.append(
                    [fact["id"], [fact["data"][key] for key in columns], quotes.index(fact["quote"]), context_refs]
                )
            groups[kind] = {"columns": columns, "constants": constants, "rows": rows}
        documents[job] = {
            "scope": book["source_scope"],
            "truncated": book["source_truncated"],
            "quotes": quotes,
            "groups": groups,
        }
    result = {"row_columns": ["fact_id", "column_values", "quote_index", "scope_quote_indices"], "documents": documents}
    # 解压结果与所有原数据、原句及来源范围完全比较；任何丢失立刻中止。
    restored = unpack_books(json.loads(json.dumps(result, ensure_ascii=False)))
    expected = {
        job + ":" + fact["id"]: [
            fact["kind"],
            fact["data"],
            fact["quote"],
            book["source_scope"],
            book["source_truncated"],
            fact.get("scope_context", []),
        ]
        for job, book in books.items()
        for fact in book["facts"]
    }
    if json.dumps(restored, sort_keys=True, ensure_ascii=False) != json.dumps(
        expected, sort_keys=True, ensure_ascii=False
    ):
        raise ValueError("LOSSLESS_FACT_PACK_FAILED")
    return result


def unpack_books(packed: dict) -> dict:
    """恢复所有行，供离线测试与审计核对；不是再次让模型摘要原文。"""
    result = {}
    for job, doc in packed["documents"].items():
        for kind, group in doc["groups"].items():
            for identifier, values, quote_index, context_refs in group["rows"]:
                data = {**group["constants"], **dict(zip(group["columns"], values, strict=True))}
                result[job + ":" + identifier] = [
                    kind,
                    data,
                    doc["quotes"][quote_index],
                    doc["scope"],
                    doc["truncated"],
                    [doc["quotes"][index] for index in context_refs],
                ]
    return result


def share_strings(value: dict) -> dict:
    """跨公告复用完全相同的长字符串；不改词、不作模型摘要，保持 JSON 类型。"""
    counts = Counter()

    def collect(item):
        if isinstance(item, str):
            counts[item] += 1
        elif isinstance(item, list):
            for child in item:
                collect(child)
        elif isinstance(item, dict):
            if set(item) == {"s"} and isinstance(item["s"], int):
                raise ValueError("SHARED_STRING_TAG_COLLISION")
            for child in item.values():
                collect(child)

    collect(value)
    strings = [text for text, count in counts.items() if count > 1 and len(text) > 12]
    lookup = {text: index for index, text in enumerate(strings)}

    def encode(item):
        if isinstance(item, str) and item in lookup:
            return {"s": lookup[item]}
        if isinstance(item, list):
            return [encode(child) for child in item]
        if isinstance(item, dict):
            return {key: encode(child) for key, child in item.items()}
        return item

    packed = {"shared_strings": strings, "value": encode(value)}
    if json.dumps(unshare_strings(packed), sort_keys=True) != json.dumps(value, sort_keys=True):
        raise ValueError("SHARED_STRING_ROUNDTRIP_FAILED")
    return packed


def unshare_strings(packed: dict) -> dict:
    """还原字符串字典；字典引用与数字、空值、原数组明确分开。"""

    def decode(item):
        if isinstance(item, dict):
            if set(item) == {"s"}:
                return packed["shared_strings"][item["s"]]
            return {key: decode(child) for key, child in item.items()}
        if isinstance(item, list):
            return [decode(child) for child in item]
        return item

    return decode(packed["value"])


def scope_context(fact: dict, document: dict, packet: dict | None) -> list[str]:
    """携带已有表头及条件完整句，避免综合只见到单个金额或悬空审批主句。

    仅选择直接对应的原文，不补造主体、期间或价格效应。档案已有表前上下文
    若本来只是摘录，明确标注；不能宣称它是完整条款。
    """
    result = []
    body, quote = document["body"], fact["quote"]
    if fact["kind"] in ("annual_average_basis", "literal_condition") and quote in body:
        start = body.index(quote)
        begin = max(body.rfind("。", 0, start), body.rfind("；", 0, start)) + 1
        end = body.find("。", start + len(quote)) if not quote.endswith("。") else start + len(quote) - 1
        if end < 0:
            end = len(body) - 1
        # 决定是否继续执行之后的列举属于同一条件组，带到下一明确小节。
        if "由股东会决定" in quote:
            boundary = re.search(r"\n\s*[（(]\s*[一二三四五六七八九十]+\s*[）)]", body[start + len(quote) :])
            end = start + len(quote) + boundary.start() - 1 if boundary else len(body) - 1
        result.append(body[begin : end + 1])
    for reference in fact.get("source_refs", []):
        if reference.get("scope"):
            result.append("原账本表前上下文摘录，范围未自动扩展：" + reference["scope"])
        if reference.get("table_rows"):
            result.extend(
                "原账本同年表前上下文摘录：" + row["preceding_scope_context"] for row in reference["table_rows"]
            )
        if packet and reference.get("headers"):
            catalog = {row["id"]: row for page in packet["pages"] for table in page["tables"] for row in table["rows"]}
            for key in reference["headers"]:
                result.append(
                    "同源PDF原表头："
                    + json.dumps(
                        [
                            {field: cell[field] for field in ("column_start", "column_end", "text")}
                            for cell in catalog[key]["cells"]
                        ],
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                )
    return list(dict.fromkeys(result))


def prepare(root: Path, parent: Path, repair: Path):
    """历史输入保持原哈希；另存本阶段首次请求及代码，最多一次本机调用。"""
    verify_freeze(parent)
    if root.exists():
        raise ValueError("NEW_SYNTHESIS_DIRECTORY_REQUIRED")
    root.mkdir(parents=True)
    sys.path.insert(0, str(repair / "dependencies"))
    budget = LocalTokenBudget(read(repair / "model-show.json")["model_info"])
    books = {f.stem: read(f) for f in sorted((parent / "books").glob("doc-*.json"))}
    for job, book in books.items():
        packet_path = parent / "pdf-packets" / f"{job}.json"
        packet = read(packet_path) if packet_path.exists() else None
        if packet:
            verified_grid = checked_grid(packet)
            save(root / "unit-checked-grids" / f"{job}.json", verified_grid)
            book = rebind_book(book, verified_grid)
            books[job] = book
        document = read(parent / "documents" / f"{job}.json")
        for fact in book["facts"]:
            fact["scope_context"] = scope_context(fact, document, packet)
    packed = pack_books(books)
    shared = share_strings(packed)
    registry = read(parent / "synthesis-facts.json")
    for job, book in books.items():
        for fact in book["facts"]:
            registry[job + ":" + fact["id"]]["text"] = fact["canonical"] + "\n原文：" + fact["quote"]
            if fact["scope_context"]:
                registry[job + ":" + fact["id"]]["text"] += "\n原文范围与表头：\n" + "\n".join(fact["scope_context"])
    snapshot = read(repair / "snapshot.json")
    timing = read(repair / "time-context.json")
    base = {key: value for key, value in snapshot["facts"].items() if value["category"] != "公告"}
    schema = Analysis.model_json_schema()
    schema["properties"].pop("evidence_units", None)
    # 不把千余个引用编号重复塞入 schema；程序仍按完整 registry 严格检查引用。
    metadata = {
        key: snapshot[key] for key in ("fund_code", "window", "as_of", "latest_nav_date", "inventory", "overflow")
    }
    metadata["time_context"] = timing
    messages = [
        {
            "role": "system",
            "content": SYNTHESIS_PROMPT + "\n这是待人工复核的历史诊断草稿。"
            "公告按 document 和事实类型分组，columns 声明行内字段，constants 是每行共有字段；"
            "row_columns 为事实编号、字段值、quotes 中的原句序号、范围和表头原句序号列表。"
            '完整引用写作 document:fact_id。shared_strings 保存重复长文字，value 中形如{"s":n}'
            "的对象代表该字典第n条完整字符串，按原字符串解释。"
            "所有字段和原句都保留，不能扩大条件或把未绑定主体期间的 literal_quantity 自行归属。"
            "程序事实只证明原值和有限关系，不能证明目标日价格效应。新样本及较晚经理资料均未纳入。",
        },
        {
            "role": "user",
            "content": json.dumps(
                {"metadata": metadata, "non_announcement_facts": base, "all_54_announcement_books": shared},
                ensure_ascii=False,
                separators=(",", ":"),
            ),
        },
    ]
    content_tokens = budget.content_tokens({"messages": messages})
    upper = content_tokens + budget.count(json.dumps(schema, ensure_ascii=False)) + 1024 + SYNTHESIS_OUTPUT_TOKENS
    ready = upper < 65536
    request = {
        "model": "qwen3.5:4b",
        "stream": False,
        "think": False,
        "keep_alive": "10m",
        "format": schema,
        "messages": messages,
        "options": {"num_ctx": 65536, "num_predict": SYNTHESIS_OUTPUT_TOKENS, "seed": 42, "temperature": 0},
    }
    save(
        root / "budget.json",
        {
            "content_tokens_reconstructed": content_tokens,
            "upper_with_reserve": upper,
            "all_documents": len(books),
            "all_announcement_facts": len(unpack_books(packed)),
            "lossless_roundtrip": True,
            "ready": ready,
            "previous_blocked_receipt_preserved": True,
        },
    )
    save(root / "requests.json", {"synthesis": request} if ready else {})
    save(
        root / "protocol.json",
        {
            "max_calls": 1,
            "retries": 0,
            "paid_calls": 0,
            "history_cutoff": snapshot["as_of"],
            "target": snapshot["window"]["target_nav_date"],
            "model_digest": read(parent / "protocol.json")["model_digest"],
            "frozen_at": datetime.now(UTC).isoformat(),
            "holdout_read": False,
            "production_switched": False,
            "pdf_units_independently_checked": True,
        },
    )
    save(root / "facts.json", registry)
    save(root / "packed.json", packed)
    save(root / "shared.json", shared)
    code = Path(__file__).resolve()
    save(
        root / "manifest.json",
        {
            "files": {
                str(p.resolve()): sha(p)
                for p in [code, Path(__file__).with_name("local_fund_pdf_units.py"), *root.rglob("*.json")]
            },
            "parent_manifest_sha256": sha(parent / "manifest.json"),
        },
    )
    (root / "source.py").write_bytes(code.read_bytes())
    print(json.dumps(read(root / "budget.json")), flush=True)


def run(root: Path, parent: Path, repair: Path):
    """只发送该完整历史请求的首次版本；首次错误不改写成验收通过。"""
    verify_freeze(parent)
    for filename, expected in read(root / "manifest.json")["files"].items():
        if sha(Path(filename)) != expected:
            raise ValueError("FROZEN_SYNTHESIS_INPUT_CHANGED")
    if (root / "check.json").exists():
        print(json.dumps(read(root / "check.json"), ensure_ascii=False))
        return
    if not read(root / "budget.json")["ready"]:
        save(root / "check.json", {"status": "BUDGET_BLOCKED", "called": False})
        return
    protocol = read(root / "protocol.json")
    model = next(m for m in local_api("tags")["models"] if m["name"] == "qwen3.5:4b")
    if model["digest"] != protocol["model_digest"]:
        raise ValueError("MODEL_CHANGED")
    raw = once(root, "synthesis", read(root / "requests.json")["synthesis"])
    issues = list(raw["error"] or [])
    try:
        parsed = json.loads(raw["response"]["message"]["content"])
        validate_analysis(parsed, read(root / "facts.json"), read(repair / "time-context.json"))
        issues.extend(
            i["code"]
            for i in synthesis_issues(
                parsed,
                read(root / "facts.json"),
                read(repair / "snapshot.json")["companies"],
                read(repair / "time-context.json"),
            )
        )
    except Exception as error:
        issues.append(f"SYNTHESIS_CONTRACT:{type(error).__name__}:{error}")
    save(
        root / "check.json",
        {
            "status": "DIAGNOSTIC_ONLY",
            "issues": issues,
            "called": True,
            "elapsed_seconds": raw["elapsed_seconds"],
            "runtime_prompt_tokens": raw["response"].get("prompt_eval_count"),
            "formal_output_adopted": False,
            "independent_prediction_passed": False,
            "first_response_sha256": sha(root / "raw/synthesis.json"),
        },
    )
    print(json.dumps(read(root / "check.json"), ensure_ascii=False), flush=True)


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
        run(args.output, args.parent, args.repair)
