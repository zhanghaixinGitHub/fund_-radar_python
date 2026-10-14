"""全量独立证据的无损文本封装；不把JSON转义换行的额外开销当资料删减理由。

标题、范围、资格、事实和引句全部可重建。只在前一综合请求尚未发送时调用一次；
上游语义检查失败时输出永远只是诊断候选。经理的较晚快照不进入此历史请求。
"""

import argparse
import hashlib
import json
import re
import sys
from copy import deepcopy
from pathlib import Path

from app.schemas.fund_information_analysis import Analysis, validate_analysis
from app.services.fund_information_analysis import SYNTHESIS_PROMPT

from scripts.local_fund_evidence_packet import fact_packet
from scripts.local_fund_repair_run import call_once, read, save, verify
from scripts.local_fund_token_budget import LocalTokenBudget

VERSION = "LOCAL_EVIDENCE_TEXT_1"
SOURCE_FIELDS = ("id", "date", "title", "body_scope", "body_truncated")


class TextTokenBudget(LocalTokenBudget):
    """正文及schema均用同一词表计数，仍另留1024模板与完整输出，不提高64K。"""

    def receipt(self, request: dict, response: dict | None = None) -> dict:
        receipt = super().receipt(request, response)
        schema_text = json.dumps(request.get("format", {}), ensure_ascii=False)
        receipt["previous_schema_bytes_upper"] = receipt["upper_with_reserve"]
        receipt["schema_tokens"] = self.count(schema_text)
        receipt["schema_utf8_bytes"] = len(schema_text.encode())
        receipt["upper_with_reserve"] = (
            receipt["content_tokens"] + receipt["schema_tokens"] + 1024 + request["options"]["num_predict"]
        )
        receipt["method"] = "GGUF_BPE_MESSAGES_AND_SCHEMA_PLUS_TEMPLATE_1024_AND_FULL_OUTPUT"
        return receipt


def encode_packet(packet: dict) -> str:
    """引句按字符长度封装，原样保留CR/LF；仅将重复键名表示为一次声明的列名。"""
    facts = deepcopy(packet["facts"])
    for fact in facts.values():
        if "text_parts" in fact:
            if any(set(part) != {"source", "quote"} for part in fact["text_parts"]):
                raise ValueError("LOCAL_PACKET_PART_FIELDS_INVALID")
            fact["text_parts"] = [[part["source"], part["quote"]] for part in fact["text_parts"]]
    if any(set(source) != set(SOURCE_FIELDS) for source in packet["sources"].values()):
        raise ValueError("LOCAL_PACKET_SOURCE_FIELDS_INVALID")
    header = {
        "facts": facts,
        "part_columns": ["source", "quote"],
        "source_columns": list(SOURCE_FIELDS),
        "sources": {key: [row[field] for field in SOURCE_FIELDS] for key, row in packet["sources"].items()},
    }
    header_text = json.dumps(header, ensure_ascii=False, separators=(",", ":"))
    parts = [VERSION + "\n", f"HEADER {len(header_text)}\n", header_text, "\n"]
    for key, text in packet["quote_pool"].items():
        if not re.fullmatch(r"P\d{4}", key):
            raise ValueError("LOCAL_PACKET_QUOTE_KEY_INVALID")
        parts.extend((f"QUOTE {key} {len(text)}\n", text, "\n"))
    encoded = "".join(parts)
    if decode_packet(encoded) != packet:
        raise ValueError("LOCAL_PACKET_ROUNDTRIP_FAILED")
    return encoded


def decode_packet(encoded: str) -> dict:
    """长度而非内容分隔引句，正文中的HEADER/QUOTE字样不能改变资料边界。"""
    prefix = VERSION + "\n"
    if not encoded.startswith(prefix):
        raise ValueError("LOCAL_PACKET_VERSION_INVALID")
    cursor = len(prefix)
    line_end = encoded.find("\n", cursor)
    match = re.fullmatch(r"HEADER (\d+)", encoded[cursor:line_end])
    if match is None:
        raise ValueError("LOCAL_PACKET_HEADER_INVALID")
    start = line_end + 1
    end = start + int(match[1])
    header = json.loads(encoded[start:end])
    if (
        encoded[end : end + 1] != "\n"
        or header["part_columns"] != ["source", "quote"]
        or header["source_columns"] != list(SOURCE_FIELDS)
    ):
        raise ValueError("LOCAL_PACKET_COLUMNS_INVALID")
    facts = deepcopy(header["facts"])
    for fact in facts.values():
        if "text_parts" in fact:
            fact["text_parts"] = [{"source": part[0], "quote": part[1]} for part in fact["text_parts"]]
    sources = {key: dict(zip(SOURCE_FIELDS, row, strict=True)) for key, row in header["sources"].items()}
    pool, cursor = {}, end + 1
    while cursor < len(encoded):
        line_end = encoded.find("\n", cursor)
        match = re.fullmatch(r"QUOTE (P\d{4}) (\d+)", encoded[cursor:line_end])
        if match is None or match[1] in pool:
            raise ValueError("LOCAL_PACKET_QUOTE_FRAME_INVALID")
        start, end = line_end + 1, line_end + 1 + int(match[2])
        if encoded[end : end + 1] != "\n":
            raise ValueError("LOCAL_PACKET_QUOTE_LENGTH_INVALID")
        pool[match[1]] = encoded[start:end]
        cursor = end + 1
    return {"facts": facts, "sources": sources, "quote_pool": pool}


def text_request(data: dict, facts: dict, timing: dict, packet: dict, budget: LocalTokenBudget) -> dict:
    """同一64K上限，完整原文不转义换行；正文计数仍另留schema、模板与4096输出。"""
    if not budget.calibrated:
        raise ValueError("LOCAL_TOKENIZER_NOT_CALIBRATED")
    schema = Analysis.model_json_schema()
    schema["properties"].pop("evidence_units", None)
    schema["$defs"]["Reason"]["properties"]["refs"]["items"]["enum"] = list(facts)
    context = {
        "fund_code": data["fund_code"],
        "window": data["window"],
        "as_of": data["as_of"],
        "latest_nav_date": data["latest_nav_date"],
        "time_context": timing,
        "inventory": data["inventory"],
        "gaps": [],
        "overflow": data["overflow"],
    }
    prompt = SYNTHESIS_PROMPT + (
        "\n输入是完整原文的无损封装，没有删事实：HEADER的facts是事实目录；有text直接阅读，"
        "text_parts的每对编号依次表示公告sources、完整引句QUOTE。source_columns说明公告每列含义。"
        "每段QUOTE后的原文按长度保持，段落换行只属于排版；不能忽略否定、条件或单位。"
        "refs只能使用facts的F编号。D/P编号仅定位资料，不能填入refs。"
        "同公告多个引句及同事项多个文件不能重复计权；上游存在疑点仍须保留不确定性。"
    )
    messages = [
        {"role": "system", "content": prompt},
        {
            "role": "user",
            "content": "TIME_AND_INVENTORY\n"
            + json.dumps(context, ensure_ascii=False, separators=(",", ":"))
            + "\nFULL_EVIDENCE\n"
            + encode_packet(packet),
        },
    ]
    count = sum(budget.count(m["content"]) for m in messages)
    # format由Ollama作为结构约束使用，仍为它预留整份schema的分词开销；
    # 不能继续用UTF-8字节数冒充token，也不通过减少正文或输出空间越过预算。
    upper = count + budget.count(json.dumps(schema, ensure_ascii=False)) + 1024 + 4096
    num_ctx = next((size for size in (16384, 32768, 65536) if upper < size), None)
    if num_ctx is None:
        raise ValueError(f"LOCAL_FULL_TEXT_PACKET_TOO_LARGE:{upper}")
    return {
        "model": "qwen3.5:4b",
        "stream": False,
        "think": False,
        "keep_alive": "10m",
        "format": schema,
        "options": {"temperature": 0, "seed": 42, "num_ctx": num_ctx, "num_predict": 4096},
        "messages": messages,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    root, output = args.run.resolve(), args.output.resolve()
    if output.exists():
        parser.error("输出目录已存在；不重复生成，不覆盖首次候选。")
    verify(root)
    if (root / "raw/independent-synthesis-started.json").exists() and not args.prepare_only:
        parser.error("前一综合已经发送；不能通过再次生成冒充首次通过。")
    sys.path.insert(0, str(root / "dependencies"))
    budget = TextTokenBudget(read(root / "model-show.json")["model_info"])
    pairs = [
        (read(root / f"raw/calibration-{i}.json")["request"], read(root / f"raw/calibration-{i}.json")["response"])
        for i in (1, 2)
    ]
    budget.calibrate(pairs)
    data, facts, timing = (
        read(root / "snapshot.json"),
        read(root / "independent-facts.json"),
        read(root / "time-context.json"),
    )
    packet = fact_packet(facts, data["documents"])
    request = text_request(data, facts, timing, packet, budget)
    save(output / "synthesis-request.json", request)
    save(output / "packet.json", packet)
    receipt = {
        "version": VERSION,
        "all_fact_keys": list(facts),
        "all_texts_reconstructed": True,
        "quote_count": len(packet["quote_pool"]),
        "source_count": len(packet["sources"]),
        "token_budget": budget.receipt(request),
        "parent_run": str(root),
        "snapshot_input_hash": read(root / "protocol.json")["input_hash"],
        "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "paid_calls": 0,
        "production_writes": False,
        "upstream_rejections_preserved": True,
    }
    save(output / "protocol.json", receipt)
    print(json.dumps(receipt, ensure_ascii=False), flush=True)
    if not args.prepare_only:
        record = call_once(output, "independent-synthesis", request, budget=budget)
        candidate = {
            "status": "DIAGNOSTIC_ONLY",
            "analysis": None,
            "issues": [],
            "independent_inputs": True,
            "independent_full_pipeline_passed": False,
        }
        try:
            if record.get("error"):
                raise ValueError(record["error"])
            candidate["analysis"] = validate_analysis(record["parsed"], facts, timing)
        except Exception as error:
            candidate["issues"] = [f"{type(error).__name__}: {error}"]
        save(output / "candidate-analysis.json", candidate)
        verify(root)
        print(json.dumps(candidate, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
