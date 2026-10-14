"""002112 本地替代验证的只读部件：本机请求、完整片段、同契约校验，不发布预测。

原始快照由独立只读导出取得。本模块不访问数据库、不调用付费接口；实验驱动脚本
必须保存输入/源码哈希与所有首次响应。缺文、超长或失败都不能伪装成完整分析。
"""

import json
from copy import deepcopy

from app.schemas.fund_information_analysis import Analysis, Event, quote_catalog
from app.services.fund_information_analysis import EVENT_PROMPT, SYNTHESIS_PROMPT

from scripts.local_announcement_reader import MODEL

VERSION = "local-fund-analysis-probe-v1"
MAX_CONTEXT = 65536
PREDICT_TOKENS = 4096


def request_payload(prompt: str, value: dict, schema: dict) -> dict:
    """UTF-8 字节上界预留完整 JSON 输出；不静默截断正文，不根据答案调整上下文。"""
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": json.dumps(value, ensure_ascii=False, separators=(",", ":"))},
    ]
    upper = sum(len(m["content"].encode("utf-8")) for m in messages)
    upper += len(json.dumps(schema, ensure_ascii=False).encode("utf-8")) + PREDICT_TOKENS + 1024
    context = next((size for size in (16384, 32768, MAX_CONTEXT) if upper < size), None)
    if context is None:
        raise ValueError(f"LOCAL_FULL_INPUT_TOO_LARGE:{upper}")
    return {
        "model": MODEL,
        "stream": False,
        "think": False,
        "keep_alive": "10m",
        "format": schema,
        "options": {"temperature": 0, "seed": 42, "num_ctx": context, "num_predict": PREDICT_TOKENS},
        "messages": messages,
    }


def event_schema(identifier: str, quote_ids: list[str]) -> dict:
    """约束文档身份及可引用编号，不向模型提供预期的事实、事件阶段或方向。"""
    event = Event.model_json_schema()
    event["properties"].pop("quotes")
    event["properties"]["quote_ids"] = {
        "type": "array",
        "minItems": 1,
        "maxItems": 3,
        "items": {"type": "string", "enum": quote_ids},
    }
    event["required"] = [key for key in event["required"] if key != "quotes"] + ["quote_ids"]
    item = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "id": {"type": "string", "enum": [identifier]},
            "events": {"type": "array", "minItems": 1, "maxItems": 4, "items": event},
        },
        "required": ["id", "events"],
    }
    return {
        "type": "object",
        "properties": {"items": {"type": "array", "minItems": 1, "maxItems": 1, "items": item}},
        "required": ["items"],
        "additionalProperties": False,
    }


def document_requests(document: dict) -> list[dict]:
    """按原顺序覆盖全部引文片段；正文只发送一次，不在 body 与引文目录里重复。

    超过预算才在既有原文片段边界拆分，分段显式标为摘录，保留原全局编号及偏移。
    没有按问题挑选“好读”的正文。分段不是全文理解等价物，须单独记录局限。
    """
    catalog = quote_catalog(document["body"])
    if "".join(row["text"] for row in catalog.values()) != document["body"]:
        raise ValueError("LOCAL_SOURCE_COVERAGE_FAILED")
    metadata = {
        key: deepcopy(document[key])
        for key in ("id", "title", "date", "codes", "kind", "body_scope", "body_truncated")
        if key in document
    }

    def build(selected: dict, split: bool) -> dict:
        doc = {**metadata, "body_location": "quote_catalogs：按编号顺序连接即为本次提供的全部原文"}
        if split:
            doc.update(body_scope="VERIFIED_EXCERPT", body_truncated=True)
        value = {"documents": [doc], "quote_catalogs": {document["id"]: selected}}
        prompt = EVENT_PROMPT + "\n原文完整保存在quote_catalogs中，按编号顺序阅读；不另重复发送body。"
        return request_payload(prompt, value, event_schema(document["id"], list(selected)))

    try:
        payload = build(catalog, False)
        return [
            {"request": payload, "quote_ids": list(catalog), "split": False, "start": 0, "end": len(document["body"])}
        ]
    except ValueError as error:
        if not str(error).startswith("LOCAL_FULL_INPUT_TOO_LARGE:"):
            raise
    groups, selected = [], {}
    for key, row in catalog.items():
        extended = {**selected, key: row}
        try:
            build(extended, True)
            selected = extended
        except ValueError as error:
            if not selected or not str(error).startswith("LOCAL_FULL_INPUT_TOO_LARGE:"):
                raise
            groups.append(selected)
            selected = {key: row}
    groups.append(selected)
    return [
        {
            "request": build(group, True),
            "quote_ids": list(group),
            "split": True,
            "start": next(iter(group.values()))["start"],
            "end": list(group.values())[-1]["end"],
        }
        for group in groups
    ]


def model_facts(facts: dict) -> dict:
    """仅省略重复审计索引与来源 URL，完整保留事实文字、关系、范围和资格标记。

    原事实包另行保存；不截短 text、不挑选方向、不删除低权重公司。原文证据仍在 text。
    """
    fields = (
        "category",
        "text",
        "relation",
        "fact_id",
        "direction_eligible",
        "required_in_analysis",
        "body_scope",
        "body_truncated",
        "identity_status",
        "weight_policy",
    )
    return {ref: {key: deepcopy(fact[key]) for key in fields if key in fact} for ref, fact in facts.items()}


def synthesis_request(data: dict, facts: dict, timing: dict, gaps: list[str]) -> dict:
    """固定完整事实目录进入综合判断；超过本地容量就失败，不以删资料换取成功。"""
    schema = Analysis.model_json_schema()
    schema["properties"].pop("evidence_units", None)
    schema["$defs"]["Reason"]["properties"]["refs"]["items"]["enum"] = list(facts)
    value = {
        "fund_code": data["fund_code"],
        "window": data["window"],
        "as_of": data["as_of"],
        "latest_nav_date": data["latest_nav_date"],
        "time_context": timing,
        "facts": model_facts(facts),
        "inventory": data["inventory"],
        "gaps": gaps,
        "overflow": data["overflow"],
    }
    return request_payload(SYNTHESIS_PROMPT, value, schema)


def decode_local(response: dict, request: dict) -> dict:
    """生成必须正常结束且未耗尽上下文；不补全截断 JSON、不掩盖错误。"""
    if response.get("done_reason") != "stop":
        raise ValueError("LOCAL_GENERATION_INCOMPLETE")
    if response.get("prompt_eval_count", 0) + response.get("eval_count", 0) >= request["options"]["num_ctx"]:
        raise ValueError("LOCAL_CONTEXT_EXHAUSTED")
    parsed = json.loads(response["message"]["content"])
    if not isinstance(parsed, dict):
        raise ValueError("LOCAL_JSON_OBJECT_REQUIRED")
    return parsed
