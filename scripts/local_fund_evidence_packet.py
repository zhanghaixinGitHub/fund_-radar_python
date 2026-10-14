"""综合阶段的无损证据字典：重复引句、标题只发送一次，不删条件或事实。

每个事实的text可逐字符重建；重建失败拒绝请求。资料正文在逐篇读取阶段全部
发送，综合阶段使用该阶段选择的全部引句。无损传输不证明抽取没有遗漏事实。
"""

from copy import deepcopy

from app.schemas.fund_information_analysis import Analysis
from app.services.fund_information_analysis import SYNTHESIS_PROMPT

from scripts.local_fund_analysis_probe import model_facts


def fact_packet(facts: dict, documents: list[dict]) -> dict:
    """在全部原文引用和资格标记不变的情况下，只消除重复存储，不做语义摘要。"""
    source = {doc["id"]: doc for doc in documents}
    if len(source) != len(documents):
        raise ValueError("LOCAL_DOCUMENT_ID_DUPLICATE")
    pool, sources, lookup, source_lookup = {}, {}, {}, {}
    projected = model_facts(facts)
    for ref, fact in facts.items():
        if not fact.get("quote_receipts"):
            continue
        parts, seen = [], set()
        for receipt in fact["quote_receipts"]:
            quote = receipt["quote"]
            identity = receipt["document_id"]
            doc = source.get(identity)
            if doc is None or quote not in doc["body"]:
                raise ValueError("LOCAL_EVIDENCE_SOURCE_MISSING")
            if quote in seen:
                continue
            seen.add(quote)
            if quote not in lookup:
                key = f"P{len(pool) + 1:04}"
                lookup[quote] = key
                pool[key] = quote
            if identity not in source_lookup:
                key = f"D{len(sources) + 1:03}"
                source_lookup[identity] = key
                sources[key] = {k: deepcopy(doc[k]) for k in ("id", "date", "title", "body_scope", "body_truncated")}
            parts.append({"source": source_lookup[identity], "quote": lookup[quote]})
        rebuilt = "\n".join(
            f"{sources[p['source']]['date']}《{sources[p['source']]['title']}》：{pool[p['quote']]}" for p in parts
        )
        if rebuilt != fact["text"]:
            raise ValueError("LOCAL_EVIDENCE_LOSSLESS_RECONSTRUCTION_FAILED")
        projected[ref].pop("text")
        projected[ref]["text_parts"] = parts
    return {"facts": projected, "sources": sources, "quote_pool": pool}


def synthesis_packet_request(data: dict, facts: dict, timing: dict, gaps: list[str], builder) -> tuple[dict, dict]:
    """保留全部事实及引句的无损表示；超预算失败，不挑选易读或同方向资料。"""
    packet = fact_packet(facts, data["documents"])
    schema = Analysis.model_json_schema()
    schema["properties"].pop("evidence_units", None)
    schema["$defs"]["Reason"]["properties"]["refs"]["items"]["enum"] = list(facts)
    value = {
        "fund_code": data["fund_code"],
        "window": data["window"],
        "as_of": data["as_of"],
        "latest_nav_date": data["latest_nav_date"],
        "time_context": timing,
        **packet,
        "inventory": data["inventory"],
        "gaps": gaps,
        "overflow": data["overflow"],
    }
    prompt = SYNTHESIS_PROMPT + (
        "\n无损引用说明：有text的事实直接阅读；有text_parts的事实按顺序读取quote_pool中的完整原文，"
        "sources给出该引句所属公告的标题、日期和资料范围。不能把不同公告混为同一事项。"
        "事实引用只使用facts中F开头的键，不使用P或D编号。所有引句和资格标记都须参加考虑，"
        "不能因按字典传输而忽略条件。不得把历史行情延续说成已验证规律。"
    )
    return builder(prompt, value, schema), packet
