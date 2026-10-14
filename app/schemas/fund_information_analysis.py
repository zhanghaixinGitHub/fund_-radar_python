"""002112综合判断的独立契约；事实引用与未来推断分开，分数不冒充概率。"""

import json
import re
from copy import deepcopy
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

PROTOCOL = "DIRECTION_1D_ANALYSIS_V1"
VERSION = "002112_INFORMATION_ANALYSIS_6"
STYLE = "PREDICTION_INFORMATION_ZH_V4"
# 引用数量是整理粒度，不是原文正确性的判据；超出时分组保留，不截掉条件或整项拒绝。
EVENT_QUOTE_GROUP_SIZE = 10
EVENT_ITEM_LIMIT = 10


def organize_text(value: str, width: int) -> str:
    """字数是单段整理目标：仅插入换行，保留全文，不能截掉末尾否定或审批条件。"""
    output = []
    for line in value.splitlines(keepends=True):
        content = line.rstrip("\r\n")
        ending = line[len(content) :]
        start = 0
        while len(content) - start > width:
            end = start + width
            cuts = [content.rfind(mark, start, end) + 1 for mark in ("。", "；", "，")]
            if max(cuts) > start:
                end = max(cuts)
            output.append(content[start:end])
            output.append("\n")
            start = end
        output.append(content[start:] + ending)
    return "".join(output)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Event(StrictModel):
    title: str = Field(min_length=1)
    assessment: Literal["POSITIVE", "NEGATIVE", "MIXED", "NEUTRAL", "UNKNOWN"]
    stage: Literal["拟议", "进行中", "已发生", "历史经营", "程序事项", "不明"]
    # 十条为常规单组上限。模型确需更多同源证据时保留全部，后续按组核对和归档。
    # 长引用按连续原文拆段，短的真实条款也保留；逐字来源及总请求预算继续严格核对。
    quotes: list[str] = Field(min_length=1)
    meaning: str = Field(min_length=1)
    mechanism: str = Field(min_length=1)
    caveat: str = Field(min_length=1)

    @field_validator("quotes")
    @classmethod
    def quote_size(cls, values):
        if any(not v for v in values):
            raise ValueError("EVENT_QUOTE_LENGTH")
        return values


class DocumentEvents(StrictModel):
    id: str
    events: list[Event] = Field(min_length=1)


class ParsedDocuments(StrictModel):
    items: list[DocumentEvents] = Field(min_length=1)


class Reason(StrictModel):
    title: str = Field(min_length=1)
    refs: list[str] = Field(min_length=1)
    role: Literal["支持上涨", "支持下跌", "双向影响", "背景观察"]
    meaning: str = Field(min_length=1)
    implication: str = Field(min_length=1)


class Analysis(StrictModel):
    direction: Literal["UP", "DOWN"]
    confidence: Literal["LOW"]
    summary: str = Field(min_length=1)
    reasons: list[Reason] = Field(min_length=1)
    counterpoints: list[Reason] = Field(default_factory=list)
    synthesis: str = Field(min_length=1)
    change_conditions: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(min_length=1)
    # 程序依据实际引用重新计算，不接受模型给出的数量作为权重。
    evidence_units: list[str] = Field(default_factory=list)

    @field_validator("change_conditions", "limitations")
    @classmethod
    def paragraph_size(cls, values):
        if any(not v for v in values):
            raise ValueError("ANALYSIS_TEXT_LENGTH")
        return values


def normalize(value: str) -> str:
    return re.sub(r"[\s\uf06c]+", "", value)


def quote_catalog(body: str) -> dict[str, dict]:
    """给连续原文分配明确编号和字符区间；不去空格，不拼句，不改变任何字符。

    优先在句号、分号或换行处结束，长段落才按上限切段。通常每段八至三百五十字；
    短于八字的完整真实条款也保留，不以长度替代真实性。
    生成器可选相邻多个片段保留条件，不能通过自己抄写或省略号改变原文。
    """
    result, start = {}, 0
    if not body:
        raise ValueError("ANALYSIS_SOURCE_TOO_SHORT")
    while start < len(body):
        end = min(start + 350, len(body))
        if end < len(body):
            cuts = [body.rfind(mark, start + 7, end) + 1 for mark in ("。", "；", "\n")]
            if max(cuts) > start + 7:
                end = max(cuts)
            if 0 < len(body) - end < 8:
                end = len(body) if len(body) - start <= 350 else end - 8
        result[f"Q{len(result) + 1:04}"] = {"text": body[start:end], "start": start, "end": end}
        start = end
    return result


def quote_groups(count: int) -> list[list[int]]:
    """每组最多十条，索引回指同一事项的完整引用列表，不拆分事项或制造额外权重。"""
    return [
        list(range(start, min(start + EVENT_QUOTE_GROUP_SIZE, count)))
        for start in range(0, count, EVENT_QUOTE_GROUP_SIZE)
    ]


def validate_events(value: dict, documents: list[dict]) -> list[dict]:
    """数量超限先去重分组；逐条同源引用仍必需，不接受拼接、改数或省略否定。"""
    source = {d["id"]: d for d in documents}
    if len(source) != len(documents):
        raise ValueError("ANALYSIS_DOCUMENT_ID_DUPLICATE")
    resolved = deepcopy(value)
    if not isinstance(resolved, dict) or not isinstance(resolved.get("items"), list):
        raise ValueError("ANALYSIS_EVENT_FORMAT_INVALID")
    selected_spans, supplied_counts, event_totals, omitted_counts = {}, {}, {}, {}
    # 新响应只能按显式编号取原文；旧缓存的文字引用仍须通过下面同样严格的逐字校验。
    for item in resolved.get("items", []):
        if not isinstance(item, dict) or not isinstance(item.get("events"), list):
            raise ValueError("ANALYSIS_EVENT_FORMAT_INVALID")
        if item.get("id") not in source:
            raise ValueError("ANALYSIS_DOCUMENT_MISMATCH")
        event_totals[item["id"]] = len(item["events"])
        # 模型按重要性给出事项。仅采用前十项，剩余仍存在不可变首次响应和原文中。
        # 这里不改传入对象；未采用事项不是已验证成功的事项，不能冒称完整覆盖。
        unique, seen = [], set()
        for event in item["events"]:
            signature = json.dumps(event, ensure_ascii=False, sort_keys=True)
            if signature not in seen:
                seen.add(signature)
                unique.append(event)
        item["events"] = unique[:EVENT_ITEM_LIMIT]
        omitted_counts[item["id"]] = len(unique) - len(item["events"])
        catalog = quote_catalog(source[item["id"]]["body"])
        for event_index, event in enumerate(item.get("events", [])):
            if not isinstance(event, dict):
                raise ValueError("ANALYSIS_EVENT_FORMAT_INVALID")
            if "quote_ids" not in event:
                continue
            ids = event.pop("quote_ids")
            if (
                "quotes" in event
                or not isinstance(ids, list)
                or not ids
                or any(not isinstance(identity, str) or identity not in catalog for identity in ids)
            ):
                raise ValueError(f"ANALYSIS_QUOTE_ID_INVALID: 文档{item['id']}只能原样选择quote_catalogs中的编号。")
            # 先核对所有编号再去重，不能取前十条而漏过尾部的虚构/异源引用。
            # 按原文位置整理，完整保留多片段中的表头、审批条件与否定；原始回答不改写。
            key = (item["id"], event_index)
            supplied_counts[key] = len(ids)
            selected = sorted(set(ids), key=lambda identity: catalog[identity]["start"])
            event["quotes"] = [catalog[identity]["text"] for identity in selected]
            selected_spans[key] = [
                {field: catalog[identity][field] for field in ("start", "end")} for identity in selected
            ]
    result = ParsedDocuments.model_validate(resolved)
    if len(result.items) != len(source) or {v.id for v in result.items} != set(source):
        raise ValueError("ANALYSIS_DOCUMENT_MISMATCH")
    events = []
    for item in result.items:
        doc = source[item.id]
        for i, event in enumerate(item.events):
            full_spans = selected_spans.get((item.id, i)) or [
                {"start": doc["body"].index(q), "end": doc["body"].index(q) + len(q)} if q in doc["body"] else {}
                for q in event.quotes
            ]
            organized_quotes, organized_spans = [], []
            for quote_index, quote in enumerate(event.quotes):
                if quote not in doc["body"]:
                    raise ValueError(
                        f"ANALYSIS_QUOTE_MISMATCH: items[{item.id}].events[{i}].quotes[{quote_index}]"
                        "必须逐字复制输入body中的连续原文，包括数字、空格、否定和条件；不得拼接或改写。"
                    )
                # 长引用拆成连续原文片段，不剪字、不拼接；短引句只要同源真实也可使用。
                base = full_spans[quote_index]["start"]
                for piece in quote_catalog(quote).values():
                    organized_quotes.append(piece["text"])
                    organized_spans.append({"start": base + piece["start"], "end": base + piece["end"]})
            body = event.model_dump()
            body["quotes"] = organized_quotes
            for field in ("title", "meaning", "mechanism", "caveat"):
                body[field] = organize_text(body[field], 120 if field == "title" else 250)
            events.append(
                {
                    **body,
                    "id": f"{item.id}:{i}",
                    "quote_spans": organized_spans,
                    "quote_groups": quote_groups(len(organized_quotes)),
                    "event_selection": {
                        "supplied_count": event_totals[item.id],
                        "retained_count": len(item.events),
                        "limit": EVENT_ITEM_LIMIT,
                        "omitted_count": omitted_counts[item.id],
                    },
                    "quote_normalization": {
                        "supplied_count": supplied_counts.get((item.id, i), len(event.quotes)),
                        "retained_count": len(organized_quotes),
                        "group_size": EVENT_QUOTE_GROUP_SIZE,
                    },
                    "document": {k: v for k, v in doc.items() if k != "body"},
                }
            )
    return events


def validate_analysis(value: dict, facts: dict, time_contract: dict | None = None) -> dict:
    """数字由程序的事实正文展示；综合文字不得新增数字、概率或确定性涨跌。"""
    result = Analysis.model_validate(value)
    for index, reason in enumerate([*result.reasons, *result.counterpoints]):
        reason.refs = list(dict.fromkeys(reason.refs))
        if not set(reason.refs) <= set(facts):
            unknown = sorted(set(reason.refs) - set(facts))
            raise ValueError(
                f"ANALYSIS_REFERENCE_INVALID: reason[{index}].refs未知编号={unknown}；"
                f"只能原样选择事实键={list(facts)}；不能漏前缀或根据相似文字补引用。"
            )
        for ref in reason.refs:
            fact = facts[ref]
            if fact.get("direction_eligible") is False and reason.role not in {"背景观察", "双向影响"}:
                raise ValueError(f"ANALYSIS_MATTER_UNRESOLVED: {ref}只可作背景或双向说明，不能单独推导次日涨跌。")
    # 重要经理背景不能只出现在输入清单却被综合判断忽略；失败进入既有有界纠正流程。
    required = {ref for ref, fact in facts.items() if fact.get("required_in_analysis") is True}
    used = {ref for reason in [*result.reasons, *result.counterpoints] for ref in reason.refs}
    if not required <= used:
        raise ValueError(
            "ANALYSIS_MANAGER_EVIDENCE_MISSING: 必须引用经理变更与表现事实：" + ",".join(sorted(required - used))
        )
    texts = [result.summary, result.synthesis, *result.change_conditions, *result.limitations]
    texts += [v for r in [*result.reasons, *result.counterpoints] for v in [r.title, r.meaning, r.implication]]
    pattern = (
        r"[0-9]|[一二三四五六七八九十百]+(?:成|个百分点)|(?:百分之|千分之)[一二三四五六七八九十百]"
        r"|必涨|必跌|稳赚|保证收益|上涨概率|胜率|跌多了.*反弹"
        r"|未披露|(?:缺乏|缺少|没有)[^，。；]{0,25}利好|数量[^，。；]{0,15}(?:占优|更多|大于|少于)"
    )

    def number_check(paragraph):
        # 指数的正式名称不是复述数值。仅对输入中确实提供的名称排除误报，其他数字仍拒绝。
        for name in ("沪深300", "中证500"):
            if any(name in fact.get("text", "") for fact in facts.values()):
                paragraph = paragraph.replace(name, "指数")
        return re.search(pattern, paragraph)

    invalid = [paragraph for paragraph in texts if number_check(paragraph)]
    if invalid:
        # 一次反馈所有问题句，避免每次只修一处却不断引入新的数字转述。
        raise ValueError(
            "ANALYSIS_UNSUPPORTED_ASSERTION: 重写以下全部句子，去掉数字/中文比例、承诺、数量比较、未披露或缺利好："
            + "\n".join(invalid)
        )
    if time_contract is not None:
        # 日期与缺口说明由程序统一生成；只拒绝矛盾断言，允许正确的基准日表述。
        for paragraph in texts:
            for clause in re.split(r"[。；]", paragraph):
                missing = r"(?:缺少|缺失|未取得|没有取得|尚未取得|未覆盖)"
                latest = r"(?:最近|最新|上一|比较基准|基准).{0,6}(?:行情|净值|交易日数据)"
                contradicts_ready = time_contract.get("baseline_complete") and re.search(
                    missing
                    + r"[^，。；]{0,8}"
                    + latest
                    + "|"
                    + latest
                    + r"[^，。；]{0,8}"
                    + missing
                    + r"|(?:净值|行情).{0,4}早于比较基准日",
                    clause,
                )
                target_missing = re.search(r"目标日(?:收盘)?行情.{0,4}(?:缺失|缺口)|缺少目标日收盘行情", clause)
                target_realized = re.search(r"已取得目标日收盘行情|目标日(?:行情已发生|已经收盘|已收盘)", clause)
                if contradicts_ready or target_missing or target_realized:
                    raise ValueError(
                        "ANALYSIS_DATE_CONTRADICTION: 此句与time_context矛盾；基准日与目标日不得混用，"
                        "目标日未收盘不算缺数：" + clause
                    )
        summary_claim = re.sub(
            r"(?:不等于|不能据此推断|不直接|并不(?:直接)?|不能(?:直接)?|不会(?:直接)?|不意味着|不代表)"
            r"(?:提升|提高|改善)每股收益",
            "",
            result.summary + result.synthesis,
        )
        if "每股收益" in summary_claim and not any(
            "每股收益" in facts[ref].get("text", "")
            for reason in [*result.reasons, *result.counterpoints]
            for ref in reason.refs
        ):
            raise ValueError("ANALYSIS_UNSUPPORTED_ASSERTION: 总结没有引用对应每股收益事实，不得讨论其改善或变动。")
    for reason in [*result.reasons, *result.counterpoints]:
        statement = re.sub(
            r"(?:不等于|不能据此推断|不直接|并不(?:直接)?|不能(?:直接)?|不会(?:直接)?|不意味着|不代表)"
            r"(?:提升|提高|改善)每股收益",
            "",
            reason.meaning + reason.implication,
        )
        if "每股收益" in statement and not any("每股收益" in facts[ref].get("text", "") for ref in reason.refs):
            raise ValueError("ANALYSIS_UNSUPPORTED_ASSERTION: 引用没有每股收益数据，不能宣称提升每股收益。")
    expected = "支持上涨" if result.direction == "UP" else "支持下跌"
    if not any(r.role == expected for r in result.reasons):
        raise ValueError("ANALYSIS_DIRECTION_UNEXPLAINED")
    # 按明确作用整理展示分组，防止同方向因素误放入“相反因素”；不改事实或预测方向。
    opposite = "支持下跌" if result.direction == "UP" else "支持上涨"
    all_reasons = [*result.reasons, *result.counterpoints]
    result.reasons = [r for r in all_reasons if r.role != opposite]
    result.counterpoints = [r for r in all_reasons if r.role == opposite]
    # 条数和字数仅作整理目标，全部真实依据和条件保留；审查先看完整原文，再调整排版。
    # 超过六条理由、四条反证或五个事实编号不构成内容错误，不增加新请求来凑格式。
    result.summary = organize_text(result.summary, 250)
    result.synthesis = organize_text(result.synthesis, 600)
    for reason in [*result.reasons, *result.counterpoints]:
        reason.title = organize_text(reason.title, 120)
        reason.meaning = organize_text(reason.meaning, 250)
        reason.implication = organize_text(reason.implication, 250)
    result.change_conditions = [organize_text(text, 250) for text in dict.fromkeys(result.change_conditions)]
    result.limitations = [organize_text(text, 250) for text in dict.fromkeys(result.limitations)]
    # 允许分别解释不同事项或正反机制，实际依据集合只保留一次；没有条数分数。
    result.evidence_units = sorted({ref for r in [*result.reasons, *result.counterpoints] for ref in r.refs})
    return result.model_dump()
