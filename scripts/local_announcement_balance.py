"""本地余额错题的窄范围原文提取：定位完整原句，保留金额的限定词、币种和单位。

仅支持明确询问当前公司总担保余额的单题，不汇总分项、不换算币种、不推断日期。
程序只提取支持的原文写法；无法确认口径时转复核，不让模型自由生成金额。
这不是权重训练，也不被正式预测服务引用。旧模型失败试验另行保留，本入口记录原文摘录范围和出处。
"""

import re

from scripts.local_announcement_continuations import link_continuations
from scripts.local_announcement_document import SUPPORTED_VERSIONS, evidence_units
from scripts.local_announcement_reader import compact

VERSION = "local-balance-source-rule-v1"
# 只路由既有拆题器生成的单一金额问题；复合题、指定子公司或指定历史日保持原流程。
QUESTION = re.compile(
    r"截至(?:公告|目前|本公告日)实际担保余额是多少[？?](?:只答本项金额，逐字保留原文金额前的限定词，不展开其他金额。)?"
)
METRIC = re.compile(r"担保(?:总)?余额")
CURRENT = re.compile(r"截至(?:目前|本公告(?:披露)?日|公告(?:披露)?日)")
TOTAL = re.compile(
    CURRENT.pattern + r"[，,]?(?:本公司|公司)(?:及控股子公司)?(?:存续的|的|实际)?(?:对外)?担保(?:总)?余额"
)
# 原文紧邻指标的金额才可核对，不能拿同句中的额度、比例或分项凑出答案。
AMOUNT = re.compile(
    r"(?:为|是|合计为|合计)?(?P<qualifier>不超过|不高于|不低于|不少于|低于|超过|约为|约|至多|至少)?"
    r"(?P<currency>人民币|美元|港币|港元|欧元)?(?P<amount>\d+(?:,\d{3})*(?:\.\d+)?)"
    r"(?P<unit>亿元|万元|元|亿美元|万美元|美元|亿港元|万港元|港元|万欧元|欧元)"
)
SLOTS = ("qualifier", "currency", "amount", "unit")


def handles_balance(questions: list[str]) -> bool:
    """不凭基金、公司或文档编号路由，避免为特定错题写死答案。"""
    return len(questions) == 1 and bool(QUESTION.fullmatch(questions[0]))


def balance_evidence(packet: dict) -> dict:
    """在完整原文中找所有余额句，只对可确认的跨页断点连读。

    返回的是显式摘录及紧凑文本偏移，不伪称全文。多处当前总余额、仅分项、仅额度、
    句子不完整、未知金额写法均进入复核。没有找到不等于公告没有披露。
    """
    if packet.get("version") not in SUPPORTED_VERSIONS or packet.get("scope") not in {"FULL_TEXT", "EXCERPT"}:
        raise ValueError("不支持的文档版本或资料范围。")
    if packet.get("issues"):
        raise ValueError("原文提取存在未解决问题。")
    units = link_continuations(evidence_units(packet))
    body = [u for u in units if u.get("kind") != "page_layout"]
    groups = []
    for unit in body:
        if groups and unit["id"] in groups[-1][-1].get("continuation_ids", []):
            groups[-1].append(unit)
        else:
            groups.append([unit])
    hits = []
    for group in groups:
        # 不将表格逐格串成一个总额句；表格任务继续使用已有完整表格读取流程。
        if any(u.get("kind") != "text" for u in group):
            continue
        text = "".join(compact(u["text"]) for u in group)
        offsets, cursor = [], 0
        for unit in group:
            end = cursor + len(compact(unit["text"]))
            offsets.append((unit, cursor, end))
            cursor = end
        # 仅中文句号等作为边界，不能在小数点处分割数字；分号后仍可能有重要条件。
        for match in re.finditer(r"[^。！？!?]+[。！？!?]|[^。！？!?]+$", text):
            sentence = match[0]
            if not METRIC.search(sentence):
                continue
            spans = []
            for unit, start, end in offsets:
                left, right = max(start, match.start()), min(end, match.end())
                if left < right:
                    spans.append(
                        {
                            "id": unit["id"],
                            "page": unit.get("page"),
                            "compact_start": left - start,
                            "compact_end": right - start,
                            "text": text[left:right],
                        }
                    )
            hits.append({"text": sentence, "spans": spans})
    candidates = []
    for hit in hits:
        sentence = hit["text"]
        # 上限授权不能冒充当前余额，历史日、子公司单项和复杂范围均不自动采用。
        if CURRENT.search(sentence) and TOTAL.search(sentence):
            candidates.append(hit)
    issues = []
    if len(candidates) != 1:
        issues.append("BALANCE_EVIDENCE_AMBIGUOUS" if candidates else "BALANCE_EVIDENCE_NOT_SUPPORTED")
    else:
        text = candidates[0]["text"]
        metric = TOTAL.search(text)
        if not text.endswith(("。", "！", "？", "!", "?")):
            issues.append("BALANCE_SENTENCE_INCOMPLETE")
        if len(list(METRIC.finditer(text))) != 1 or re.search(
            r"拟|计划|预计|经审批|经批准|更正|不含|不包括|未包括|仅为|仅含|并非|不是|不再|尚未|假设|如果", text
        ):
            issues.append("BALANCE_SCOPE_REQUIRES_REVIEW")
        amount = AMOUNT.match(text[metric.end() :])
        if not amount:
            issues.append("BALANCE_AMOUNT_FORMAT_UNSUPPORTED")
        elif re.match(r"以上|以下|左右|上下|至|到|以内|以外|起|或", text[metric.end() + amount.end() :]):
            issues.append("BALANCE_TRAILING_QUALIFIER_UNSUPPORTED")
        elif not re.match(r"[。！？!?；;，,（(]", text[metric.end() + amount.end() :]):
            # 金额后必须进入正文边界，禁止只识别范围前半段或“亿元/年”等复合口径。
            issues.append("BALANCE_AMOUNT_BOUNDARY_UNSUPPORTED")
    return {
        "scope": "EXCERPT",
        "source_scope": packet["scope"],
        "hits": hits,
        "candidates": candidates,
        "issues": issues,
    }


def extract_balance(title: str, packet: dict, questions: list[str]) -> dict:
    """仅从唯一、当前、明确主体的余额原句提取；不调用模型，不汇总或换算。

    模型多轮实测仍丢限定词，故金额不再交给自由生成。source_rule 明示结果来源，
    不能把这里的通过率计为模型单独答对率。缺句、多句、分项、期间或写法不支持
    时返回复核状态，不用首项、最大值或模型猜测兜底。
    """
    if not handles_balance(questions):
        raise ValueError("原文余额入口只接受当前公司总担保余额的单题。")
    selected = balance_evidence(packet)
    issues = list(selected["issues"])
    slots, evidence = {}, []
    answer = "无法从当前摘录确定。"
    amount_source = None
    if not issues:
        candidate = selected["candidates"][0]
        sentence = candidate["text"]
        metric = TOTAL.search(sentence)
        match = AMOUNT.match(sentence[metric.end() :])
        slots = {key: match[key] or "" for key in SLOTS}
        # 丢弃的仅是谓语“为/是/合计”；不改变限定词、币种、数字或单位。
        answer = "".join(slots[key] for key in SLOTS)
        amount_source = {"text": match[0], "sentence_start": metric.end(), "sentence_end": metric.end() + match.end()}
        evidence = candidate["spans"]
    row = {
        "question_number": 1,
        "answer": answer,
        "answer_status": "insufficient" if issues else "answered",
        "issues": issues,
        "safe_answer": answer if not issues else None,
        "review_state": "needs_review" if issues else "checks_passed",
        "semantic_verified": False,
        "evidence": evidence,
        "evidence_ids": [span["id"] for span in evidence],
        "quotes": [span["text"] for span in evidence],
        "source_slots": slots,
        "amount_source": amount_source,
        "answer_origin": "source_rule",
        "model_called": False,
    }
    return {
        "version": VERSION,
        "title": title,
        "source_sha256": packet["source_sha256"],
        "source_scope": packet["scope"],
        "selection": selected,
        "attempts": [],
        "answers": [row],
        "issues": issues,
        "review_state": row["review_state"],
        "semantic_verified": False,
        "elapsed_seconds": 0.0,
        "calls": 0,
        "answer_origin": "source_rule",
    }
