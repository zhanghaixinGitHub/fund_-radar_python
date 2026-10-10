"""生成后的保守检查；不修改答案，也不把评测标签或核对答案反馈给模型。

可直接由已保存的生成结果重放。这些检查只能识别一部分高风险表述，不能代替
语义复核；正确的转述也可能被拦截，结果必须同时报告覆盖率和误报。
"""

import re
from copy import deepcopy
from decimal import Decimal


def compact(value: str) -> str:
    return re.sub(r"\s+", "", value)


def literal_chinese_counts(text: str) -> set[str]:
    """只识别原文明示的零至十九的小计数，不猜测名单长度、不把编号当金额。"""
    digits = {c: n for n, c in enumerate("零一二三四五六七八九")}
    digits["两"] = 2
    result = set()
    for token in re.findall(r"([零一二三四五六七八九十两]{1,3})(?=名|人|个|年|月|日|次|票)", compact(text)):
        if token in digits:
            result.add(str(digits[token]))
        elif token == "十":
            result.add("10")
        elif len(token) == 2 and token.startswith("十") and token[1] in digits:
            result.add(str(10 + digits[token[1]]))
    return result


def final_checks(question: str, raw: dict, body: str = "") -> dict:
    """只调整检查状态与可采用字段，逐字保留answer和evidence以供审计。

    因果转述难以靠词匹配证明时直接交由复核，不猜测哪一种因果关系才正确；
    上下限检查也只做风险提醒，不把模型数值自动替换成另一个数值。
    """
    answer = deepcopy(raw)
    text = compact(answer.get("answer", ""))
    quote_list = answer.get("quotes", [])
    evidence = compact("\n".join(quote_list))
    issues = list(answer.get("issues", []))
    explicit_counts = literal_chinese_counts(evidence)
    cleaned = []
    for issue in issues:
        if issue.startswith("NUMBER_NOT_IN_EVIDENCE:"):
            numbers = set(issue.partition(":")[2].split(",")) - explicit_counts
            if numbers:
                cleaned.append("NUMBER_NOT_IN_EVIDENCE:" + ",".join(sorted(numbers)))
        else:
            cleaned.append(issue)
    # 先限定是原因类问题，再要求因果子句能在原文中直接定位。不能据此判断
    # 不匹配就一定答错；本规则刻意把难以验证的转述交给人工或更强模型复核。
    if re.search(r"为何|为什么|原因|因何", question):
        clauses = re.split(r"[。；;\n]", text)
        causal = [s for s in clauses if re.search(r"因为|由于|鉴于|导致|原因|^因", s)]
        supported = []
        for clause in causal:
            phrase = re.sub(r"^(?:调整)?原因(?:是|为|：|:)?", "", clause)
            supported.append(bool(phrase) and any(phrase in compact(q) for q in quote_list))
        if not causal or not all(supported):
            cleaned.append("CAUSAL_PARAPHRASE_REQUIRES_REVIEW")
    # 若问题本身已明确询问上限/下限，可以用数值作简短回答；其他情形必须
    # 在对应数值附近保留界限词，不能把最多可达的值写成实际已经发生的值。
    if not re.search(r"上限|下限|最高|最低|不超过|不低于", question):
        bound_groups = [
            (r"不超过|不高于|最多|至多", r"不超过|不高于|最多|至多|上限|≤"),
            (r"不低于|不少于|至少", r"不低于|不少于|至少|下限|≥"),
        ]
        for source_bound, answer_bound in bound_groups:
            pattern = rf"(?:{source_bound})[^\d。；]{{0,8}}([\d,]+(?:\.\d+)?)"
            for match in re.finditer(pattern, evidence):
                number = match.group(1).replace(",", "")
                answer_no_commas = text.replace(",", "")
                for occurrence in re.finditer(rf"(?<![\d.]){re.escape(number)}(?![\d.])", answer_no_commas):
                    before = answer_no_commas[max(0, occurrence.start() - 14) : occurrence.start()]
                    if not re.search(answer_bound, before):
                        cleaned.append("BOUND_QUALIFIER_REQUIRES_REVIEW:" + number)
    # 仅对明确问“总额和两部分”的三金额回答做加总检查；不假定任意列出的
    # 几个分项就是全部，不换算不同单位，也不把表内日期、比例当金额。
    if re.search(r"总金额|总额|合计", question) and "两部分" in question:
        amounts = re.findall(r"([\d,]+(?:\.\d+)?)(亿元|万元|元)", text)
        if len(amounts) == 3 and len({unit for _, unit in amounts}) == 1:
            total, first, second = [Decimal(value.replace(",", "")) for value, _ in amounts]
            if abs(total - first - second) > Decimal("0.02"):
                cleaned.append("COMPONENTS_DO_NOT_SUM")
    if re.search(r"考核|条件|要求", question) and re.search(r"同行业平均|行业均值", evidence):
        if re.search(r"\d.*%", text) and not re.search(r"同行业|行业均值|行业平均", text):
            cleaned.append("BENCHMARK_CONDITION_MISSING")
    full_source = compact(body)
    if re.search(r"窗口|禁止买卖|交易限制", question):
        if "香港联交所" in full_source and re.search(r"深圳证券交易所|上海证券交易所", full_source):
            cleaned.append("MULTI_MARKET_RULES_REQUIRES_REVIEW")
    if "手续" in question and re.search(r"还需|后续|尚需", question):
        pending = "。".join(s for s in re.split(r"[。；]", full_source) if "尚需" in s)
        terms = ("项目备案", "规划许可", "施工许可", "权属证书", "出让合同", "缴纳")
        missing = [term for term in terms if term in pending and term not in text]
        if missing:
            cleaned.append("PENDING_PROCEDURES_REQUIRES_REVIEW:" + ",".join(missing))
    answer["issues"] = list(dict.fromkeys(cleaned))
    answer["review_state"] = "needs_review" if answer["issues"] else "checks_passed"
    answer["safe_answer"] = answer.get("answer") if not answer["issues"] else None
    answer["semantic_verified"] = False
    return answer
