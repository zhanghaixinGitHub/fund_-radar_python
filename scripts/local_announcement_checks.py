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


def equivalent_amount_numbers(text: str, evidence: str) -> set[str]:
    """识别元/万元/亿元与股/万股/亿股的精确等值，不把汇率或取整当作等值。

    仅用于数字必要条件检查，不能证明金额的主体或期间正确。出现外币单位时
    不作隐含币种换算；同一数字若还在答案其他非金额位置出现，保留原告警。
    """
    if re.search(r"美元|港元|港币|日元|欧元|英镑|澳元|美金", text + evidence):
        return set()
    pattern = r"(?<![\d.])(\d+(?:\.\d+)?)(亿|万)?(元|股)"
    scale = {None: Decimal(1), "万": Decimal(10000), "亿": Decimal(100000000)}

    def clean(value):
        return compact(value).replace(",", "")

    source = {(Decimal(m[1]) * scale[m[2]], m[3]) for m in re.finditer(pattern, clean(evidence))}
    answer_text = clean(text)
    matches = list(re.finditer(pattern, answer_text))
    allowed = set()
    for match in matches:
        if (Decimal(match[1]) * scale[match[2]], match[3]) not in source:
            continue
        number = Decimal(match[1])
        occurrences = [m for m in re.finditer(r"\d+(?:\.\d+)?", answer_text) if Decimal(m[0]) == number]
        # 全部同值出现位置都必须是得到支持的金额，避免顺便放过同值的天数或比例。
        if all(
            any(m.start(1) == occurrence.start() and (Decimal(m[1]) * scale[m[2]], m[3]) in source for m in matches)
            for occurrence in occurrences
        ):
            allowed.add(format(number, "f").rstrip("0").rstrip(".") if number % 1 else str(int(number)))
    return allowed


def final_checks(question: str, raw: dict, body: str = "") -> dict:
    """只调整检查状态与可采用字段，逐字保留answer和evidence以供审计。

    因果转述难以靠词匹配证明时直接交由复核，不猜测哪一种因果关系才正确；
    上下限检查也只做风险提醒，不把模型数值自动替换成另一个数值。
    """
    answer = deepcopy(raw)
    text = compact(answer.get("answer", ""))
    quote_list = answer.get("quotes", [])
    # 引用之间不是一个连续句子，尤其不能把页末金额和另一引用中的数字直接拼接。
    evidence = "；".join(compact(q) for q in quote_list)
    issues = list(answer.get("issues", []))
    explicit_counts = literal_chinese_counts(evidence) | equivalent_amount_numbers(text, evidence)
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
    # “因考核原因未归属的股票如何处理”问的是处理办法，不能仅因出现“原因”
    # 两字就要求答复解释因果。真正询问原因的短问句及疑问结构仍接受严格核对。
    asks_cause = re.search(r"为何|为什么|因何|什么原因|哪些原因|原因(?:是|为|有哪些|是什么|[？?]|$)", question)
    if asks_cause:
        clauses = re.split(r"[。；;\n]", text)
        causal = [s for s in clauses if re.search(r"因为|由于|鉴于|导致|原因|^因", s)]
        supported = []
        for clause in causal:
            # 只去掉“……的原因是：”这种答题前缀；核心因果子句仍须逐字在证据中。
            # 不用同义词、关键词重合或任意逗号拆分来放过模型捏造的因果关系。
            phrase = re.sub(r"^(?:[^，；。:：]{1,40}的)?(?:调整)?原因(?:是|为)?[：:]?", "", clause)

            # 仅容纳复制原句时省略逗号/顿号，不删否定、程度词，不替换同义词。
            def literal(s):
                return re.sub(r"[，,、]", "", compact(s))

            supported.append(bool(phrase) and any(literal(phrase) in literal(q) for q in quote_list))
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
    # 同一表格行常包含多个指标。单问研发强度时，不能把净资产收益率的行业约束
    # 强加给研发指标；只有明确匹配问到的指标条款才检查，泛问考核时保留全范围。
    metrics = ("净资产收益率", "利润总额增长率", "净利润增长率", "研发投入强度", "研发占比", "营业收入增长率")
    asked = [m for m in metrics if m in question]
    if "研发占比" in asked:
        asked.append("研发投入强度")
    benchmark_source = evidence
    if asked:
        benchmark_source = "；".join(s for s in re.split(r"[；;。]", evidence) if any(m in s for m in asked))
    if re.search(r"考核|条件|要求", question) and re.search(r"同行业平均|行业均值", benchmark_source):
        if re.search(r"\d.*%", text) and not re.search(r"同行业|行业均值|行业平均", text):
            cleaned.append("BENCHMARK_CONDITION_MISSING")
    full_source = compact(body)
    # 只核对与答案同一谓语相接的括号例外，不把整篇其他业务的括号强加给当前答案。
    # 条件转述无法逐字确认时保守标记，不补写答案，更不能把规则通过当语义证明。
    if re.search(r"条件|要求|限制|需.*满足", question):
        for match in re.finditer(r"([^。；（）()]{6,24})[（(]([^（）()]{2,80})[）)]", full_source):
            prefix, exception = match.groups()
            if not re.search(r"按.*执行|另.*规定|除外|例外|豁免|但|除.+外", exception):
                continue
            if prefix[-8:] in text and exception not in text:
                cleaned.append("PARENTHETICAL_EXCEPTION_REQUIRES_REVIEW:" + exception)
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
