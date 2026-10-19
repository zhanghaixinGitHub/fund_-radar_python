"""读取同一报告的现金篮子原句与保证金范围，不生成现金目标或评分。

股指期货、国债期货、股票期权和“全部保证金”分别保存。报告未写扣减
范围或现金排除项时，保留已识别原句及明确缺口，不能推断无保证金要求。
调用方先验证原件身份、摘要、公开日及报告期；文本位置仅是规范化后的
字符位置，不冒充PDF坐标、合同版本审核或实际逐日合规证明。
"""

import hashlib
import re
import unicodedata
from decimal import Decimal

VERSION = "QUALITY_REPORTED_CASH_CLAUSE_20261017_2"
INSTRUMENTS = {"股指期货": "INDEX_FUTURES", "国债期货": "TREASURY_FUTURES", "股票期权": "STOCK_OPTIONS"}
INSTRUMENT = r"(?:股指期货|国债期货|股票期权)(?:合约)?"
INSTRUMENT_LIST = rf"{INSTRUMENT}(?:[、和及]{INSTRUMENT})*"
EXCLUDED_ITEMS = r"结算备付金[、,]存出保证金(?:[、,]|和|以及)应收申购款等"
INLINE_DEFINITION = rf"(?P<inline>\((?:不包括|不含){EXCLUDED_ITEMS}\))"
BASKET = rf"现金(?:{INLINE_DEFINITION})?(?:或者|或)(?:投资于)?到期日在一年以内的政府债券"
NUMBER = r"\d+(?:\.\d+)?"
FORWARD = rf"{BASKET}(?:的)?(?:(?:投资比例|比例)(?:合计)?)?不低于基金资产净值的?(?P<floor>{NUMBER})%"
REVERSE = rf"不低于基金资产净值的?(?P<floor>{NUMBER})%的{BASKET}"
DEDUCTION = (rf"(?:(?P<daily>每个交易日日终|任何交易日日终)[,]?)?(?:在)?"
             rf"扣除(?P<margin>{INSTRUMENT_LIST}|全部|)"
             r"需缴纳的(?:交易)?保证金(?:以后|后)[,]?(?:(?:本)?基金)?"
             r"(?:(?:应当)?(?:保持|持有)|保留的)?")
DEFINITION = rf"[,;。.]?(?:其中[,]?|前述|本基金所指的)现金(?:类资产)?不包括{EXCLUDED_ITEMS}"


def compact(text: str) -> str:
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", text))


def _subject(body: str, start: int) -> str | None:
    """目标基金字样附近只接纳明确属于联接基金自身的目标ETF配置主语链。"""
    context = body[max(0, start - 700):start]
    if "本基金" not in context:
        return None
    if any(word in context[-350:] for word in ("目标ETF", "目标基金")):
        match = re.search(
            r"本基金(?:的投资组合比例为:)?投资于目标ETF的比例不低于基金资产净值的?"
            r"(?:100|\d{1,2})(?:\.\d+)?%[;,。](?:本基金)?$", context
        )
        return match[0] if match else None
    return context[max(0, context.rfind("本基金")):]


def cash_basket_clause(text: str) -> dict:
    """返回本报告明示的合计下限、排除项和保证金范围，或精确拒收原因。

    完整扣减条款优先于其中嵌套的简写下限；同一报告不同下限或不同完整
    保证金范围不能静默取一条。缺排除项、未明示扣减范围仅形成原句事实，
    不足以调用现金缺口计算器。全部保证金不得缩为列举的三种衍生工具。
    """
    body = compact(text)
    complete, partial = [], []
    for base in (FORWARD, REVERSE):
        for match in re.finditer(DEDUCTION + r"(?:投资于)?" + base, body):
            if subject := _subject(body, match.start()):
                complete.append((match, subject))
    for base in (FORWARD, REVERSE):
        for match in re.finditer(base, body):
            if any(other.start() <= match.start() < other.end() for other, _ in complete):
                continue
            if subject := _subject(body, match.start()):
                partial.append((match, subject))
    matches = complete + partial
    if not matches:
        raise ValueError("QUALITY_CASH_REPORTED_BASKET_OWN_SUBJECT_OR_WORDING_UNRESOLVED")
    floors = {Decimal(match["floor"]) / 100 for match, _ in matches}
    if len(floors) != 1 or not 0 <= next(iter(floors)) <= 1:
        raise ValueError("QUALITY_CASH_REPORTED_BASKET_FLOOR_CONFLICT_OR_DOMAIN")
    observations = []
    for match, subject in matches:
        deduction = match.groupdict().get("margin")
        if deduction == "全部":
            scope, required = "ALL_REQUIRED_TRADING_MARGIN", None
        elif deduction == "":
            # “扣除需缴纳的保证金”确实明示了扣减，原句没有按工具限定范围；
            # 与完全没写扣减的简写下限不同，但仍不能由三个具名工具推导保证金为零。
            scope, required = "UNQUALIFIED_REQUIRED_TRADING_MARGIN", None
        elif deduction:
            scope = "EXPLICIT_NAMED_INSTRUMENTS"
            names = re.findall(r"股指期货|国债期货|股票期权", deduction)
            if len(names) != len(set(names)):
                raise ValueError("QUALITY_CASH_REPORTED_MARGIN_INSTRUMENT_REPEATED")
            required = sorted(INSTRUMENTS[name] for name in names)
        else:
            scope, required = "DEDUCTION_NOT_SPECIFIED_IN_MATCHED_CLAUSE", None
        definition = re.match(DEFINITION, body[match.end():])
        inline = match.groupdict().get("inline")
        stop = match.end() + (definition.end() if definition else 0)
        original = body[match.start():stop]
        observations.append({
            "reported_floor_ratio": str(next(iter(floors))), "original_clause_normalized": original,
            "subject_evidence_normalized": subject, "margin_requirement_scope": scope,
            "required_margin_instruments": required,
            "margin_deduction_original": deduction,
            "cash_exclusion_definition_normalized": inline or (definition[0] if definition else None),
            "cash_exclusion_placement": "INLINE_PARENTHETICAL" if inline
            else "FOLLOWING_CLAUSE" if definition else None,
            "timing_in_matched_clause": "DAILY_CLOSE_EXPLICIT" if match.groupdict().get("daily")
            else "NO_DAILY_TIME_PHRASE_IN_MATCHED_CLAUSE",
            "normalized_character_span": [match.start(), stop],
            "following_unresolved_fragment": body[match.end():match.end() + 160]
            if not (inline or definition) else None,
        })
    explicit = [row for row in observations
                if row["margin_requirement_scope"] != "DEDUCTION_NOT_SPECIFIED_IN_MATCHED_CLAUSE"]
    signatures = {(row["margin_requirement_scope"], tuple(row["required_margin_instruments"] or []))
                  for row in explicit}
    if len(signatures) > 1:
        raise ValueError("QUALITY_CASH_REPORTED_MARGIN_SCOPE_CONFLICT_REQUIRES_APPLICABLE_VERSION_REVIEW")
    selected = explicit or observations
    defined = [row for row in selected if row["cash_exclusion_definition_normalized"]]
    chosen = (defined or selected)[0]
    missing = []
    if not explicit:
        missing.append("MARGIN_DEDUCTION_SCOPE_NOT_EXPLICIT")
    if not defined:
        missing.append("CASH_EXCLUSION_DEFINITION_NOT_IDENTIFIED")
    if Decimal(chosen["reported_floor_ratio"]) == 0:
        missing.append("ZERO_FLOOR_CLAUSE_REQUIRES_SEPARATE_APPLICABILITY_REVIEW")
    return {"version": VERSION, **chosen, "observations": observations, "missing_basis": missing,
            "state": "REPORTED_CLAUSE_BASIS_IDENTIFIED" if not missing else "REPORTED_CLAUSE_ONLY_BASIS_INCOMPLETE",
            "scope": "REPORT_STATED_REQUIREMENT_NOT_CONTRACT_VERSION_OR_DAILY_COMPLIANCE_ACCEPTANCE",
            "normalized_view": "NFKC_AND_WHITESPACE_REMOVAL_NO_ARBITRARY_NUMERIC_DELETION",
            "score_or_grade_created": False}


def cash_clause_page_view(pages: list[str]) -> tuple[str, dict]:
    """去除有真实页序依据的报告页眉与页码，只供条款原句衔接，不改金额表。

    先沿用已核的重复页眉/明确页码规则。裸整数仅在已去页眉后的页首、
    数字等于物理页序且至少三页连续成立时去除；单页转载文字中的21、
    普通正文数字、表内金额均不能按此规则删除。原页号、行号和原文留存。
    """
    from app.services.fund_quality_v2_restrictions import restriction_page_header_view

    _, prior = restriction_page_header_view(pages)
    removed = {(r["page"], r["line"]) for r in prior["removed_headers"]}
    first = {}
    for number, page in enumerate(pages, 1):
        first[number] = next(((line, text) for line, text in enumerate(page.splitlines(), 1)
                              if text.strip() and (number, line) not in removed), (0, ""))
    report_identity = re.search(r"(?:20\d{2}|[〇零一二三四五六七八九0]{4})年(?:年度|中期)报告",
                                compact("\n".join(pages[:2])))
    candidates = {number for number, (_, text) in first.items() if report_identity and text.strip() == str(number)}
    sequential = {number for number in candidates if any(
        {start, start+1, start+2} <= candidates for start in (number-2, number-1, number))}
    additions = [{"page": number, "line": first[number][0], "removed_header": first[number][1],
                  "position_rule": "FIRST_AFTER_VERIFIED_HEADERS_EQUALS_PHYSICAL_PAGE_IN_CONSECUTIVE_RUN"}
                 for number in sorted(sequential)]
    removed.update((r["page"], r["line"]) for r in additions)
    view = "\n".join("\n".join(text for line, text in enumerate(page.splitlines(), 1)
                              if (number, line) not in removed)
                     for number, page in enumerate(pages, 1))
    return view, {**prior, "removed_headers": prior["removed_headers"]+additions,
                  "parser_text_sha256": hashlib.sha256(view.encode()).hexdigest(),
                  "scope": "CLAUSE_ONLY_VERIFIED_HEADER_AND_SEQUENTIAL_PAGE_NUMBER_REMOVAL_NO_AMOUNT_REWRITE"}


def cash_report_clause(pages: list[str]) -> dict:
    """保留条款读取视图和逐行处理依据，字符区间属于该视图，不冒充PDF坐标。"""
    view, evidence = cash_clause_page_view(pages)
    return cash_basket_clause(view) | {"report_page_view": evidence}
