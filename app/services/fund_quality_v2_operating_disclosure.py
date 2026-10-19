"""从定期报告读取持续运作预警事实；研究入口，不生成分数或替代策略容量。

报告期内是否触发、期末是否恢复及合同适用条件是三个不同事实。这里只
读取报告明示内容；笼统的“无”保留为无适用预警披露，不能证明统一数值
门槛、持有人逐日数量或全部合同义务均已核查。任何未知状态不填作零风险。
"""

import re
import unicodedata
from datetime import date

from app.services.fund_quality_v2_disclosures import report_period
from app.services.fund_quality_v2_restrictions import restriction_page_header_view

VERSION = "QUALITY_OPERATING_DISCLOSURE_20261017_5"
_TITLE = "基金持有人数或基金资产净值预警情形的说明"
_DAYS = r"连续(?:二十|20)个(?:工作|交易)日(?:以上)?(?:出现)?"
_HOLDERS = r"基金(?:份额)?持有人(?:人数|数量|数)?(?:不满|低于)(?:二百|200)人"
_ASSETS = r"基金资产净值低于(?:五千万(?:元)?|5[，,]?000万元|50000000元)"
_NEGATIVE = r"(?:未出现|未发生|不存在|无)"
_PERIOD = r"(?:(?:本基金)?(?:本报告期|报告期)(?:内)?[，,]?(?:本基金)?)"


def _compact(text: str) -> str:
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", text))


def _section(pages: list[str]) -> tuple[str, str, list[dict]]:
    """只读独占行的预警标题到托管人报告；目录与正文交叉引用不算正文。

    标题允许实际换行及可选章号；不依赖某年必须使用4.8或某页码。页眉
    删除严格限定为带年度的定期报告标题及明确页码行，逐行保留删除轨迹。
    """
    normalized_pages = [unicodedata.normalize("NFKC", page) for page in pages]
    text = "\n".join(normalized_pages)
    _, header_trace = restriction_page_header_view(normalized_pages)
    # 部分原PDF的页首是“重复报告名 / 裸页码”两行。只有报告名已由
    # 公共规则核实为页眉、第二行等于真实物理页号，且相邻三页连续成立，
    # 才补记该页码的删除；不删除章节内部的裸整数或凭单页猜页序偏移。
    verified = {(edit["page"], edit["line"]) for edit in header_trace["removed_headers"]}
    paired = {}
    for page_number, page in enumerate(normalized_pages, 1):
        first = [(number, line) for number, line in enumerate(page.splitlines(), 1) if line.strip()][:2]
        if (len(first) == 2 and (page_number, first[0][0]) in verified
                and _compact(first[1][1]) == str(page_number)):
            paired[page_number] = {"page": page_number, "line": first[1][0], "removed_header": first[1][1],
                                   "position_rule": "VERIFIED_REPORT_TITLE_THEN_PHYSICAL_PAGE_IN_CONSECUTIVE_RUN"}
    consecutive = {page for page in paired if any(
        {start, start + 1, start + 2} <= paired.keys() for start in (page - 2, page - 1, page)
    )}
    header_trace["removed_headers"].extend(paired[page] for page in sorted(consecutive))
    # 仅复用逐页身份及连续页序已经核实的删除位置。章节中的裸整数即使
    # 看起来像页码也保留，防止把真实披露中的数字吞掉后错误匹配“无”。
    positions, page_offset = {}, 0
    for page_number, page in enumerate(normalized_pages, 1):
        line_offset = 0
        edits = {item["line"]: item for item in header_trace["removed_headers"] if item["page"] == page_number}
        for line_number, line in enumerate(page.splitlines(keepends=True), 1):
            if line_number in edits:
                positions[page_offset + line_offset] = edits[line_number]
            line_offset += len(line)
        page_offset += len(page) + 1
    title = r"[ \t\r\n]*".join(map(re.escape, _TITLE))
    short_title = r"[ \t\r\n]*".join(map(re.escape, "报告期内基金持有人数或基金资产净值预警说明"))
    prefix = r"(?:(?:报告期内管理人对本)[ \t\r\n]*)?"
    headings = list(re.finditer(
        rf"(?m)^[ \t]*(?:\d+\.\d+[ \t]*)?(?:{prefix}{title}|{short_title})[ \t]*$", text
    ))
    candidates = []
    for head in headings:
        tail = text[head.end():]
        # 原PDF的自动章节号可能未被抽取。完整、独占行的托管人报告标题
        # 仍能确定边界；带引导点/页码的目录行不能作为标题或章节正文。
        end = re.search(r"(?m)^[ \t]*(?:(?:§[ \t]*)?\d+[ \t]*)?托管人报告[ \t]*$", tail)
        if not end or end.start() > 4000:
            continue
        raw = tail[:end.start()]
        removed, lines = [], []
        for line in re.finditer(r"(?m)^[^\r\n]+", raw):
            absolute = head.end() + line.start()
            if absolute in positions:
                removed.append(positions[absolute])
            else:
                lines.append(line[0])
        body = _compact("\n".join(lines))
        # 正文中不能残留目录域代码、引导点或跨越其他财报章节。
        if not body or re.search(r"\.{3,}|PAGEREF|HYPERLINK|资产负债表|利润表", body):
            continue
        candidates.append((raw, body, removed))
    if not candidates:
        raise ValueError("QUALITY_OPERATING_DISCLOSURE_BODY_SECTION_UNRESOLVED")
    if len({value[1] for value in candidates}) != 1:
        raise ValueError("QUALITY_OPERATING_DISCLOSURE_MULTIPLE_CONFLICTING_SECTIONS")
    return candidates[0]


def _negative_criteria(sentence: str) -> set[str]:
    """按同一句的否定作用域核两项条件，不将“未整改”误作“未出现”。"""
    match = re.fullmatch(
        rf"{_PERIOD}{_NEGATIVE}{_DAYS}(?P<body>.+?)(?:的)?情形", sentence
    )
    # “本基金无连续二十个工作日出现……”没有显式本报告期，但其标题与
    # 报告覆盖期已绑定；依然只作为这个报告期的明示陈述。
    if not match:
        match = re.fullmatch(rf"本基金{_NEGATIVE}{_DAYS}(?P<body>.+?)(?:的)?情形", sentence)
    if not match:
        return set()
    body = match["body"]
    if re.fullmatch(rf"{_HOLDERS}(?:或者|或){_ASSETS}", body):
        return {"holders", "assets"}
    if re.fullmatch(_HOLDERS, body):
        return {"holders"}
    if re.fullmatch(_ASSETS, body):
        return {"assets"}
    return set()


def _reported_exception(body: str, pages: list[str], observed: date) -> dict | None:
    """保留发起式三周年前的不适用声明，并以本报告生效日交叉核日期。

    只确认报告实际作出这个声明，不能把该法条例外推成无其他合同义务。
    缺生效日、已满三周年或正文日期矛盾时明确拒收，不能据基金名称猜测。
    """
    # 简写声明只能证明报告作出了“不适用”说明。没有法条、期限和依据时，
    # 不从“发起式”推断三周年前例外成立，也不赋予无预警或无风险状态。
    if re.fullmatch(r"本基金为发起式基金[，,]不适用本项说明[。.]?", body):
        return {"reported_exception": "SPONSOR_FUND_EXCEPTION_WITH_UNVERIFIED_RULE_BASIS",
                "statement": body, "report_effective_date": None, "third_anniversary_bound": None,
                "scope": "REPORT_ASSERTION_ONLY_NO_LEGAL_BASIS_OR_DURATION_INFERRED"}
    statement = (r"本基金为发起式基金[，,]且截至本报告期末[，,]本基金基金合同生效未满3年[，,]"
                 r"暂不适用《公开募集证券投资基金运作管理办法》第四十一条第一款的规定[。.]?")
    if not re.fullmatch(statement, body):
        return None
    text = unicodedata.normalize("NFKC", "\n".join(pages))
    # 联接基金的报告还列目标ETF的另一个生效日，不能把两者混成冲突，
    # 更不能按最早/最晚日期猜。本基金基本情况限定到目标基金子节之前。
    blocks = []
    for heading in re.finditer(r"(?m)^[ \t]*(?:2\.1[ \t]*)?基金基本情况[ \t]*$", text):
        tail = text[heading.end():]
        stop = re.search(r"(?m)^[ \t]*(?:2\.1\.1[ \t]*目标基金基本情况|2\.2[ \t]*基金产品说明|"
                         r"目标基金基本情况|基金产品说明)[ \t]*$", tail)
        if stop and stop.start() < 5000:
            blocks.append(_compact(tail[:stop.start()]))
    if len(set(blocks)) != 1:
        raise ValueError("QUALITY_OPERATING_OWN_PRODUCT_BASIC_SECTION_UNRESOLVED")
    matches = list(re.finditer(r"基金合同生效日(?:期)?[：:]?(20\d{2})年(\d{1,2})月(\d{1,2})日", blocks[0]))
    dates = {date(*map(int, match.groups())) for match in matches}
    if len(dates) != 1:
        raise ValueError("QUALITY_OPERATING_REPORTED_EXCEPTION_EFFECTIVE_DATE_UNVERIFIED")
    effective = next(iter(dates))
    # 闰日的三周年按随后非闰年的2月末作保守界限；本函数不为该边界日
    # 自动适用例外，仍保留报告的具体声明。
    try:
        anniversary = effective.replace(year=effective.year + 3)
    except ValueError:
        anniversary = date(effective.year + 3, 2, 28)
    if not effective <= observed < anniversary:
        raise ValueError("QUALITY_OPERATING_REPORTED_EXCEPTION_DATE_CONFLICT")
    return {"reported_exception": "ARTICLE41_PARAGRAPH1_BEFORE_THIRD_ANNIVERSARY",
            "report_effective_date": str(effective), "third_anniversary_bound": str(anniversary),
            "effective_date_excerpt": matches[0][0], "statement": body,
            "effective_date_scope": "OWN_PRODUCT_BASIC_SECTION_EXCLUDING_TARGET_FUND_SUBSECTION",
            "scope": "REPORT_ASSERTION_DATE_CORROBORATED_NOT_ALL_CONTRACT_OBLIGATIONS"}


def operating_disclosure(pages: list[str], *, observed_on: str, source_hash: str) -> dict:
    """返回原文支持的预警及恢复状态，保留期内条件和未知分量。

    调用方核基金身份与原字节摘要；本方法核报告经济期间、摘要格式及章节。
    report_alert_disclosed是报告陈述，而非独立监管合规结论。triggered为
    True/False/None仅表示原句的肯定/否定/未知，不接受外部核验布尔入参。
    发起式例外只在本报告有明示声明与对应生效日期时保留；不将该声明
    扩大为全部合同适用性核验，不生成数值指标，不参加八维评分或校准。
    """
    observed = date.fromisoformat(observed_on)
    start, end = report_period("\n".join(pages))
    if end != observed or not re.fullmatch(r"[a-f0-9]{64}", source_hash):
        raise ValueError("QUALITY_OPERATING_SOURCE_OR_PERIOD")
    raw, body, trace = _section(pages)
    exception = _reported_exception(body, pages, observed)
    sentences = [value for value in re.split(r"[。；;]", body) if value]
    negative, positive, recovery = set(), set(), {}
    unresolved_sentences, explicit = [], []
    summary_none = re.fullmatch(
        r"(?:无|(?:本基金)?本报告期(?:内)?(?:本基金管理人|基金管理人)?无应说明的?预警信息|"
        r"本基金本报告期无需要说明的相关情形|报告期内[，,]本基金无需要说明的情形|"
        r"本基金本报告期内不存在基金持有人数或基金资产净值预警的情况)[。.]?", body
    ) is not None
    if not summary_none and not exception:
        for sentence in sentences:
            denied = _negative_criteria(sentence)
            if denied:
                negative.update(denied)
                explicit.append({"statement": sentence, "facts": {key: False for key in sorted(denied)}})
                continue
            # 只读过去/当期的肯定事实；“如果”“可能”“将”“应当”之类政策
            # 和假设不能变为实际触发；否定复合句也不能退回肯定匹配。
            event = re.fullmatch(
                rf"{_PERIOD}{_DAYS}(?P<criterion>{_HOLDERS}|{_ASSETS})(?:的)?情形(?:[，,](?P<tail>.*))?",
                sentence,
            )
            if event:
                key = "assets" if re.fullmatch(_ASSETS, event["criterion"]) else "holders"
                positive.add(key)
                explicit.append({"statement": sentence, "facts": {key: True}})
                rest = event["tail"]
                if rest:
                    # 恢复必须明示期末及触发的同一种条件；期末资产大并不能
                    # 证明持有人人数已恢复，单独的未来恢复计划也不算。
                    recovered = (r"截至本报告期末[，,](?:本基金)?基金资产净值已恢复至五千万元以上"
                                 if key == "assets" else
                                 r"截至本报告期末[，,](?:本基金)?基金(?:份额)?持有人(?:人数|数量|数)?已恢复至二百人以上")
                    ongoing = (r"截至本报告期末[，,](?:本基金)?基金资产净值仍低于五千万元"
                               if key == "assets" else
                               r"截至本报告期末[，,](?:本基金)?基金(?:份额)?持有人(?:人数|数量|数)?仍不满二百人")
                    if re.fullmatch(recovered, rest):
                        recovery[key] = "EXPLICITLY_RECOVERED_AT_PERIOD_END"
                    elif re.fullmatch(ongoing, rest):
                        recovery[key] = "EXPLICITLY_STILL_BELOW_AT_PERIOD_END"
                    else:
                        unresolved_sentences.append(rest)
                continue
            unresolved_sentences.append(sentence)
    if positive & negative:
        raise ValueError("QUALITY_OPERATING_CONFLICTING_TRIGGER_STATEMENTS")
    triggers = {key: True if key in positive else False if key in negative else None for key in ("holders", "assets")}
    if exception:
        state = ("REPORTED_EXCEPTION_BEFORE_THIRD_ANNIVERSARY"
                 if exception["reported_exception"] == "ARTICLE41_PARAGRAPH1_BEFORE_THIRD_ANNIVERSARY"
                 else "REPORTED_EXCEPTION_WITH_UNVERIFIED_RULE_BASIS")
        disclosed = None
    elif summary_none:
        state, disclosed = "SUMMARY_NO_APPLICABLE_ALERT_DISCLOSED", False
    elif unresolved_sentences:
        state, disclosed = "UNRESOLVED_DISCLOSURE", True if positive else None
    elif positive:
        disclosed = True
        if all(recovery.get(key) == "EXPLICITLY_RECOVERED_AT_PERIOD_END" for key in positive):
            state = "REPORTED_ALERT_EXPLICITLY_RECOVERED"
        elif any(recovery.get(key) == "EXPLICITLY_STILL_BELOW_AT_PERIOD_END" for key in positive):
            state = "REPORTED_ALERT_EXPLICITLY_ONGOING"
        else:
            state = "REPORTED_ALERT_RECOVERY_UNKNOWN"
    elif negative == {"holders", "assets"}:
        state, disclosed = "EXPLICIT_TWO_CRITERIA_NO_TRIGGER_REPORTED", False
    else:
        state, disclosed = "UNRESOLVED_DISCLOSURE", None
    return {
        "version": VERSION, "source_hashes": [source_hash], "observed_on": observed_on,
        "report_period_start": str(start), "report_period_end": str(end), "state": state,
        "report_alert_disclosed": disclosed, "criteria_triggered": triggers,
        "period_end_recovery": {key: recovery.get(key) for key in ("holders", "assets")},
        "explicit_statements": explicit, "unresolved_statements": unresolved_sentences,
        "raw_section": raw, "normalized_section": body, "parser_view_trace": trace,
        "reported_applicability_exception": exception,
        "threshold_application": "REPORT_WORDING_ONLY_CONTRACT_APPLICABILITY_NOT_ESTABLISHED",
        "score_or_grade_created": False, "raw4_metric_replaced": False,
    }
