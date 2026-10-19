"""治理文书事实、原文覆盖期和监管决定适用对象；不自动形成治理评分。

年度/中期报告的声明和监管决定是不同证据。公司声称完成整改，不自动变成
监管验收原件；覆盖期完整也不代表检索完整、严重性核准或独立复核已经完成。
"""

import re
from datetime import date, timedelta
from decimal import Decimal

from bs4 import BeautifulSoup

SCOPES = {
    "管理人受调查或处罚等情况": "MANAGER",
    "管理人相关从业人员受调查或处罚等情况": "MANAGER_RELATED_STAFF",
    "管理人、托管人及其高级管理人员受稽查或处罚等情况": "COMBINED_MANAGER_CUSTODIAN_OR_STAFF",
    "管理人、托管人及其高级管理人员受稽查或处罚的情况": "COMBINED_MANAGER_CUSTODIAN_OR_STAFF",
}
END_TITLES = {
    "托管人受调查或处罚等情况",
    "托管人相关从业人员受调查或处罚等情况",
    "基金租用证券公司交易单元的有关情况",
    "其他重大事件",
}
FIELDS = {
    "受到调查或处罚等措施的主体": "reported_subject",
    "受到调查或处罚等措施的时间": "reported_event_time",
    "采取调查或处罚等措施的机构": "reported_authorities",
    "受到调查或处罚等措施类型": "reported_measure_type",
    "受到的具体措施类型": "reported_measures",
    "受到调查或处罚等措施的原因": "reported_reasons",
    "受到处罚的依据": "reported_legal_basis",
    "管理人采取整改措施的情况（如提出整改意见）": "reported_remediation",
    "管理人采取整改措施的情况": "reported_remediation",
}


def report_manager_name(pages: list[str]) -> str | None:
    """从封面/基本信息读法定管理人全称，不把持仓发行人、托管人或简称当管理人。"""
    text = re.sub(r"\s+", "", "\n".join(pages))[:16000]
    company = r"([^。:：]{2,28}?基金管理(?:股份有限公司|有限责任公司|有限公司))"
    explicit = re.search(r"基金管理人(?:名称)?[:：]" + company, text)
    if explicit:
        return explicit[1]
    table = re.search(r"基金管理人(?:名称)?" + company + r"基金托管人", text)
    return table[1] if table else None


def _reported_event_rows(quote: str) -> list[dict]:
    """保留报告表格里的事项、日期和整改原话，不把自述整改升级为独立验收。"""
    # “其他问题”可能出现在违规原因里，不把“其他”当无条件字段分隔符。
    # 只去掉表尾独立的“其他 -”占位行，完整excerpt仍由调用方保留。
    quote = re.sub(r"(?m)^\s*其他[ \t]+[-—－][ \t]*$", "", quote)
    text = re.sub(r"\s+", "", quote)
    labels = sorted(FIELDS, key=len, reverse=True)
    found = list(re.finditer("|".join(map(re.escape, labels)), text))
    rows, row = [], {}
    for index, match in enumerate(found):
        key = FIELDS[match[0]]
        if key == "reported_subject" and row:
            rows.append(row)
            row = {}
        if not row and key != "reported_subject":
            continue
        row[key] = text[match.end() : found[index + 1].start() if index + 1 < len(found) else len(text)]
    if row:
        rows.append(row)
    for item in rows:
        dates = re.findall(r"(20[0-9]{2})[-年]([0-9]{1,2})[-月]([0-9]{1,2})日?", item.get("reported_event_time", ""))
        item["event_dates"] = [str(date(*map(int, parts))) for parts in dates]
        item["remediation_evidence_kind"] = "ISSUER_REPORT_STATEMENT_NOT_REGULATOR_ACCEPTANCE_ORIGINAL"
        remediation = item.get("reported_remediation", "")
        # 转载正文可能将左列换行备注和右列内容交错，不能删去中间文字后假称还原了单元格。
        warnings = []
        if re.search(r"整改意见[）)]|[（(]如提出整改意|见[）)]", remediation):
            warnings.append("REMEDIATION_LABEL_AND_VALUE_INTERLEAVED_REQUIRES_ATTACHMENT_REVIEW")
        # 完整法规标题出现在原因列，且依据列从法规残句开始，是已见转载错列形态。
        # 只标歧义，不删去文字后冒充已还原单元格，清洁值仍须读取原PDF。
        if "《" in item.get("reported_reasons", "") and not item.get("reported_legal_basis", "").startswith("《"):
            warnings.append("REASONS_AND_LEGAL_BASIS_INTERLEAVED_REQUIRES_ATTACHMENT_REVIEW")
        item["layout_warnings"] = warnings
        item["field_state"] = ("LAYOUT_AMBIGUOUS_RAW_TEXT_ONLY" if warnings
                               else "PARSED_REPORT_DECLARATION_NOT_INDEPENDENT_REVIEW")
    return rows


def report_governance_sections(
    pages: list[str], *, coverage: tuple[date, date], source_hash: str, source_url: str
) -> list[dict]:
    """按真实标题截取治理章节，保留跨页正文和准确页号；无章节不视作无事项。"""
    text = "\n".join(pages)
    from app.services.fund_quality_v2_restrictions import restriction_page_header_view

    # 复用已按整份报告身份及连续物理页序验证的页码规则；原文excerpt保持不变。
    # 不能删除字段中的一般整数，也不能仅因某行是两位数便当页码。
    _, header_trace = restriction_page_header_view(pages)
    boundaries = []
    for line in re.finditer(r"(?m)^[^\r\n]+", text):
        raw = line[0].strip()
        prefix = re.match(r"[0-9]{1,2}(?:\.[0-9]{1,2}){1,3}\s*", raw)
        title = re.sub(r"\s+", "", raw[prefix.end() :] if prefix else raw)
        # 不把22.50万元之类金额误当作下一节；只使用已知的治理对象/后续标题。
        if title in SCOPES or title in END_TITLES:
            boundaries.append((line.start(), title))
    selected = [(offset, title) for offset, title in boundaries if title in SCOPES]
    if any(SCOPES[title] != "COMBINED_MANAGER_CUSTODIAN_OR_STAFF" for _, title in selected):
        selected = [
            (offset, title) for offset, title in selected if SCOPES[title] != "COMBINED_MANAGER_CUSTODIAN_OR_STAFF"
        ]
    result = []
    for start, title in selected:
        stop = next((offset for offset, other in boundaries if offset > start and other != title), None)
        if stop is None or not 0 < stop - start <= 12000:
            raise ValueError("GOVERNANCE_REPORT_SECTION_END_UNRESOLVED")
        quote, touched, offset, field_parts = text[start:stop], [], 0, []
        for index, page in enumerate(pages, 1):
            if offset < stop and offset + len(page) > start:
                touched.append(index)
                left, right = max(0, start - offset), min(len(page), stop - offset)
                head_lines = list(re.finditer(r"(?m)^[^\r\n]+", page))[:4]
                header = next(
                    (line for line in head_lines if re.search(r"20[0-9]{2}\s*年\s*(?:年度|中期)报告", line[0])), None
                )
                page_number = next(
                    (
                        line
                        for line in head_lines
                        if re.fullmatch(
                            r"\s*(?:第\s*[0-9]+\s*页\s*共\s*[0-9]+\s*页|[0-9]{1,3}(?:/[0-9]{1,3})?)\s*", line[0]
                        )
                    ),
                    None,
                )
                # 仅剔除该页页首实际“报告名+页码”对，避免把字段中引用文书的文字删除。
                removals = (
                    [(line.start(), line.end()) for line in (header, page_number) if line is not None]
                    if header and page_number
                    else []
                )
                edits = {edit["line"]: edit for edit in header_trace["removed_headers"] if edit["page"] == index}
                line_offset = 0
                for number, original_line in enumerate(page.splitlines(keepends=True), 1):
                    if number in edits:
                        removals.append((line_offset, line_offset + len(original_line.rstrip("\r\n"))))
                    line_offset += len(original_line)
                cursor = left
                parts = []
                for a, b in sorted(removals):
                    if b <= left or a >= right:
                        continue
                    parts.append(page[cursor : max(cursor, a)])
                    cursor = max(cursor, min(b, right))
                parts.append(page[cursor:right])
                field_parts.append("".join(parts))
            offset += len(page) + 1
        result.append(
            {
                "scope": SCOPES[title],
                "coverage": list(map(str, coverage)),
                "reported_manager_name": report_manager_name(pages),
                "source_hashes": [source_hash],
                "source_url": source_url,
                "pages": touched,
                "excerpt": quote,
                "reported_events": _reported_event_rows("\n".join(field_parts)),
                "field_parsing_note": "字段值仅去除成对页眉/页码及整份报告已验证页码；完整excerpt保留原文",
                "parser_header_edits": [edit for edit in header_trace["removed_headers"] if edit["page"] in touched],
                "evidence_kind": "PERIODIC_REPORT_DECLARATION_NOT_COMPLETE_REGULATORY_OR_INDEPENDENT_REVIEW",
            }
        )
    return result


def declaration_coverage(declarations: list[dict], *, start: date, end: date) -> dict:
    """保留原始标题统计，再按标题真实包含的主体汇总；不推定无事项或全监管覆盖。

    旧“管理人、托管人及其高级管理人员”包括管理人公司及高管，但不等于所有
    相关从业人员；因此不能用旧合并标题填满现代的全从业人员覆盖。重复报告及
    A/C不增加自然日数，完全处于窗口外的报告也不为本窗口贡献来源哈希。
    """
    if end < start:
        raise ValueError("GOVERNANCE_COVERAGE_WINDOW_REVERSED")
    grouped = {}
    for item in declarations:
        left, right = map(date.fromisoformat, item["coverage"])
        if right < left or right > end or not item.get("source_hashes"):
            raise ValueError("GOVERNANCE_DECLARATION_PERIOD_OR_SOURCE")
        if right < start or left > end:
            continue
        grouped.setdefault(item["scope"], []).append((max(left, start), min(right, end), item["source_hashes"]))

    def summarize(intervals):
        days = [start + timedelta(days=index) for index in range((end - start).days + 1)]
        missing = [day for day in days if not any(left <= day <= right for left, right, _ in intervals)]
        gaps = []
        for day in missing:
            if gaps and date.fromisoformat(gaps[-1][1]) + timedelta(days=1) == day:
                gaps[-1][1] = str(day)
            else:
                gaps.append([str(day), str(day)])
        return {
            "covered_days": len(days) - len(missing),
            "window_days": len(days),
            "missing_intervals": gaps,
            "source_hashes": sorted({key for _, _, hashes in intervals for key in hashes}),
        }

    result = {scope: summarize(intervals) for scope, intervals in grouped.items()}
    subject_membership = {
        "MANAGER_COMPANY": {"MANAGER", "COMBINED_MANAGER_CUSTODIAN_OR_STAFF"},
        "MANAGER_SENIOR_MANAGEMENT": {"MANAGER_RELATED_STAFF", "COMBINED_MANAGER_CUSTODIAN_OR_STAFF"},
        "MANAGER_ALL_RELATED_STAFF": {"MANAGER_RELATED_STAFF"},
    }
    subjects = {}
    for subject, scopes in subject_membership.items():
        intervals = [interval for scope in sorted(scopes) for interval in grouped.get(scope, [])]
        if intervals:
            subjects[subject] = summarize(intervals) | {"included_report_scopes": sorted(scopes & grouped.keys())}
    return {
        "window": [str(start), str(end)],
        "by_report_scope": result,
        "by_subject_scope": subjects,
        "subject_scope_rule": "按标题包含关系并集统计；旧合并标题仅含公司/高管，不替代全部从业人员",
        "not_proven": ["监管事项检索完整性", "事件严重性及具体基金适用范围", "整改结论的独立证据", "独立复核结论"],
        "scope": "REPORT_DECLARATION_COVERAGE_NOT_GOVERNANCE_STATE",
    }


def safe_decision(raw: bytes) -> dict:
    """重读外汇局处罚表；日期只取处罚决定日期，金额由万元转元，保留原始单位。"""
    soup = BeautifulSoup(raw, "html.parser")
    required = {
        "行政处罚决定书文号",
        "违规主体名称",
        "作出处罚决定的行政机关名称",
        "违法事实",
        "处罚类别",
        "罚没款金额（万元）",
        "处罚决定日期",
    }
    fields = {}
    for row in soup.find_all("tr"):
        cells = row.find_all(["th", "td"], recursive=False)
        if len(cells) == 2:
            key, value = (re.sub(r"\s+", "", cell.get_text()) for cell in cells)
            if key in required:
                if key in fields:
                    raise ValueError("GOVERNANCE_SAFE_DUPLICATE_FIELD")
                fields[key] = value
    if not required <= fields.keys() or any(not fields[key] for key in required):
        raise ValueError("GOVERNANCE_SAFE_REQUIRED_TABLE_FIELDS")
    amount = Decimal(fields["罚没款金额（万元）"])
    if not amount.is_finite() or amount < 0:
        raise ValueError("GOVERNANCE_SAFE_PENALTY_AMOUNT")
    return {
        "document_number": fields["行政处罚决定书文号"],
        "legal_entity": fields["违规主体名称"],
        "authority": fields["作出处罚决定的行政机关名称"],
        "event_date": str(date.fromisoformat(fields["处罚决定日期"])),
        "reported_violation": fields["违法事实"],
        "measures": fields["处罚类别"],
        "fine_amount_cny": str(amount * 10000),
        "original_amount": fields["罚没款金额（万元）"],
        "original_unit": "万元",
        "scope": "REGULATOR_DECISION_NOT_REMEDIATION_OR_SEVERITY_REVIEW",
    }
