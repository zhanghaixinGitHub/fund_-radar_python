"""同步报告的版式兼容入口；保留旧研究解析器和原件，不放宽金额、身份及完整性校验。"""

import hashlib
import re

from app.integrations.dbfund_reports import parse_text as parse_original
from app.integrations.fund_report_sections_v2 import normalize_pages

PARSER_VERSION = "FUND_REPORT_LAYOUT_V3"
_HAN = r"\u4e00-\u9fff"
_ROW = re.compile(r"^(\d{1,4})\s+(\d{5,6})(?=\s|[\u4e00-\u9fff])\s*(.*)$")
_NUMBER = re.compile(r"[\d,]+(?:\.\d*)?%?$")
_QUANTITY = re.compile(r"(?:\d+|\d{1,3}(?:,\d{3})+)(?:\.\d+)?$")
_VALUE = re.compile(r"(?:\d+|\d{1,3}(?:,\d{3})+)\.\d{2}$")


def _holding_row(lines):
    """只拼接同一股票行中明确折行的单元格；尾数必须恰好补齐，歧义和残缺仍拒绝。"""
    match = _ROW.fullmatch(lines[0])
    rank, code, first = match.groups()
    tokens = first.split()
    # 股票名称可以折成两行，也可以和代码粘连；数量、金额、权重始终是最后三列。
    first_numeric = next((i for i, token in enumerate(tokens) if _NUMBER.fullmatch(token)), len(tokens))
    name = tokens[:first_numeric]
    numbers = tokens[first_numeric:]
    continuations = []
    for line in lines[1:]:
        for token in line.split():
            if _NUMBER.fullmatch(token):
                continuations.append(token)
            elif re.fullmatch(rf"[{_HAN}Ａ-ＺA-Za-z*－—-]+", token):
                name.append(token)
            else:
                raise ValueError("REPORT_LAYOUT_ROW_AMBIGUOUS")
    if not numbers:
        numbers, continuations = continuations, []
    if len(numbers) != 3 or not name:
        raise ValueError("REPORT_LAYOUT_ROW_INCOMPLETE")
    quantity, value, weight = numbers
    # 不把行后任意数字当尾数：只接受十进制位数/千位分组本身能确定缺位的情况。
    for index, (token, pattern) in enumerate(((quantity, _QUANTITY), (value, _VALUE))):
        if pattern.fullmatch(token):
            continue
        if not continuations or not continuations[0].isdigit():
            raise ValueError("REPORT_LAYOUT_NUMBER_INCOMPLETE")
        repaired = token + continuations.pop(0)
        if not pattern.fullmatch(repaired):
            raise ValueError("REPORT_LAYOUT_NUMBER_AMBIGUOUS")
        numbers[index] = repaired
    if continuations or not re.fullmatch(r"\d+(?:\.\d+)?%?", weight):
        raise ValueError("REPORT_LAYOUT_ROW_AMBIGUOUS")
    return f"{rank} {code} {''.join(name)} {' '.join(numbers)}"


def normalize_layout(pages, *, fund_name):
    """限定股票和行业章节规范化，保存变更前后摘要及逐块证据；不推算任何财务数值。"""
    normalized, heading_edits = normalize_pages(pages)
    text = "\n".join(normalized)
    lines = [re.sub(r"[ \t\u3000]+", " ", line).strip() for line in text.splitlines()]
    edits = []
    # 行业名称跨列换行：名称上半行、行业代码和金额行、名称下半行必须连续。
    text = "\n".join(lines)
    for section in list(re.finditer(r"(?ms)^[578]\.2\s+(?:报告期末|期末)?按行业分类.*?(?=^[578]\.3\s)", text))[::-1]:
        original = section[0]
        body = re.sub(
            rf"(?m)^([{_HAN}、，]+)\n\s*([A-S])\s+([\d,]+\.\d{{2}})\s+([\d.]+)\n([{_HAN}、，]+)$",
            r"\2 \1\5 \3 \4", original,
        )
        body = re.sub(r"(?m)^(合计\s+[\d,]+\.\d)\s+([\d.]+)\n(\d)\s*$", r"\1\3 \2", body)
        if body != original:
            edits.append({"rule": "INDUSTRY_WRAPPED_CELLS", "before": original, "after": body})
            text = text[:section.start()] + body + text[section.end():]
    # 从后往前替换，避免目录或前面替换改变坐标；不处理找不到结束标题的截断报告。
    for section in list(re.finditer(
        r"(?ms)^[578]\.3\s+(?:期末|报告期末)[^\n]*股票\s*投\s*资\s*明\s*细[^\n]*\n(.*?)(?=^[578]\.4\s)", text
    ))[::-1]:
        body = section[1]
        row_lines = body.splitlines()
        clean = []
        for line in row_lines:
            compact = re.sub(r"\s", "", line)
            # 已绑定基金名称的报告页眉及明确页码可跳过；普通注释不作为股票名称。
            if not line or re.fullmatch(r"第?\s*\d+\s*页(?:\s*共\s*\d+\s*页)?", line):
                continue
            if fund_name in compact and re.search(r"20\d{2}年.*报告$", compact):
                continue
            clean.append(line)
        starts = [i for i, line in enumerate(clean) if _ROW.fullmatch(line)]
        if not starts:
            continue
        output = clean[:starts[0]]
        for ordinal, start in enumerate(starts):
            end = starts[ordinal + 1] if ordinal + 1 < len(starts) else len(clean)
            block = clean[start:end]
            # 报告末尾说明保留原位，不能拼进最后一只股票；只切分明确的“注”提示。
            note = next((i for i, line in enumerate(block[1:], 1)
                         if line.startswith(("注：", "注:", "注 ")) or re.match(r"[578]\.3\.\d+\s", line)), len(block))
            row = _holding_row(block[:note])
            output.extend([row, *block[note:]])
        repaired = "\n".join(output) + "\n"
        if repaired != body:
            edits.append({"rule": "HOLDING_WRAPPED_CELLS", "before": body, "after": repaired})
            text = text[:section.start(1)] + repaired + text[section.end(1):]
    return text, {"headings": heading_edits, "blocks": edits}


def parse_text(pages, title, *, fund_code="002112", fund_name="德邦鑫星价值", master_code="001412"):
    """对已核验原文进行版式恢复，再执行原解析器所有金额、比例、序号和来源身份校验。"""
    # 已能严格通过的版式保持原业务结果；仅对明确的表格解析失败启用兼容路径。
    # 日期、基金身份和权重越界等错误不能通过文本恢复绕过。
    try:
        result = parse_original(pages, title, fund_code=fund_code, fund_name=fund_name, master_code=master_code)
    except ValueError as error:
        if str(error) not in {
            "REPORT_HOLDINGS_SECTION_MISSING", "REPORT_STOCK_NAV_RATIO_MISSING",
            "REPORT_HOLDINGS_RANK_OR_DUPLICATE", "REPORT_HOLDINGS_PARSE_EMPTY",
            "REPORT_FULL_HOLDING_TOTAL_MISMATCH", "REPORT_INDUSTRY_TOTAL_MISMATCH",
        }:
            raise
    else:
        result.update(parser_version=PARSER_VERSION, layout_normalization={"used_original_layout": True})
        return result
    normalized, evidence = normalize_layout(pages, fund_name=fund_name)
    result = parse_original([normalized], title, fund_code=fund_code, fund_name=fund_name, master_code=master_code)
    result.update(parser_version=PARSER_VERSION, layout_normalization={
        "original_text_sha256": hashlib.sha256("\n".join(pages).encode()).hexdigest(),
        "normalized_text_sha256": hashlib.sha256(normalized.encode()).hexdigest(),
        **evidence,
    })
    return result
