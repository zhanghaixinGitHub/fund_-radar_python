"""独立报告解析 V2：仅兼容章节号后省略空格，沿用原金额、比例和股票身份校验。

旧解析器和采集入口不改。新研究必须显式选择此入口，并保存本版本和正文摘要，
不能将新输出回写成旧实验当时的解析结果。
"""

import hashlib
import re

from app.integrations.dbfund_reports import parse_text as parse_v1

PARSER_VERSION = "FUND_REPORT_SECTIONS_V2"
_HEADINGS = (
    r"(?m)^([578]\.1)(?=(?:报告期末|期末)基金资产组合情况)",
    r"(?m)^([578]\.2)(?=(?:报告期末|期末)?按行业分类)",
    r"(?m)^([578]\.3)(?=(?:报告期末|期末)[^\n]*股票\s*投\s*资\s*明\s*细)",
    r"(?m)^([578]\.4)(?=(?:报告期末|期末|报告期内|本报告期内))",
)


def normalize_pages(pages: list[str]) -> tuple[list[str], list[dict]]:
    """返回新文本和变更位置；不处理 5.3.1 等子章节，不变更表格数值或已有空格。

    pages 来自已核验的正文提取。改动仅发生于行首明确的顶层章节标题；空正文拒绝，
    没有匹配的原文继续交给 V1 自己判断，不靠猜测补持仓表。
    """
    if not pages or any(not isinstance(page, str) for page in pages) or not any(p.strip() for p in pages):
        raise ValueError("REPORT_EMPTY_OR_INVALID_PAGES")
    normalized, edits = [], []
    for page_number, original in enumerate(pages, 1):
        text = original
        for pattern in _HEADINGS:
            matches = list(re.finditer(pattern, text))
            edits.extend(
                {"page": page_number, "line": text.count("\n", 0, m.start()) + 1, "section": m[1]} for m in matches
            )
            text = re.sub(pattern, r"\1 ", text)
        if re.sub(r"\s", "", text) != re.sub(r"\s", "", original):
            raise ValueError("REPORT_NORMALIZATION_CHANGED_CONTENT")
        normalized.append(text)
    return normalized, sorted(edits, key=lambda e: (e["page"], e["line"]))


def parse_text(
    pages: list[str],
    title: str,
    *,
    fund_code: str = "002112",
    fund_name: str = "德邦鑫星价值",
    master_code: str = "001412",
    derive_missing_weights: bool = False,
) -> dict:
    """解析已核验的报告正文，保留原返回字段并增加新版本与空格规范化证据。

    缺百分比仍按原默认拒绝；此修复不启用推导权重、不填零、不更改报告公开时间。
    调用方仍须核对正文来源、基金/份额身份、目录日期及版本修订情况。
    """
    normalized, edits = normalize_pages(pages)
    result = parse_v1(
        normalized,
        title,
        fund_code=fund_code,
        fund_name=fund_name,
        master_code=master_code,
        derive_missing_weights=derive_missing_weights,
    )
    return {
        **result,
        "parser_version": PARSER_VERSION,
        "section_normalization": {
            "edits": edits,
            "original_text_sha256": hashlib.sha256("\n".join(pages).encode("utf-8")).hexdigest(),
            "normalized_text_sha256": hashlib.sha256("\n".join(normalized).encode("utf-8")).hexdigest(),
            "only_whitespace_changed": True,
        },
    }
