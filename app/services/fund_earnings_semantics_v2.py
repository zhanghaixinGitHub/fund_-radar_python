"""正文语义补充：业绩预增/预减同属预告，表头百分比独立于金额单位。"""

import re
from decimal import Decimal

from app.services.fund_earnings_evidence_v1 import earnings_kind
from app.services.fund_information_history_v1 import normalize


def earnings_title_kind(title):
    """扩充公司自身业绩预告的公开写法；说明会、业绩补偿、考核不作为预告。"""
    if any(w in title for w in ("持续督导", "保荐机构", "保荐工作")):
        return None
    old = earnings_kind(title)
    if old:
        return old
    if any(w in title for w in ("说明会", "核查", "回复", "独立意见", "承诺", "考核", "补偿")):
        return None
    return "FORECAST" if re.search(r"业绩(?:预增|预减|预盈|预亏|扭亏)", title) else None


def review_table_yoy(spec, pages):
    """原件百分比在表头、行内不带 % 时，固定表头、指标行和明确列序一起复核。

    row_values 是该指标行按阅读顺序的数值；value_index 指同比列，不能只在
    全页搜索一个碰巧相同的数字。负数直接保留，不从金额除法制造同比。
    """
    if spec["comparison"] != "YEAR_ON_YEAR" or spec["unit"] != "PERCENT":
        raise ValueError("NOT_YEAR_ON_YEAR_PERCENT")
    if spec["kind"] not in {"REPORTED_RESULT", "PRELIMINARY_RESULT"}:
        raise ValueError("TABLE_KIND_NOT_REGISTERED")
    if not spec.get("period_start") or not spec.get("period_end") or not spec.get("metric") or not spec.get("basis"):
        raise ValueError("TABLE_BASIS_MISSING")
    if not 1 <= spec["page"] <= len(pages):
        raise ValueError("TABLE_PAGE_INVALID")
    page = normalize(pages[spec["page"] - 1])
    header, row = normalize(spec["header"]), normalize(spec["row_quote"])
    if not header or not row or page.count(header) != 1 or page.count(row) != 1:
        raise ValueError("TABLE_ANCHOR_NOT_UNIQUE")
    if "上年同期" not in header or not any(s in header for s in ("%", "％")):
        raise ValueError("TABLE_NOT_YOY_PERCENT_HEADER")
    if "百分点" in row:
        raise ValueError("PERCENTAGE_POINTS_NOT_YOY")
    # 从带空白的原行抽取，保留单元格分隔；规范化只用于原件锚点核对。
    literals = re.findall(r"[-+]?\d[\d,，]*(?:\.\d+)?", spec["row_quote"].replace("−", "-"))
    numbers = [Decimal(n.replace(",", "").replace("，", "")) for n in literals]
    expected = [Decimal(str(n)) for n in spec["row_values"]]
    if numbers != expected:
        raise ValueError("TABLE_ROW_VALUES_MISMATCH")
    index = spec["value_index"]
    if not isinstance(index, int) or index < 0 or index >= len(numbers):
        raise ValueError("TABLE_COLUMN_INVALID")
    if Decimal(str(spec["value"])) != numbers[index]:
        raise ValueError("TABLE_YOY_COLUMN_MISMATCH")
    return {
        **spec,
        "reported_value": str(numbers[index]),
        "header_anchor": {"page": spec["page"], "offset": page.index(header), "text": header},
        "row_anchor": {"page": spec["page"], "offset": page.index(row), "text": row},
        "derived_from_money": False,
        "training_ready": False,
    }
