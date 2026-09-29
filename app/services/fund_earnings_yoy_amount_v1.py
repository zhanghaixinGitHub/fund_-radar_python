"""同比增加额的有限核验：预告增量与当期年报比较值分开，禁止倒算历史利润。"""

import re
from datetime import date, datetime, timedelta
from decimal import Decimal

from app.services.fund_information_history_v1 import normalize

LABELS = {
    "NET_PROFIT_ATTRIBUTABLE_TO_PARENT": "归属于上市公司股东的净利润",
    "ADJUSTED_NET_PROFIT_ATTRIBUTABLE_TO_PARENT": "归属于上市公司股东的扣除非经常性损益后的净利润",
}


def unique(pages, page, quote):
    """锚点必须唯一存在；不把缺失原文或重复段落视为核验通过。"""
    if isinstance(page, bool) or not isinstance(page, int) or not 1 <= page <= len(pages):
        raise ValueError("YOY_ANCHOR_PAGE_INVALID")
    text, needle = normalize(pages[page - 1]), normalize(quote)
    if not needle or text.count(needle) != 1:
        raise ValueError("YOY_ANCHOR_NOT_UNIQUE")
    return dict(page=page, offset=text.index(needle), text=needle)


def common(spec):
    """限定人民币、全年、中国企业会计准则，公开日次日 08:00 才可用。"""
    if spec["metric"] not in LABELS or not re.fullmatch(r"\d{6}", spec["issuer"]):
        raise ValueError("YOY_METRIC_OR_ISSUER_INVALID")
    start, end, public = [date.fromisoformat(spec[k]) for k in ("period_start", "period_end", "published_date")]
    if (start.month, start.day, end.year, end.month, end.day) != (1, 1, start.year, 12, 31) or public <= end:
        raise ValueError("YOY_COMPLETED_ANNUAL_PERIOD_REQUIRED")
    if spec["basis"] != "CONSOLIDATED_CN_GAAP" or spec["currency"] != "CNY":
        raise ValueError("YOY_ACCOUNTING_BASIS_OR_CURRENCY_INVALID")
    return {
        k: spec[k] for k in ("issuer", "metric", "period_start", "period_end", "basis", "currency", "document_id")
    } | {
        "published_date": str(public),
        "available_at": str(public + timedelta(days=1)) + "T08:00:00+08:00",
        "training_ready": False,
        "eligible_for_absolute_profit_chain": False,
    }


def review_increase_range(spec, pages):
    """核验明确同比增加额区间，金额、单位和同比百分比均来自同一段原文。"""
    base = common(spec)
    anchor = unique(pages, spec["page"], spec["quote"])
    period = unique(pages, spec["period_page"], spec["period_quote"])
    year = spec["period_start"][:4]
    if normalize(spec["period_quote"]).rstrip("。") != f"{year}年1月1日至{year}年12月31日":
        raise ValueError("YOY_PERIOD_ANCHOR_MISMATCH")
    basis = unique(pages, spec["basis_page"], spec["basis_quote"])
    if normalize(spec["basis_quote"]) != f"公司按照中国企业会计准则对{year}年度经营业绩进行测算":
        raise ValueError("YOY_FORECAST_BASIS_UNSUPPORTED")
    # 归母及扣非分别匹配；期间在独立锚点中核对，不凭标题或旧值补出区间。
    label = LABELS[spec["metric"]].replace("损益后的", "损益的")
    suffix = (
        label + r"与上年同期（法定披露数据）相比，?将增加(\d+\.\d+)亿元人民币～"
        r"(\d+\.\d+)亿元人民币，同比增加(\d+(?:\.\d+)?)%～(\d+(?:\.\d+)?)%。"
    )
    match = re.fullmatch(
        r"(?:[12]、)?(?:经本公司财务部门初步测算，预计" + year + r"年年度实现)?" + suffix, normalize(spec["quote"])
    )
    if not match or list(match.groups()) != spec["values"]:
        raise ValueError("YOY_INCREASE_RANGE_OR_MEANING_MISMATCH")
    low, high, percent_low, percent_high = map(Decimal, match.groups())
    if not 0 < low <= high or not 0 < percent_low <= percent_high:
        raise ValueError("YOY_RANGE_NOT_ORDERED_POSITIVE")
    return {
        **base,
        "measure": "FORECAST_YOY_INCREASE_RANGE",
        "original_values": spec["values"],
        "unit": "亿元",
        "lower_cny": str(low * 100000000),
        "upper_cny": str(high * 100000000),
        "reported_resolution_cny": "1000000",
        "anchor": anchor,
        "period_anchor": period,
        "basis_anchor": basis,
        "audit_anchor": unique(pages, spec["audit_page"], spec["audit_quote"]),
    }


def review_reported_increase(spec, pages):
    """在中国准则区块核对完整四列表；比较数仅在本年报公开后参与同比计算。

    只支持千元、整千元金额、至多连续两页；scope_quote 由中国准则小节至目标行末。
    禁止借用邻近国际准则表、季度表或元/股单位，也不将比较数另写为早期事实。
    """
    base = common(spec)
    first, last = spec["first_page"], spec["last_page"]
    if not 1 <= first <= last <= len(pages) or last - first > 1:
        raise ValueError("YOY_TABLE_PAGE_SPAN_INVALID")
    text, scope = normalize("\n".join(pages[first - 1 : last])), normalize(spec["scope_quote"])
    section = "（十一）按中国会计准则编制的会计数据"
    year = int(spec["period_start"][:4])
    header = f"项目{year}年{year - 1}年本年比上年增减（％）{year - 2}年"
    row, label = normalize(spec["row_quote"]), LABELS[spec["metric"]]
    if not scope or text.count(scope) != 1 or not scope.startswith(section) or not scope.endswith(row):
        raise ValueError("YOY_CN_TABLE_SCOPE_INVALID")
    if scope.count("单位：") != 1 or scope.count("（单位：千元）") != 1 or scope.count(header) != 1:
        raise ValueError("YOY_TABLE_UNIT_OR_HEADER_MISMATCH")
    if not scope.index("（单位：千元）") < scope.index(header) < scope.index(row):
        raise ValueError("YOY_TABLE_ORDER_INVALID")
    if not row.startswith(label) or scope.count(row) != 1:
        raise ValueError("YOY_TABLE_METRIC_MISMATCH")
    raw = spec["row_quote"]
    end = next((i for i in range(1, len(raw) + 1) if normalize(raw[:i]) == label), None)
    cells = raw[end:].split() if end else []
    if cells != spec["cells"] or len(cells) != 4:
        raise ValueError("YOY_TABLE_CELL_SEQUENCE_MISMATCH")
    if any(not re.fullmatch(r"\d{1,3}(?:,\d{3})*|\d+", cells[i]) for i in (0, 1, 3)):
        raise ValueError("YOY_TABLE_AMOUNT_INVALID")
    current, prior = [Decimal(v.replace(",", "")) for v in cells[:2]]
    if prior <= 0 or not re.fullmatch(r"\d+\.\d{2}", cells[2]):
        raise ValueError("YOY_TABLE_PERCENT_INVALID")
    computed_percent = (100 * (current - prior) / prior).quantize(Decimal("0.01"))
    if computed_percent != Decimal(cells[2]):
        raise ValueError("YOY_TABLE_PERCENT_INCONSISTENT")
    return {
        **base,
        "measure": "REPORTED_YOY_INCREASE_FROM_CURRENT_REPORT_COMPARATIVES",
        "unit": "千元",
        "original_cells": cells,
        "current_cny": str(current * 1000),
        "previous_comparative_cny": str(prior * 1000),
        "increase_cny": str((current - prior) * 1000),
        "reported_resolution_cny": "1000",
        "percent": cells[2],
        "comparative_is_not_independent_prior_disclosure": True,
        "anchor": dict(first_page=first, last_page=last, text=scope),
        "audit_anchor": unique(pages, spec["audit_page"], spec["audit_quote"]),
    }


def compare_increases(forecast, report, as_of):
    """比较同公司、年度、指标及会计准则的增量，不能把增量当绝对利润比较。"""
    keys = ("issuer", "metric", "period_start", "period_end", "basis", "currency")
    if any(forecast[k] != report[k] for k in keys):
        raise ValueError("INCOMPARABLE_YOY_AMOUNTS")
    if (
        forecast["measure"] != "FORECAST_YOY_INCREASE_RANGE"
        or report["measure"] != "REPORTED_YOY_INCREASE_FROM_CURRENT_REPORT_COMPARATIVES"
        or forecast["document_id"] == report["document_id"]
    ):
        raise ValueError("YOY_INDEPENDENT_DISCLOSURE_ROLES_REQUIRED")
    a, b, cutoff = [datetime.fromisoformat(v) for v in (forecast["available_at"], report["available_at"], as_of)]
    if any(t.tzinfo is None for t in (a, b, cutoff)) or not a < b <= cutoff:
        raise ValueError("YOY_NOT_AVAILABLE_IN_ORDER")
    value, low, high = [Decimal(v) for v in (report["increase_cny"], forecast["lower_cny"], forecast["upper_cny"])]
    return dict(
        issuer=report["issuer"],
        metric=report["metric"],
        as_of=as_of,
        forecast_id=forecast["document_id"],
        report_id=report["document_id"],
        increase_cny=str(value),
        forecast_lower_cny=str(low),
        forecast_upper_cny=str(high),
        relation="BELOW_FORECAST_INCREASE"
        if value < low
        else "ABOVE_FORECAST_INCREASE"
        if value > high
        else "WITHIN_FORECAST_INCREASE_RANGE",
        training_ready=False,
        is_market_consensus_surprise=False,
        whole_history_admitted=False,
    )
