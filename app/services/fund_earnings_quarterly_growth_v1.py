"""先保存一季度/上半年预告的同比口径，再观察同一期报告；未知准则不补推。"""

import re
from datetime import date, datetime, timedelta
from decimal import Decimal

from app.services.fund_earnings_identity_v1 import occurrence
from app.services.fund_earnings_yoy_amount_v1 import unique
from app.services.fund_information_history_v1 import normalize

LABEL = "归属于上市公司股东的净利润"
NUMBER = r"(?:\d{1,3}(?:,\d{3})+|\d+)"


def _base(spec):
    """期间来自登记及原文核对；可用日统一保守延后至公开次日 08:00。"""
    start, end, public = [date.fromisoformat(spec[k]) for k in ("period_start", "period_end", "published_date")]
    if (
        str(start) != f"{start.year}-01-01"
        or str(end) not in {f"{start.year}-03-31", f"{start.year}-06-30"}
        or public <= end
    ):
        raise ValueError("COMPLETED_Q1_OR_H1_REQUIRED")
    if not re.fullmatch(r"\d{6}", spec["issuer"]):
        raise ValueError("ISSUER_REQUIRED")
    return {k: spec[k] for k in ("document_id", "issuer", "period_start", "period_end", "published_date")} | {
        "metric": "NET_PROFIT_ATTRIBUTABLE_TO_PARENT",
        "available_at": str(public + timedelta(days=1)) + "T08:00:00+08:00",
        "accounting_basis": "UNKNOWN",
        "accounting_standard_verified": False,
        "eligible_for_profit_level_comparison": False,
        "training_ready": False,
    }


def review_percentage_forecast(spec, pages):
    """只提取原文约增幅和上年数，绝不据此倒算本期精确利润。"""
    base = _base(spec)
    year = spec["period_start"][:4]
    half = spec["period_end"].endswith("06-30")
    label, end = ("上半年", "6月30日") if half else ("第一季度", "3月31日")
    expected = f"{year}年1月1日至{year}年{end}。"
    if normalize(spec["period_quote"]) != expected:
        raise ValueError("FORECAST_PERIOD_MISMATCH")
    pattern = (
        f"经财务部门初步测算，本公司预计{year}年{label}{LABEL}与上年同期相比，增长"
        r"(\d+(?:\.\d+)?)%左右。"
    )
    match = re.fullmatch(pattern, normalize(spec["quote"]))
    prior = re.fullmatch("二、上年同期业绩情况1、" + LABEL + "：(" + NUMBER + ")元", normalize(spec["prior_quote"]))
    if not match or match[1] != spec["percent"] or not prior or prior[1] != spec["prior_value"]:
        raise ValueError("FORECAST_MEANING_OR_VALUE_MISMATCH")
    if Decimal(match[1]) <= 0 or Decimal(prior[1].replace(",", "")) <= 0:
        raise ValueError("POSITIVE_GROWTH_AND_REFERENCE_REQUIRED")
    audit = "本次所预计的业绩未经注册会计师审计。"
    return {
        **base,
        "kind": "FORECAST",
        "yoy_percent": match[1],
        "approximate": True,
        "approximation_marker": "左右",
        "currency": None,
        "unit": "元",
        "previous_comparative_original": prior[1],
        "current_profit_derived": False,
        "comparative_is_independent_earlier_disclosure": False,
        "audit_status": "UNAUDITED",
        "anchors": [
            unique(pages, spec["page"], spec["quote"]),
            unique(pages, spec["page"], spec["period_quote"]),
            unique(pages, spec["page"], spec["prior_quote"]),
            unique(pages, spec["page"], audit),
        ],
    }


def review_q1_report(spec, pages):
    """核对一季报同表元/人民币、年初至期末三列及归母行，避免净资产错行。"""
    base = _base(spec)
    if not spec["period_end"].endswith("03-31"):
        raise ValueError("Q1_REPORT_REQUIRED")
    year = spec["period_start"][:4]
    if normalize(spec["title_quote"]) != year + "年第一季度报告":
        raise ValueError("REPORT_PERIOD_MISMATCH")
    scope = normalize(spec["scope_quote"])
    header = "年初至报告期末上年初至上年报告期末比上年同期增减（%）"
    row = normalize(spec["row_quote"])
    if (
        not scope.startswith("2.1主要财务数据单位：元币种：人民币")
        or not scope.endswith(row)
        or scope.count(header) != 1
        or scope.count("单位：") != 1
        or scope.index(header) > scope.index(row)
    ):
        raise ValueError("REPORT_TABLE_SCOPE_MISMATCH")
    # 按空白解析单元格，不在移除空白后猜测长整数边界。
    raw = spec["row_quote"]
    offset = next((i for i in range(1, len(raw) + 1) if normalize(raw[:i]) == LABEL), None)
    cells = raw[offset:].split() if offset else []
    if (
        cells != spec["cells"]
        or len(cells) != 3
        or any(not re.fullmatch(NUMBER, v) for v in cells[:2])
        or not re.fullmatch(r"\d+\.\d{2}", cells[2])
    ):
        raise ValueError("REPORT_CELL_SEQUENCE_MISMATCH")
    current, prior = [Decimal(v.replace(",", "")) for v in cells[:2]]
    if prior <= 0 or (100 * (current / prior - 1)).quantize(Decimal("0.01")) != Decimal(cells[2]):
        raise ValueError("REPORT_YOY_RECALCULATION_FAILED")
    return {
        **base,
        "kind": "REPORTED_RESULT",
        "yoy_percent": cells[2],
        "approximate": False,
        "currency": "CNY",
        "unit": "元",
        "current_cny": str(current),
        "previous_comparative_original": cells[1],
        "comparative_is_independent_earlier_disclosure": False,
        "audit_status": "UNAUDITED",
        "original_cells": cells,
        "anchors": [
            occurrence(pages, spec["title_page"], spec["title_quote"]),
            unique(pages, spec["page"], spec["scope_quote"]),
            unique(pages, spec["page"], "本公司第一季度报告未经审计。"),
        ],
    }


def observe_same_period(forecast, report, as_of):
    """只记录两次披露的数值差；未知准则、近似区间未明，不能成为合格模型输入。"""
    keys = ("issuer", "metric", "period_start", "period_end", "previous_comparative_original")
    if (
        any(forecast[k] != report[k] for k in keys)
        or forecast["kind"] != "FORECAST"
        or report["kind"] != "REPORTED_RESULT"
        or forecast["document_id"] == report["document_id"]
    ):
        raise ValueError("SAME_PERIOD_AND_SAME_COMPARATIVE_REQUIRED")
    a, b, cutoff = [datetime.fromisoformat(v) for v in (forecast["available_at"], report["available_at"], as_of)]
    if any(v.tzinfo is None for v in (a, b, cutoff)) or not a < b <= cutoff:
        raise ValueError("DISCLOSURES_NOT_AVAILABLE_IN_ORDER")
    return {
        "forecast_id": forecast["document_id"],
        "report_id": report["document_id"],
        "as_of": as_of,
        "forecast_percent_approximate": forecast["yoy_percent"],
        "reported_percent": report["yoy_percent"],
        "difference_from_stated_number_percentage_points": str(
            Decimal(report["yoy_percent"]) - Decimal(forecast["yoy_percent"])
        ),
        "forecast_range_established": False,
        "range_exceedance_established": False,
        "is_market_consensus_surprise": False,
        "accounting_basis_equivalence_verified": False,
        "qualified_numeric_comparison": False,
        "training_ready": False,
    }
