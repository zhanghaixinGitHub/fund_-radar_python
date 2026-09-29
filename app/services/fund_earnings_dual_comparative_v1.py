"""核验同一披露中的法定、重述两种同比基数，不生成跨披露业绩变化。

限定正利润、全年归母指标以及两种明确表格布局。原单位、公开日、未审计
状态分别保存；“比较期按准则重述”不等于本期准则和全部合并范围已核准。
"""

import re
from datetime import date, timedelta
from decimal import Decimal

from app.services.fund_earnings_yoy_amount_v1 import unique
from app.services.fund_information_history_v1 import normalize

LABEL = "归属于上市公司股东的净利润"
MONEY = re.compile(r"(?:\d{1,3}(?:,\d{3})+|\d+)\.\d{2}")


def growth_rounding_check(current, comparative, reported):
    """按原数小数位构造舍入区间，避免把亿元展示精度当作完整计算精度。

    仅核正数基数；负基数、零基数不能偷偷沿用普通增长率公式。输出是
    本次披露的算术一致性证据，不是原始未舍入数或独立历史事实。
    """
    if not MONEY.fullmatch(current) or not MONEY.fullmatch(comparative):
        raise ValueError("DUAL_COMPARATIVE_AMOUNT_FORMAT")
    if not re.fullmatch(r"-?\d+\.\d{2}", reported):
        raise ValueError("DUAL_COMPARATIVE_PERCENT_FORMAT")
    c, p, r = [Decimal(x.replace(",", "")) for x in (current, comparative, reported)]
    half = Decimal("0.005")
    if min(c, p) <= half:
        raise ValueError("DUAL_COMPARATIVE_POSITIVE_BASIS_REQUIRED")
    low = ((c - half) / (p + half) - 1) * 100
    high = ((c + half) / (p - half) - 1) * 100
    if r + half < low or r - half > high:
        raise ValueError("DUAL_COMPARATIVE_PERCENT_INCONSISTENT")
    return {
        "possible_percent_low": str(low),
        "possible_percent_high": str(high),
        "reported_percent": reported,
        "is_rounding_consistency_only": True,
    }


def review_dual_comparative(spec, pages):
    """核对完整行、表头列序、原单位及重述解释；保留当前与两个比较列。

    FORECAST 布局为本期、法定、重述，快报布局为本期、重述、法定、两同比。
    冻结计划必须提供连续原文和所有单元格。不能只挑一个增长率作为结论。
    """
    year = spec["year"]
    public = date.fromisoformat(spec["published_date"])
    if public <= date(year, 12, 31) or not re.fullmatch(r"\d{6}", spec["issuer"]):
        raise ValueError("DUAL_COMPARATIVE_PERIOD_OR_ISSUER")
    kind = spec["kind"]
    if kind == "FORECAST":
        title = f"{year}年年度业绩预增公告"
        header = "项目本报告期上年同期法定披露数据重述调整数据"
        label, unit, currency = LABEL + "(万元)", "万元", None
        roles = ["CURRENT", "PRIOR_STATUTORY", "PRIOR_RESTATED"]
        period = unique(pages, spec["period_page"], f"{year}年1月1日至{year}年12月31日。")
    elif kind == "PRELIMINARY_RESULT":
        title = f"{year}年度业绩快报公告"
        header = "项目本报告期上年同期增减变动幅度重述调整数据法定披露数据重述调整数据法定披露数据"
        label, unit, currency = LABEL, "亿元", "CNY"
        roles = ["CURRENT", "PRIOR_RESTATED", "PRIOR_STATUTORY", "YOY_RESTATED", "YOY_STATUTORY"]
        period = unique(pages, spec["title_page"], title)
    else:
        raise ValueError("DUAL_COMPARATIVE_STAGE_UNSUPPORTED")
    if spec["column_roles"] != roles or normalize(spec["header_quote"]) != header:
        raise ValueError("DUAL_COMPARATIVE_COLUMN_ORDER")
    anchors = {
        "title": unique(pages, spec["title_page"], title),
        "period": period,
        "header": unique(pages, spec["page"], spec["header_quote"]),
        "row": unique(pages, spec["page"], spec["row_quote"]),
    }
    if anchors["header"]["offset"] + len(header) > anchors["row"]["offset"]:
        raise ValueError("DUAL_COMPARATIVE_HEADER_POSITION")
    raw = spec["row_quote"]
    end = next((i for i in range(1, len(raw) + 1) if normalize(raw[:i]) == label), None)
    cells = raw[end:].split() if end else []
    if cells != spec["cells"] or len(cells) != len(roles) or any(not MONEY.fullmatch(x) for x in cells[:3]):
        raise ValueError("DUAL_COMPARATIVE_FULL_ROW_MISMATCH")
    if kind == "PRELIMINARY_RESULT":
        scope = normalize(spec["scope_quote"])
        if not scope.startswith("单位：人民币亿元") or not scope.endswith(normalize(raw)):
            raise ValueError("DUAL_COMPARATIVE_UNIT_SCOPE")
        if scope.count("单位：") != 1 or scope.count(header) != 1:
            raise ValueError("DUAL_COMPARATIVE_UNIT_SCOPE")
        anchors["scope"] = unique(pages, spec["page"], spec["scope_quote"])
        rates = {"PRIOR_RESTATED": cells[3].removesuffix("%"), "PRIOR_STATUTORY": cells[4].removesuffix("%")}
        if any(not c.endswith("%") for c in cells[3:]):
            raise ValueError("DUAL_COMPARATIVE_PERCENT_UNIT")
    else:
        rates = spec["reported_percent"]
        if set(rates) != {"PRIOR_STATUTORY", "PRIOR_RESTATED"}:
            raise ValueError("DUAL_COMPARATIVE_BOTH_RATES_REQUIRED")
        for role, word in (("PRIOR_STATUTORY", "法定披露数据"), ("PRIOR_RESTATED", "重述调整数据")):
            value = Decimal(rates[role])
            verb = "增加" if value >= 0 else "减少"
            quote = f"预计{year}年实现{LABEL}与上年同期（{word}）相比，将{verb}约{abs(value)}%"
            anchors[role] = unique(pages, spec["rate_page"], quote)
    audit = normalize(spec["audit_quote"])
    if "未经会计师事务所审计" not in audit:
        raise ValueError("DUAL_COMPARATIVE_UNAUDITED_REQUIRED")
    anchors["audit"] = unique(pages, spec["audit_page"], spec["audit_quote"])
    restatement = (
        "重述调整数据：为公司按照企业会计准则对同一控制下企业合并的要求，"
        f"对{year - 1}年年度报告数据进行重述调整后的数据。"
    )
    anchors["restatement"] = unique(pages, spec["restatement_page"], restatement)
    amounts = dict(zip(roles[:3], cells[:3], strict=True))
    checks = {
        role: growth_rounding_check(amounts["CURRENT"], amounts[role], rates[role])
        for role in ("PRIOR_STATUTORY", "PRIOR_RESTATED")
    }
    return {
        "document_id": spec["document_id"],
        "issuer": spec["issuer"],
        "kind": kind,
        "period_start": f"{year}-01-01",
        "period_end": f"{year}-12-31",
        "published_date": str(public),
        "available_at": str(public + timedelta(days=1)) + "T08:00:00+08:00",
        "unit": unit,
        "currency": currency,
        "amounts_in_source_unit": amounts,
        "reported_percent": rates,
        "rounding_checks": checks,
        "anchors": anchors,
        "audit_status": "NOT_AUDITED_BY_ACCOUNTING_FIRM",
        "comparative_period": {"start": f"{year - 1}-01-01", "end": f"{year - 1}-12-31"},
        "comparatives_are_independent_earlier_disclosures": False,
        "accounting_standard_verified": False,
        "consolidation_scope_verified": False,
        "basis": "CURRENT_ACCOUNTING_AND_CONSOLIDATION_UNSPECIFIED",
        "metric": "NET_PROFIT_ATTRIBUTABLE_TO_PARENT",
        "training_ready": False,
    }


def same_day_order_status(first, second):
    """同一公开日的不同文件不能依公告编号排序为先后业绩变化。"""
    if {first["kind"], second["kind"]} != {"FORECAST", "PRELIMINARY_RESULT"}:
        raise ValueError("SAME_DAY_PAIR_STAGE_MISMATCH")
    if first["document_id"] == second["document_id"] or any(
        first[k] != second[k]
        for k in ("issuer", "metric", "period_start", "period_end", "published_date", "available_at")
    ):
        raise ValueError("SAME_DAY_PAIR_SCOPE_MISMATCH")
    return {
        "document_ids": [first["document_id"], second["document_id"]],
        "available_at": first["available_at"],
        "intraday_order": None,
        "eligible_for_sequential_change": False,
        "new_comparable_numeric_changes": 0,
        "reason": "SAME_PUBLIC_DAY_WITHOUT_VERIFIED_INTRADAY_ORDER",
        "training_ready": False,
    }
