"""核验三季报内全年预告的原始字段；未核准的口径不进入金额变化计算。"""

from datetime import date, timedelta
from decimal import Decimal

from app.services.fund_earnings_evidence_v1 import compare_claims
from app.services.fund_earnings_identity_v1 import occurrence
from app.services.fund_information_history_v1 import normalize


def review_annual_forecast(spec, pages):
    """只核全年归母预告行，保留万元与同比，不拿前三季度已实现数替代。

    本阶段不猜会计准则、合并范围或币种。原文只有“万元”的情况下先保存
    原单位及金额，不自动输出带确定币种的金额特征。上年数属于本次比较栏。
    """
    year = spec["year"]
    expected_title = {f"{year}年第三季度报告全文", f"{year}年第三季度报告正文"}
    if normalize(spec["source_title"]) not in expected_title:
        raise ValueError("ANNUAL_FORECAST_CARRIER_MISMATCH")
    if spec["period_start"] != f"{year}-01-01" or spec["period_end"] != f"{year}-12-31":
        raise ValueError("ANNUAL_FORECAST_PERIOD_MISMATCH")
    published = date.fromisoformat(spec["published_date"])
    if published.year != year or published >= date(year, 12, 31):
        raise ValueError("ANNUAL_FORECAST_PUBLIC_DATE_INVALID")
    label = "归属于上市公司股东的净利润"
    expected = {
        "heading": f"四、对{year}年度经营业绩的预计",
        "stage": f"{year}年度预计的经营业绩情况：{label}为正值且不属于扭亏为盈的情形",
        "range": f"{year}年度{label}变动区间（万元）{spec['values'][0]}至{spec['values'][1]}",
        "growth": f"{year}年度{label}变动幅度{spec['yoy_percent'][0]}%至{spec['yoy_percent'][1]}%",
        "comparative": f"{year - 1}年度{label}（万元）{spec['comparative_value']}",
    }
    anchors = {k: occurrence(pages, spec["page"], text) for k, text in expected.items()}
    if any(len(a["offsets"]) != 1 for a in anchors.values()):
        raise ValueError("ANNUAL_FORECAST_ANCHOR_NOT_UNIQUE")
    if (
        not anchors["heading"]["offsets"][0]
        < anchors["stage"]["offsets"][0]
        < anchors["growth"]["offsets"][0]
        < anchors["range"]["offsets"][0]
        < anchors["comparative"]["offsets"][0]
    ):
        raise ValueError("ANNUAL_FORECAST_ROW_ORDER_CONFLICT")
    values = [Decimal(x.replace(",", "")) for x in spec["values"]]
    growth = [Decimal(x) for x in spec["yoy_percent"]]
    if len(values) != 2 or values != sorted(values) or min(values) <= 0 or growth != sorted(growth):
        raise ValueError("ANNUAL_FORECAST_RANGE_CONFLICT")
    return {
        **spec,
        "kind": "FORECAST",
        "metric": "NET_PROFIT_ATTRIBUTABLE_TO_PARENT",
        "unit": "万元",
        "amount_values_in_source_unit": [str(x) for x in values],
        "reported_resolution_in_source_unit": [str(Decimal(1).scaleb(x.as_tuple().exponent)) for x in values],
        "yoy_percent": [str(x) for x in growth],
        "anchors": anchors,
        "available_at": str(published + timedelta(days=1)) + "T08:00:00+08:00",
        "audit_status": "FORECAST_NOT_ACTUAL_RESULT",
        "accounting_standard_verified": False,
        "consolidation_scope_verified": False,
        "currency": None,
        "basis": "FORECAST_STANDARD_AND_CONSOLIDATION_UNSPECIFIED",
        "comparative_is_independent_earlier_disclosure": False,
        "training_ready": False,
    }


def comparison_readiness(older, newer):
    """缺口返回明确原因；相同的未知字符串不能证明口径相同。"""
    reasons = []
    for name, claim in (("older", older), ("newer", newer)):
        if claim.get("accounting_standard_verified") is not True or not claim.get("accounting_basis_anchor"):
            reasons.append(name + ":ACCOUNTING_STANDARD_UNVERIFIED")
        if "UNSPECIFIED" in claim.get("basis", "") or not claim.get("basis"):
            reasons.append(name + ":ACCOUNTING_BASIS_UNSPECIFIED")
        if claim.get("consolidation_scope_verified") is not True:
            reasons.append(name + ":CONSOLIDATION_SCOPE_UNVERIFIED")
        if claim.get("currency") != "CNY":
            reasons.append(name + ":CURRENCY_UNVERIFIED")
    for key in ("issuer", "period_start", "period_end", "metric", "basis", "currency"):
        if older.get(key) != newer.get(key):
            reasons.append("MISMATCH:" + key)
    return {"ready": not reasons, "reasons": reasons, "training_ready": False}


def compare_verified_claims(older, newer, as_of):
    """只有同口径核验通过才调用旧金额比较器；旧冻结实现和结果保持不变。"""
    readiness = comparison_readiness(older, newer)
    if not readiness["ready"]:
        raise ValueError("COMPARISON_BASIS_NOT_VERIFIED:" + ",".join(readiness["reasons"]))
    return compare_claims(older, newer, as_of)
