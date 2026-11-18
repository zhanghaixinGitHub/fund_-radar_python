"""002112原347法律产品的滚动股票预算准入；只重放原件，不计算分数或删缺数据同行。"""

import re
import unicodedata
from calendar import monthrange
from datetime import date
from decimal import Decimal, localcontext
from pathlib import Path

VERSION = "R5_ACTIVE_ROLLING_COHORT_ORIGINALS_V1"
PARENT = "093f5dbe000c881ad7aff93f1e5178079c18c0d59aaec008bafba8f58f47c8f9"


def exact_target_ratio(raw, *, code, aliases, period_end, as_of):
    """分母为报告全部份额净资产之和；不拿基金总资产或单份额净资产替代。"""
    from app.services.fund_quality_v2_disclosures import report_period

    data = raw.get("data", {}) or {}
    body = unicodedata.normalize("NFKC", data.get("notice_content", ""))
    if (
        raw.get("success") != 1
        or code not in {r["stock"] for r in data.get("security", [])}
        or not period_end <= data.get("notice_date", "")[:10] <= as_of
        or str(report_period(body)[1]) != period_end
    ):
        raise ValueError("MONTHLY_ADMISSION_TARGET_ORIGINAL_IDENTITY_PERIOD")
    money = r"(?:\d{1,3}(?:,\d{3})+|\d+)\.\d{2}"
    table = body[body.rfind("期末基金资产组合情况") :]
    equity = re.search(r"权益投资\s+(" + money + ")", table)
    stock = re.search(r"其中\s*[:：]\s*股票\s+(" + money + ")", table)
    nav = re.search(r"4[.、]?\s*期末基金资产净值\s+((" + money + r"\s*)+)", body)
    if not equity or not stock or not nav:
        raise ValueError("MONTHLY_ADMISSION_EXACT_NET_OR_EQUITY_ROWS")
    columns = re.findall(money, nav[1])
    if len(columns) != len(aliases):
        raise ValueError("MONTHLY_ADMISSION_ALL_SHARE_NET_COLUMNS")
    # 本次原协议包含存托凭证；有差额时不能静默按普通股票行缩小目标。
    equity_amount, stock_amount = Decimal(equity[1].replace(",", "")), Decimal(stock[1].replace(",", ""))
    if equity_amount != stock_amount:
        raise ValueError("MONTHLY_ADMISSION_EQUITY_CDR_BRIDGE_REQUIRED")
    with localcontext() as context:
        context.prec = 70
        net = sum(Decimal(v.replace(",", "")) for v in columns)
        if net <= 0:
            raise ValueError("MONTHLY_ADMISSION_NONPOSITIVE_NET")
        return equity_amount / net * 100


def classify_exposure(item, target, *, precise=None):
    """原门槛：均差5pp、最大差10pp；风险境内90%，结构纯境内100%。"""
    quarters = item["quarters"]
    if len(quarters) != 12 or {q["date"] for q in quarters} != set(target):
        raise ValueError("MONTHLY_ADMISSION_TWELVE_QUARTERS_REQUIRED")
    precise = precise or {}
    differences = [
        (
            abs(precise.get(q["date"], Decimal(q["equity_nav_pct"])) - target[q["date"]]),
            Decimal(0) if q["date"] in precise else Decimal(q["rounding_radius_pp"]),
        )
        for q in quarters
    ]
    upper = [difference + radius for difference, radius in differences]
    lower = [max(Decimal(0), difference - radius) for difference, radius in differences]
    budget_passed = max(upper) <= 10 and sum(upper) / 12 <= 5
    budget_failed = max(lower) > 10 or sum(lower) / 12 > 5
    return {
        "code": item["code"],
        "budget_passed": budget_passed,
        "budget_definitely_failed": budget_failed,
        "budget_precision_review_required": not budget_passed and not budget_failed,
        "mean_gap_upper_pp": str(sum(upper) / 12),
        "maximum_gap_upper_pp": str(max(upper)),
        "risk_admitted": budget_passed
        and not item["gaps"]
        and item["region_status_by_cutoff"]["90"].startswith("REGION_PASS"),
        "structure_admitted": budget_passed
        and not item["gaps"]
        and item["region_status_by_cutoff"]["100"].startswith("REGION_PASS"),
        "region_status_by_cutoff": item["region_status_by_cutoff"],
        "gaps": item["gaps"],
        "exact_financial_exposure_periods": {day: str(value) for day, value in precise.items()},
    }


def replay_monthly_admission(inputs, *, root):
    """读取完整原件图，独立重建准入；源数量可超过主研究包，逐源校摘要并惰性读取。"""
    from app.services.fund_rating_active_exposure import replay_exposures
    from app.services.fund_rating_active_performance import _bytes, _read

    root = Path(root)
    if (
        inputs.get("version") != VERSION
        or inputs.get("fund_code") != "002112"
        or inputs.get("parent_cohort_ref") != PARENT
    ):
        raise ValueError("MONTHLY_ADMISSION_ORIGINAL_PARENT_REQUIRED")
    parent = _read(inputs, root, PARENT)
    codes = parent["peer_representatives"]
    if len(codes) != 347 or len(set(codes)) != 347:
        raise ValueError("MONTHLY_ADMISSION_FULL_347_UNIVERSE_REQUIRED")
    target = {}
    for quarter in inputs["target_quarters"]:
        day = quarter["period_end"]
        if day in target:
            raise ValueError("MONTHLY_ADMISSION_TARGET_DUPLICATE_QUARTER")
        target[day] = exact_target_ratio(
            _read(inputs, root, quarter["source_ref"]),
            code="002112",
            aliases=["001412", "002112"],
            period_end=day,
            as_of=inputs["as_of_date"],
        )
    ends = sorted(target)
    if any(
        (parsed := date.fromisoformat(day)).month not in {3, 6, 9, 12}
        or parsed.day != monthrange(parsed.year, parsed.month)[1]
        for day in ends
    ):
        raise ValueError("MONTHLY_ADMISSION_ACTUAL_QUARTER_END_REQUIRED")
    quarter_numbers = [date.fromisoformat(day).year * 4 + (date.fromisoformat(day).month - 1) // 3 for day in ends]
    if len(ends) != 12 or quarter_numbers != list(range(quarter_numbers[0], quarter_numbers[0] + 12)):
        raise ValueError("MONTHLY_ADMISSION_CONSECUTIVE_REAL_QUARTERS")
    exposure, audit = replay_exposures(inputs, root, _read, _bytes)
    if len(exposure) != 347 or {item["code"] for item in exposure} != set(codes):
        raise ValueError("MONTHLY_ADMISSION_CANNOT_REMOVE_PARENT_PEER")
    admission = {item["code"]: item for item in _read(inputs, root, inputs["admission_ref"])}
    exact = {}
    for review in inputs.get("precision_originals", []):
        code, day = review["code"], review["period_end"]
        if code not in codes or day not in target or day in exact.get(code, {}):
            raise ValueError("MONTHLY_ADMISSION_PRECISION_BINDING")
        value = exact_target_ratio(
            _read(inputs, root, review["source_ref"]),
            code=code,
            aliases=admission[code]["share_codes"],
            period_end=day,
            as_of=inputs["as_of_date"],
        )
        exact.setdefault(code, {})[day] = value
    results = [classify_exposure(item, target, precise=exact.get(item["code"])) for item in exposure]
    for item in results:
        decision = admission.get(item["code"], {}).get("decision")
        item["mandate_decision"] = decision
        if decision != "COMPARABLE_MANDATE_EVIDENCE_AVAILABLE":
            item["risk_admitted"] = item["structure_admitted"] = False
    legal_pending = [
        item["code"]
        for item in results
        if not item["mandate_decision"]
        or any(word in item["mandate_decision"] for word in ("PENDING", "REQUIRED", "MISSING", "UNRESOLVED"))
    ]
    precision_pending = [item["code"] for item in results if item["budget_precision_review_required"]]
    return {
        "version": VERSION,
        "fund_code": "002112",
        "as_of_date": inputs["as_of_date"],
        "status": "ADMISSION_PENDING" if legal_pending or precision_pending else "EXPOSURE_ADMISSION_COMPLETE",
        "target_quarter_exposure": {day: str(value) for day, value in target.items()},
        "risk_codes": [item["code"] for item in results if item["risk_admitted"]],
        "structure_codes": [item["code"] for item in results if item["structure_admitted"]],
        "liquidity_codes": None,
        "liquidity_review_required": "仍须独立核准原普通日开放合同条件；不能默认等同结构名单。",
        "precision_pending_codes": precision_pending,
        "legal_review_pending_codes": legal_pending,
        "rows": results,
        "source_audit": audit,
        "time_contract": "当前研究对历史公开季度的复算，不是当时交易可得信息或新分数。",
    }
