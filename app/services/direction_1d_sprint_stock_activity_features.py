"""从既有SH/SZ报价构造成交活跃程度和集中变化，严格只用T及其之前20个交易日。"""

import json
import math
import statistics

from app.services import direction_1d_sprint as base
from app.services import direction_1d_sprint_stock_turnover_features as old

KEYS = ("log_activity", "concentration_change")
SCOPE = "PROVIDER_RETURNED_SH_SZ_DAILY_QUOTES"


def require(condition, reason):
    if not condition:
        raise ValueError("STOCK_ACTIVITY_" + reason)


def parse(raw, day, minimum_included=3000):
    """复用旧解析器验证每行及固定SH/SZ集合；返回总额与最大20只占比，金额单位为千元。"""
    checked = old.parse(raw, day, minimum_included=minimum_included)
    payload = json.loads(raw)
    index = old.source.FIELDS.index("amount")
    amounts = [row[index] for row in payload["data"]["items"] if row[0].endswith((".SH", ".SZ"))]
    total = math.fsum(amounts)
    require(total == checked["total_amount"], "AMOUNT_CHANGED")
    top20 = math.fsum(sorted(amounts, reverse=True)[:20])
    return {
        "date": day,
        "scope": SCOPE,
        "available": checked["available"],
        "total_amount": total,
        "top20_amount": top20,
        "top20_share": top20 / total if checked["available"] else None,
        "unavailable_reason": checked["unavailable_reason"],
        "included_rows": checked["included_rows"],
        "included_set_sha256": checked["included_set_sha256"],
        "source_turnover_hash": base.digest(checked),
        "amount_unit": "THOUSAND_CNY",
        "full_registry_verified": False,
        "historical_first_publication_verified": False,
    }


def validate_day(row, day):
    require(row["date"] == day and row["scope"] == SCOPE and type(row["available"]) is bool, "DAY_IDENTITY_INVALID")
    if row["available"]:
        require(
            type(row["total_amount"]) in (int, float)
            and math.isfinite(row["total_amount"])
            and row["total_amount"] > 0
            and type(row["top20_share"]) in (int, float)
            and math.isfinite(row["top20_share"])
            and 0 < row["top20_share"] <= 1,
            "DAY_RANGE_INVALID",
        )
    else:
        require(
            row["total_amount"] == 0
            and row["top20_share"] is None
            and row["unavailable_reason"] == "ZERO_TOTAL_RETURNED_AMOUNT",
            "MISSING_DAY_INVALID",
        )


def rolling(t, u, points, days):
    """精确取T前20个日历交易日；任何缺日或零总额即不可用，不向更早日期凑足窗口。

    活跃度用自然对数比值，集中变化用占比之差；按各自训练标准差缩放由模型完成。
    未来字典条目完全不参与计算。基线明确排除T，避免今日金额同时污染分子和分母。
    """
    require(t in days and u in days and days.index(u) == days.index(t) + 1, "ADJACENCY_INVALID")
    index = days.index(t)
    past = days[index - 20 : index] if index >= 20 else []
    empty = {"date": t, "available": False, "baseline_dates": past, "log_activity": None, "concentration_change": None}
    if len(past) != 20:
        return empty | {"reason": "INSUFFICIENT_CALENDAR_PREFIX"}
    if t not in points:
        return empty | {"reason": "MISSING_T_DAY"}
    validate_day(points[t], t)
    if not points[t]["available"]:
        return empty | {"reason": "ZERO_T_AMOUNT"}
    missing = [d for d in past if d not in points]
    if missing:
        return empty | {"reason": "MISSING_BASELINE_DAY", "missing_dates": missing}
    for day in past:
        validate_day(points[day], day)
    if any(not points[d]["available"] for d in past):
        return empty | {"reason": "ZERO_BASELINE_AMOUNT"}
    amount = statistics.median(points[d]["total_amount"] for d in past)
    concentration = statistics.median(points[d]["top20_share"] for d in past)
    # 对数之差与正数比值的对数等价，并避免极端有限金额相除发生溢出/下溢。
    raw_activity = math.log(points[t]["total_amount"]) - math.log(amount)
    delta = points[t]["top20_share"] - concentration
    require(math.isfinite(raw_activity) and math.isfinite(delta) and -1 <= delta <= 1, "ROLLING_RANGE_INVALID")
    return empty | {
        "available": True,
        "reason": None,
        "amount_baseline_median": amount,
        "concentration_baseline_median": concentration,
        "raw_log_activity": raw_activity,
        "log_activity": max(-3.0, min(3.0, raw_activity)),
        "concentration_change": delta,
    }
