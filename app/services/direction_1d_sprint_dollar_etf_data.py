"""美元相关ETF的独立研究输入：仅累计新增美股会话各自开收到收盘，不包含跨日价差。"""

import hashlib
import math
from datetime import datetime

from app.integrations import sina_sprint_dollar_etf as api
from app.services import direction_1d_sprint as b
from app.services import direction_1d_sprint_overnight as overnight


def root():
    return b.ROOT / "dollar-etf-feasibility-v1"


def require(condition, reason):
    if not condition:
        raise ValueError("DOLLAR_ETF_" + reason)


def features(t, u, points):
    """只选T收盘后、相邻U的08:30前已经结束的美国会话。

    使用100*sum(log(close/open))，单位为百分数对数收益，固定截断±20；它不含隔夜跳空，
    也不是分红再投资总回报。单日统一拆合股或比例复权因子在同日比值中抵消，
    不据此声称供应商历史OHLC每个字段的调整规则均已证实。
    缺任何一个会话、无新增会话或任何价格质量异常都明确不可用，不前填、不跳过坏行。
    """
    aligned = overnight.alignment(t, u)
    days = aligned["new_us_dates"]
    unavailable = {"available": False, "intraday_log_pct": 0.0, "new_us_dates": days}
    if not days:
        return unavailable | {"reason": "NO_NEW_US_SESSION"}
    if any(day not in points for day in days):
        return unavailable | {"reason": "MISSING_US_SESSION"}
    for day in days:
        row = points[day]
        require(
            all(
                type(row.get(k)) in (int, float) and math.isfinite(row[k]) and row[k] > 0
                for k in ("open", "high", "low", "close")
            )
            and type(row.get("volume")) in (int, float)
            and math.isfinite(row["volume"])
            and row["volume"] >= 0,
            "INVALID_PRICE_OR_VOLUME",
        )
        valid = (
            row["volume"] > 0
            and row["low"] <= min(row["open"], row["close"]) <= max(row["open"], row["close"]) <= row["high"]
        )
        require(
            row.get("available") is valid
            and row.get("unavailable_reason") == (None if valid else "OHLC_RANGE_OR_NO_TRADE"),
            "QUALITY_FLAG_CHANGED",
        )
        if not valid:
            return unavailable | {"reason": "INVALID_OHLC_OR_NO_TRADE"}
    value = 100.0 * sum(math.log(points[d]["close"] / points[d]["open"]) for d in days)
    return {"available": True, "intraday_log_pct": min(20.0, max(-20.0, value)), "new_us_dates": days, "reason": None}


def reconstruct():
    """重新核验三次实际响应；身份和开收盘用发行方明确字段，不混用NAV或买卖中点。"""
    import json

    p = b.read(root() / "qualification-plan.json")
    for name, expected in p["code_hashes"].items():
        require(hashlib.sha256((b.PROJECT / name).read_bytes()).hexdigest() == expected, "SOURCE_CODE_CHANGED")
    probe = b.read(root() / "probe-plan.json")
    for name, expected in probe["code"].items():
        require(hashlib.sha256((b.PROJECT / name).read_bytes()).hexdigest() == expected, "PROBE_CODE_CHANGED")
    require(b.digest(probe) == p["probe_plan_hash"], "PROBE_PLAN_CHANGED")
    for key in ("issuer-uup", "sina-uup", "issuer-price"):
        request = b.read(root() / "requests" / (key + ".json"))
        receipt = b.read(root() / "responses" / (key + ".json"))
        raw = (root() / "raw" / (key + ".bin")).read_bytes()
        spec = p["sources"][key]
        require(
            b.digest(request) == spec["request_hash"] and b.digest(receipt) == spec["receipt_hash"], "RECEIPT_CHANGED"
        )
        require(
            receipt["http_status"] == 200
            and receipt["request_hash"] == b.digest(request)
            and receipt["bytes"] == len(raw)
            and receipt["sha256"] == spec["raw_sha256"] == hashlib.sha256(raw).hexdigest(),
            "SOURCE_RAW_CHANGED",
        )
        require(datetime.fromisoformat(request["at"]) <= datetime.fromisoformat(receipt["at"]), "SOURCE_TIME_CHANGED")
    price_plan = b.read(root() / "issuer-price-plan.json")
    require(b.digest(price_plan) == p["issuer_price_plan_hash"], "ISSUER_PLAN_CHANGED")
    require(
        hashlib.sha256((root() / "read_issuer_price.py").read_bytes()).hexdigest() == price_plan["script_sha256"],
        "ISSUER_SCRIPT_CHANGED",
    )
    require(b.read(root() / "requests/issuer-price.json")["url"] == price_plan["url"], "ISSUER_URL_CHANGED")
    request = b.read(root() / "requests/sina-uup.json")
    require(
        request["query"]["url"] == api.url_for("UUP") and request["plan_hash"] == b.digest(probe), "QUOTE_URL_CHANGED"
    )
    points = api.parse((root() / "raw/sina-uup.bin").read_bytes(), "UUP", p["cutoff"])
    require(points == b.read(root() / "parsed.json")["points"], "PARSED_HISTORY_CHANGED")
    price = json.loads((root() / "raw/issuer-price.bin").read_bytes())
    require(
        price["cusip"] == "46141D203"
        and price["currency"] == "USD"
        and price["effectiveDate"] == price["openingPriceEffectiveBusinessDate"] == "2026-09-15",
        "ISSUER_IDENTITY_OR_DATE_CHANGED",
    )
    require(
        price["closingPrice"] == points["2026-09-15"]["close"] == 28.22
        and price["openingPrice"] == points["2026-09-15"]["open"] == 28.21,
        "ISSUER_PRICE_MISMATCH",
    )
    return points


def history():
    """资格仅限于已核对近期价格的供应商同日比值研究，不等于正式发布或首发时间证明。"""
    p, result, snapshot = (
        b.read(root() / name) for name in ("qualification-plan.json", "qualification-result.json", "history.json")
    )
    require(
        result["status"] == "QUALIFIED_INTRADAY_RESEARCH_WITH_LIMITS"
        and result["plan_hash"] == snapshot["plan_hash"] == b.digest(p)
        and result["history_hash"] == b.digest(snapshot),
        "HISTORY_MANIFEST_CHANGED",
    )
    points = reconstruct()
    require(points == snapshot["points"], "HISTORY_POINTS_CHANGED")
    return snapshot


def extend(market, t, u, points):
    return market | {"dollar_etf": features(t, u, points)}


def original_row(row):
    return row | {"market": {k: v for k, v in row["market"].items() if k != "dollar_etf"}}
