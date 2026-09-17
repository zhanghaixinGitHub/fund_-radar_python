"""沪市减深市的当日相对强弱；只表示供应商返回交易所集合，不冒充基金持仓暴露。"""

import json
import math

from app.integrations import tushare_sprint_stock_breadth_v2 as source
from app.services import direction_1d_sprint as base

KEYS = ("breadth_spread", "return_spread")


def parse(raw, day, minimum_included=3000):
    """使用冻结日线校验器，按.SH/.SZ分组后保留两个交易所各自的计数、金额与收益证据。

    第一个差值各用自己的股票行数作分母，平盘和零成交额仍计数；第二个差值各用
    成交金额加权，单只股票涨幅预先截断±20个百分点，不裁剪股票集合。金额单位
    相同，按组归一后抵消；任何一组缺失或总金额非正都不能作为中性的零差。
    """
    checked = source.parse(raw, day, minimum_included=minimum_included)
    groups = {}
    items = json.loads(raw)["data"]["items"]
    for exchange in ("SH", "SZ"):
        rows = sorted((r for r in items if r[0].endswith("." + exchange)), key=lambda r: r[0])
        total = math.fsum(r[-1] for r in rows)
        up, down = sum(r[4] > 0 for r in rows), sum(r[4] < 0 for r in rows)
        weighted_sum = math.fsum(r[-1] * max(-20.0, min(20.0, r[4])) for r in rows)
        groups[exchange] = {
            "rows": len(rows),
            "up": up,
            "down": down,
            "total_amount": total,
            "weighted_clipped_return_sum": weighted_sum,
            "breadth": (up - down) / len(rows) if rows else None,
            "weighted_return": weighted_sum / total if total > 0 else None,
        }
    available = all(x["rows"] > 0 and x["total_amount"] > 0 for x in groups.values())
    return {
        "date": day,
        "available": available,
        "unavailable_reason": None if available else "MISSING_EXCHANGE_OR_ZERO_AMOUNT",
        "scope": checked["scope"],
        "included_rows": checked["included_rows"],
        "included_set_sha256": checked["included_set_sha256"],
        "source_distribution_hash": base.digest(checked),
        "groups": groups,
        "breadth_spread": groups["SH"]["breadth"] - groups["SZ"]["breadth"] if available else None,
        "return_spread": groups["SH"]["weighted_return"] - groups["SZ"]["weighted_return"] if available else None,
        "all_listed_stocks_independently_verified": False,
        "historical_first_publication_verified": False,
    }


def validate_day(value, day):
    if (
        value["date"] != day
        or type(value["available"]) is not bool
        or value["scope"] != "PROVIDER_RETURNED_SH_SZ_DAILY_QUOTES"
        or set(value["groups"]) != {"SH", "SZ"}
    ):
        raise ValueError("STOCK_EXCHANGE_SPREAD_IDENTITY_CHANGED")
    for group in value["groups"].values():
        n, up, down = (group[k] for k in ("rows", "up", "down"))
        if not all(type(x) is int for x in (n, up, down)) or min(n, up, down) < 0 or up + down > n:
            raise ValueError("STOCK_EXCHANGE_SPREAD_COUNTS_INVALID")
        total, weighted = group["total_amount"], group["weighted_clipped_return_sum"]
        if not all(type(x) in (int, float) and math.isfinite(x) for x in (total, weighted)) or total < 0:
            raise ValueError("STOCK_EXCHANGE_SPREAD_AMOUNT_INVALID")
        if abs(weighted) > total * 20 + 1e-6:
            raise ValueError("STOCK_EXCHANGE_SPREAD_CLIP_INVALID")
        if group["breadth"] != ((up - down) / n if n else None):
            raise ValueError("STOCK_EXCHANGE_SPREAD_BREADTH_CHANGED")
        if group["weighted_return"] != (weighted / total if total > 0 else None):
            raise ValueError("STOCK_EXCHANGE_SPREAD_RETURN_CHANGED")
    groups = value["groups"]
    available = all(x["rows"] > 0 and x["total_amount"] > 0 for x in groups.values())
    if value["available"] != available or value["included_rows"] != sum(x["rows"] for x in groups.values()):
        raise ValueError("STOCK_EXCHANGE_SPREAD_UNIVERSE_CHANGED")
    if not available:
        if value["unavailable_reason"] != "MISSING_EXCHANGE_OR_ZERO_AMOUNT" or any(value[k] is not None for k in KEYS):
            raise ValueError("STOCK_EXCHANGE_SPREAD_MISSING_NOT_NULL")
        return
    if (
        value["breadth_spread"] != groups["SH"]["breadth"] - groups["SZ"]["breadth"]
        or value["return_spread"] != groups["SH"]["weighted_return"] - groups["SZ"]["weighted_return"]
    ):
        raise ValueError("STOCK_EXCHANGE_SPREAD_FEATURE_CHANGED")


def aligned(t, u, points, days):
    if t not in days or u not in days or days.index(u) != days.index(t) + 1:
        raise ValueError("STOCK_EXCHANGE_SPREAD_ADJACENCY_INVALID")
    row = points.get(t)
    if row is None:
        return {"date": t, "available": False}
    validate_day(row, t)
    return dict(row)
