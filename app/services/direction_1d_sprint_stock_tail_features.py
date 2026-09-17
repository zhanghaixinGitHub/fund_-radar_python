"""极端涨跌的股票比例：固定5个百分点阈值，不将其解释为交易所涨跌停。"""

import json

from app.integrations import tushare_sprint_stock_breadth_v2 as source
from app.services import direction_1d_sprint as base

KEYS = ("tail_balance", "tail_mass")
THRESHOLD_PCT = 5.0


def parse(raw, day, minimum_included=3000):
    """先按冻结来源校验再计数；每只股票等权，平盘及零成交额仍保留在分母。

    pct_chg单位为百分点。边界包含恰好+5和-5；上涨尾部数减下跌尾部数为
    方向差，两尾之和为极端变化占比。使用当日实际返回SH/SZ集合，固定排除BJ；
    不引入当前上市清单、不要求跨日股票集合相同，也不删除大幅变化的新股。
    """
    checked = source.parse(raw, day, minimum_included=minimum_included)
    values = [row[4] for row in json.loads(raw)["data"]["items"] if row[0].endswith((".SH", ".SZ"))]
    n = len(values)
    up, down = sum(x >= THRESHOLD_PCT for x in values), sum(x <= -THRESHOLD_PCT for x in values)
    return {
        "date": day,
        "available": True,
        "scope": checked["scope"],
        "threshold_pct": THRESHOLD_PCT,
        "included_rows": n,
        "included_set_sha256": checked["included_set_sha256"],
        "source_distribution_hash": base.digest(checked),
        "tail_up": up,
        "tail_down": down,
        "tail_balance": (up - down) / n,
        "tail_mass": (up + down) / n,
        "all_listed_stocks_independently_verified": False,
        "historical_first_publication_verified": False,
    }


def validate_day(value, day):
    """核对派生计数与特征代数关系；原文摘要由历史/实时来源边界另行校验。"""
    if (
        value["date"] != day
        or value["available"] is not True
        or value["scope"] != "PROVIDER_RETURNED_SH_SZ_DAILY_QUOTES"
        or value["threshold_pct"] != THRESHOLD_PCT
    ):
        raise ValueError("STOCK_TAIL_IDENTITY_CHANGED")
    n, up, down = (value[k] for k in ("included_rows", "tail_up", "tail_down"))
    if not all(type(x) is int for x in (n, up, down)) or n <= 0 or min(up, down) < 0 or up + down > n:
        raise ValueError("STOCK_TAIL_COUNTS_INVALID")
    if value["tail_balance"] != (up - down) / n or value["tail_mass"] != (up + down) / n:
        raise ValueError("STOCK_TAIL_FEATURE_CHANGED")


def aligned(t, u, points, days):
    """只取相邻下一交易日目标之前的T日；缺失明确不可用，不借更早或目标日替补。"""
    if t not in days or u not in days or days.index(u) != days.index(t) + 1:
        raise ValueError("STOCK_TAIL_ADJACENCY_INVALID")
    row = points.get(t)
    if row is None:
        return {"date": t, "available": False}
    validate_day(row, t)
    return dict(row)
