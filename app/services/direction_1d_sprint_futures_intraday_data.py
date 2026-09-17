"""第96轮期货日内信息；复用已验证原文和第95轮真实采集，不另发请求。"""

import hashlib

import numpy as np

from app.services import direction_1d_sprint as base
from app.services import direction_1d_sprint_current_basis_data as current
from app.services import direction_1d_sprint_futures_basis_data as old

scope, require = old.scope, old.require


def root():
    return base.ROOT / "futures-intraday-feasibility-v1"


def features(q):
    """同一日同一合约的原价开收比与收盘位置；零振幅只表示中性位置。"""
    require(
        all(type(q[k]) in (int, float) and np.isfinite(q[k]) and q[k] > 0 for k in ("open", "close", "high", "low")),
        "INTRADAY_PRICE_INVALID",
    )
    require(q["low"] <= min(q["open"], q["close"]) <= max(q["open"], q["close"]) <= q["high"], "INTRADAY_OHLC_INVALID")
    span = q["high"] - q["low"]
    return {
        "intraday_return_pct": 100 * (q["close"] / q["open"] - 1),
        "close_location": (2 * q["close"] - q["high"] - q["low"]) / span if span else 0.0,
        "zero_range": span == 0,
    }


def history():
    source, p, result = old.history(), base.read(root() / "plan.json"), base.read(root() / "result.json")
    require(
        hashlib.sha256((root() / "check.py").read_bytes()).hexdigest() == p["script_sha256"], "INTRADAY_SCRIPT_CHANGED"
    )
    require(
        result["plan_hash"] == base.digest(p) and result["history_hash"] == p["history_hash"] == base.digest(source),
        "INTRADAY_HISTORY_CHANGED",
    )
    quotes = {}
    for q in base.read(old.root() / "plan.json")["queries"]:
        if q["api"] == "fut_daily":
            quotes.update(base.read(old.root() / "parsed" / (q["key"] + ".json"))["rows"])
    require(set(quotes) == set(source["snapshot"]["rows"]), "INTRADAY_DATE_COVERAGE_CHANGED")
    rebuilt = {day: features(q) for day, q in quotes.items()}
    require(rebuilt == result["features"] and len(rebuilt) == 1383, "INTRADAY_FEATURES_CHANGED")
    return {
        "at": result["at"],
        "snapshot": {"rows": rebuilt},
        "source_history_hash": base.digest(source),
        "feasibility_hash": base.digest(result),
    }


def extend(market, t, u, points):
    days = list(map(str, base.calendar()[0]))
    require(t in days and u in days and days.index(u) == days.index(t) + 1, "ADJACENCY_INVALID")
    row = points.get(t)
    value = {"date": t, "available": row is not None}
    if row is not None:
        require(
            np.isfinite(row["intraday_return_pct"])
            and np.isfinite(row["close_location"])
            and abs(row["close_location"]) <= 1.000000000001,
            "INTRADAY_FEATURE_INVALID",
        )
        value.update(
            row
            | {
                "intraday_return_pct": float(np.clip(row["intraday_return_pct"], -20, 20)),
                "close_location": float(np.clip(row["close_location"], -1, 1)),
            }
        )
    return market | {"futures_intraday": value}


def original_row(row):
    return row | {"market": {k: v for k, v in row["market"].items() if k != "futures_intraday"}}


def live(t, u, parent_plan_hash):
    """只有第95轮实际采集的原响应能产生新实时特征，缺源保持不可用。"""
    parent = current.load_live(t, u, parent_plan_hash)
    points = {}
    if parent["available"]:
        raw = (current.live_root() / u / "raw/fut_daily.json").read_bytes()
        day = t.replace("-", "")
        q = current.parser.parse(raw, "fut_daily", day, day)[t]
        require(q["close"] == parent["snapshot"]["points"][t]["futures_close"], "LIVE_CLOSE_CHANGED")
        points[t] = features(q)
    return {
        "at": parent["at"],
        "t": t,
        "u": u,
        "parent_plan_hash": parent_plan_hash,
        "parent_source_hash": base.digest(parent),
        "snapshot": {"points": points},
        "available": bool(points),
        "new_source_requests": 0,
    }
