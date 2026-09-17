"""期货方向与同合约成交持仓活跃度；复用已核验来源，禁止跨主力差分。"""

import hashlib

import numpy as np

from app.services import direction_1d_sprint as base
from app.services import direction_1d_sprint_current_basis_data as current
from app.services import direction_1d_sprint_futures_basis_data as old
from app.services import direction_1d_sprint_futures_intraday_data as intra

scope, require = old.scope, old.require


def root():
    return base.ROOT / "futures-activity-feasibility-v1"


def features(q, contract):
    """手数比是同日同合约的无量纲活动量；与方向相乘，不推断买方或卖方身份。"""
    require(
        all(type(q[k]) in (int, float) and np.isfinite(q[k]) and q[k] > 0 for k in ("vol", "oi")), "ACTIVITY_INVALID"
    )
    direction = intra.features(q)
    activity = float(np.clip(np.log(q["vol"] / q["oi"]), -5, 5))
    return direction | {
        "volume": q["vol"],
        "open_interest": q["oi"],
        "main_contract": contract,
        "log_volume_to_oi": activity,
        "return_activity": float(np.clip(direction["intraday_return_pct"], -20, 20) * activity),
        "location_activity": float(np.clip(direction["close_location"], -1, 1) * activity),
    }


def history():
    source, p, result = old.history(), base.read(root() / "plan.json"), base.read(root() / "result.json")
    require(
        hashlib.sha256((root() / "check.py").read_bytes()).hexdigest() == p["script_sha256"], "ACTIVITY_SCRIPT_CHANGED"
    )
    require(
        hashlib.sha256((root() / "design-before-analysis.md").read_bytes()).hexdigest() == p["design_sha256"],
        "ACTIVITY_DESIGN_CHANGED",
    )
    require(
        result["plan_hash"] == base.digest(p) and result["history_hash"] == p["history_hash"] == base.digest(source),
        "ACTIVITY_HISTORY_CHANGED",
    )
    quotes = {}
    for q in base.read(old.root() / "plan.json")["queries"]:
        if q["api"] == "fut_daily":
            quotes.update(base.read(old.root() / "parsed" / (q["key"] + ".json"))["rows"])
    require(set(quotes) == set(source["snapshot"]["rows"]), "ACTIVITY_DATE_COVERAGE_CHANGED")
    rebuilt = {day: features(q, source["snapshot"]["rows"][day]["main_contract"]) for day, q in quotes.items()}
    require(rebuilt == result["features"] and len(rebuilt) == 1383, "ACTIVITY_FEATURES_CHANGED")
    return {
        "at": result["at"],
        "snapshot": {"rows": rebuilt},
        "source_history_hash": base.digest(source),
        "feasibility_hash": base.digest(result),
    }


def extend(market, t, u, points):
    """保留基础日内日期检查；只追加同T日活动交互，缺失保持不可用。"""
    intraday = intra.extend({}, t, u, points)["futures_intraday"]
    if intraday["available"]:
        require(
            all(
                type(intraday[k]) in (int, float) and np.isfinite(intraday[k])
                for k in ("return_activity", "location_activity")
            ),
            "ACTIVITY_FEATURE_INVALID",
        )
        require(
            abs(intraday["return_activity"]) <= 100 and abs(intraday["location_activity"]) <= 5,
            "ACTIVITY_FEATURE_RANGE",
        )
    return market | {"futures_activity": intraday}


def original_row(row):
    return row | {"market": {k: v for k, v in row["market"].items() if k != "futures_activity"}}


def live(t, u, parent_plan_hash):
    """实际原文来自第95轮有界采集；没有新增HTTP或历史数据冒充实时响应。"""
    parent = current.load_live(t, u, parent_plan_hash)
    points = {}
    if parent["available"]:
        raw = (current.live_root() / u / "raw/fut_daily.json").read_bytes()
        day = t.replace("-", "")
        q = current.parser.parse(raw, "fut_daily", day, day)[t]
        source = parent["snapshot"]["points"][t]
        require(q["close"] == source["futures_close"], "LIVE_CLOSE_CHANGED")
        points[t] = features(q, source["main_contract"])
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
