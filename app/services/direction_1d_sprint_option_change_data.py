"""期权比例的相邻交易日变化；历史与实际源都复用第97轮，不新增采集。"""

import hashlib

import numpy as np

from app.services import direction_1d_sprint as base
from app.services import direction_1d_sprint_option_position_data as old

scope, require = old.scope, old.require


def root():
    return base.ROOT / "option-position-change-feasibility-v1"


def features(day, points):
    """T与前一中国交易日均需实际存在；变化是对数比的差，不跨缺日计算。"""
    days = list(map(str, base.calendar()[0]))
    require(day in days, "CHANGE_DATE_INVALID")
    index = days.index(day)
    prior = days[index - 1] if index else None
    previous, current = points.get(prior), points.get(day)
    if previous is None or current is None:
        return None
    value = {"date": day, "previous_date": prior}
    for key in ("log_put_call_volume", "log_put_call_oi"):
        require(
            all(type(v[key]) in (int, float) and np.isfinite(v[key]) for v in (previous, current)),
            "CHANGE_VALUE_INVALID",
        )
        value[key + "_change"] = float(np.clip(current[key] - previous[key], -20, 20))
    return value


def history():
    source, p, result = old.history(), base.read(root() / "plan.json"), base.read(root() / "result.json")
    require(
        hashlib.sha256((root() / "check.py").read_bytes()).hexdigest() == p["script_sha256"], "CHANGE_SCRIPT_CHANGED"
    )
    require(
        hashlib.sha256((root() / "design-before-analysis.md").read_bytes()).hexdigest() == p["design_sha256"],
        "CHANGE_DESIGN_CHANGED",
    )
    require(
        result["plan_hash"] == base.digest(p) and result["history_hash"] == p["history_hash"] == base.digest(source),
        "CHANGE_HISTORY_CHANGED",
    )
    points = source["snapshot"]["rows"]
    rebuilt = {day: value for day in points if (value := features(day, points)) is not None}
    require(rebuilt == result["features"] and len(rebuilt) == 1382, "CHANGE_FEATURES_CHANGED")
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
    feature = {"date": t, "available": row is not None}
    if row is not None:
        require(row["date"] == t and row["previous_date"] == days[days.index(t) - 1], "CHANGE_INTERVAL_INVALID")
        require(
            all(
                type(row[k]) in (int, float) and np.isfinite(row[k]) and abs(row[k]) <= 20
                for k in ("log_put_call_volume_change", "log_put_call_oi_change")
            ),
            "CHANGE_FEATURE_INVALID",
        )
        feature.update(row)
    return market | {"option_change": feature}


def original_row(row):
    return row | {"market": {k: v for k, v in row["market"].items() if k != "option_change"}}


def live(t, u, parent_plan_hash):
    """未来T日必须来自第97轮真实捕获，D来自更早保存的不可变历史；两者分别绑定。"""
    from datetime import datetime

    parent = old.load_live(t, u, parent_plan_hash)
    source = old.history()
    require(
        datetime.fromisoformat(source["at"]) < datetime.fromisoformat(parent["at"]),
        "PREVIOUS_HISTORY_NOT_SAVED_BEFORE_CAPTURE",
    )
    points = {}
    if parent["available"]:
        # 只允许新增的实际T日覆盖该日；旧历史不得已含T以掩盖收到时间。
        require(t not in source["snapshot"]["rows"], "LIVE_T_ALREADY_IN_HISTORY")
        combined = source["snapshot"]["rows"] | parent["snapshot"]["points"]
        value = features(t, combined)
        if value is not None:
            points[t] = value
    return {
        "at": parent["at"],
        "t": t,
        "u": u,
        "parent_plan_hash": parent_plan_hash,
        "parent_source_hash": base.digest(parent),
        "previous_history_hash": base.digest(source),
        "previous_history_at": source["at"],
        "snapshot": {"points": points},
        "available": bool(points),
        "new_source_requests": 0,
    }
