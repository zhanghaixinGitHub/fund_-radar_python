"""联合IF日内价格与IO比例变化，分别保留两条实际来源的证据。"""

import hashlib
from datetime import datetime

from app.services import direction_1d_sprint as base
from app.services import direction_1d_sprint_futures_intraday_data as intra
from app.services import direction_1d_sprint_option_change_data as change

scope, require = change.scope, change.require


def root():
    return base.ROOT / "option-futures-combination-feasibility-v1"


def combine(day, changes, intraday):
    """两组特征必须来自同T日；缺任一来源就明确不可用，不用旧值或零补全。"""
    c, f = changes.get(day), intraday.get(day)
    if c is None or f is None:
        return None
    require(c["date"] == day, "COMBINATION_DATE_MISMATCH")
    return c | f


def reconstruct():
    c, f = change.history(), intra.history()
    points = {
        day: value
        for day in c["snapshot"]["rows"]
        if (value := combine(day, c["snapshot"]["rows"], f["snapshot"]["rows"])) is not None
    }
    return {"rows": points, "change_history_hash": base.digest(c), "intraday_history_hash": base.digest(f)}


def history():
    p, result = base.read(root() / "plan.json"), base.read(root() / "result.json")
    require(
        hashlib.sha256((root() / "check.py").read_bytes()).hexdigest() == p["script_sha256"],
        "COMBINATION_SCRIPT_CHANGED",
    )
    for name, digest in p["code"].items():
        require(hashlib.sha256((base.PROJECT / name).read_bytes()).hexdigest() == digest, "COMBINATION_CODE_CHANGED")
    rebuilt = reconstruct()
    require(
        result["plan_hash"] == base.digest(p) and result["snapshot"] == rebuilt and len(rebuilt["rows"]) == 1382,
        "COMBINATION_HISTORY_CHANGED",
    )
    require(result["history_hash"] == base.digest(rebuilt), "COMBINATION_SOURCE_CHANGED")
    return {
        "at": result["at"],
        "snapshot": {"rows": rebuilt["rows"]},
        "source_history_hash": result["history_hash"],
        "feasibility_hash": base.digest(result),
    }


def extend(market, t, u, points):
    # 两套旧日期/数值边界分别复用，价格截断和比例变化单位保持既定口径。
    c = change.extend({}, t, u, points)["option_change"]
    f = intra.extend({}, t, u, points)["futures_intraday"]
    require(c["available"] == f["available"], "COMBINATION_AVAILABILITY_CHANGED")
    return market | {"option_futures": c | f}


def original_row(row):
    return row | {"market": {k: v for k, v in row["market"].items() if k != "option_futures"}}


def live(t, u, option_plan_hash, futures_plan_hash):
    """只复用R97实际期权及R95实际期货；收到时间取两源较晚者，不产生HTTP。"""
    c, f = change.live(t, u, option_plan_hash), intra.live(t, u, futures_plan_hash)
    at = max(datetime.fromisoformat(c["at"]), datetime.fromisoformat(f["at"]))
    points = {}
    if c["available"] and f["available"]:
        value = combine(t, c["snapshot"]["points"], f["snapshot"]["points"])
        if value is not None:
            points[t] = value
    return {
        "at": at.isoformat(),
        "t": t,
        "u": u,
        "change_source_hash": base.digest(c),
        "intraday_source_hash": base.digest(f),
        "previous_history_at": c["previous_history_at"],
        "snapshot": {"points": points},
        "available": bool(points),
        "new_source_requests": 0,
    }
