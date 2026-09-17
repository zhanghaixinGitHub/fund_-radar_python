"""交易所收益差复用R109实际原文；每次将派生值与已实际收到的响应绑定，新增请求为零。"""

import hashlib

from app.services import direction_1d_sprint as b
from app.services import direction_1d_sprint_stock_breadth_live as parent
from app.services import direction_1d_sprint_stock_exchange_spread_features as features


def in_window(t, u, at):
    return parent.in_window(t, u, at)


def ready(u):
    return (parent.root() / u / "snapshot.json").is_file()


def load_live(t, u, ph):
    """按父计划校验实际T原文及收据；不加载历史T，不将缺失替换成中性差值。"""
    parent.require(ph == b.digest(b.read(b.ROOT / "round-119/plan.json")), "EXCHANGE_SPREAD_MODEL_PLAN_CHANGED")
    parent_plan = b.digest(b.read(b.ROOT / "round-109/plan.json"))
    source = parent.load_live(t, u, parent_plan)
    points = {}
    if source["available"]:
        raw = (parent.root() / u / "raw.bin").read_bytes()
        receipt = b.read(parent.root() / u / "response.json")
        parent.require(b.digest(receipt) == source["receipt_hash"], "EXCHANGE_SPREAD_ACTUAL_RECEIPT_CHANGED")
        parent.require(
            receipt["raw"] == {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()},
            "EXCHANGE_SPREAD_ACTUAL_RAW_CHANGED",
        )
        value = features.parse(raw, t)
        parent.require(
            value["source_distribution_hash"] == b.digest(source["points"][t]),
            "EXCHANGE_SPREAD_ACTUAL_DISTRIBUTION_CHANGED",
        )
        points[t] = value
    return {
        "at": source["at"],
        "t": t,
        "u": u,
        "plan_hash": ph,
        "parent_plan_hash": parent_plan,
        "parent_source_hash": b.digest(source),
        "receipt_hash": source["receipt_hash"],
        "points": points,
        "available": bool(points.get(t, {}).get("available")),
        "received_at": source["received_at"],
        "reserved_request_slots": 0,
        "new_source_requests": 0,
    }
