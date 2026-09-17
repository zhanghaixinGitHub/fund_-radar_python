"""R122零新增请求：R121真实流量加截止16日已冻结的基金历史反应。"""

from app.services import direction_1d_sprint as b
from app.services import direction_1d_sprint_fund_moneyflow_data as data
from app.services import direction_1d_sprint_stock_moneyflow_live_v2 as parent


def in_window(t, u, at):
    return parent.in_window(t, u, at)


def ready(u):
    return parent.ready(u)


def load_live(t, u, ph):
    """没有网络路径或历史流量替代；原来源原文及成熟统计均现场绑定。"""
    data.require(ph == b.digest(b.read(b.ROOT / "round-122/plan.json")), "LIVE_MODEL_PLAN_CHANGED")
    parent_plan = b.digest(b.read(b.ROOT / "round-121/plan.json"))
    source = parent.load_live(t, u, parent_plan)
    _, _, binding, current = data.reference()
    data.require(current["cutoff"] == "2026-09-16" and current["cutoff"] <= u, "LIVE_CONTEXT_NOT_MATURE")
    return {
        "at": source["at"],
        "t": t,
        "u": u,
        "plan_hash": ph,
        "parent_plan_hash": parent_plan,
        "parent_source_hash": b.digest(source),
        "receipt_hash": source["receipt_hash"],
        "points": source["points"],
        "available": source["available"],
        "received_at": source["received_at"],
        "context_reference_hash": b.digest(binding),
        "current_contexts_hash": b.digest(current),
        "frozen_current": current,
        "reserved_request_slots": 0,
        "new_source_requests": 0,
    }
