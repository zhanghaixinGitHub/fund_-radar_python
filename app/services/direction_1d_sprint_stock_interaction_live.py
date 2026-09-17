"""交互模型复用R109的股票实际响应；父源校验不变，不增加网络请求或数据写入。"""

from app.services import direction_1d_sprint as b
from app.services import direction_1d_sprint_stock_breadth_live as parent


def in_window(t, u, at):
    return parent.in_window(t, u, at)


def ready(u):
    return (parent.root() / u / "snapshot.json").is_file()


def load_live(t, u, ph):
    """先校验本轮计划，再按原父计划读取实际行情；保留父原文/收据摘要，不冒充新实收。"""
    parent.require(ph == b.digest(b.read(b.ROOT / "round-114/plan.json")), "INTERACTION_MODEL_PLAN_CHANGED")
    parent_plan = b.digest(b.read(b.ROOT / "round-109/plan.json"))
    source = parent.load_live(t, u, parent_plan)
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
        "reserved_request_slots": 0,
        "new_source_requests": 0,
    }
