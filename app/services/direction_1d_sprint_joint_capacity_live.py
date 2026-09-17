"""联合输入只复用父链实际收据；历史资格文件不进入未来输入，无新增供应商请求。"""

from datetime import datetime

from app.services import direction_1d_sprint as b
from app.services import direction_1d_sprint_credit_pair_live as credit
from app.services import direction_1d_sprint_dollar_etf_live as dollar
from app.services import direction_1d_sprint_futures_intraday_data as futures
from app.services import direction_1d_sprint_joint_market_data_v2 as data


def require(condition, reason):
    if not condition:
        raise ValueError("JOINT_ACTUAL_" + reason)


def in_window(t, u, at):
    return credit.in_window(t, u, at)


def ready(u):
    """只表示已有实际来源槽；可用性及完整性还必须由各来源读回校验。"""
    return all(
        (root / u / "snapshot.json").is_file() for root in (futures.current.live_root(), dollar.root(), credit.root())
    )


def load_live(t, u, ph):
    """逐一验证三个父来源原文和实际时点，再形成确定性的联合视图，不覆盖父文件。"""
    require(ph == b.digest(b.read(b.ROOT / "round-113/plan.json")), "MODEL_PLAN_CHANGED")
    plans = {n: b.digest(b.read(b.ROOT / f"round-{n}/plan.json")) for n in (95, 111, 112)}
    sources = {
        "futures": futures.live(t, u, plans[95]),
        "dollar": dollar.load_live(t, u, plans[111]),
        "credit": credit.load_live(t, u, plans[112]),
    }
    require(all(v["t"] == t and v["u"] == u for v in sources.values()), "SOURCE_TARGET_CHANGED")
    require(all(in_window(t, u, datetime.fromisoformat(v["at"])) for v in sources.values()), "SOURCE_TIME_INVALID")
    points = {
        "futures": sources["futures"]["snapshot"]["points"],
        "dollar": sources["dollar"]["points"],
        "credit": sources["credit"]["points"],
    }
    feature = data.features(t, u, points)
    hashes = {k: b.digest(v) for k, v in sources.items()}
    return {
        "at": max(datetime.fromisoformat(v["at"]) for v in sources.values()).isoformat(),
        "t": t,
        "u": u,
        "plan_hash": ph,
        "source_plan_hashes": plans,
        "source_hashes": hashes,
        "receipt_hash": b.digest(hashes),
        "points": points,
        "feature": feature,
        "available": feature["available"],
        "source_received_snapshot_at": {k: v["at"] for k, v in sources.items()},
        "reserved_request_slots": 0,
        "new_source_requests": 0,
    }
