"""股指期货日内走势与基金个体医疗反应联合输入，保留两条原始来源身份。"""

from datetime import datetime

from app.services import direction_1d_sprint as b
from app.services import direction_1d_sprint_futures_intraday_data as futures
from app.services import direction_1d_sprint_health_response_data as response
from app.services import direction_1d_sprint_market_health_response as previous

current = response.current


def root():
    return b.ROOT / "health-response-futures-feasibility-v1"


def joint(health, future):
    """分别保留缺失状态；训练仅将缺失分量设0，真实推断缺任一新源时回退原SIGN。"""
    return {
        "available": health["available"] and future["available"],
        "response_available": health["available"],
        "if_available": future["available"],
        "response_signal": health["response_signal"],
        "intraday_return_pct": future["intraday_return_pct"] if future["available"] else 0.0,
        "close_location": future["close_location"] if future["available"] else 0.0,
        "date": future["date"],
        "new_us_dates": health["new_us_dates"],
        "max_prior_mature": health["max_prior_mature"],
        "context_cutoff": health["context_cutoff"],
        "context_hash": health["context_hash"],
    }


def original_row(row):
    return row | {"market": {k: v for k, v in row["market"].items() if k != "health_futures"}}


def reconstruct():
    """逐题重建R105个体统计，再按T日连接已合格IF原行情，不改变任何原标签或训练行。"""
    rows, proof = previous.reconstruct_dataset()
    previous.verify_identity(rows, proof, b.read(previous.root() / "source-identity.json")["identity"])
    history = futures.history()
    cache, output = {}, []
    for row in rows:
        key = row["t"], row["u"]
        if key not in cache:
            cache[key] = futures.extend({}, *key, history["snapshot"]["rows"])["futures_intraday"]
        value = joint(row["market"]["health_response"], cache[key])
        old = response.original_row(row)
        output.append(old | {"market": old["market"] | {"health_futures": value}})
    return output, proof | {"health_futures_if_history_hash": b.digest(history)}


def historical_inputs():
    """预检使用已冻结历史与更早保存的个体快照，不产生供应商请求。"""
    return {
        "health_points": response.source.data.history()["points"],
        "futures_points": futures.history()["snapshot"]["rows"],
        "contexts": current()["contexts"],
    }


def extend(market, t, u, combined, code):
    health = response.extend({}, t, u, combined["health_points"], code, combined["contexts"])["health_response"]
    future = futures.extend({}, t, u, combined["futures_points"])["futures_intraday"]
    return market | {"health_futures": joint(health, future)}


def live(t, u, parent_plan_hash):
    """同日医疗实收与IF实收分别回读，合并时间取较晚实际来源；新增请求0。"""
    previous.require(parent_plan_hash == b.digest(b.read(previous.root() / "plan.json")), "JOINT_PARENT_PLAN_CHANGED")
    health = response.live(t, u, b.digest(b.read(b.ROOT / "round-104/plan.json")))
    future = futures.live(t, u, b.digest(b.read(b.ROOT / "round-95/plan.json")))
    return {
        "at": max(health["at"], future["at"], key=datetime.fromisoformat),
        "t": t,
        "u": u,
        "response_source_hash": b.digest(health),
        "intraday_source_hash": b.digest(future),
        "context_manifest_hash": health["context_manifest_hash"],
        "contexts": health["contexts"],
        "health_points": health["points"],
        "futures_points": future["snapshot"]["points"],
        "available": health["available"] and future["available"],
        "new_source_requests": 0,
    }
