"""已验证跨市场信息联合：三个原始市场输入之外加入IF、美元及债券价格，零新增数据请求。"""

import hashlib

from app.services import direction_1d_sprint as b
from app.services import direction_1d_sprint_credit_pair_etf_data as credit
from app.services import direction_1d_sprint_dollar_etf_data as dollar
from app.services import direction_1d_sprint_futures_intraday_data as futures


def root():
    return b.ROOT / "joint-cross-asset-feasibility-v1"


def require(condition, reason):
    if not condition:
        raise ValueError("JOINT_MARKET_" + reason)


def features(t, u, points):
    """所有来源齐全才使用新增五维，缺任何来源不制造零变化；下游按预定原模型回退。

    顺序为IF同日涨幅(百分比)、IF收盘区间位置[-1,1]、UUP同日对数变化(百分数)、
    HYG减IEF同日对数变化(百分数)、IEF同日对数变化(百分数)。各来源沿用已冻结时点和限幅。
    """
    fi = futures.extend({}, t, u, points.get("futures", {}))["futures_intraday"]
    usd = dollar.features(t, u, points.get("dollar", {}))
    bond = credit.features(t, u, points.get("credit", {}))
    require(usd["new_us_dates"] == bond["new_us_dates"], "US_SESSION_MISMATCH")
    empty = {"available": False, "values": [0.0] * 5, "if_date": t, "new_us_dates": usd["new_us_dates"]}
    if not fi["available"]:
        return empty | {"reason": "MISSING_T_IF"}
    if not usd["available"]:
        return empty | {"reason": "DOLLAR_" + usd["reason"]}
    if not bond["available"]:
        return empty | {"reason": "CREDIT_" + bond["reason"]}
    return {
        "available": True,
        "values": [
            fi["intraday_return_pct"],
            fi["close_location"],
            usd["intraday_log_pct"],
            bond["relative_log_pct"],
            bond["treasury_log_pct"],
        ],
        "if_date": t,
        "new_us_dates": usd["new_us_dates"],
        "reason": None,
    }


def reconstruct():
    """重新校验三份已资格历史；只引用其实际冻结产物，不读取临时行情或供应商。"""
    p = b.read(root() / "qualification-plan.json")
    for name, expected in p["code_hashes"].items():
        require(hashlib.sha256((b.PROJECT / name).read_bytes()).hexdigest() == expected, "SOURCE_CODE_CHANGED")
    sources = {"futures": futures.history(), "dollar": dollar.history(), "credit": credit.history()}
    require({k: b.digest(v) for k, v in sources.items()} == p["source_history_hashes"], "PARENT_HISTORY_CHANGED")
    return {
        "futures": sources["futures"]["snapshot"]["rows"],
        "dollar": sources["dollar"]["points"],
        "credit": sources["credit"]["points"],
    }


def history():
    p, result, snapshot = (
        b.read(root() / n) for n in ("qualification-plan.json", "qualification-result.json", "history.json")
    )
    require(
        result["status"] == "QUALIFIED_JOINT_MARKET_WITH_LIMITS"
        and result["plan_hash"] == snapshot["plan_hash"] == b.digest(p)
        and result["history_hash"] == b.digest(snapshot),
        "HISTORY_MANIFEST_CHANGED",
    )
    require(reconstruct() == snapshot["points"], "HISTORY_POINTS_CHANGED")
    return snapshot


def extend(market, t, u, points):
    return market | {"joint_market": features(t, u, points)}


def original_row(row):
    return row | {"market": {k: v for k, v in row["market"].items() if k != "joint_market"}}
