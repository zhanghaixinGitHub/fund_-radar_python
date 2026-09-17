"""沪深300海外价格残差：包含开盘缺口，扣除中国当日已知价格变化。

美元ETF与人民币价格指数并非同一资产收益；差值还包含汇率、折溢价和跟踪误差。
现金分红仅按每股现金加回，不假称已取得复权/拆股全史或历史首次公告版本。
"""

import hashlib
import json
from datetime import date, datetime, time

import numpy as np

from app.services import direction_1d_sprint as base
from app.services import direction_1d_sprint_ashare_data as source
from app.services import direction_1d_sprint_overnight as overnight


def cash_snapshot():
    """现金必须与原始发行方响应重新解析一致，不能只重算摘要后篡改分红表。"""
    folder = base.ROOT / "ashr-cash-residual-source-v1"
    value = base.read(folder / "cash-distributions.json")
    raw = (base.ROOT / value["source_file"]).read_bytes()
    if hashlib.sha256(raw).hexdigest() != value["raw_sha256"]:
        raise ValueError("ASHR_CASH_RAW_CHANGED")
    data = json.loads(raw)["pdpResult"]
    table = next(
        x["table"]
        for x in data["pageSections"]["keyFacts"]["accordionItems"]
        if x["id"] == "fundinformation-distributions"
    )
    if [c["value"] for c in table["columns"]][:4] != ["Ex-Date", "Record date", "Pay date", "US$ / Share"]:
        raise ValueError("ASHR_CASH_COLUMNS_CHANGED")
    cash = {
        r["column_0"]["sortValue"][:10]: float(r["column_3"]["sortValue"])
        for r in table["values"]
        if r["column_0"]["sortValue"][:10] >= "2021-01-01"
    }
    if cash != value["cash"] or any(not np.isfinite(v) or v < 0 for v in cash.values()):
        raise ValueError("ASHR_CASH_PARSED_CHANGED")
    return value


def history():
    return source.history() | {"cash": cash_snapshot()["cash"]}


def interval(base_day, target):
    """取T中国收盘之前的最后美国收盘作起点，包含长假中全部已结束美国时段。"""
    china_close = datetime.combine(date.fromisoformat(base_day), time(15), base.ZONE)
    end = source.required(base_day, target)[-1]
    sessions = overnight.sessions()
    if sessions[end] <= china_close:
        return []
    anchor = max(day for day, close in sessions.items() if close < china_close)
    return sorted(day for day in sessions if anchor <= day <= end)


def features(base_day, target, points, china_return_pct):
    """返回残差百分数和可用标记，缺失/异常时保留原问题并使用父级预测。

    现金日期必须落在(起始收盘日,结束收盘日]，起点已除息的现金不重复加回。
    大于40%的单日原价跳变只用于拦截明显异常，不能证明细小拆股不存在。
    """
    if not np.isfinite(china_return_pct) or set(points) != {"ASHR", "cash"}:
        raise ValueError("ASHR_RESIDUAL_CONTEXT_INVALID")
    if not source.features(base_day, target, {"ASHR": points["ASHR"]})[1]:
        return [0.0, 0.0]
    days = interval(base_day, target)
    if not days or any(day not in points["ASHR"] for day in days):
        raise ValueError("ASHR_RESIDUAL_INTERVAL_MISSING")
    rows = [points["ASHR"][day] for day in days]
    for row in rows:
        prices = [row[k] for k in ("open", "high", "low", "close")]
        if not np.isfinite(prices).all() or min(prices) <= 0:
            raise ValueError("ASHR_RESIDUAL_PRICE_INVALID")
        valid = (
            row["low"] <= min(row["open"], row["close"]) <= max(row["open"], row["close"]) <= row["high"]
            and row["volume"] > 0
        )
        if row["available"] is not valid:
            raise ValueError("ASHR_RESIDUAL_QUALITY_CHANGED")
        if not valid:
            return [0.0, 0.0]
    if any(abs(right["close"] / left["close"] - 1) > 0.4 for left, right in zip(rows[:-1], rows[1:], strict=True)):
        return [0.0, 0.0]
    cash = sum(value for day, value in points["cash"].items() if days[0] < day <= days[-1])
    if not np.isfinite(cash) or cash < 0:
        raise ValueError("ASHR_RESIDUAL_CASH_INVALID")
    residual = 100 * ((rows[-1]["close"] + cash) / rows[0]["close"] - 1) - china_return_pct
    return [float(np.clip(residual, -20, 20)), 1.0]


def load(target):
    """复用第63轮真实HTTP字节，读取全区间，不发新的供应商请求。"""
    original = source.load(target)
    ref = original["raw_refs"]["ASHR"]
    # 上层load已校验请求、原始字节摘要、到账时间以及08:30边界，再解码全区间。
    raw = (source.root() / target / "ASHR" / f"raw/{ref['slot']}.bin").read_bytes()
    meta = base.read(source.root() / target / "ASHR" / f"raw/{ref['slot']}.json")
    if hashlib.sha256(raw).hexdigest() != meta["body_sha256"] or base.digest(meta) != ref["hash"]:
        raise ValueError("ASHR_RESIDUAL_LIVE_RAW_CHANGED")
    points = source.parse(raw, "ASHR", source.required(original["base"], target)[-1])
    cash = cash_snapshot()
    return original | {"rows": {"ASHR": points, "cash": cash["cash"]}, "cash_snapshot_hash": base.digest(cash)}


def capture(at):
    original = source.capture(at)
    return load(original["target"]) if original is not None else None
