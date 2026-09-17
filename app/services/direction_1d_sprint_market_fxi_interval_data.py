"""完整海外交易区间的FXI特征，同时保留旧输入供全部控制分支使用。"""

import hashlib
from datetime import datetime

import numpy as np

from app.services import direction_1d_sprint as base
from app.services import direction_1d_sprint_market_current_refresh as current
from app.services import direction_1d_sprint_market_only_data as parent

scope = parent.scope


def require(condition, reason):
    if not condition:
        raise ValueError("FXI_INTERVAL_" + reason)


def extend(market, t, u, points):
    """原价首个新增美国交易日开盘至最后收盘；单日严格等价，不能跳过中间缺失日。"""
    result = market | {"features": list(market["features"])}
    require(len(result["features"]) == 3 and np.isfinite(result["features"]).all(), "ORIGINAL_VECTOR_INVALID")
    if not market["available"]:
        return result
    days = parent.overnight.alignment(t, u)["new_us_dates"]
    require(bool(days), "NO_NEW_SESSION_BUT_MARKED_AVAILABLE")
    for day in days:
        require(day in points, "MISSING_INTERMEDIATE_DATE")
        row = points[day]
        require(
            all(
                type(row.get(k)) in (float, int) and np.isfinite(row[k]) and row[k] > 0
                for k in ("open", "high", "low", "close")
            ),
            "INVALID_OHLC",
        )
        require(type(row.get("volume")) in (float, int) and np.isfinite(row["volume"]), "INVALID_VOLUME")
        require(
            row["available"] is True
            and row["unavailable_reason"] is None
            and row["volume"] > 0
            and row["low"] <= min(row["open"], row["close"]) <= max(row["open"], row["close"]) <= row["high"],
            "INVALID_PRICE_RANGE",
        )
    result["features"][1] = float(np.clip(100.0 * (points[days[-1]]["close"] / points[days[0]]["open"] - 1), -20, 20))
    if len(days) == 1:
        require(result == market, "SINGLE_SESSION_MUST_MATCH_ORIGINAL")
    return result


def pair(original, interval):
    """新特征只进入候选，控制分支必须显式读original_market，不能随候选改变。"""
    return interval | {"original_market": original}


def original_row(row):
    return {k: v for k, v in row.items() if k != "market3"} | {"market": row["market3"]}


def dataset():
    original, proof = current.dataset()
    folder = base.ROOT / "fxi-full-interval-feasibility-v1"
    plan, result, saved = (base.read(folder / name) for name in ("plan.json", "result.json", "rows.json"))
    require(
        plan["script_sha256"] == hashlib.sha256((folder / "check.py").read_bytes()).hexdigest(),
        "FEASIBILITY_SCRIPT_CHANGED",
    )
    require(
        result["plan_hash"] == saved["plan_hash"] == base.digest(plan) and result["rows_hash"] == base.digest(saved),
        "FEASIBILITY_CHANGED",
    )
    require(plan["fresh_rows_hash"] == saved["fresh_rows_hash"] == proof["fresh_rows_hash"], "ORIGINAL_ROWS_CHANGED")
    etfs = parent.etfs.history()
    require(saved["etf_history_hash"] == base.digest(etfs), "ETF_HISTORY_CHANGED")
    rows, cache = [], {}
    for old, captured in zip(original, saved["rows"], strict=True):
        require(
            {
                k: v
                for k, v in captured.items()
                if k not in ("market_interval", "interval_alignment", "invalid_intermediate_dates")
            }
            == old,
            "ORIGINAL_LABEL_OR_ROW_CHANGED",
        )
        key = old["t"], old["u"]
        if key not in cache:
            cache[key] = extend(old["market"], *key, etfs["FXI"])
        require(
            captured["market_interval"] == cache[key] and not captured["invalid_intermediate_dates"],
            "INTERVAL_FEATURE_CHANGED",
        )
        require(captured["interval_alignment"] == parent.overnight.alignment(*key), "INTERVAL_ALIGNMENT_CHANGED")
        rows.append(old | {"market3": old["market"], "market": pair(old["market"], cache[key])})
    require(len(rows) == 38979, "TRAINING_SCOPE_CHANGED")
    return rows, proof | {"interval_rows_hash": base.digest(saved), "feature_proof_hash": base.digest(result)}


def live_market(source):
    """父输入只保存最后两行；重验相应HTTP字节后重新解析完整已有历史，不额外请求。"""
    target, t = source["target"], source["base"]
    require(parent.load(target) == source, "LIVE_PARENT_CHANGED")
    etfs = parent.etfs.load(target)
    require(base.digest(etfs) == source["etf_hash"] and etfs["base"] == t, "LIVE_ETF_BINDING_CHANGED")
    slot = etfs["raw_refs"]["FXI"]["slot"]
    _, meta = parent.etfs.load_piece(target, t, "FXI", slot)
    raw = (parent.etfs.root() / target / "FXI" / "raw" / f"{slot}.bin").read_bytes()
    require(
        hashlib.sha256(raw).hexdigest() == meta["body_sha256"] and len(raw) == meta["bytes"], "LIVE_RAW_BYTES_CHANGED"
    )
    aligned = parent.overnight.alignment(t, target)
    received = datetime.fromisoformat(meta["received_at"])
    require(
        all(datetime.fromisoformat(stamp) <= received for stamp in aligned["close_events"].values()),
        "UNFINISHED_SESSION",
    )
    points = parent.etfs.parse(raw, "FXI", aligned["required_us_dates"][-1])
    interval = extend(source["market"], t, target, points)
    return pair(source["market"], interval) | {
        "interval_raw_hash": meta["body_sha256"],
        "interval_us_dates": aligned["new_us_dates"],
    }
