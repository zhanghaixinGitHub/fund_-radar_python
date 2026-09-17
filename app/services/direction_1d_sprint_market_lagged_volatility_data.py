"""过去20个完整市场观察的波动尺度；未来仅消费已保存的原始来源，不发起请求。"""

import hashlib
from datetime import datetime, time

import numpy as np

from app.services import direction_1d_sprint as base
from app.services import direction_1d_sprint_market_fxi_interval_data as interval
from app.services import direction_1d_sprint_market_only_data as parent

scope = parent.scope
deadline = parent.deadline
LOOKBACK = 20
FLOOR = 1e-6
SCAN_LIMIT = 80


def root():
    return base.ROOT / "round-84" / "market-inputs"


def require(condition, reason):
    if not condition:
        raise ValueError("LAGGED_VOLATILITY_" + reason)


def context(market, u, previous):
    """尺度不包含当前目标；0波动用固定正下限，不减均值，不改变方向符号。"""
    if not market["available"]:
        return {"available": False, "reason": "PARENT_SOURCE_UNAVAILABLE"}
    if len(previous) < LOOKBACK:
        return {"available": False, "reason": "LESS_THAN20_PRIOR_COMPLETE_DATES"}
    prior = previous[-LOOKBACK:]
    dates = [r["u"] for r in prior]
    require(dates == sorted(set(dates)) and max(dates) < u, "CURRENT_OR_DUPLICATE_CONTEXT_DATE")
    require(all(r["market"]["available"] for r in prior), "CONTEXT_NOT_COMPLETE")
    x = np.asarray([r["market"]["features"] for r in prior], dtype=float)
    require(x.shape == (LOOKBACK, 3) and np.isfinite(x).all(), "CONTEXT_VECTOR_INVALID")
    scale = np.maximum(x.std(axis=0, ddof=0), FLOOR)
    normalized = np.asarray(market["features"], dtype=float) / scale
    require(normalized.shape == (3,) and np.isfinite(normalized).all(), "NORMALIZED_VECTOR_INVALID")
    return {
        "available": True,
        "dates": dates,
        "market_context_hash": base.digest(prior),
        "scale": scale.tolist(),
        "normalized_features": normalized.tolist(),
    }


def paired(market, captured):
    """候选、完整区间学习对照、原固定规则三份输入分别保留。"""
    return market | {
        "features": captured["normalized_features"] if captured["available"] else list(market["features"]),
        "interval_market": market,
        "volatility_context": captured,
    }


def original_row(row):
    return {k: v for k, v in row.items() if k != "interval_market"} | {"market": row["interval_market"]}


def dataset():
    """逐日期重建上下文并比对事前可行性快照；基金标签完全不参与尺度计算。"""
    rows, proof = interval.dataset()
    folder = base.ROOT / "lagged-market-volatility-feasibility-v1"
    p, result, saved = (base.read(folder / n) for n in ("plan.json", "result.json", "contexts.json"))
    require(p["script_sha256"] == hashlib.sha256((folder / "check.py").read_bytes()).hexdigest(), "SCRIPT_CHANGED")
    require(result["plan_hash"] == saved["plan_hash"] == base.digest(p), "FEASIBILITY_PLAN_CHANGED")
    require(result["contexts_hash"] == base.digest(saved), "CONTEXTS_CHANGED")
    require(saved["interval_rows_hash"] == proof["interval_rows_hash"], "INTERVAL_ROWS_CHANGED")
    unique = {}
    for r in rows:
        v = {"t": r["t"], "u": r["u"], "market": r["market"]}
        require(r["u"] not in unique or unique[r["u"]] == v, "DATE_MARKET_DIFFERS_BY_FUND")
        unique[r["u"]] = v
    previous, rebuilt = [], {}
    for u, v in sorted(unique.items()):
        rebuilt[u] = context(v["market"], u, previous)
        if v["market"]["available"]:
            previous.append(v)
    require(rebuilt == saved["contexts"], "RECONSTRUCTED_CONTEXT_CHANGED")
    transformed = [r | {"interval_market": r["market"], "market": paired(r["market"], rebuilt[r["u"]])} for r in rows]
    return transformed, proof | {"contexts_hash": base.digest(saved), "volatility_proof_hash": base.digest(result)}


def merge_prices(existing, extra):
    """旧历史与真实增量重叠价格必须相同；不以较晚响应静默覆盖过去记录。"""
    for day, row in extra.items():
        require(
            all(
                type(row.get(k)) in (int, float) and np.isfinite(row[k]) and row[k] > 0 for k in ("close", "pre_close")
            ),
            "SPX_INVALID_PRICE",
        )
        if day in existing:
            require(all(existing[day][k] == row[k] for k in ("close", "pre_close")), "SPX_OVERLAP_CONFLICT")
        else:
            existing[day] = row


def full_sources(source, before, source_targets):
    """重验所有绑定输入及HTTP原字节，再解析全历史；裁剪后的两行不能充作20日上下文。"""
    target, t = source["target"], source["base"]
    require(parent.load(target) == source, "PARENT_CHANGED")
    require(
        before.tzinfo is not None and datetime.fromisoformat(source["at"]) <= before < deadline(target),
        "SOURCE_TIME_INVALID",
    )
    history = base.read(parent.overnight.root() / "spx.json")
    spec = base.read(base.ROOT / "nav-independent-market-feasibility-v1/plan.json")
    require(base.digest(history) == spec["source_hashes"]["spx"], "SPX_HISTORY_CHANGED")
    require(before < datetime.fromisoformat(history["expires_at"]), "SPX_HISTORY_EXPIRED")
    spx, refs = dict(history["rows"]), {}
    days = list(map(str, base.calendar()[0]))
    require(
        target in days and target in source_targets and source_targets == sorted(set(source_targets)),
        "SOURCE_TARGETS_INVALID",
    )
    bounded = days[max(1, days.index(target) - SCAN_LIMIT) : days.index(target) + 1]
    for day in source_targets:
        require(day in bounded, "SOURCE_TARGET_OUTSIDE_BOUND")
        saved = parent.load(day)
        observation, ref = parent.spx_observation(
            day, saved["base"], saved["spx_ref"]["origin"], saved["spx_ref"]["slot"]
        )
        require(
            datetime.fromisoformat(observation["received_at"]) <= datetime.fromisoformat(saved["at"]) <= before,
            "FUTURE_SOURCE_RESPONSE",
        )
        merge_prices(spx, observation["parsed"]["rows"])
        refs[day] = {"input_hash": base.digest(saved), "spx_ref": ref}
    e, c = parent.etfs.load(target), parent.cnya.load(target)
    require(base.digest(e) == source["etf_hash"] and base.digest(c) == source["cnya_hash"], "RAW_INPUT_BINDING_CHANGED")
    aligned = parent.overnight.alignment(t, target)
    etf_points, raw_refs = {}, {}
    for symbol in parent.etfs.SYMBOLS:
        slot = e["raw_refs"][symbol]["slot"]
        _, meta = parent.etfs.load_piece(target, t, symbol, slot)
        raw = (parent.etfs.root() / target / symbol / "raw" / f"{slot}.bin").read_bytes()
        require(hashlib.sha256(raw).hexdigest() == meta["body_sha256"] and len(raw) == meta["bytes"], "ETF_RAW_CHANGED")
        require(
            datetime.fromisoformat(meta["received_at"]) <= datetime.fromisoformat(e["at"]) <= before, "ETF_TIME_INVALID"
        )
        etf_points[symbol] = parent.etfs.parse(raw, symbol, aligned["required_us_dates"][-1])
        raw_refs[symbol] = {"meta_hash": base.digest(meta), "body_sha256": meta["body_sha256"]}
    slot = c["raw_ref"]["slot"]
    folder = parent.cnya.root() / target / "raw"
    meta, raw = base.read(folder / f"{slot}.json"), (folder / f"{slot}.xls").read_bytes()
    require(hashlib.sha256(raw).hexdigest() == meta["body_sha256"] and len(raw) == meta["bytes"], "CNYA_RAW_CHANGED")
    require(
        datetime.fromisoformat(meta["received_at"]) <= datetime.fromisoformat(c["at"]) <= before, "CNYA_TIME_INVALID"
    )
    cn = parent.cnya.parse(raw)
    original = parent.feature_values(t, target, spx, etf_points, cn)
    require(original == source["market"], "RAW_PARENT_VECTOR_CHANGED")
    return (
        spx,
        etf_points,
        cn,
        {
            "spx_history_hash": base.digest(history),
            "spx_inputs": refs,
            "etf_raw": raw_refs,
            "cnya_raw": {"meta_hash": base.digest(meta), "body_sha256": meta["body_sha256"]},
        },
    )


def past_vectors(target, spx, etfs, cn):
    """最多80个中国相邻交易日；缺整行可以跳过，非法数值或价格链不能当缺失吞掉。"""
    days = list(map(str, base.calendar()[0]))
    index = days.index(target)
    values, missing = [], []
    for i in range(max(1, index - SCAN_LIMIT), index):
        t, u = days[i - 1], days[i]
        aligned = parent.overnight.alignment(t, u)
        needed = parent.etfs.required(t, u)
        if (
            any(d not in spx for d in aligned["required_us_dates"])
            or any(d not in etfs[s] for s in parent.etfs.SYMBOLS for d in needed)
            or any(d not in cn for d in parent.cnya.required(t, u))
            or any(d not in etfs["FXI"] for d in aligned["new_us_dates"])
        ):
            missing.append({"u": u, "reason": "SOURCE_DATE_ABSENT"})
            continue
        market = parent.feature_values(t, u, spx, etfs, cn)
        if not market["available"]:
            missing.append({"u": u, "reason": "PARENT_SOURCE_UNAVAILABLE"})
            continue
        values.append({"t": t, "u": u, "market": interval.pair(market, interval.extend(market, t, u, etfs["FXI"]))})
    return values[-LOOKBACK:], missing


def rebuild(source, before, source_targets):
    spx, e, c, refs = full_sources(source, before, source_targets)
    previous, missing = past_vectors(source["target"], spx, e, c)
    current = interval.pair(
        source["market"], interval.extend(source["market"], source["base"], source["target"], e["FXI"])
    )
    captured = context(current, source["target"], previous)
    require(len(previous) == LOOKBACK, "INSUFFICIENT_COMPLETE_CONTEXT")
    return {
        "market": paired(current, captured),
        "context_rows": previous,
        "skipped_dates": missing,
        "source_refs": refs,
    }


def load(target):
    path = root() / target / "input.json"
    value, receipt, source = base.read(path), base.read(path.with_name("receipt.json")), parent.load(target)
    created, checked = datetime.fromisoformat(value["at"]), datetime.fromisoformat(receipt["readback_at"])
    require(
        created.tzinfo is not None
        and checked.tzinfo is not None
        and datetime.fromisoformat(source["at"]) <= created <= checked < deadline(target),
        "INPUT_TIME_INVALID",
    )
    require(
        value["target"] == target and value["base"] == source["base"] and value["parent_hash"] == base.digest(source),
        "INPUT_BINDING_CHANGED",
    )
    require(receipt["status"] == "VERIFIED" and receipt["input_hash"] == base.digest(value), "INPUT_RECEIPT_INVALID")
    rebuilt = rebuild(source, created, value["source_targets"])
    require(all(value[k] == v for k, v in rebuilt.items()), "CONTEXT_OR_SOURCE_CHANGED")
    return value


def capture(at):
    """已有来源不足则记录原因等待；不新增请求，也不伪造准时的输入回执。"""
    at = at.astimezone(base.ZONE)
    target = str(at.date())
    if parent.ended(at) or not time(7) <= at.time() < time(8, 30):
        return None
    path = root() / target / "input.json"
    if path.exists():
        return load(target)
    if not (parent.root() / target / "input.json").exists():
        return None
    source = parent.load(target)
    days = list(map(str, base.calendar()[0]))
    index = days.index(target)
    sources = [d for d in days[max(1, index - SCAN_LIMIT) : index + 1] if (parent.root() / d / "input.json").exists()]
    try:
        rebuilt = rebuild(source, base.now(), sources)
    except (ValueError, FileNotFoundError) as exc:
        base.save(
            root() / target / "context-unavailable.json",
            {"at": base.now().isoformat(), "error": base.error_code(exc)},
            replace=True,
        )
        return None
    created = base.now()
    if created >= deadline(target):
        return None
    value = {
        "at": created.isoformat(),
        "target": target,
        "base": source["base"],
        "parent_hash": base.digest(source),
        "source_targets": sources,
    } | rebuilt
    base.save(path, value)
    # 回读包括所有原始源与上下文重建；不能仅回读外层JSON就声称源校验完成。
    saved = base.read(path)
    verified = rebuild(source, created, sources)
    checked = base.now()
    status = (
        "VERIFIED"
        if saved == value and all(value[k] == v for k, v in verified.items()) and checked < deadline(target)
        else "LATE_OR_INVALID"
    )
    base.save(
        path.with_name("receipt.json"),
        {"readback_at": checked.isoformat(), "input_hash": base.digest(value), "status": status},
    )
    return load(target)


def live_market(source):
    value = load(source["target"])
    require(value["parent_hash"] == base.digest(source), "LIVE_PARENT_CHANGED")
    return value["market"] | {"volatility_input_hash": base.digest(value)}
