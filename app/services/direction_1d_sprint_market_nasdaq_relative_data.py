"""第80轮纳指相对标普特征：重验已有历史，未来仅复用已收到的第6轮响应。"""

import hashlib
import json
from datetime import date, datetime, time

import numpy as np

from app.services import direction_1d_sprint as base
from app.services import direction_1d_sprint_ixic_data as ixic
from app.services import direction_1d_sprint_market_current_refresh as current
from app.services import direction_1d_sprint_market_only_data as parent

scope = parent.scope
deadline = parent.deadline


def root():
    return base.ROOT / "round-80" / "market-inputs"


def require(condition, reason):
    if not condition:
        raise ValueError("NASDAQ_RELATIVE_" + reason)


def history():
    """六个已保存年度原文分别解析，并检查跨年收盘连续性；不会触发数据采集。"""
    value, spec = base.read(ixic.root() / "history.json"), ixic.plan()
    proof = base.read(ixic.root() / "source-proof.json")
    require(value["code"] == ixic.fingerprint() and value["plan_hash"] == base.digest(spec), "HISTORY_CODE_CHANGED")
    require(proof["history_hash"] == base.digest(value), "HISTORY_PROOF_CHANGED")
    points = {}
    for year in range(2021, 2027):
        raw = base.read(ixic.root() / "raw" / f"{year}.json")
        saved = base.read(ixic.root() / "data" / f"{year}.json")
        rows = ixic.validate(
            json.dumps(raw["response"]).encode(), date(year, 1, 1), min(date(year, 12, 31), date(2026, 9, 11))
        )
        require(
            saved["raw_hash"] == base.digest(raw)
            and saved["rows"] == rows
            and saved["received_at"] == raw["received_at"],
            "ANNUAL_SOURCE_CHANGED",
        )
        points.update(rows)
    require(points == value["rows"] and len(points) == 1429, "HISTORY_ROWS_CHANGED")
    dates = sorted(points)
    require(
        all(
            parent.overnight.same_previous_close(points[a]["close"], points[z]["pre_close"])
            for a, z in zip(dates, dates[1:], strict=False)
        ),
        "HISTORY_PRICE_CHAIN_INVALID",
    )
    return value


def extend(market, t, u, points):
    """第四项为两个指数同一现金收盘区间的变化之差，单位百分点，不使用不同日历的日涨幅。"""
    aligned = parent.overnight.alignment(t, u)
    extra = parent.overnight.extend([0.0] * 30, aligned, points)
    require(len(market["features"]) == 3 and extra[31] == market["us_sessions"], "SESSION_OR_WIDTH_MISMATCH")
    relative = extra[30] * 100.0 - market["features"][0]
    vector = market["features"] + [relative]
    require(np.isfinite(vector).all(), "NONFINITE_FEATURE")
    return market | {"features": vector}


def original_row(row):
    return {k: v for k, v in row.items() if k != "market3"} | {"market": row["market3"]}


def dataset():
    """按已验证的近期增量重建四项输入；旧2025题和标签不变，2026仅作成熟训练行。"""
    original, proof = current.dataset()
    folder = base.ROOT / "nasdaq-relative-market-feasibility-v1"
    p, result, saved = (base.read(folder / n) for n in ("plan.json", "result.json", "rows.json"))
    require(
        p["script_sha256"] == hashlib.sha256((folder / "check.py").read_bytes()).hexdigest(),
        "FEASIBILITY_SCRIPT_CHANGED",
    )
    require(
        result["plan_hash"] == saved["plan_hash"] == base.digest(p) and result["rows_hash"] == base.digest(saved),
        "FEASIBILITY_CHANGED",
    )
    require(p["fresh_rows_hash"] == saved["fresh_rows_hash"] == proof["fresh_rows_hash"], "FRESH_LABELS_CHANGED")
    source = history()
    require(p["ixic_history_hash"] == base.digest(source), "IXIC_HISTORY_CHANGED")
    rows, cache = [], {}
    for old, captured in zip(original, saved["rows"], strict=True):
        require({k: v for k, v in captured.items() if k not in ("market4", "ixic")} == old, "ORIGINAL_ROW_CHANGED")
        key = old["t"], old["u"]
        if key not in cache:
            cache[key] = extend(old["market"], *key, source["rows"])
        require(cache[key] == captured["market4"], "HISTORICAL_VECTOR_CHANGED")
        aligned = parent.overnight.alignment(*key)
        expected_ixic = {
            "ixic_return_pct": (parent.overnight.extend([0.0] * 30, aligned, source["rows"])[30] * 100.0),
            "relative_spx_pp": cache[key]["features"][3],
            "alignment": aligned,
        }
        require(captured["ixic"] == expected_ixic, "HISTORICAL_ALIGNMENT_CHANGED")
        rows.append(old | {"market3": old["market"], "market": cache[key]})
    require(len(rows) == 38979, "HISTORICAL_SCOPE_CHANGED")
    return rows, proof | {"four_feature_rows_hash": base.digest(saved), "feature_proof_hash": base.digest(result)}


def observation(t, u, slot, before):
    """验证第6轮真实响应及请求占位。未收到合格当日行情不能填昨天或零值。"""
    require(slot in ("0700", "0730", "0800"), "INVALID_SLOT")
    folder = base.ROOT / "round-06" / "live" / u
    value = base.read(folder / f"{slot}-response.json")
    reserved = base.read(folder / f"{slot}-reserved.json")
    aligned = parent.overnight.alignment(t, u)
    received = datetime.fromisoformat(value["received_at"])
    requested = datetime.fromisoformat(reserved["at"])
    require(received.tzinfo is not None and requested.tzinfo is not None and before.tzinfo is not None, "NAIVE_TIME")
    require(requested <= received <= before < deadline(u), "LATE_OR_FUTURE_RESPONSE")
    require(
        value["alignment"] == reserved["alignment"] == aligned and reserved["maximum_requests"] == 1,
        "LIVE_ALIGNMENT_CHANGED",
    )
    require(
        all(datetime.fromisoformat(close) <= received for close in aligned["close_events"].values()),
        "UNFINISHED_SESSION",
    )
    required = aligned["required_us_dates"]
    points = ixic.validate(
        json.dumps(value["response"]).encode(), date.fromisoformat(required[0]), date.fromisoformat(required[-1])
    )
    return value, reserved, points


def load(target):
    """独立输入实际保存/回读时间与原响应摘要一起校验，后续可重读但不能改写。"""
    path = root() / target / "input.json"
    value = base.read(path)
    receipt = base.read(path.with_name("receipt.json"))
    source = parent.load(target)
    created, checked = datetime.fromisoformat(value["at"]), datetime.fromisoformat(receipt["readback_at"])
    require(value["base"] == source["base"] and value["target"] == source["target"] == target, "INPUT_DATES_CHANGED")
    require(datetime.fromisoformat(source["at"]) <= created <= checked < deadline(target), "INPUT_TIME_INVALID")
    require(receipt["status"] == "VERIFIED" and receipt["input_hash"] == base.digest(value), "INPUT_RECEIPT_INVALID")
    raw, reserved, points = observation(value["base"], target, value["ixic_slot"], created)
    require(
        value["parent_hash"] == base.digest(source)
        and value["ixic_hash"] == base.digest(raw)
        and value["ixic_reservation_hash"] == base.digest(reserved),
        "LIVE_SOURCE_CHANGED",
    )
    require(value["market"] == extend(source["market"], source["base"], target, points), "LIVE_VECTOR_CHANGED")
    return value


def capture(at):
    """仅消费已有文件，无供应商调用；缺纳指时等待，截止后由报告计缺失。"""
    at = at.astimezone(base.ZONE)
    target = str(at.date())
    if parent.ended(at) or not time(7) <= at.timetz().replace(tzinfo=None) < time(8, 30):
        return None
    path = root() / target / "input.json"
    if path.exists():
        return load(target)
    if not (parent.root() / target / "input.json").exists():
        return None
    source = parent.load(target)
    for slot in ("0700", "0730", "0800"):
        raw_path = base.ROOT / "round-06" / "live" / target / f"{slot}-response.json"
        if not raw_path.exists():
            continue
        try:
            raw, reserved, points = observation(source["base"], target, slot, base.now())
        except Exception as exc:
            rejection = root() / target / f"{slot}-rejected.json"
            if not rejection.exists():
                base.save(
                    rejection,
                    {
                        "at": base.now().isoformat(),
                        "error": base.error_code(exc),
                        "source_hash": base.digest(base.read(raw_path)),
                    },
                )
            continue
        created = base.now()
        if created >= deadline(target):
            return None
        value = {
            "at": created.isoformat(),
            "base": source["base"],
            "target": target,
            "parent_hash": base.digest(source),
            "ixic_slot": slot,
            "ixic_hash": base.digest(raw),
            "ixic_reservation_hash": base.digest(reserved),
            "market": extend(source["market"], source["base"], target, points),
        }
        base.save(path, value)
        loaded, checked = base.read(path), base.now()
        base.save(
            path.with_name("receipt.json"),
            {
                "readback_at": checked.isoformat(),
                "input_hash": base.digest(value),
                "status": "VERIFIED" if loaded == value and checked < deadline(target) else "LATE_OR_INVALID",
            },
        )
        return load(target)
    return None


def live_market(source):
    value = load(source["target"])
    require(value["parent_hash"] == base.digest(source), "PARENT_CHANGED")
    # 摘要随市场向量写入预测，原输入即使数值相同也不能换成另一次较晚响应。
    return value["market"] | {"nasdaq_input_hash": base.digest(value)}
