"""最新已收盘美股日期的FXI估值差，优先复用旧分支原始文件，独立记录可用时点。

历史文件缺少首发时刻，只能作为重建开发数据；真实输入必须在08:30前取得并回读。
"""

import hashlib
import math
from bisect import bisect_right
from datetime import date, datetime, time

import numpy as np

from app.integrations.ishares_sprint_fxi import MAX_BYTES, URL, fetch_fxi
from app.services import direction_1d_sprint as base
from app.services import direction_1d_sprint_fxi_gap_data as parent
from app.services import direction_1d_sprint_overnight as overnight

SOURCE = "ISHARES_FXI_LATEST_COMPLETED_US_VALUATION_GAP"
SLOTS = ("0700", "0730", "0800")
parse = parent.parse


def root():
    return base.ROOT / "round-42/fxi-inputs"


def required(base_day, target):
    """目标必须是相邻国内交易日；选目标日08:30前已结束的最新两个美股现金交易日。"""
    parent.required(base_day, target)
    events = overnight.sessions()
    keys, closes = list(events), list(events.values())
    deadline = datetime.combine(date.fromisoformat(target), time(8, 30), base.ZONE)
    index = bisect_right(closes, deadline) - 1
    if index < 1:
        raise ValueError("FXI_OVERNIGHT_CASH_CALENDAR_UNCOVERED")
    return [keys[index - 1], keys[index]]


def features(base_day, target, points):
    needed = required(base_day, target)
    if any(day not in points for day in needed):
        raise ValueError("FXI_REQUIRED_ROW_ABSENT")
    available = True
    for day in needed:
        row = points[day]
        nav, nonfv = row["nav_per_share"], row["non_fv_nav"]
        if (
            isinstance(nav, bool)
            or not isinstance(nav, (float, int))
            or not math.isfinite(nav)
            or nav <= 0
            or (
                nonfv is not None
                and (
                    isinstance(nonfv, bool)
                    or not isinstance(nonfv, (float, int))
                    or not math.isfinite(nonfv)
                    or nonfv < 0
                )
            )
        ):
            raise ValueError("FXI_FEATURE_NUMBER_INVALID")
        reason = "EXPLICIT_MISSING" if nonfv is None else "INVALID_ZERO" if nonfv == 0 else None
        if row["available"] is not (reason is None) or row["unavailable_reason"] != reason:
            raise ValueError("FXI_AVAILABILITY_REASON_CHANGED")
        available &= reason is None
    if not available:
        return [0.0, 0.0, 0.0]
    prior, anchor = [100 * (points[d]["nav_per_share"] / points[d]["non_fv_nav"] - 1) for d in needed]
    return np.clip([anchor, anchor - prior], -20, 20).tolist() + [1.0]


def history():
    """沿用原发行方原始文件与缺失语义，同时核对本轮独立时点原型。"""
    proposal = base.read(base.ROOT / "round-42/proposal-before-training.json")
    feasible = base.read(base.ROOT / "round-42/input-feasibility.json")
    if (
        hashlib.sha256((base.ROOT / "round-42/feature-feasibility.py").read_bytes()).hexdigest()
        != proposal["script_sha256"]
        or proposal["parent_history_plan_hash"]
        != base.digest(base.read(base.ROOT / "fxi-masked-feasibility-v2/plan.json"))
        or proposal["parent_history_hash"]
        != base.digest(base.read(base.ROOT / "fxi-masked-feasibility-v2/history.json"))
        or feasible["proposal_hash"] != base.digest(proposal)
        or feasible["status"] != "FEASIBLE_NOT_TRAINED"
    ):
        raise ValueError("FXI_OVERNIGHT_HISTORICAL_SOURCE_CHANGED")
    return parent.history()


def closed_before(received_at, base_day, target):
    """日历截止只是计划；实际采集时点也必须晚于选中的所有现金收盘。"""
    events = overnight.sessions()
    if any(events[day] > received_at for day in required(base_day, target)):
        raise ValueError("FXI_OVERNIGHT_SESSION_NOT_CLOSED")


def raw_input(target, base_day, ref):
    """按来源分别回读原请求和响应，复用父文件不能伪造成一次新的供应商请求。"""
    slot = ref["slot"]
    if slot not in SLOTS:
        raise ValueError("FXI_OVERNIGHT_SLOT_INVALID")
    deadline = datetime.combine(date.fromisoformat(target), time(8, 30), base.ZONE)
    if ref["kind"] == "ROUND31":
        parent_input = parent.load(target)
        if (
            ref["parent_input_hash"] != base.digest(parent_input)
            or parent_input["base"] != base_day
            or parent_input["raw_ref"] != {"slot": slot, "hash": ref["hash"]}
        ):
            raise ValueError("FXI_OVERNIGHT_PARENT_INPUT_CHANGED")
        folder = parent.root() / target
        meta = base.read(folder / f"raw/{slot}.json")
    elif ref["kind"] == "OWN":
        folder = root() / target
        meta = base.read(folder / f"raw/{slot}.json")
        request = base.read(folder / f"requests/{slot}.json")
        if (
            meta["request_hash"] != base.digest(request)
            or meta["source"] != SOURCE
            or request["url"] != URL
            or request["target"] != target
            or request["base"] != base_day
            or request["required_dates"] != required(base_day, target)
            or not datetime.fromisoformat(request["at"]) <= datetime.fromisoformat(meta["received_at"]) < deadline
        ):
            raise ValueError("FXI_OVERNIGHT_RESPONSE_CHANGED_OR_LATE")
    else:
        raise ValueError("FXI_OVERNIGHT_RAW_OWNER_INVALID")
    raw = (folder / f"raw/{slot}.xls").read_bytes()
    if (
        not raw
        or len(raw) > MAX_BYTES
        or ref["hash"] != base.digest(meta)
        or len(raw) != meta["bytes"]
        or hashlib.sha256(raw).hexdigest() != meta["body_sha256"]
    ):
        raise ValueError("FXI_OVERNIGHT_RAW_CHANGED")
    received = datetime.fromisoformat(meta["received_at"])
    if received >= deadline:
        raise ValueError("FXI_OVERNIGHT_RESPONSE_LATE")
    closed_before(received, base_day, target)
    return raw, received


def load(target):
    """回读需要最新两个美股日期，旧分支成功不代表本轮最新日期已到达。"""
    value = base.read(root() / target / "input.json")
    deadline = datetime.combine(date.fromisoformat(target), time(8, 30), base.ZONE)
    if value["target"] != target or value["source"] != SOURCE or datetime.fromisoformat(value["at"]) >= deadline:
        raise ValueError("FXI_OVERNIGHT_INPUT_SCOPE_OR_TIME_INVALID")
    raw, received = raw_input(target, value["base"], value["raw_ref"])
    points = parse(raw)
    features(value["base"], target, points)
    wanted = {d: points[d] for d in required(value["base"], target)}
    if wanted != value["rows"] or received > datetime.fromisoformat(value["at"]):
        raise ValueError("FXI_OVERNIGHT_PARSED_INPUT_CHANGED")
    return value


def persist(target, base_day, ref, points):
    """输入保存后立即按原始证据回读；迟到文件可以保留，但不作为合格预测输入。"""
    deadline = datetime.combine(date.fromisoformat(target), time(8, 30), base.ZONE)
    features(base_day, target, points)
    value = {
        "at": base.now().isoformat(),
        "target": target,
        "base": base_day,
        "source": SOURCE,
        "rows": {d: points[d] for d in required(base_day, target)},
        "raw_ref": ref,
    }
    if base.now() >= deadline:
        raise ValueError("FXI_OVERNIGHT_DEADLINE_REACHED")
    base.save(root() / target / "input.json", value)
    checked = load(target)
    if base.now() >= deadline:
        raise ValueError("FXI_OVERNIGHT_READBACK_LATE")
    return checked


def capture(at):
    """先复用已验证父原文；最新现金日期尚未到达时最多三个独立时槽补取，无自动重试。"""
    window = base.window(at)
    end = datetime.fromisoformat(base.read(base.ROOT / "protocol.json")["deadline_at"])
    if at >= end or window["status"] != "OPEN" or not time(7) <= at.time() < time(8, 30):
        raise ValueError("FXI_OVERNIGHT_WINDOW_CLOSED")
    target, base_day = window["target_nav_date"], window["base_nav_date"]
    folder = root() / target
    if (folder / "input.json").exists():
        return load(target)
    deadline = datetime.combine(date.fromisoformat(target), time(8, 30), base.ZONE)
    if base.now() >= deadline:
        raise ValueError("FXI_OVERNIGHT_DEADLINE_REACHED")
    closed_before(base.now(), base_day, target)
    needed = required(base_day, target)
    if (parent.root() / target / "input.json").exists():
        parent_input = parent.load(target)
        ref = {"kind": "ROUND31", "parent_input_hash": base.digest(parent_input)} | parent_input["raw_ref"]
        raw, _ = raw_input(target, base_day, ref)
        points = parse(raw)
        if all(day in points for day in needed):
            return persist(target, base_day, ref, points)
    slot = "0800" if at.time() >= time(8) else "0730" if at.time() >= time(7, 30) else "0700"
    attempt = folder / f"requests/{slot}.json"
    if attempt.exists() or len(list((folder / "requests").glob("*.json"))) >= 3:
        raise ValueError("FXI_OVERNIGHT_SLOT_OR_BUDGET_EXHAUSTED")
    request = {"at": base.now().isoformat(), "url": URL, "target": target, "base": base_day, "required_dates": needed}
    base.save(attempt, request)
    try:
        body, headers = fetch_fxi()
        meta = {
            "received_at": base.now().isoformat(),
            "source": SOURCE,
            "body_sha256": hashlib.sha256(body).hexdigest(),
            "bytes": len(body),
            "headers": headers,
            "request_hash": base.digest(request),
        }
        raw_path = folder / f"raw/{slot}.xls"
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        with raw_path.open("xb") as file:
            file.write(body)
        base.save(folder / f"raw/{slot}.json", meta)
        if datetime.fromisoformat(meta["received_at"]) >= deadline:
            raise ValueError("FXI_OVERNIGHT_RESPONSE_LATE")
        ref = {"kind": "OWN", "slot": slot, "hash": base.digest(meta)}
        return persist(target, base_day, ref, parse(body))
    except Exception as exc:
        base.save(folder / f"errors-{slot}.json", {"at": base.now().isoformat(), "error": base.error_code(exc)})
        raise
