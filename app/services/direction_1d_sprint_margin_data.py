"""两融历史快照和真实到达证据，只使用基准日前一个交易日及更早的六日记录。

上一交易日记录可能在早上09:05才完整；不把它用于当日08:30截止的预测。
每个目标日最多三个时槽请求，原始响应共享给所有基金，失败不自动追加请求。
"""

import hashlib
import json
import math
from datetime import date, datetime, time

import numpy as np

from app.integrations.tushare_sprint_margin import FIELDS, MAX_BYTES, URL, fetch_margin
from app.services import direction_1d_sprint as base

SOURCE = "TUSHARE_MARGIN_EXISTING_PERMISSION_RESEARCH"
SLOTS = ("0700", "0730", "0800")


def root():
    return base.ROOT / "round-53" / "margin-inputs"


def slot_for(at):
    return "0800" if at.time() >= time(8) else "0730" if at.time() >= time(7, 30) else "0700"


def source_folder():
    return base.ROOT / "margin-source-v1"


def parse(raw, start="2021-01-01", end="2026-12-31"):
    """只使用沪深交易所金额汇总；北交所原始行保留在响应中，不混入研究输入。"""
    if not isinstance(raw, bytes) or not raw or len(raw) > MAX_BYTES:
        raise ValueError("MARGIN_RAW_SIZE_INVALID")
    value = json.loads(raw)
    if not isinstance(value, dict) or type(value.get("code")) is not int or value["code"] != 0:
        raise ValueError("MARGIN_PROVIDER_REJECTED")
    data = value.get("data") or {}
    if (
        data.get("fields") != FIELDS
        or data.get("has_more", False)
        or not isinstance(data.get("items"), list)
        or not 1 <= len(data["items"]) <= 1100
    ):
        raise ValueError("MARGIN_SCHEMA_OR_ROWS_INVALID")
    points = {}
    seen = set()
    for row in data["items"]:
        if not isinstance(row, list) or len(row) != len(FIELDS) or row[1] not in ("SSE", "SZSE", "BSE"):
            raise ValueError("MARGIN_ROW_INVALID")
        if not isinstance(row[0], str) or len(row[0]) != 8 or not row[0].isdigit():
            raise ValueError("MARGIN_DATE_FORMAT_INVALID")
        day = datetime.strptime(row[0], "%Y%m%d").date().isoformat()
        if not start <= day <= end or (day, row[1]) in seen:
            raise ValueError("MARGIN_DUPLICATE_OR_OUTSIDE_REQUEST")
        seen.add((day, row[1]))
        if row[1] == "BSE":
            continue
        values = row[2:]
        if any(isinstance(v, bool) or not isinstance(v, (float, int)) or not math.isfinite(v) or v < 0 for v in values):
            raise ValueError("MARGIN_AMOUNT_INVALID")
        if row[2] <= 0 or abs(row[2] + row[5] - row[6]) > max(1, abs(row[6]) * 1e-10):
            raise ValueError("MARGIN_BALANCE_IDENTITY_INVALID")
        points.setdefault(day, {})[row[1]] = dict(zip(FIELDS[2:], map(float, values), strict=True))
    if not points or any(set(v) != {"SSE", "SZSE"} for v in points.values()):
        raise ValueError("MARGIN_EXCHANGE_PAIR_MISSING")
    return dict(sorted(points.items()))


def required(base_day, target):
    """T->U只取T前一国内交易日及此前5日，避开T的数据在U08:30后才发布的问题。"""
    days = list(map(str, base.calendar()[0]))
    if base_day not in days:
        raise ValueError("MARGIN_BASE_OUTSIDE_CALENDAR")
    index = days.index(base_day)
    if index < 6 or index + 1 >= len(days) or days[index + 1] != target:
        raise ValueError("MARGIN_TARGET_NOT_ADJACENT_OR_HISTORY_SHORT")
    return days[index - 6 : index]


def features(base_day, target, points):
    """每交易所两项：1日和5日融资余额变化，均百分数并限幅±20%；排除与原表有差异的偿还额。"""
    needed = required(base_day, target)
    if any(day not in points for day in needed):
        raise ValueError("MARGIN_REQUIRED_DATE_MISSING")
    out = []
    for exchange in ("SSE", "SZSE"):
        rows = [points[d][exchange] for d in needed]
        if any(
            any(
                isinstance(v, bool) or not isinstance(v, (float, int)) or not math.isfinite(v) or v < 0
                for v in r.values()
            )
            or r["rzye"] <= 0
            for r in rows
        ):
            raise ValueError("MARGIN_FEATURE_AMOUNT_INVALID")
        first, previous, anchor = rows[0], rows[-2], rows[-1]
        out += [
            100 * (anchor["rzye"] / previous["rzye"] - 1),
            100 * (anchor["rzye"] / first["rzye"] - 1),
        ]
    return np.clip(out, -20, 20).tolist()


def raw_history():
    """逐字节复核已有六次历史响应及请求，不在训练时重新下载。"""
    folder = source_folder()
    plan = base.read(folder / "history-plan.json")
    if hashlib.sha256((folder / "fetch-history.py").read_bytes()).hexdigest() != plan["script_sha256"]:
        raise ValueError("MARGIN_HISTORY_SCRIPT_CHANGED")
    points = {}
    for i, (start, end) in enumerate(plan["windows"], 1):
        request = base.read(folder / f"history-request-{i}.json")
        receipt = base.read(folder / f"history-receipt-{i}.json")
        raw = (folder / f"history-response-{i}.json").read_bytes()
        if (
            request["url"] != URL
            or request["api_name"] != "margin"
            or request["fields"] != FIELDS
            or request["params"] != {"start_date": start, "end_date": end}
            or request["plan_hash"] != base.digest(plan)
            or receipt["request_hash"] != base.digest(request)
            or receipt["body_sha256"] != hashlib.sha256(raw).hexdigest()
            or receipt["bytes"] != len(raw)
        ):
            raise ValueError("MARGIN_HISTORICAL_RAW_CHANGED")

        def as_date(text):
            return datetime.strptime(text, "%Y%m%d").date().isoformat()

        yearly = parse(raw, as_date(start), as_date(end))
        if set(points).intersection(yearly):
            raise ValueError("MARGIN_YEARLY_OVERLAP")
        points.update(yearly)
    return dict(sorted(points.items()))


def history():
    folder = source_folder()
    snapshot = base.read(folder / "history.json")
    qualified = base.read(folder / "qualification-result.json")
    if qualified["history_hash"] != base.digest(snapshot) or qualified["status"] != "QUALIFIED_WITH_ONE_CN_SESSION_LAG":
        raise ValueError("MARGIN_QUALIFICATION_CHANGED")
    points = raw_history()
    if points != snapshot["rows"]:
        raise ValueError("MARGIN_HISTORY_CHANGED")
    return points


def load(target):
    """回读原始JSON和实际请求时点，拒绝仅重写外层摘要的解析值或已过截止的响应。"""
    folder = root() / target
    value = base.read(folder / "input.json")
    deadline = datetime.combine(date.fromisoformat(target), time(8, 30), base.ZONE)
    if value["target"] != target or value["source"] != SOURCE or datetime.fromisoformat(value["at"]) >= deadline:
        raise ValueError("MARGIN_LIVE_INPUT_SCOPE_OR_TIME_INVALID")
    needed = required(value["base"], target)
    slot = value["raw_ref"]["slot"]
    if slot not in SLOTS:
        raise ValueError("MARGIN_LIVE_SLOT_INVALID")
    meta = base.read(folder / f"raw/{slot}.json")
    request = base.read(folder / f"requests/{slot}.json")
    request_at = datetime.fromisoformat(request["at"])
    if (
        value["raw_ref"]["hash"] != base.digest(meta)
        or meta["request_hash"] != base.digest(request)
        or request["target"] != target
        or request["base"] != value["base"]
        or request["url"] != URL
        or request["api_name"] != "margin"
        or request["fields"] != FIELDS
        or request["required_dates"] != needed
        or meta["source"] != SOURCE
        or not datetime.combine(date.fromisoformat(target), time(7), base.ZONE) <= request_at < deadline
        or slot_for(request_at) != slot
        or not request_at <= datetime.fromisoformat(meta["received_at"]) < deadline
        or datetime.fromisoformat(meta["received_at"]) > datetime.fromisoformat(value["at"])
    ):
        raise ValueError("MARGIN_LIVE_RESPONSE_CHANGED_OR_LATE")
    raw = (folder / f"raw/{slot}.response.json").read_bytes()
    if hashlib.sha256(raw).hexdigest() != meta["body_sha256"] or len(raw) != meta["bytes"]:
        raise ValueError("MARGIN_LIVE_RAW_CHANGED")
    points = parse(raw, needed[0], needed[-1])
    if any(day not in points for day in needed) or {d: points[d] for d in needed} != value["rows"]:
        raise ValueError("MARGIN_LIVE_PARSED_INPUT_CHANGED")
    return value


def capture(at):
    """每目标最多三个固定时槽，每槽一次请求；已成功则全部基金共享，不自动重试。"""
    window = base.window(at)
    end = datetime.fromisoformat(base.read(base.ROOT / "protocol.json")["deadline_at"])
    if at >= end or window["status"] != "OPEN" or not time(7) <= at.time() < time(8, 30):
        raise ValueError("MARGIN_LIVE_WINDOW_CLOSED")
    target, base_day = window["target_nav_date"], window["base_nav_date"]
    folder = root() / target
    if (folder / "input.json").exists():
        return load(target)
    deadline = datetime.combine(date.fromisoformat(target), time(8, 30), base.ZONE)
    request_at = base.now()
    if not datetime.combine(date.fromisoformat(target), time(7), base.ZONE) <= request_at < deadline:
        raise ValueError("MARGIN_LIVE_REQUEST_TIME_INVALID")
    slot = slot_for(request_at)
    attempt = folder / f"requests/{slot}.json"
    if attempt.exists() or len(list((folder / "requests").glob("*.json"))) >= 3:
        raise ValueError("MARGIN_LIVE_SLOT_OR_BUDGET_EXHAUSTED")
    needed = required(base_day, target)
    request = {
        "at": request_at.isoformat(),
        "url": URL,
        "api_name": "margin",
        "fields": FIELDS,
        "target": target,
        "base": base_day,
        "required_dates": needed,
    }
    base.save(attempt, request)
    try:
        body = fetch_margin(date.fromisoformat(needed[0]), date.fromisoformat(needed[-1]))
        meta = {
            "received_at": base.now().isoformat(),
            "source": SOURCE,
            "body_sha256": hashlib.sha256(body).hexdigest(),
            "bytes": len(body),
            "request_hash": base.digest(request),
        }
        raw_path = folder / f"raw/{slot}.response.json"
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        with raw_path.open("xb") as file:
            file.write(body)
        base.save(folder / f"raw/{slot}.json", meta)
        if datetime.fromisoformat(meta["received_at"]) >= deadline:
            raise ValueError("MARGIN_LIVE_RESPONSE_LATE")
        points = parse(body, needed[0], needed[-1])
        features(base_day, target, points)
        value = {
            "at": base.now().isoformat(),
            "target": target,
            "base": base_day,
            "source": SOURCE,
            "rows": {d: points[d] for d in needed},
            "raw_ref": {"slot": slot, "hash": base.digest(meta)},
        }
        if base.now() >= deadline:
            raise ValueError("MARGIN_LIVE_DEADLINE_REACHED")
        base.save(folder / "input.json", value)
        checked = load(target)
        if base.now() >= deadline:
            raise ValueError("MARGIN_LIVE_READBACK_LATE")
        return checked
    except Exception as exc:
        base.save(folder / f"errors-{slot}.json", {"at": base.now().isoformat(), "error": base.error_code(exc)})
        raise
