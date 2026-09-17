"""Shibor历史快照和真实到达证据，严格使用基准国内交易日及前一交易日。

供应商也可能发布调休工作日记录；特征必须命中两个明确国内交易日，不按供应商相邻行推算。
每个目标日最多三个时槽请求，原始响应共享给所有基金，失败不自动追加请求。
"""

import hashlib
import json
import math
from datetime import date, datetime, time
from functools import lru_cache

import numpy as np

from app.integrations.tushare_sprint_shibor import FIELDS, MAX_BYTES, URL, fetch_shibor
from app.services import direction_1d_sprint as base

SOURCE = "TUSHARE_SHIBOR_EXISTING_PERMISSION_RESEARCH"
SLOTS = ("0700", "0730", "0800")


def root():
    return base.ROOT / "round-30" / "shibor-inputs"


def slot_for(at):
    return "0800" if at.time() >= time(8) else "0730" if at.time() >= time(7, 30) else "0700"


@lru_cache(maxsize=8)
def json_rows(raw):
    """缓存不可变行；空响应、重复日期和非法利率均拒绝，不缓存可变字典。"""
    if not isinstance(raw, bytes) or not raw or len(raw) > MAX_BYTES:
        raise ValueError("SHIBOR_RAW_SIZE_INVALID")
    value = json.loads(raw)
    if not isinstance(value, dict) or type(value.get("code")) is not int or value["code"] != 0:
        raise ValueError("SHIBOR_PROVIDER_REJECTED")
    data = value.get("data") or {}
    if data.get("fields") != FIELDS or not isinstance(data.get("items"), list) or not 1 <= len(data["items"]) <= 366:
        raise ValueError("SHIBOR_SCHEMA_OR_ROWS_INVALID")
    points = {}
    for row in data["items"]:
        if not isinstance(row, list) or len(row) != len(FIELDS):
            raise ValueError("SHIBOR_ROW_WIDTH_INVALID")
        text = row[0]
        if not isinstance(text, str) or len(text) != 8 or not text.isdigit():
            raise ValueError("SHIBOR_DATE_FORMAT_INVALID")
        day = datetime.strptime(text, "%Y%m%d").date().isoformat()
        if day in points:
            raise ValueError("SHIBOR_DUPLICATE_DATE")
        numbers = row[1:]
        if any(
            isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or not 0 < v < 100
            for v in numbers
        ):
            raise ValueError("SHIBOR_RATE_INVALID")
        points[day] = tuple(map(float, numbers))
    return tuple((day, *values) for day, values in sorted(points.items()))


def parse(raw, start="2021-01-01", end="2026-12-31"):
    """利率日期必须落在请求范围，拒绝供应商超范围返回的未来日期。"""
    rows = json_rows(raw)
    if any(not start <= row[0] <= end for row in rows):
        raise ValueError("SHIBOR_DATE_OUTSIDE_REQUEST")
    return {row[0]: dict(zip(FIELDS[1:], row[1:], strict=True)) for row in rows}


def required(base_day, target):
    days = list(map(str, base.calendar()[0]))
    if base_day not in days:
        raise ValueError("SHIBOR_BASE_OUTSIDE_CALENDAR")
    index = days.index(base_day)
    if index < 1 or index + 1 >= len(days) or days[index + 1] != target:
        raise ValueError("SHIBOR_TARGET_NOT_ADJACENT")
    return [days[index - 1], base_day]


def features(base_day, target, points):
    """百分数利率变化乘100得到基点；八项次序和±1000bp限幅在实验前固定。"""
    prior, anchor = required(base_day, target)
    if any(day not in points for day in (prior, anchor)):
        raise ValueError("SHIBOR_REQUIRED_DATE_MISSING")
    p, a = points[prior], points[anchor]
    values = [row[k] for row in (p, a) for k in FIELDS[1:]]
    if any(
        isinstance(v, bool) or not isinstance(v, (float, int)) or not math.isfinite(v) or not 0 < v < 100
        for v in values
    ):
        raise ValueError("SHIBOR_FEATURE_RATE_INVALID")
    differences = [100 * (a[k] - p[k]) for k in ("on", "1w", "3m", "1y")]
    spreads = [100 * (a[x] - a[y]) for x, y in (("1w", "on"), ("3m", "1w"), ("1y", "3m"))]
    return np.clip(differences, -1000, 1000).tolist() + [a["on"]] + np.clip(spreads, -1000, 1000).tolist()


def history():
    """逐年核对原始字节、请求范围及派生快照，不重复请求已取得的历史。"""
    folder = base.ROOT / "shibor-feasibility-v1"
    snapshot, plan = base.read(folder / "history.json"), base.read(folder / "history-plan.json")
    coverage = base.read(folder / "coverage-result.json")
    if (
        snapshot["plan_hash"] != base.digest(plan)
        or coverage["history_hash"] != base.digest(snapshot)
        or coverage["plan_hash"] != base.digest(plan)
        or hashlib.sha256((folder / "acquire_history.py").read_bytes()).hexdigest() != plan["script_sha256"]
    ):
        raise ValueError("SHIBOR_HISTORICAL_MANIFEST_CHANGED")
    points = {}
    for year in range(2021, 2027):
        meta = base.read(folder / f"receipt-{year}.json")
        attempt = base.read(folder / f"attempt-{year}.json")
        start, end = f"{year}-01-01", f"{year}-12-31" if year < 2026 else "2026-09-14"
        raw = (folder / f"response-{year}.json").read_bytes()
        if (
            meta["request_hash"] != base.digest(attempt)
            or attempt["plan_hash"] != base.digest(plan)
            or attempt["api_name"] != "shibor"
            or attempt["params"] != {"start_date": start.replace("-", ""), "end_date": end.replace("-", "")}
            or hashlib.sha256(raw).hexdigest() != meta["response_sha256"]
            or len(raw) != meta["bytes"]
        ):
            raise ValueError("SHIBOR_HISTORICAL_RAW_CHANGED")
        yearly = parse(raw, start, end)
        if len(yearly) != meta["rows"] or set(points).intersection(yearly):
            raise ValueError("SHIBOR_HISTORICAL_DATES_CHANGED")
        points.update(yearly)
    if points != snapshot["rows"]:
        raise ValueError("SHIBOR_HISTORICAL_PARSED_CHANGED")
    return points


def load(target):
    """回读原始JSON和实际请求时点，拒绝仅重写外层摘要的解析值或已过截止的响应。"""
    folder = root() / target
    value = base.read(folder / "input.json")
    deadline = datetime.combine(date.fromisoformat(target), time(8, 30), base.ZONE)
    if value["target"] != target or value["source"] != SOURCE or datetime.fromisoformat(value["at"]) >= deadline:
        raise ValueError("SHIBOR_LIVE_INPUT_SCOPE_OR_TIME_INVALID")
    needed = required(value["base"], target)
    slot = value["raw_ref"]["slot"]
    if slot not in SLOTS:
        raise ValueError("SHIBOR_LIVE_SLOT_INVALID")
    meta = base.read(folder / f"raw/{slot}.json")
    request = base.read(folder / f"requests/{slot}.json")
    request_at = datetime.fromisoformat(request["at"])
    if (
        value["raw_ref"]["hash"] != base.digest(meta)
        or meta["request_hash"] != base.digest(request)
        or request["target"] != target
        or request["base"] != value["base"]
        or request["url"] != URL
        or request["api_name"] != "shibor"
        or request["fields"] != FIELDS
        or request["required_dates"] != needed
        or meta["source"] != SOURCE
        or not datetime.combine(date.fromisoformat(target), time(7), base.ZONE) <= request_at < deadline
        or slot_for(request_at) != slot
        or not request_at <= datetime.fromisoformat(meta["received_at"]) < deadline
        or datetime.fromisoformat(meta["received_at"]) > datetime.fromisoformat(value["at"])
    ):
        raise ValueError("SHIBOR_LIVE_RESPONSE_CHANGED_OR_LATE")
    raw = (folder / f"raw/{slot}.response.json").read_bytes()
    if hashlib.sha256(raw).hexdigest() != meta["body_sha256"] or len(raw) != meta["bytes"]:
        raise ValueError("SHIBOR_LIVE_RAW_CHANGED")
    points = parse(raw, *needed)
    if any(day not in points for day in needed) or {d: points[d] for d in needed} != value["rows"]:
        raise ValueError("SHIBOR_LIVE_PARSED_INPUT_CHANGED")
    return value


def capture(at):
    """每目标最多三个固定时槽，每槽一次请求；已成功则全部基金共享，不自动重试。"""
    window = base.window(at)
    end = datetime.fromisoformat(base.read(base.ROOT / "protocol.json")["deadline_at"])
    if at >= end or window["status"] != "OPEN" or not time(7) <= at.time() < time(8, 30):
        raise ValueError("SHIBOR_LIVE_WINDOW_CLOSED")
    target, base_day = window["target_nav_date"], window["base_nav_date"]
    folder = root() / target
    if (folder / "input.json").exists():
        return load(target)
    deadline = datetime.combine(date.fromisoformat(target), time(8, 30), base.ZONE)
    request_at = base.now()
    if not datetime.combine(date.fromisoformat(target), time(7), base.ZONE) <= request_at < deadline:
        raise ValueError("SHIBOR_LIVE_REQUEST_TIME_INVALID")
    slot = slot_for(request_at)
    attempt = folder / f"requests/{slot}.json"
    if attempt.exists() or len(list((folder / "requests").glob("*.json"))) >= 3:
        raise ValueError("SHIBOR_LIVE_SLOT_OR_BUDGET_EXHAUSTED")
    needed = required(base_day, target)
    request = {
        "at": request_at.isoformat(),
        "url": URL,
        "api_name": "shibor",
        "fields": FIELDS,
        "target": target,
        "base": base_day,
        "required_dates": needed,
    }
    base.save(attempt, request)
    try:
        body = fetch_shibor(date.fromisoformat(needed[0]), date.fromisoformat(needed[1]))
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
            raise ValueError("SHIBOR_LIVE_RESPONSE_LATE")
        points = parse(body, *needed)
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
            raise ValueError("SHIBOR_LIVE_DEADLINE_REACHED")
        base.save(folder / "input.json", value)
        checked = load(target)
        if base.now() >= deadline:
            raise ValueError("SHIBOR_LIVE_READBACK_LATE")
        return checked
    except Exception as exc:
        base.save(folder / f"errors-{slot}.json", {"at": base.now().isoformat(), "error": base.error_code(exc)})
        raise
