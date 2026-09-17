"""美国国债年化收益率：独立发行日历、相邻两期变化及预测前真实到达证据。

历史按18:00纽约时间可得作开发假设，实际仍须08:30前取得原始响应；不前向填充缺失报价。
"""

import csv
import hashlib
import io
import json
import math
from bisect import bisect_left
from datetime import date, datetime, time
from functools import lru_cache
from zoneinfo import ZoneInfo

import numpy as np

from app.integrations.tushare_sprint_us_tycr import FIELDS, MAX_BYTES, URL, fetch_us_tycr
from app.services import direction_1d_sprint as base

SOURCE = "TUSHARE_US_TREASURY_EXISTING_PERMISSION_RESEARCH"
SLOTS = ("0700", "0730", "0800")


def root():
    return base.ROOT / "round-52" / "us_tycr-inputs"


def slot_for(at):
    return "0800" if at.time() >= time(8) else "0730" if at.time() >= time(7, 30) else "0700"


@lru_cache(maxsize=8)
def json_rows(raw):
    """缓存不可变行；空响应、重复日期和非法利率均拒绝，不缓存可变字典。"""
    if not isinstance(raw, bytes) or not raw or len(raw) > MAX_BYTES:
        raise ValueError("US_TREASURY_RAW_SIZE_INVALID")
    value = json.loads(raw)
    if not isinstance(value, dict) or type(value.get("code")) is not int or value["code"] != 0:
        raise ValueError("US_TREASURY_PROVIDER_REJECTED")
    data = value.get("data") or {}
    if (
        data.get("has_more", False)
        or data.get("fields") != FIELDS
        or not isinstance(data.get("items"), list)
        or not 1 <= len(data["items"]) <= 366
    ):
        raise ValueError("US_TREASURY_SCHEMA_OR_ROWS_INVALID")
    points = {}
    for row in data["items"]:
        if not isinstance(row, list) or len(row) != len(FIELDS):
            raise ValueError("US_TREASURY_ROW_WIDTH_INVALID")
        text = row[0]
        if not isinstance(text, str) or len(text) != 8 or not text.isdigit():
            raise ValueError("US_TREASURY_DATE_FORMAT_INVALID")
        day = datetime.strptime(text, "%Y%m%d").date().isoformat()
        if day in points:
            raise ValueError("US_TREASURY_DUPLICATE_DATE")
        numbers = row[1:]
        if any(
            isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or not 0 <= v <= 30
            for v in numbers
        ):
            raise ValueError("US_TREASURY_RATE_INVALID")
        points[day] = tuple(map(float, numbers))
    return tuple((day, *values) for day, values in sorted(points.items()))


def parse(raw, start="2021-01-01", end="2026-12-31"):
    """利率日期必须落在请求范围，拒绝供应商超范围返回的未来日期。"""
    rows = json_rows(raw)
    if any(not start <= row[0] <= end for row in rows):
        raise ValueError("US_TREASURY_DATE_OUTSIDE_REQUEST")
    return {row[0]: dict(zip(FIELDS[1:], row[1:], strict=True)) for row in rows}


def source_folder():
    return base.ROOT / "us-treasury-source-v1"


def publication(day):
    """仅作历史开发的可得时点假设；实际发布可能延迟，真实预测必须另核对接收时间。"""
    return datetime.combine(date.fromisoformat(day), time(18), ZoneInfo("America/New_York"))


def required(base_day, target):
    """根据独立美债日期表取截止前最近两期，不把纽约股票交易日直接当美债日期。"""
    days = list(map(str, base.calendar()[0]))
    if base_day not in days or days.index(base_day) + 1 >= len(days) or days[days.index(base_day) + 1] != target:
        raise ValueError("US_TREASURY_TARGET_NOT_ADJACENT")
    calendar = base.read(source_folder() / "calendar.json")["dates"]
    # 仅覆盖冻结研究区间，避免日历用尽后悄悄复用旧数据。
    if not "2021-01-06" <= target <= "2026-09-17":
        raise ValueError("US_TREASURY_CALENDAR_RANGE")
    deadline = datetime.combine(date.fromisoformat(target), time(8, 30), base.ZONE)
    index = bisect_left(calendar, target)
    while index and publication(calendar[index - 1]) >= deadline:
        index -= 1
    if index < 2:
        raise ValueError("US_TREASURY_HISTORY_TOO_SHORT")
    return calendar[index - 2 : index]


def features(base_day, target, points):
    """五期限日变化、10年减2年及10年减3月期限差（基点），10年水平（年化%），最后为新数据标识。"""
    prior, anchor = required(base_day, target)
    if any(day not in points for day in (prior, anchor)):
        raise ValueError("US_TREASURY_REQUIRED_DATE_MISSING")
    p, a = points[prior], points[anchor]
    values = [row[k] for row in (p, a) for k in FIELDS[1:]]
    if any(
        isinstance(v, bool) or not isinstance(v, (float, int)) or not math.isfinite(v) or not 0 <= v <= 30
        for v in values
    ):
        raise ValueError("US_TREASURY_FEATURE_RATE_INVALID")
    base_close = datetime.combine(date.fromisoformat(base_day), time(15), base.ZONE)
    if publication(anchor) <= base_close:
        return [0.0] * 9
    changes = [100 * (a[k] - p[k]) for k in FIELDS[1:]]
    spreads = [100 * (a["y10"] - a[k]) for k in ("y2", "m3")]
    return np.clip(changes + spreads, -1000, 1000).tolist() + [a["y10"], 1.0]


def raw_history():
    """重读六年原始响应，逐项绑定日期范围和哈希；仅缺失的2021-06-28由原始财政部CSV补齐。"""
    folder = source_folder()
    plan = base.read(folder / "history-plan.json")
    if hashlib.sha256((folder / "fetch-history.py").read_bytes()).hexdigest() != plan["script_sha256"]:
        raise ValueError("US_TREASURY_ACQUISITION_CODE_CHANGED")
    points = {}
    for i, (start, end) in enumerate(plan["windows"], 1):
        request = base.read(folder / f"history-request-{i}.json")
        receipt = base.read(folder / f"history-receipt-{i}.json")
        raw = (folder / f"history-response-{i}.json").read_bytes()
        if (
            request["plan_hash"] != base.digest(plan)
            or receipt["request_hash"] != base.digest(request)
            or request["api_name"] != "us_tycr"
            or request["fields"] != FIELDS
            or request["url"] != URL
            or request["params"] != {"start_date": start, "end_date": end}
            or receipt["body_sha256"] != hashlib.sha256(raw).hexdigest()
            or receipt["bytes"] != len(raw)
        ):
            raise ValueError("US_TREASURY_HISTORICAL_RAW_CHANGED")

        def as_date(text):
            return datetime.strptime(text, "%Y%m%d").date().isoformat()

        rows = parse(raw, as_date(start), as_date(end))
        if set(points).intersection(rows) or len(rows) != receipt["rows"]:
            raise ValueError("US_TREASURY_HISTORICAL_ROWS_CHANGED")
        points.update(rows)
    meta = base.read(folder / "primary-june2021-receipt.json")
    request = base.read(folder / "primary-june2021-request.json")
    raw = (folder / "primary-june2021.csv").read_bytes()
    if (
        meta["request_hash"] != base.digest(request)
        or hashlib.sha256(raw).hexdigest() != meta["sha256"]
        or meta["bytes"] != len(raw)
        or not request["url"].startswith("https://home.treasury.gov/resource-center/")
    ):
        raise ValueError("US_TREASURY_PRIMARY_CSV_CHANGED")
    primary = {}
    for row in csv.DictReader(io.StringIO(raw.decode("utf-8-sig"))):
        day = datetime.strptime(row["Date"], "%m/%d/%Y").date().isoformat()
        if not "2021-06-01" <= day <= "2021-06-30" or day in primary:
            raise ValueError("US_TREASURY_PRIMARY_DATES_INVALID")
        primary[day] = dict(
            zip(FIELDS[1:], (float(row[k]) for k in ("3 Mo", "2 Yr", "5 Yr", "10 Yr", "30 Yr")), strict=True)
        )
    if len(primary) != 22 or set(primary) - set(points) != {"2021-06-28"}:
        raise ValueError("US_TREASURY_PRIMARY_PATCH_SCOPE_CHANGED")
    if any(primary[d] != points[d] for d in set(primary) & set(points)):
        raise ValueError("US_TREASURY_PRIMARY_PROVIDER_DISAGREEMENT")
    points["2021-06-28"] = primary["2021-06-28"]
    return dict(sorted(points.items()))


def history():
    """正式训练和运行只接受已通过日期、时点、原始数据复算校验的快照。"""
    folder = source_folder()
    snapshot, qualified = base.read(folder / "history.json"), base.read(folder / "qualification-result.json")
    if (
        qualified["history_hash"] != base.digest(snapshot)
        or qualified["calendar_hash"] != base.digest(base.read(folder / "calendar.json"))
        or qualified["status"] != "QUALIFIED_FOR_HISTORICAL_DEVELOPMENT_WITH_TIMING_ASSUMPTION"
    ):
        raise ValueError("US_TREASURY_QUALIFICATION_CHANGED")
    points = raw_history()
    if snapshot["rows"] != points:
        raise ValueError("US_TREASURY_HISTORY_CHANGED")
    return points


def load(target):
    """回读原始JSON和实际请求时点，拒绝仅重写外层摘要的解析值或已过截止的响应。"""
    folder = root() / target
    value = base.read(folder / "input.json")
    deadline = datetime.combine(date.fromisoformat(target), time(8, 30), base.ZONE)
    if value["target"] != target or value["source"] != SOURCE or datetime.fromisoformat(value["at"]) >= deadline:
        raise ValueError("US_TREASURY_LIVE_INPUT_SCOPE_OR_TIME_INVALID")
    needed = required(value["base"], target)
    slot = value["raw_ref"]["slot"]
    if slot not in SLOTS:
        raise ValueError("US_TREASURY_LIVE_SLOT_INVALID")
    meta = base.read(folder / f"raw/{slot}.json")
    request = base.read(folder / f"requests/{slot}.json")
    request_at = datetime.fromisoformat(request["at"])
    if (
        value["raw_ref"]["hash"] != base.digest(meta)
        or meta["request_hash"] != base.digest(request)
        or request["target"] != target
        or request["base"] != value["base"]
        or request["url"] != URL
        or request["api_name"] != "us_tycr"
        or request["fields"] != FIELDS
        or request["required_dates"] != needed
        or meta["source"] != SOURCE
        or not datetime.combine(date.fromisoformat(target), time(7), base.ZONE) <= request_at < deadline
        or slot_for(request_at) != slot
        or not request_at <= datetime.fromisoformat(meta["received_at"]) < deadline
        or datetime.fromisoformat(meta["received_at"]) > datetime.fromisoformat(value["at"])
    ):
        raise ValueError("US_TREASURY_LIVE_RESPONSE_CHANGED_OR_LATE")
    raw = (folder / f"raw/{slot}.response.json").read_bytes()
    if hashlib.sha256(raw).hexdigest() != meta["body_sha256"] or len(raw) != meta["bytes"]:
        raise ValueError("US_TREASURY_LIVE_RAW_CHANGED")
    points = parse(raw, *needed)
    if any(day not in points for day in needed) or {d: points[d] for d in needed} != value["rows"]:
        raise ValueError("US_TREASURY_LIVE_PARSED_INPUT_CHANGED")
    return value


def capture(at):
    """每目标最多三个固定时槽，每槽一次请求；已成功则全部基金共享，不自动重试。"""
    window = base.window(at)
    end = datetime.fromisoformat(base.read(base.ROOT / "protocol.json")["deadline_at"])
    if at >= end or window["status"] != "OPEN" or not time(7) <= at.time() < time(8, 30):
        raise ValueError("US_TREASURY_LIVE_WINDOW_CLOSED")
    target, base_day = window["target_nav_date"], window["base_nav_date"]
    folder = root() / target
    if (folder / "input.json").exists():
        return load(target)
    deadline = datetime.combine(date.fromisoformat(target), time(8, 30), base.ZONE)
    request_at = base.now()
    if not datetime.combine(date.fromisoformat(target), time(7), base.ZONE) <= request_at < deadline:
        raise ValueError("US_TREASURY_LIVE_REQUEST_TIME_INVALID")
    slot = slot_for(request_at)
    attempt = folder / f"requests/{slot}.json"
    if attempt.exists() or len(list((folder / "requests").glob("*.json"))) >= 3:
        raise ValueError("US_TREASURY_LIVE_SLOT_OR_BUDGET_EXHAUSTED")
    needed = required(base_day, target)
    request = {
        "at": request_at.isoformat(),
        "url": URL,
        "api_name": "us_tycr",
        "fields": FIELDS,
        "target": target,
        "base": base_day,
        "required_dates": needed,
    }
    base.save(attempt, request)
    try:
        body = fetch_us_tycr(date.fromisoformat(needed[0]), date.fromisoformat(needed[1]))
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
            raise ValueError("US_TREASURY_LIVE_RESPONSE_LATE")
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
            raise ValueError("US_TREASURY_LIVE_DEADLINE_REACHED")
        base.save(folder / "input.json", value)
        checked = load(target)
        if base.now() >= deadline:
            raise ValueError("US_TREASURY_LIVE_READBACK_LATE")
        return checked
    except Exception as exc:
        base.save(folder / f"errors-{slot}.json", {"at": base.now().isoformat(), "error": base.error_code(exc)})
        raise
