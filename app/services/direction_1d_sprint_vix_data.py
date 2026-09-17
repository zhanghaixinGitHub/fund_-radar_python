"""滞后VIX历史与实际到达证据；只使用基准日15:00前已结束的现金交易日。"""

import csv
import hashlib
import io
import math
from bisect import bisect_right
from datetime import date, datetime, time
from functools import lru_cache

import numpy as np

from app.integrations.cboe_sprint_vix import MAX_BYTES, URL, fetch_vix
from app.services import direction_1d_sprint as base
from app.services import direction_1d_sprint_overnight as overnight

SOURCE = "CBOE_VIX_PUBLIC_CSV_PERSONAL_RESEARCH"
SLOTS = ("0700", "0730", "0800")


def root():
    return base.ROOT / "round-22" / "vix-inputs"


@lru_cache(maxsize=4)
def csv_rows(raw):
    """缓存原始字节对应的不可变行，避免30只基金重复解析同一个文件；不缓存可变字典。"""
    if not isinstance(raw, bytes) or not raw or len(raw) > MAX_BYTES:
        raise ValueError("VIX_RAW_SIZE_INVALID")
    reader = csv.DictReader(io.StringIO(raw.decode("utf-8-sig")))
    if reader.fieldnames != ["DATE", "OPEN", "HIGH", "LOW", "CLOSE"]:
        raise ValueError("VIX_HEADER_INVALID")
    result, prior = [], ""
    for item in reader:
        day = datetime.strptime(item["DATE"], "%m/%d/%Y").date().isoformat()
        if day <= prior or len(result) >= 12000:
            raise ValueError("VIX_DATE_ORDER_OR_ROW_LIMIT")
        prior = day
        values = tuple(float(item[key]) for key in ("OPEN", "HIGH", "LOW", "CLOSE"))
        # 研究只需2021年以来记录；更早文件可能采用旧发布口径，仍保留原始字节。
        if day >= "2021-01-01":
            if any(not math.isfinite(v) or v <= 0 for v in values):
                raise ValueError("VIX_PRICE_INVALID")
            opening, high, low, close = values
            if not low <= min(opening, close) <= max(opening, close) <= high:
                raise ValueError("VIX_OHLC_ORDER_INVALID")
        result.append((day, *values))
    if not result:
        raise ValueError("VIX_EMPTY_RESPONSE")
    return tuple(result)


def parse(raw):
    """从源文件投影出已验证现金交易日；休市行和未来超出日历的行不作为新交易日。"""
    calendar = overnight.sessions()
    return {
        day: {"open": opening, "high": high, "low": low, "close": close}
        for day, opening, high, low, close in csv_rows(raw)
        if day >= "2021-01-01" and day in calendar
    }


def required(base_day, target):
    """相邻内地交易日校验后，选T日15:00前最近的美股现金收盘及前一收盘。"""
    days = list(map(str, base.calendar()[0]))
    if base_day not in days or days.index(base_day) + 1 >= len(days) or days[days.index(base_day) + 1] != target:
        raise ValueError("VIX_TARGET_NOT_ADJACENT")
    events = overnight.sessions()
    keys, closes = list(events), list(events.values())
    index = bisect_right(closes, datetime.combine(date.fromisoformat(base_day), time(15), base.ZONE)) - 1
    if index < 1:
        raise ValueError("VIX_ANCHOR_CALENDAR_UNCOVERED")
    return [keys[index - 1], keys[index]]


def features(base_day, target, points):
    prior, anchor = required(base_day, target)
    if any(day not in points for day in (prior, anchor)):
        raise ValueError("VIX_REQUIRED_CASH_CLOSE_MISSING")
    p, a = float(points[prior]["close"]), float(points[anchor]["close"])
    if not np.isfinite([p, a]).all() or min(p, a) <= 0:
        raise ValueError("VIX_FEATURE_PRICE_INVALID")
    return [a / 20, float(np.clip((a / p - 1) * 100, -100, 100))]


def history():
    """复用已采集文件，验证原始字节、响应清单与解析快照；不重复下载历史数据。"""
    folder = base.ROOT / "vix-data-feasibility-v1"
    record, response = base.read(folder / "result.json"), base.read(folder / "response.json")
    raw = (folder / "response.csv").read_bytes()
    if record["response_hash"] != base.digest(response) or hashlib.sha256(raw).hexdigest() != response["sha256"]:
        raise ValueError("VIX_HISTORICAL_RAW_CHANGED")
    points = parse(raw)
    expected = {day: row for day, row in record["selected_rows"].items() if day in overnight.sessions()}
    if points != expected:
        raise ValueError("VIX_HISTORICAL_PARSED_CHANGED")
    return points


def load(target):
    """回读原始CSV和请求时点，拒绝只修改外层哈希的解析值或已过截止的响应。"""
    folder = root() / target
    value = base.read(folder / "input.json")
    deadline = datetime.combine(date.fromisoformat(target), time(8, 30), base.ZONE)
    if value["target"] != target or value["source"] != SOURCE or datetime.fromisoformat(value["at"]) >= deadline:
        raise ValueError("VIX_LIVE_INPUT_SCOPE_OR_TIME_INVALID")
    needed = required(value["base"], target)
    slot = value["raw_ref"]["slot"]
    if slot not in SLOTS:
        raise ValueError("VIX_LIVE_SLOT_INVALID")
    meta = base.read(folder / f"raw/{slot}.json")
    request = base.read(folder / f"requests/{slot}.json")
    if (
        value["raw_ref"]["hash"] != base.digest(meta)
        or meta["request_hash"] != base.digest(request)
        or request["target"] != target
        or request["base"] != value["base"]
        or request["url"] != URL
        or request["required_dates"] != needed
        or meta["source"] != SOURCE
        or not datetime.fromisoformat(request["at"]) <= datetime.fromisoformat(meta["received_at"]) < deadline
        or datetime.fromisoformat(meta["received_at"]) > datetime.fromisoformat(value["at"])
    ):
        raise ValueError("VIX_LIVE_RESPONSE_CHANGED_OR_LATE")
    raw = (folder / f"raw/{slot}.csv").read_bytes()
    if hashlib.sha256(raw).hexdigest() != meta["body_sha256"] or len(raw) != meta["bytes"]:
        raise ValueError("VIX_LIVE_RAW_CHANGED")
    points = parse(raw)
    if any(day not in points for day in needed) or {d: points[d] for d in needed} != value["rows"]:
        raise ValueError("VIX_LIVE_PARSED_INPUT_CHANGED")
    return value


def capture(at):
    """每目标最多三个固定时槽，每槽一次原始文件请求；已成功则全基金共享，无自动重试。"""
    window = base.window(at)
    end = datetime.fromisoformat(base.read(base.ROOT / "protocol.json")["deadline_at"])
    if at >= end or window["status"] != "OPEN" or not time(7) <= at.time() < time(8, 30):
        raise ValueError("VIX_LIVE_WINDOW_CLOSED")
    target, base_day = window["target_nav_date"], window["base_nav_date"]
    folder = root() / target
    if (folder / "input.json").exists():
        return load(target)
    deadline = datetime.combine(date.fromisoformat(target), time(8, 30), base.ZONE)
    if base.now() >= deadline:
        raise ValueError("VIX_LIVE_DEADLINE_REACHED")
    slot = "0800" if at.time() >= time(8) else "0730" if at.time() >= time(7, 30) else "0700"
    attempt = folder / f"requests/{slot}.json"
    if attempt.exists() or len(list((folder / "requests").glob("*.json"))) >= 3:
        raise ValueError("VIX_LIVE_SLOT_OR_BUDGET_EXHAUSTED")
    needed = required(base_day, target)
    request = {"at": base.now().isoformat(), "url": URL, "target": target, "base": base_day, "required_dates": needed}
    base.save(attempt, request)
    try:
        body, headers = fetch_vix()
        meta = {
            "received_at": base.now().isoformat(),
            "source": SOURCE,
            "body_sha256": hashlib.sha256(body).hexdigest(),
            "bytes": len(body),
            "headers": headers,
            "request_hash": base.digest(request),
        }
        raw_path = folder / f"raw/{slot}.csv"
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        with raw_path.open("xb") as file:
            file.write(body)
        base.save(folder / f"raw/{slot}.json", meta)
        if datetime.fromisoformat(meta["received_at"]) >= deadline:
            raise ValueError("VIX_LIVE_RESPONSE_LATE")
        points = parse(body)
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
            raise ValueError("VIX_LIVE_DEADLINE_REACHED")
        base.save(folder / "input.json", value)
        checked = load(target)
        if base.now() >= deadline:
            raise ValueError("VIX_LIVE_READBACK_LATE")
        return checked
    except Exception as exc:
        base.save(folder / f"errors-{slot}.json", {"at": base.now().isoformat(), "error": base.error_code(exc)})
        raise
