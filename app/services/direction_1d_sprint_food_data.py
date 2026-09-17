"""食品饮料基准行情的历史核验和真实提前采集，所有基金共用原始响应。

每目标日三时槽，每槽每指数至多一次，总计三次，不自动重试。
特征只命中基准日及之前五个国内交易日；缺失拒绝，不以前值或未来值填补。
"""

import hashlib
import json
import math
import time as sleep_time
from datetime import date, datetime, time
from functools import lru_cache

import numpy as np

from app.integrations.tushare_sprint_food import CODES, FIELDS, MAX_BYTES, URL, fetch_food
from app.services import direction_1d_sprint as base
from app.services import direction_1d_sprint_market as market

SOURCE = "TUSHARE_FOOD_BENCHMARK_EXISTING_PERMISSION_RESEARCH"
SLOTS = ("0700", "0730", "0800")


def root():
    return base.ROOT / "round-46" / "benchmark-inputs"


def slot_for(at):
    return "0800" if at.time() >= time(8) else "0730" if at.time() >= time(7, 30) else "0700"


@lru_cache(maxsize=64)
def json_rows(raw, code):
    """缓存不可变解析行，验证固定标识、日期格式及正数收盘，避免可变缓存污染。"""
    if code not in CODES or not isinstance(raw, bytes) or not raw or len(raw) > MAX_BYTES:
        raise ValueError("FOOD_RAW_SIZE_OR_CODE_INVALID")
    obj = json.loads(raw)
    if not isinstance(obj, dict) or type(obj.get("code")) is not int or obj["code"] != 0:
        raise ValueError("FOOD_PROVIDER_REJECTED")
    data = obj.get("data") or {}
    if data.get("fields") != FIELDS or not isinstance(data.get("items"), list) or not 1 <= len(data["items"]) <= 366:
        raise ValueError("FOOD_SCHEMA_OR_ROWS_INVALID")
    rows = {}
    for row in data["items"]:
        if not isinstance(row, list) or len(row) != 4 or row[0] != code:
            raise ValueError("FOOD_ROW_WIDTH_OR_CODE_INVALID")
        text = row[1]
        if not isinstance(text, str) or len(text) != 8 or not text.isdigit():
            raise ValueError("FOOD_DATE_FORMAT_INVALID")
        day = datetime.strptime(text, "%Y%m%d").date().isoformat()
        if day in rows or any(type(v) not in (float, int) or not math.isfinite(v) or v <= 0 for v in row[2:]):
            raise ValueError("FOOD_DUPLICATE_OR_PRICE_INVALID")
        rows[day] = tuple(map(float, row[2:]))
    return tuple((day, *values) for day, values in sorted(rows.items()))


def parse(raw, code, start="2021-01-01", end="2026-12-31"):
    rows = json_rows(raw, code)
    if any(not start <= day <= end for day, _, _ in rows):
        raise ValueError("FOOD_DATE_OUTSIDE_REQUEST")
    return {day: {"close": close, "pre_close": prior} for day, close, prior in rows}


def required(base_day, target):
    days = list(map(str, base.calendar()[0]))
    if base_day not in days:
        raise ValueError("FOOD_BASE_OUTSIDE_CALENDAR")
    index = days.index(base_day)
    if index < 5 or index + 1 >= len(days) or days[index + 1] != target:
        raise ValueError("FOOD_TARGET_NOT_ADJACENT")
    return days[index - 5 : index + 1]


def features(x, fund_code, base_day, target, points):
    """001632追加自身指数的1/5日超额表现及基金偏离，百分数；其它基金明确走原HK分支。

    95%来自历史基准公式，只用于定义偏离，不声称每日实际仓位恰为95%。
    未计入5%存款利息和分红调整，因此该差不是会计意义完整跟踪误差。
    """
    days = required(base_day, target)
    if len(x) != 32 or not np.isfinite(x).all():
        raise ValueError("FOOD_PARENT_FEATURE_INVALID")
    if fund_code != "001632":
        return [0.0] * 5
    code = CODES[0]
    if code not in points or any(day not in points[code] for day in days):
        raise ValueError("FOOD_REQUIRED_DATE_MISSING")
    prices = [points[code][day] for day in days]
    if any(
        type(p[k]) not in (int, float) or not math.isfinite(p[k]) or p[k] <= 0
        for p in prices
        for k in ("close", "pre_close")
    ):
        raise ValueError("FOOD_FEATURE_PRICE_INVALID")
    if any(abs(a["pre_close"] - p["close"]) > 0.001 for p, a in zip(prices[:-1], prices[1:], strict=True)):
        raise ValueError("FOOD_PRECLOSE_DISCONTINUITY")
    r1 = prices[-1]["close"] / prices[-2]["close"] - 1
    r5 = prices[-1]["close"] / prices[0]["close"] - 1
    return [float(np.clip(v * 100, -20, 20)) for v in (r1 - x[23], r5 - x[24], x[7] - 0.95 * r1, x[0] - 0.95 * r5)] + [
        1.0
    ]


def history():
    """核对8个已保存的API请求、精确名称/后缀和原始响应，不再次查询历史。"""
    folder = base.ROOT / "food-benchmark-data-v1"
    plan, identity = base.read(folder / "plan.json"), base.read(folder / "identity.json")
    snapshot, result = base.read(folder / "history.json"), base.read(folder / "result.json")
    if (
        hashlib.sha256((folder / "acquire.py").read_bytes()).hexdigest() != plan["script_sha256"]
        or identity["plan_hash"] != base.digest(plan)
        or identity["row"]["ts_code"] != CODES[0]
        or identity["row"]["fullname"] != "中证食品饮料指数"
        or identity["row"]["market"] != "CSI"
        or snapshot["plan_hash"] != base.digest(plan)
        or snapshot["code"] != CODES[0]
        or result["plan_hash"] != base.digest(plan)
        or result["history_hash"] != base.digest(snapshot)
        or result["status"] != "ACQUIRED_NOT_TRAINED"
    ):
        raise ValueError("FOOD_HISTORICAL_MANIFEST_CHANGED")
    combined = {}
    pieces = [("probe", "2025-01-02", "2025-01-10")]
    pieces += [(str(y), f"{y}-01-01", f"{y}-12-31" if y < 2026 else "2026-09-14") for y in range(2021, 2027)]
    for label, start, end in pieces:
        request = base.read(folder / f"attempt-{label}.json")
        receipt = base.read(folder / f"receipt-{label}.json")
        raw = (folder / f"response-{label}.json").read_bytes()
        if (
            request["plan_hash"] != base.digest(plan)
            or request["api"] != "index_daily"
            or request["url"] != URL
            or request["fields"] != FIELDS
            or request["params"]
            != {"ts_code": CODES[0], "start_date": start.replace("-", ""), "end_date": end.replace("-", "")}
            or receipt["request_hash"] != base.digest(request)
            or receipt["status"] != "OK"
            or receipt["bytes"] != len(raw)
            or receipt["sha256"] != hashlib.sha256(raw).hexdigest()
        ):
            raise ValueError("FOOD_HISTORICAL_RAW_CHANGED")
        parsed = parse(raw, CODES[0], start, end)
        if label == "probe":
            probe = parsed
        else:
            if set(combined) & set(parsed):
                raise ValueError("FOOD_HISTORICAL_DUPLICATE")
            combined.update(parsed)
    if combined != snapshot["points"] or any(combined.get(k) != v for k, v in probe.items()):
        raise ValueError("FOOD_HISTORICAL_PARSED_CHANGED")
    return {CODES[0]: combined}


def load_piece(target, base_day, code, slot):
    """核对单只指数的原始字节及请求/接收顺序；已成功的同槽请求可复用。"""
    if slot not in SLOTS or code not in CODES:
        raise ValueError("FOOD_LIVE_SLOT_OR_CODE_INVALID")
    folder = root() / target
    meta = base.read(folder / f"raw/{slot}-{code}.json")
    request = base.read(folder / f"requests/{slot}-{code}.json")
    needed = required(base_day, target)
    start = datetime.combine(date.fromisoformat(target), time(7), base.ZONE)
    deadline = datetime.combine(date.fromisoformat(target), time(8, 30), base.ZONE)
    request_at, received_at = datetime.fromisoformat(request["at"]), datetime.fromisoformat(meta["received_at"])
    if (
        meta["source"] != SOURCE
        or meta["request_hash"] != base.digest(request)
        or request["url"] != URL
        or request["api_name"] != "index_daily"
        or request["fields"] != FIELDS
        or request["target"] != target
        or request["base"] != base_day
        or request["code"] != code
        or request["required_dates"] != needed
        or not start <= request_at <= received_at < deadline
        or slot_for(request_at) != slot
    ):
        raise ValueError("FOOD_LIVE_RESPONSE_CHANGED_OR_LATE")
    raw = (folder / f"raw/{slot}-{code}.response.json").read_bytes()
    if meta["bytes"] != len(raw) or meta["body_sha256"] != hashlib.sha256(raw).hexdigest():
        raise ValueError("FOOD_LIVE_RAW_CHANGED")
    points = parse(raw, code, needed[0], needed[-1])
    if any(day not in points for day in needed):
        raise ValueError("FOOD_REQUIRED_DATE_MISSING")
    return meta, {day: points[day] for day in needed}


def load(target):
    value = base.read(root() / target / "input.json")
    deadline = datetime.combine(date.fromisoformat(target), time(8, 30), base.ZONE)
    at = datetime.fromisoformat(value["at"])
    if value["target"] != target or value["source"] != SOURCE or at >= deadline or set(value["rows"]) != set(CODES):
        raise ValueError("FOOD_LIVE_INPUT_SCOPE_OR_TIME_INVALID")
    for code in CODES:
        ref = value["raw_refs"][code]
        meta, points = load_piece(target, value["base"], code, ref["slot"])
        if ref["hash"] != base.digest(meta) or datetime.fromisoformat(meta["received_at"]) > at:
            raise ValueError("FOOD_LIVE_INPUT_RECEIPT_CHANGED")
        if points != value["rows"][code]:
            raise ValueError("FOOD_LIVE_PARSED_INPUT_CHANGED")
    features([0.0] * 32, "001632", value["base"], target, value["rows"])
    return value


def capture(at):
    """每指数每槽至多一次，单只失败不会阻止其他指数采集，整体成功后才可预测。"""
    window = base.window(at)
    end = datetime.fromisoformat(base.read(base.ROOT / "protocol.json")["deadline_at"])
    if at >= end or window["status"] != "OPEN" or not time(7) <= at.time() < time(8, 30):
        raise ValueError("FOOD_LIVE_WINDOW_CLOSED")
    target, base_day = window["target_nav_date"], window["base_nav_date"]
    folder = root() / target
    if (folder / "input.json").exists():
        return load(target)
    deadline = datetime.combine(date.fromisoformat(target), time(8, 30), base.ZONE)
    needed = required(base_day, target)
    source = market.active()
    slot = slot_for(base.now())
    combined, refs, errors = {}, {}, {}
    for code in CODES:
        meta_path, attempt = folder / f"raw/{slot}-{code}.json", folder / f"requests/{slot}-{code}.json"
        try:
            request_at = base.now()
            if (
                not datetime.combine(date.fromisoformat(target), time(7), base.ZONE) <= request_at < deadline
                or slot_for(request_at) != slot
            ):
                raise ValueError("FOOD_LIVE_REQUEST_TIME_INVALID")
            if not meta_path.exists():
                if attempt.exists() or len(list((folder / "requests").glob("*.json"))) >= 3:
                    raise ValueError("FOOD_LIVE_SLOT_OR_BUDGET_EXHAUSTED")
                request = {
                    "at": request_at.isoformat(),
                    "url": URL,
                    "api_name": "index_daily",
                    "fields": FIELDS,
                    "target": target,
                    "base": base_day,
                    "code": code,
                    "required_dates": needed,
                }
                base.save(attempt, request)
                try:
                    raw = fetch_food(code, date.fromisoformat(needed[0]), date.fromisoformat(needed[-1]))
                    received = base.now().isoformat()
                    raw_path = folder / f"raw/{slot}-{code}.response.json"
                    raw_path.parent.mkdir(parents=True, exist_ok=True)
                    with raw_path.open("xb") as file:
                        file.write(raw)
                    base.save(
                        meta_path,
                        {
                            "received_at": received,
                            "source": SOURCE,
                            "request_hash": base.digest(request),
                            "body_sha256": hashlib.sha256(raw).hexdigest(),
                            "bytes": len(raw),
                        },
                    )
                finally:
                    sleep_time.sleep(max(1, 60 / min(source["rate_limit_per_minute"], 60)))
            meta, combined[code] = load_piece(target, base_day, code, slot)
            refs[code] = {"slot": slot, "hash": base.digest(meta)}
        except Exception as exc:
            errors[code] = base.error_code(exc)
    if errors:
        base.save(folder / f"errors-{slot}.json", {"at": base.now().isoformat(), "errors": errors}, replace=True)
        raise ValueError("FOOD_LIVE_INPUT_INCOMPLETE")
    features([0.0] * 32, "001632", base_day, target, combined)
    if base.now() >= deadline:
        raise ValueError("FOOD_LIVE_DEADLINE_REACHED")
    value = {
        "at": base.now().isoformat(),
        "target": target,
        "base": base_day,
        "source": SOURCE,
        "rows": combined,
        "raw_refs": refs,
    }
    base.save(folder / "input.json", value)
    checked = load(target)
    if base.now() >= deadline:
        raise ValueError("FOOD_LIVE_READBACK_LATE")
    return checked
