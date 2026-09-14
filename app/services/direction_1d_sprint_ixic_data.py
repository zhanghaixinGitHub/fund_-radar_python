"""第六轮前置数据准备：先两个小样本，再六次按年读取，最多八次真实请求。

请求发出前留占位，成功原文在数值校验前保存，失败不自动重复占用供应商额度。
只采集到固定2026-09-11历史终点；现有五轮预测不依赖这个目录。
"""

import hashlib
import json
import time
from datetime import date, datetime

import numpy as np

from app.integrations.tushare_sprint_ixic import FIELDS, fetch_ixic
from app.services import direction_1d_sprint as base
from app.services import direction_1d_sprint_overnight as overnight


def root():
    return base.ROOT / "ixic-data-v1"


def fingerprint():
    result = overnight.fingerprint()
    for name in (
        "app/integrations/tushare_sprint_ixic.py",
        "app/services/direction_1d_sprint_ixic_data.py",
        "scripts/direction_1d_sprint_ixic_data.py",
    ):
        result[name] = hashlib.sha256((base.PROJECT / name).read_bytes()).hexdigest()
    return result


def plan():
    path = root() / "plan.json"
    if path.exists():
        value = base.read(path)
        if value["code"] != fingerprint():
            raise ValueError("IXIC_DATA_CODE_CHANGED")
        return value
    value = {
        "at": base.now().isoformat(),
        "code": fingerprint(),
        "source_id": str(overnight.source()["source_id"]),
        "purpose": "USER_AUTHORIZED_PERSONAL_LOCAL_ONE_DAY_RESEARCH",
        "index": "IXIC",
        "api": "index_global",
        "start": "2021-01-01",
        "end": "2026-09-11",
        "maximum_requests": 8,
        "minimum_spacing_seconds": 7,
        "probes": [["2026-09-11", "2026-09-11"], ["2025-01-02", "2025-01-10"]],
        "fields": FIELDS,
        "maximum_rows_per_request": 270,
        "maximum_bytes": 262144,
        "documents": [
            "https://tushare.pro/document/2?doc_id=211",
            "https://www.nasdaqtrader.com/Trader.aspx?id=Calendar",
            "https://ir.nasdaq.com/news-releases/news-release-details/nasdaq-announces-closure-its-us-markets-honor-national-day-0",
        ],
        "calendar_policy": "require same cash-session dates as existing US calendar; any difference blocks use",
        "historical_first_availability_verified": False,
        "new_cost_cny": 0,
        "training_permitted_by_this_plan": False,
    }
    base.save(path, value)
    return value


def validate(body, start, end):
    value = json.loads(body)
    if type(value.get("code")) is not int or value["code"] != 0:
        raise ValueError("IXIC_PROVIDER_BUSINESS_FAILED")
    data = value.get("data") or {}
    if data.get("fields") != FIELDS or not isinstance(data.get("items"), list) or len(data["items"]) > 270:
        raise ValueError("IXIC_FIELDS_OR_ROWS_INVALID")
    required = [d for d in overnight.sessions() if str(start) <= d <= str(end)]
    rows = {}
    for values in data["items"]:
        if not isinstance(values, list) or len(values) != 4:
            raise ValueError("IXIC_ROW_INVALID")
        code, raw_day, close, previous = values
        if not isinstance(raw_day, str) or len(raw_day) != 8 or not raw_day.isdigit():
            raise ValueError("IXIC_DATE_INVALID")
        day = datetime.strptime(raw_day, "%Y%m%d").date().isoformat()
        if code != "IXIC" or day not in required or day in rows:
            raise ValueError("IXIC_UNEXPECTED_OR_DUPLICATE_DATE")
        if (
            any(type(v) not in (int, float) for v in (close, previous))
            or not np.isfinite([close, previous]).all()
            or min(close, previous) <= 0
        ):
            raise ValueError("IXIC_PRICE_INVALID")
        rows[day] = {"close": close, "pre_close": previous}
    if set(rows) != set(required):
        raise ValueError("IXIC_CALENDAR_COVERAGE_INCOMPLETE")
    for a, b in zip(required, required[1:], strict=False):
        if not overnight.same_previous_close(rows[a]["close"], rows[b]["pre_close"]):
            raise ValueError("IXIC_PREVIOUS_CLOSE_CHANGED")
    return rows


def query(key, start, end):
    p = plan()
    source = overnight.source()
    if str(source["source_id"]) != p["source_id"]:
        raise ValueError("IXIC_SOURCE_CHANGED")
    output = root() / "data" / f"{key}.json"
    if output.exists():
        return base.read(output)
    raw_path = root() / "raw" / f"{key}.json"
    reservation = root() / "requests" / f"{key}.json"
    if not raw_path.exists():
        if reservation.exists():
            raise ValueError("IXIC_PREVIOUS_REQUEST_NO_SUCCESS_NO_AUTO_RETRY")
        if len(list((root() / "requests").glob("*.json"))) >= p["maximum_requests"]:
            raise ValueError("IXIC_REQUEST_BUDGET_EXHAUSTED")
        base.save(
            reservation,
            {"at": base.now().isoformat(), "start": str(start), "end": str(end), "index": "IXIC", "maximum_calls": 1},
        )
        body = fetch_ixic(start, end)
        raw = {
            "received_at": base.now().isoformat(),
            "body_sha256": hashlib.sha256(body).hexdigest(),
            "response": json.loads(body),
        }
        base.save(raw_path, raw)
        time.sleep(max(p["minimum_spacing_seconds"], 60 / source["rate_limit_per_minute"]))
    raw = base.read(raw_path)
    rows = validate(json.dumps(raw["response"]).encode(), start, end)
    value = {
        "received_at": raw["received_at"],
        "start": str(start),
        "end": str(end),
        "raw_hash": base.digest(raw),
        "rows": rows,
    }
    base.save(output, value)
    return value


def acquire():
    p = plan()
    if (root() / "history.json").exists():
        return {"status": "ALREADY_CAPTURED"}
    for i, (start, end) in enumerate(p["probes"]):
        query(f"probe-{i}", date.fromisoformat(start), date.fromisoformat(end))
    rows = {}
    for year in range(2021, 2027):
        value = query(str(year), date(year, 1, 1), min(date(year, 12, 31), date.fromisoformat(p["end"])))
        rows.update(value["rows"])
    dates = sorted(rows)
    if set(rows) != {d for d in overnight.sessions() if p["start"] <= d <= p["end"]}:
        raise ValueError("IXIC_FULL_HISTORY_INCOMPLETE")
    if any(
        not overnight.same_previous_close(rows[a]["close"], rows[b]["pre_close"])
        for a, b in zip(dates, dates[1:], strict=False)
    ):
        raise ValueError("IXIC_FULL_HISTORY_DISCONTINUITY")
    value = {
        "at": base.now().isoformat(),
        "plan_hash": base.digest(p),
        "code": fingerprint(),
        "rows": rows,
        "requests": len(list((root() / "requests").glob("*.json"))),
        "new_cost_cny": 0,
        "historical_first_availability_verified": False,
        "model_trained": False,
    }
    base.save(root() / "history.json", value)
    return {"rows": len(rows), "requests": value["requests"], "new_cost_cny": 0}
