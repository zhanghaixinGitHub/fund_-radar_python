"""市场风格指数补数：四次小样本已确认权限，本阶段固定十二次按年请求。

缺失、接口失败和价格链不连续均留下实际证据，不自动重复调用或伪造缺失值。
本模块仅采集已有权限下的研究数据，不写业务数据源登记或模型发布状态。
"""

import hashlib
import json
import time
from datetime import date, datetime

import numpy as np

from app.integrations.tushare_sprint_style import FIELDS, fetch_style
from app.services import direction_1d_sprint as base
from app.services import direction_1d_sprint_overnight as overnight
from app.services import direction_1d_sprint_return_target as regression

INDICES = ("RUT", "DJI")


def root():
    return base.ROOT / "market-style-data-v1"


def fingerprint():
    return {
        n: hashlib.sha256((base.PROJECT / n).read_bytes()).hexdigest()
        for n in (
            "app/integrations/tushare_sprint_style.py",
            "app/services/direction_1d_sprint_style_data.py",
            "scripts/direction_1d_sprint_style_data.py",
            "tests/test_direction_1d_sprint_style_data.py",
        )
    }


def plan():
    path = root() / "plan.json"
    if path.exists():
        p = base.read(path)
        if p["code"] != fingerprint():
            raise ValueError("STYLE_DATA_CODE_CHANGED")
        return p
    regression.active()
    source = overnight.source()
    probe = base.read(base.ROOT / "market-style-probe-v1/result.json")
    if len(probe["queries"]) != 4 or any(q["status"] != "AVAILABLE" for q in probe["queries"]):
        raise ValueError("STYLE_PROBES_NOT_READY")
    p = {
        "at": base.now().isoformat(),
        "code": fingerprint(),
        "source_id": str(source["source_id"]),
        "indices": INDICES,
        "start": "2021-01-01",
        "end": "2026-09-11",
        "max_requests": 12,
        "max_rows_per_request": 270,
        "max_bytes": 262144,
        "minimum_spacing_seconds": 7,
        "probe_hash": base.digest(probe),
        "document": "https://tushare.pro/document/2?doc_id=211",
        "calendar": "require exact equality to existing US cash sessions, including temporary closures",
        "historical_first_availability_verified": False,
        "new_cost_cny": 0,
        "training_permitted_by_this_plan": False,
    }
    base.save(path, p)
    return p


def validate(code, body, start, end):
    """逐日验证指数标识、日期、价格及前收盘链；不按不完整样本缩小预定考试范围。"""
    value = json.loads(body)
    if code not in INDICES or type(value.get("code")) is not int or value["code"] != 0:
        raise ValueError("STYLE_PROVIDER_FAILED")
    data = value.get("data") or {}
    if data.get("fields") != FIELDS or not isinstance(data.get("items"), list) or len(data["items"]) > 270:
        raise ValueError("STYLE_FIELDS_OR_ROWS_INVALID")
    required = [d for d in overnight.sessions() if str(start) <= d <= str(end)]
    points = {}
    for row in data["items"]:
        if not isinstance(row, list) or len(row) != 4:
            raise ValueError("STYLE_ROW_INVALID")
        index, raw_date, close, previous = row
        if not isinstance(raw_date, str) or len(raw_date) != 8 or not raw_date.isdigit():
            raise ValueError("STYLE_DATE_INVALID")
        day = datetime.strptime(raw_date, "%Y%m%d").date().isoformat()
        if index != code or day not in required or day in points:
            raise ValueError("STYLE_UNEXPECTED_OR_DUPLICATE_DATE")
        if (
            any(type(v) not in (int, float) for v in (close, previous))
            or not np.isfinite([close, previous]).all()
            or min(close, previous) <= 0
        ):
            raise ValueError("STYLE_PRICE_INVALID")
        points[day] = {"close": close, "pre_close": previous}
    if set(points) != set(required):
        raise ValueError("STYLE_SESSION_COVERAGE_INCOMPLETE")
    for a, b in zip(required, required[1:], strict=False):
        if not overnight.same_previous_close(points[a]["close"], points[b]["pre_close"]):
            raise ValueError("STYLE_PREVIOUS_CLOSE_CHANGED")
    return points


def query(code, year):
    p = plan()
    regression.active()
    if str(overnight.source()["source_id"]) != p["source_id"]:
        raise ValueError("STYLE_SOURCE_CHANGED")
    start, end = date(year, 1, 1), min(date(year, 12, 31), date.fromisoformat(p["end"]))
    key = f"{code}-{year}"
    output = root() / f"data/{key}.json"
    if output.exists():
        return base.read(output)
    raw_path, reservation = root() / f"raw/{key}.json", root() / f"requests/{key}.json"
    if not raw_path.exists():
        if reservation.exists():
            raise ValueError("STYLE_PREVIOUS_ATTEMPT_NO_AUTO_RETRY")
        if len(list((root() / "requests").glob("*.json"))) >= p["max_requests"]:
            raise ValueError("STYLE_REQUEST_BUDGET_EXHAUSTED")
        base.save(reservation, {"at": base.now().isoformat(), "code": code, "start": str(start), "end": str(end)})
        try:
            body = fetch_style(code, start, end)
            base.save(
                raw_path,
                {
                    "received_at": base.now().isoformat(),
                    "response": json.loads(body),
                    "body_sha256": hashlib.sha256(body).hexdigest(),
                },
            )
        finally:
            # 失败同样消耗一次请求，不马上重试；沿用当前来源更严格的速率限制。
            time.sleep(max(p["minimum_spacing_seconds"], 60 / overnight.source()["rate_limit_per_minute"]))
    raw = base.read(raw_path)
    rows = validate(code, json.dumps(raw["response"]).encode(), start, end)
    result = {"received_at": raw["received_at"], "raw_hash": base.digest(raw), "code": code, "rows": rows}
    base.save(output, result)
    return result


def acquire():
    p = plan()
    if (root() / "history.json").exists():
        return {"status": "ALREADY_CAPTURED"}
    points, errors = {code: {} for code in INDICES}, []
    for code in INDICES:
        for year in range(2021, 2027):
            try:
                points[code].update(query(code, year)["rows"])
                print(json.dumps({"index": code, "year": year, "status": "CAPTURED"}), flush=True)
            except Exception as exc:
                error = {"index": code, "year": year, "error": base.error_code(exc), "at": base.now().isoformat()}
                base.save(root() / f"errors/{code}-{year}.json", error, replace=True)
                errors.append(error)
    if errors:
        raise ValueError("STYLE_HISTORY_INCOMPLETE_SEE_SAVED_ERRORS")
    required = [d for d in overnight.sessions() if p["start"] <= d <= p["end"]]
    for code in INDICES:
        if set(points[code]) != set(required):
            raise ValueError("STYLE_FULL_HISTORY_SESSION_MISMATCH")
        for a, b in zip(required, required[1:], strict=False):
            if not overnight.same_previous_close(points[code][a]["close"], points[code][b]["pre_close"]):
                raise ValueError("STYLE_FULL_HISTORY_DISCONTINUITY")
        for path in (base.ROOT / "market-style-probe-v1/raw").glob(f"{code}-*.json"):
            for _, day, close, previous in base.read(path)["response"]["data"]["items"]:
                day = datetime.strptime(day, "%Y%m%d").date().isoformat()
                if points[code][day] != {"close": close, "pre_close": previous}:
                    raise ValueError("STYLE_PROBE_OVERLAP_REVISED")
    result = {
        "at": base.now().isoformat(),
        "plan_hash": base.digest(p),
        "code": fingerprint(),
        "rows": points,
        "requests": len(list((root() / "requests").glob("*.json"))),
        "new_cost_cny": 0,
        "historical_first_availability_verified": False,
    }
    base.save(root() / "history.json", result)
    return {"rows_per_index": {c: len(v) for c, v in points.items()}, "requests": result["requests"], "new_cost_cny": 0}
