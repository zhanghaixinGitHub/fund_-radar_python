"""港股参考行情的独立只读采集，不改变既有模型和美股交易日历。

先用四次样本确认已有权限，再按固定十二次年度请求保存历史。
原始响应、接收时间、请求预算和解析结果同时留档，不自动重试失败请求。
"""

import hashlib
import json
import time
from datetime import date, datetime

import numpy as np

from app.integrations.tushare_sprint_hk import FIELDS, fetch_hk
from app.services import direction_1d_sprint as base
from app.services import direction_1d_sprint_overnight as overnight
from app.services import direction_1d_sprint_return_target as regression

INDICES = ("HSI", "HKTECH")


def root(kind):
    if kind not in ("probe", "history"):
        raise ValueError("HK_DATA_STAGE_INVALID")
    return base.ROOT / f"hk-{kind}-v1"


def fingerprint():
    return {
        n: hashlib.sha256((base.PROJECT / n).read_bytes()).hexdigest()
        for n in (
            "app/integrations/tushare_sprint_hk.py",
            "app/services/direction_1d_sprint_hk_data.py",
            "tests/test_direction_1d_sprint_hk_data.py",
        )
    }


def plan(kind):
    path = root(kind) / "plan.json"
    if path.exists():
        value = base.read(path)
        if value["code"] != fingerprint():
            raise ValueError("HK_DATA_CODE_CHANGED")
        return value
    regression.active()
    source = overnight.source()
    # 业务同步白名单不包含既有独立研究的index_global，不能据此宣称港股权限已验证。
    # 用户已授权零新增费用的数据研究；先做四次受限能力探针，成功后才允许补全历史。
    # 不改业务来源白名单，也不把已有SPX能力凭据当成HSI/HKTECH已成功的证明。
    if kind == "history":
        probe = base.read(root("probe") / "result.json")
        if probe["status"] != "AVAILABLE" or probe["query_count"] != 4:
            raise ValueError("HK_PROBES_NOT_READY")
    specs = []
    for code in INDICES:
        periods = (
            [("2025-01-02", "2025-01-10"), ("2026-09-14", "2026-09-14")]
            if kind == "probe"
            else [(f"{y}-01-01", min(f"{y}-12-31", "2026-09-14")) for y in range(2021, 2027)]
        )
        for start, end in periods:
            specs.append({"key": f"{code}-{start}", "code": code, "start": start, "end": end})
    value = {
        "at": base.now().isoformat(),
        "code": fingerprint(),
        "kind": kind,
        "source_id": str(source["source_id"]),
        "queries": specs,
        "max_requests": len(specs),
        "max_rows": 270,
        "max_bytes": 262144,
        "minimum_spacing_seconds": 7,
        "document": "https://tushare.pro/document/2?doc_id=211",
        "new_cost_cny": 0,
        "training_authorized_by_this_plan": False,
        "authorization_basis": "User-authorized personal zero-cost data research; independent HK capability probes",
        "application_api_registry_snapshot": source["authorized_api_names"],
        "calendar_status": "Observed HK rows, not independently verified HK calendar; never use US calendar",
        "historical_availability": "Current vendor snapshot, not original publication receipts",
    }
    base.save(path, value)
    return value


def validate(code, body, start, end):
    """检查标识、请求范围、正价格、重复及前收盘链；不把港股缺行当美股休市。

    这里只证明供应商记录内部一致，不能声称已核对完整港交所日历。
    所有后续特征必须另外保证行情日期不晚于基金基准日，并保留陈旧天数。
    """
    payload = json.loads(body)
    if code not in INDICES or type(payload.get("code")) is not int or payload["code"] != 0:
        raise ValueError("HK_PROVIDER_FAILED")
    data = payload.get("data") or {}
    if data.get("fields") != FIELDS or not isinstance(data.get("items"), list) or not 1 <= len(data["items"]) <= 270:
        raise ValueError("HK_FIELDS_OR_ROWS_INVALID")
    rows = {}
    for row in data["items"]:
        if not isinstance(row, list) or len(row) != 4:
            raise ValueError("HK_ROW_INVALID")
        symbol, raw_day, close, previous = row
        if not isinstance(raw_day, str) or len(raw_day) != 8 or not raw_day.isdigit():
            raise ValueError("HK_DATE_INVALID")
        day = datetime.strptime(raw_day, "%Y%m%d").date()
        if symbol != code or not str(start) <= str(day) <= str(end) or day.weekday() > 4 or str(day) in rows:
            raise ValueError("HK_UNEXPECTED_OR_DUPLICATE_DATE")
        if (
            any(type(v) not in (float, int) for v in (close, previous))
            or not np.isfinite([close, previous]).all()
            or min(close, previous) <= 0
        ):
            raise ValueError("HK_PRICE_INVALID")
        rows[str(day)] = {"close": close, "pre_close": previous}
    rows = dict(sorted(rows.items()))
    validate_chain(rows)
    return rows


def validate_chain(rows):
    values = list(dict(sorted(rows.items())).values())
    for a, b in zip(values, values[1:], strict=False):
        if not overnight.same_previous_close(a["close"], b["pre_close"]):
            raise ValueError("HK_PREVIOUS_CLOSE_CHANGED")


def query(kind, spec):
    p = plan(kind)
    regression.active()
    if spec not in p["queries"]:
        raise ValueError("HK_QUERY_NOT_RESERVED")
    source = overnight.source()
    if str(source["source_id"]) != p["source_id"]:
        raise ValueError("HK_SOURCE_CHANGED")
    key = spec["key"]
    output = root(kind) / f"data/{key}.json"
    if output.exists():
        return base.read(output)
    raw_path, attempt = root(kind) / f"raw/{key}.json", root(kind) / f"requests/{key}.json"
    if not raw_path.exists():
        if attempt.exists():
            raise ValueError("HK_FAILED_ATTEMPT_NO_AUTO_RETRY")
        if len(list((root(kind) / "requests").glob("*.json"))) >= p["max_requests"]:
            raise ValueError("HK_REQUEST_BUDGET_EXHAUSTED")
        base.save(attempt, {"at": base.now().isoformat(), "query": spec})
        try:
            body = fetch_hk(spec["code"], date.fromisoformat(spec["start"]), date.fromisoformat(spec["end"]))
            base.save(
                raw_path,
                {
                    "received_at": base.now().isoformat(),
                    "response": json.loads(body),
                    "body_sha256": hashlib.sha256(body).hexdigest(),
                },
            )
        finally:
            time.sleep(max(p["minimum_spacing_seconds"], 60 / source["rate_limit_per_minute"]))
    raw = base.read(raw_path)
    rows = validate(spec["code"], json.dumps(raw["response"]).encode(), spec["start"], spec["end"])
    value = {"code": spec["code"], "rows": rows, "received_at": raw["received_at"], "raw_hash": base.digest(raw)}
    base.save(output, value)
    return value


def capture(kind):
    p = plan(kind)
    path = root(kind) / "result.json"
    if path.exists():
        return base.read(path)
    combined = {code: {} for code in INDICES}
    query_hashes = {}
    for spec in p["queries"]:
        value = query(kind, spec)
        combined[value["code"]].update(value["rows"])
        query_hashes[spec["key"]] = base.digest(value)
    if kind == "history":
        for rows in combined.values():
            validate_chain(rows)
        probe = base.read(root("probe") / "result.json")
        for code in INDICES:
            for day, row in probe["rows"][code].items():
                if combined[code].get(day) != row:
                    raise ValueError("HK_PROBE_CHANGED")
    value = {
        "at": base.now().isoformat(),
        "status": "AVAILABLE",
        "kind": kind,
        "rows": combined,
        "query_count": len(query_hashes),
        "query_hashes": query_hashes,
        "plan_hash": base.digest(p),
        "new_cost_cny": 0,
        "counts": {c: len(r) for c, r in combined.items()},
        "calendar_status": p["calendar_status"],
        "historical_availability": p["historical_availability"],
    }
    base.save(path, value)
    return value
