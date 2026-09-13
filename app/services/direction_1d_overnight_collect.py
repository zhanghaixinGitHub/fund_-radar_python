"""真实隔夜数据留档：以本机实际接收/落盘时间证明观测，不改历史可用性假设或业务预测。"""

import hashlib
import json
import os
import traceback
from datetime import date, datetime, time, timedelta
from math import isfinite
from pathlib import Path
from time import monotonic
from uuid import uuid4
from zoneinfo import ZoneInfo

from sqlalchemy import text

from app.db.session import get_nav_preview_engine
from app.integrations.tushare_overnight import FIELDS, fetch_spx
from app.services.direction_1d_protocol import ZONE, calendar, canonical, digest
from app.services.direction_1d_training import read, write_new
from app.services.direction_training_artifacts import file_hash

CALENDAR = Path(__file__).resolve().parents[1] / "data/calendars/nyse_cash_2026_v1.json"
SLOTS = ("0730", "0750", "0758", "0805")
PROTOCOL = "SPX_OBSERVED_BEFORE_U08_V1"


def now() -> datetime:
    return datetime.now(ZONE)


def market_sessions() -> dict[str, datetime]:
    """独立2026日历，避免修改绑定旧历史实验的2021—2024日历及指纹。"""
    spec = read(CALENDAR)
    if spec["calendar_id"] != "NYSE_CASH_2026_V1":
        raise ValueError("OVERNIGHT_CALENDAR_INVALID")
    holidays, early = set(spec["holidays"]), set(spec["early_closes"])
    result, day = {}, date(2026, 1, 1)
    while day.year == 2026:
        key = day.isoformat()
        if day.weekday() < 5 and key not in holidays:
            result[key] = datetime.combine(
                day, time(13 if key in early else 16), ZoneInfo("America/New_York")
            ).astimezone(ZONE)
        day += timedelta(days=1)
    return result


def source_status() -> dict:
    """只读原来源启用状态；SPX研究能力由本轮独立契约限定，不更改业务13项登记。"""
    with get_nav_preview_engine().connect() as connection, connection.begin():
        connection.execute(text("SET TRANSACTION READ ONLY"))
        connection.execute(text("SET LOCAL statement_timeout='5s'"))
        row = (
            connection.execute(
                text("""SELECT source_id,source_code,enabled,authorization_verified_at,
            retention_days,rate_limit_per_minute FROM source_registry WHERE source_code='TUSHARE_PRO_FUND'""")
            )
            .mappings()
            .one()
        )
    return json.loads(canonical(dict(row)))


def validate_source(source: dict) -> None:
    if (
        source.get("source_code") != "TUSHARE_PRO_FUND"
        or source.get("enabled") is not True
        or not source.get("authorization_verified_at")
        or source.get("retention_days", 0) <= 0
        or source.get("rate_limit_per_minute", 0) <= 0
    ):
        raise ValueError("OVERNIGHT_SOURCE_UNAVAILABLE")


def code_files() -> dict[str, str]:
    project = Path(__file__).resolve().parents[2]
    names = (
        "app/services/direction_1d_overnight_collect.py",
        "app/integrations/tushare_overnight.py",
        "scripts/direction_1d_overnight_collect.py",
        "app/data/calendars/nyse_cash_2026_v1.json",
        "app/data/calendars/cn_a_share_2026_v1.json",
    )
    return {n: file_hash(project / n) for n in names}


def initialize(root: Path, probe: Path, *, clock=now, source_reader=source_status) -> dict:
    """建立不能覆盖的个人研究契约；安装日期之后才开始计数，不回填此前缺失时点。"""
    at, source, evidence = clock(), source_reader(), read(probe)
    validate_source(source)
    if (
        evidence.get("status") != "PERMISSION_AVAILABLE_NOT_USAGE_OR_TIMING_PROOF"
        or evidence.get("http_status") != 200
        or evidence.get("provider_code") != 0
        or evidence.get("row_count") != 1
        or evidence.get("reservation", {}).get("api_name") != "index_global"
        or evidence["reservation"].get("params", {}).get("ts_code") != "SPX"
    ):
        raise ValueError("OVERNIGHT_CAPABILITY_PROOF_INVALID")
    if at.tzinfo is None or at.year != 2026:
        raise ValueError("OVERNIGHT_CALENDAR_UNAVAILABLE")
    root.mkdir(parents=True, exist_ok=False)
    contract = {
        "protocol": PROTOCOL,
        "installed_at": at.isoformat(),
        "source_id": source["source_id"],
        "purpose": "USER_AUTHORIZED_PERSONAL_LOCAL_SPX_RESEARCH",
        "api_name": "index_global",
        "ts_code": "SPX",
        "slots": list(SLOTS),
        "deadline": "08:00:00+08:00",
        "last_calendar_date": "2026-12-31",
        "retention_days": min(365, source["retention_days"]),
        "max_requests_per_trading_day": 4,
        "code_files": code_files(),
        "capability_proof_hash": digest(evidence),
        "database_writes": 0,
        "terms": "https://tushare.pro/document/1?doc_id=405",
        "provider_first_publication_claimed": False,
    }
    write_new(root / "capability-proof.json", evidence)
    write_new(root / "contract.json", contract)
    write_new(root / "contract-receipt.json", {"hash": digest(contract), "at": at.isoformat()})
    return contract


def load_contract(root: Path) -> dict:
    contract = read(root / "contract.json")
    if (
        digest(contract) != read(root / "contract-receipt.json")["hash"]
        or contract.get("protocol") != PROTOCOL
        or contract.get("slots") != list(SLOTS)
        or contract.get("code_files") != code_files()
        or digest(read(root / "capability-proof.json")) != contract.get("capability_proof_hash")
    ):
        raise ValueError("OVERNIGHT_CONTRACT_CHANGED")
    return contract


def day_plan(day: date) -> dict:
    """按中国交易日构造所需美股日期；完整覆盖基准收盘和长假内全部美股收盘。"""
    if day.year != 2026:
        raise ValueError("OVERNIGHT_CALENDAR_UNAVAILABLE")
    days = calendar()[0]
    if day not in days:
        return {"target_date": str(day), "status": "NON_TRADING_DAY"}
    t = days[days.index(day) - 1]
    market = market_sessions()
    start, deadline = datetime.combine(t, time(15), ZONE), datetime.combine(day, time(8), ZONE)
    baseline = [d for d, close in market.items() if close <= start]
    if not baseline:
        raise ValueError("OVERNIGHT_BASELINE_OUTSIDE_CALENDAR")
    selected = [d for d, close in market.items() if start < close <= deadline]
    required = [baseline[-1], *selected]
    if (date.fromisoformat(required[-1]) - date.fromisoformat(required[0])).days > 31:
        raise ValueError("OVERNIGHT_QUERY_RANGE_INVALID")
    return {
        "target_date": str(day),
        "base_date": str(t),
        "status": "TRADING_DAY",
        "deadline": deadline.isoformat(),
        "required_us_dates": required,
        "new_us_dates": selected,
        "close_events": {d: market[d].isoformat() for d in required},
    }


def validate_response(body: bytes, plan: dict) -> dict:
    """验证原文内容与应有日期；多日数据保留全部版本，缺日不能当成休市或沿用昨日。"""
    payload = json.loads(body)
    if not isinstance(payload, dict) or type(payload.get("code")) is not int or payload["code"] != 0:
        raise ValueError("OVERNIGHT_PROVIDER_BUSINESS_FAILED")
    data = payload.get("data") or {}
    if data.get("fields") != FIELDS or not isinstance(data.get("items"), list) or len(data["items"]) > 32:
        raise ValueError("OVERNIGHT_FIELDS_OR_ROWS_INVALID")
    rows = {}
    for values in data["items"]:
        if not isinstance(values, list) or len(values) != len(FIELDS):
            raise ValueError("OVERNIGHT_ROW_INVALID")
        code, raw_day, close, previous, pct = values
        if not isinstance(raw_day, str) or len(raw_day) != 8 or not raw_day.isdigit():
            raise ValueError("OVERNIGHT_DATE_INVALID")
        day = datetime.strptime(raw_day, "%Y%m%d").date().isoformat()
        if code != "SPX" or day not in plan["required_us_dates"] or day in rows:
            raise ValueError("OVERNIGHT_UNEXPECTED_OR_DUPLICATE_DATE")
        if (
            any(type(v) not in (int, float) or not isfinite(v) for v in (close, previous, pct))
            or min(close, previous) <= 0
        ):
            raise ValueError("OVERNIGHT_PRICE_INVALID")
        if abs((close / previous - 1) * 100 - pct) > 0.0001:
            raise ValueError("OVERNIGHT_RETURN_INCONSISTENT")
        rows[day] = {"close": close, "pre_close": previous, "pct_chg": pct}
    missing = sorted(set(plan["required_us_dates"]) - rows.keys())
    if missing:
        return {"status": "INCOMPLETE", "missing_dates": missing, "diagnostic_return": None, "rows": rows}
    dates = plan["required_us_dates"]
    if any(abs(rows[b]["pre_close"] - rows[a]["close"]) > 0.0001 for a, b in zip(dates, dates[1:], strict=False)):
        raise ValueError("OVERNIGHT_PREVIOUS_CLOSE_CHANGED")
    return {
        "status": "COMPLETE",
        "missing_dates": [],
        "rows": rows,
        "diagnostic_return": rows[dates[-1]]["close"] / rows[dates[0]]["close"] - 1,
    }


def write_body(path: Path, body: bytes) -> None:
    """独占保存原始响应并刷入磁盘；完成时间在调用返回后读取，写失败不能算已取得。"""
    with path.open("xb") as stream:
        stream.write(body)
        stream.flush()
        os.fsync(stream.fileno())


def collect_slot(
    root: Path, plan: dict, slot: str, contract: dict, *, query=fetch_spx, clock=now, source_reader=source_status
) -> dict:
    """同日同槽先独占预算；响应收到与成功落盘均早于截止才可用，失败保留且不隐式重试。"""
    at = clock()
    if slot not in SLOTS or plan != day_plan(at.date()) or contract != load_contract(root):
        raise ValueError("OVERNIGHT_CAPTURE_SCOPE_INVALID")
    directory = root / "days" / plan["target_date"]
    directory.mkdir(parents=True, exist_ok=True)
    reservation = directory / f"{slot}-reserved.json"
    if reservation.exists():
        return {"status": "ALREADY_ATTEMPTED", "slot": slot, "api_calls": 0}
    try:
        write_new(reservation, {"started_at": at.isoformat(), "slot": slot, "plan": plan, "max_requests": 1})
    except FileExistsError:
        return {"status": "ALREADY_ATTEMPTED", "slot": slot, "api_calls": 0}
    receipt = {
        "slot": slot,
        "target_date": plan["target_date"],
        "started_at": at.isoformat(),
        "api_calls": 0,
        "usable_before_u08": False,
        "provider_first_publication_verified": False,
        "contract_hash": digest(contract),
        "plan": plan,
    }
    try:
        source = source_reader()
        validate_source(source)
        if source["source_id"] != contract["source_id"]:
            raise ValueError("OVERNIGHT_SOURCE_ID_CHANGED")
        started, monotonic_start = clock(), monotonic()
        receipt.update(api_calls=1, request_started_at=started.isoformat())
        body = query(
            date.fromisoformat(plan["required_us_dates"][0]), date.fromisoformat(plan["required_us_dates"][-1])
        )
        received, elapsed = clock(), monotonic() - monotonic_start
        if received < started or abs((received - started).total_seconds() - elapsed) > 5:
            raise ValueError("OVERNIGHT_CLOCK_DISCONTINUITY")
        receipt.update(response_received_at=received.isoformat(), response_sha256=hashlib.sha256(body).hexdigest())
        raw_path = directory / f"{slot}-response.json"
        write_body(raw_path, body)
        validation = validate_response(body, plan)
        persisted = clock()
        if persisted < received or persisted.date().isoformat() != plan["target_date"]:
            raise ValueError("OVERNIGHT_CLOCK_DISCONTINUITY")
        receipt.update(
            raw_file=raw_path.name,
            persisted_at=persisted.isoformat(),
            validation=validation,
            retention_until=(
                persisted + timedelta(days=min(contract["retention_days"], source["retention_days"]))
            ).isoformat(),
        )
        deadline = datetime.fromisoformat(plan["deadline"])
        timely = started <= received <= persisted < deadline
        if any(datetime.fromisoformat(v) > started for v in plan["close_events"].values()):
            raise ValueError("OVERNIGHT_UNFINISHED_MARKET_SESSION")
        receipt["usable_before_u08"] = timely and validation["status"] == "COMPLETE"
        receipt["status"] = (
            "OBSERVED_BEFORE_DEADLINE" if receipt["usable_before_u08"] else ("LATE" if not timely else "INCOMPLETE")
        )
    except Exception as error:
        # 不记录可能带URL参数/凭据的外部异常文本；保留稳定业务码和类型用于本地排障。
        receipt.update(
            status="FAILED",
            error_type=type(error).__name__,
            error_code=str(error)
            if isinstance(error, ValueError) and str(error).startswith("OVERNIGHT_")
            else "OVERNIGHT_CAPTURE_FAILED",
            error_stack=[
                {"file": Path(frame.filename).name, "function": frame.name, "line": frame.lineno}
                for frame in traceback.extract_tb(error.__traceback__)[-8:]
            ],
        )
    write_new(directory / f"{slot}-receipt.json", receipt)
    write_new(directory / f"{slot}-seal.json", {"receipt_hash": digest(receipt)})
    return {k: receipt[k] for k in ("status", "slot", "api_calls", "usable_before_u08")}


def day_status(root: Path, day: date, at: datetime) -> dict:
    """从完整原文与回执重算当天状态，尚未到截止与错过采集分开；不信任缓存ready标记。"""
    contract, plan = load_contract(root), day_plan(day)
    if plan["status"] == "NON_TRADING_DAY":
        return {"date": str(day), "status": "NON_TRADING_DAY", "usable_before_u08": False}
    directory = root / "days" / str(day)
    records, usable, payloads = [], [], []
    for slot in SLOTS:
        path = directory / f"{slot}-receipt.json"
        if not path.exists():
            continue
        item = read(path)
        if digest(item) != read(directory / f"{slot}-seal.json")["receipt_hash"]:
            raise ValueError("OVERNIGHT_RECEIPT_HASH_CHANGED")
        reservation = read(directory / f"{slot}-reserved.json")
        if any(reservation.get(k) != item.get(k) for k in ("slot", "plan", "started_at")):
            raise ValueError("OVERNIGHT_RESERVATION_CHANGED")
        if item.get("contract_hash") != digest(contract) or item.get("plan") != plan or item.get("slot") != slot:
            raise ValueError("OVERNIGHT_RECEIPT_CHANGED")
        if "raw_file" in item:
            if item["raw_file"] != f"{slot}-response.json":
                raise ValueError("OVERNIGHT_RAW_PATH_INVALID")
            body = (directory / item["raw_file"]).read_bytes()
            if hashlib.sha256(body).hexdigest() != item["response_sha256"]:
                raise ValueError("OVERNIGHT_RAW_HASH_CHANGED")
            validation = validate_response(body, plan)
            if validation != item["validation"]:
                raise ValueError("OVERNIGHT_VALIDATION_CHANGED")
            started, received, saved = (
                datetime.fromisoformat(item[k]) for k in ("request_started_at", "response_received_at", "persisted_at")
            )
            good = (
                validation["status"] == "COMPLETE"
                and started <= received <= saved < datetime.fromisoformat(plan["deadline"])
                and saved <= at
                and saved.date() == day
                and all(datetime.fromisoformat(v) <= started for v in plan["close_events"].values())
            )
            if good != item["usable_before_u08"]:
                raise ValueError("OVERNIGHT_TIMING_FLAG_CHANGED")
            if good:
                usable.append((saved, slot, validation["diagnostic_return"]))
            payloads.append(digest(validation["rows"]))
        records.append({k: item[k] for k in ("slot", "status", "api_calls", "usable_before_u08")})
    chosen = max(usable, default=None)
    deadline_passed = at >= datetime.fromisoformat(plan["deadline"])
    return {
        "date": str(day),
        "status": "READY_OBSERVED" if chosen else ("MISSED_DEADLINE" if deadline_passed else "WAITING"),
        "usable_before_u08": bool(chosen),
        "selected_slot": chosen[1] if chosen else None,
        "observed_at": chosen[0].isoformat() if chosen else None,
        "diagnostic_return": chosen[2] if chosen else None,
        "observed_value_versions": len(set(payloads)),
        "attempts": records,
        "provider_first_publication_verified": False,
    }


def write_health(root: Path, value: dict) -> None:
    """状态文件只是可替换索引；原始响应、预算和回执一直保留，不被新状态覆盖。"""
    temporary = root / ("health-" + uuid4().hex + ".tmp")
    temporary.write_text(canonical(value), encoding="utf-8")
    os.replace(temporary, root / "health.json")


def tick(root: Path, *, clock=now, query=fetch_spx, source_reader=source_status) -> dict:
    """定时任务/登录调用同一入口；只执行当前五分钟内到期的一个槽，不回放错过的槽。"""
    contract, at = load_contract(root), clock()
    if at.tzinfo is None or at < datetime.fromisoformat(contract["installed_at"]):
        raise ValueError("OVERNIGHT_CLOCK_BEFORE_INSTALL")
    at = at.astimezone(ZONE)
    plan = day_plan(at.date())
    attempted = None
    if plan["status"] == "TRADING_DAY":
        due = [(slot, datetime.combine(at.date(), time(int(slot[:2]), int(slot[2:])), ZONE)) for slot in SLOTS]
        eligible = [(slot, stamp) for slot, stamp in due if timedelta(0) <= at - stamp < timedelta(minutes=5)]
        if eligible:
            slot, _ = eligible[-1]
            if slot == "0805" or at.time() < time(8):
                attempted = collect_slot(
                    root, plan, slot, contract, query=query, clock=clock, source_reader=source_reader
                )
    finished = clock()
    result = {
        "at": finished.isoformat(),
        "operation": "tick",
        "attempt": attempted,
        "day": day_status(root, at.date(), finished),
        "database_writes": 0,
    }
    write_health(root, result)
    return result


def probe(root: Path, *, clock=now, query=fetch_spx, source_reader=source_status) -> dict:
    """安装验收专用的一次当日诊断；只查最近已收盘SPX一天，永不计入08:00可用记录。"""
    contract, at = load_contract(root), clock()
    source = source_reader()
    validate_source(source)
    if source["source_id"] != contract["source_id"] or at.year != 2026:
        raise ValueError("OVERNIGHT_PROBE_SCOPE_INVALID")
    latest = max(d for d, close in market_sessions().items() if close <= at)
    out = root / "diagnostics" / str(at.date())
    out.mkdir(parents=True, exist_ok=False)
    write_new(out / "reserved.json", {"at": at.isoformat(), "max_requests": 1, "us_date": latest})
    body = query(date.fromisoformat(latest), date.fromisoformat(latest))
    received = clock()
    write_body(out / "response.json", body)
    validation = validate_response(body, {"required_us_dates": [latest]})
    receipt = {
        "at": received.isoformat(),
        "us_date": latest,
        "validation_status": validation["status"],
        "raw_sha256": hashlib.sha256(body).hexdigest(),
        "api_calls": 1,
        "status": "DIAGNOSTIC_ONLY",
        "usable_before_u08": False,
        "training_data": False,
    }
    write_new(out / "receipt.json", receipt)
    return receipt
