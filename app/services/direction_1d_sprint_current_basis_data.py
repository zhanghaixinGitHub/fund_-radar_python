"""第95轮取T日价差；历史开发与17日实际提前采集分开，不能相互冒充。"""

import hashlib
import time
from datetime import date, datetime, timedelta
from time import monotonic

import httpx
import numpy as np

from app.core.config import get_settings
from app.integrations import tushare_sprint_futures as parser
from app.services import direction_1d_sprint as base
from app.services import direction_1d_sprint_futures_basis_data as old
from app.services import direction_1d_sprint_market_only_data as market_data

root, history, scope, original_row, require = old.root, old.history, old.scope, old.original_row, old.require
TARGET = "2026-09-17"


def live_root():
    return base.ROOT / "round-95" / "live-basis"


def extend(market, t, u, points):
    """只取预测日前一交易日T，不取U，也不以更早价差代替缺失T。"""
    days = list(map(str, base.calendar()[0]))
    require(t in days and u in days and days.index(u) == days.index(t) + 1, "ADJACENCY_INVALID")
    row = points.get(t)
    feature = {"date": t, "available": row is not None}
    if row is not None:
        require(t < u and np.isfinite(row["basis_pct"]), "FEATURE_TIME_OR_VALUE_INVALID")
        feature.update(row | {"basis_pct": float(np.clip(row["basis_pct"], -20, 20))})
    return market | {"futures_basis": feature}


def queries(t):
    d = t.replace("-", "")
    return [
        {
            "api": api,
            "params": {"ts_code": "000300.SH" if api == "index_daily" else "IF.CFX", "start_date": d, "end_date": d},
        }
        for api in parser.FIELDS
    ]


def in_window(t, u, at):
    """本轮只有17日一次三请求预算，0800开始，0830关闭；不扩展到其他日期。"""
    return (
        t == "2026-09-16"
        and u == TARGET
        and at.date().isoformat() == u
        and at.hour == 8
        and at < market_data.deadline(u)
    )


def point(values, t, contracts):
    require(set(values) == set(parser.FIELDS), "LIVE_APIS_INCOMPLETE")
    require(all(set(rows) == {t} for rows in values.values()), "LIVE_DATE_CHANGED")
    quote, spot = values["fut_daily"][t], values["index_daily"][t]
    code = values["fut_mapping"][t]["mapping_ts_code"]
    require(code in contracts, "LIVE_CONTRACT_UNKNOWN")
    c = contracts[code]
    require(
        c["list_date"] <= t.replace("-", "") <= c["delist_date"]
        and c["exchange"] == "CFFEX"
        and c["quote_unit"] == "指数点"
        and c["multiplier"] == 300,
        "LIVE_CONTRACT_INVALID",
    )
    return {
        "main_contract": code,
        "futures_close": quote["close"],
        "spot_close": spot["close"],
        "basis_pct": float(100 * (quote["close"] / spot["close"] - 1)),
        "calendar_days_to_expiry": (datetime.strptime(c["delist_date"], "%Y%m%d").date() - date.fromisoformat(t)).days,
    }


def contracts():
    folder = base.ROOT / "stock-index-futures-feasibility-v1"
    q, basic = base.read(folder / "qualification-result.json"), base.read(folder / "results/basic.json")
    require(base.digest(basic) == q["source_result_hashes"]["basic"], "LIVE_CONTRACT_METADATA_CHANGED")
    return {r["ts_code"]: r for r in basic["rows"]}


def reconstruct_live(t, u, ph, saved_at):
    """逐个重验预约、原响应、实际收到时间；失败不能从旧历史拼出T日价格。"""
    values, receipts, reservations = {}, {}, {}
    folder = live_root() / u
    require({p.stem for p in (folder / "requests").glob("*.json")} <= set(parser.FIELDS), "LIVE_REQUEST_BUDGET_CHANGED")
    for q in queries(t):
        api = q["api"]
        req_path, response_path = folder / "requests" / f"{api}.json", folder / "responses" / f"{api}.json"
        if not req_path.exists():
            continue
        request = base.read(req_path)
        require(
            request["query"] == q and request["plan_hash"] == ph and request["max_retries"] == 0, "LIVE_REQUEST_CHANGED"
        )
        requested = datetime.fromisoformat(request["at"])
        require(in_window(t, u, requested) and requested <= saved_at, "LIVE_REQUEST_TIME_INVALID")
        reservations[api] = base.digest(request)
        if not response_path.exists():
            continue
        meta = base.read(response_path)
        raw = (folder / "raw" / f"{api}.json").read_bytes()
        received = datetime.fromisoformat(meta["received_at"])
        require(
            meta["request_hash"] == base.digest(request)
            and meta["sha256"] == hashlib.sha256(raw).hexdigest()
            and meta["bytes"] == len(raw),
            "LIVE_RAW_CHANGED",
        )
        require(requested <= received <= saved_at and received < market_data.deadline(u), "LIVE_RESPONSE_LATE")
        receipts[api] = base.digest(meta)
        if meta["http_status"] == 200:
            try:
                values[api] = parser.parse(raw, api, q["params"]["start_date"], q["params"]["end_date"])
            except (ValueError, TypeError, KeyError):
                # 原始拒绝/缺行保留在响应中；只表示本次新特征不可用，不能填零。
                continue
    row = None
    if len(values) == 3:
        try:
            row = point(values, t, contracts())
        except (ValueError, TypeError, KeyError):
            row = None
    return {"points": {t: row} if row is not None else {}, "receipts": receipts, "reservations": reservations}


def load_live(t, u, ph):
    value = base.read(live_root() / u / "snapshot.json")
    at = datetime.fromisoformat(value["at"])
    require(
        value["t"] == t and value["u"] == u and value["plan_hash"] == ph and in_window(t, u, at),
        "LIVE_SNAPSHOT_SCOPE_OR_TIME_CHANGED",
    )
    require(value["snapshot"] == reconstruct_live(t, u, ph, at), "LIVE_SNAPSHOT_CHANGED")
    require(value["available"] == bool(value["snapshot"]["points"]), "LIVE_AVAILABILITY_CHANGED")
    return value


def capture(t, u, ph):
    """只在实际时间窗口预约一次；已预约请求不可重试，失败留痕并交给原模型回退。"""
    target = live_root() / u / "snapshot.json"
    if target.exists():
        return load_live(t, u, ph)
    if not in_window(t, u, base.now()):
        return None
    folder = target.parent
    errors = []
    reserved = list((folder / "requests").glob("*.json"))
    # 中断后若尚未完成快照，只核对已保存响应；不再次访问供应商。
    if reserved:
        errors.append("RESERVED_BATCH_NOT_RETRIED")
    else:
        cfg = get_settings()
        require(cfg.tushare_api_url == "https://api.tushare.pro", "LIVE_ENDPOINT_CHANGED")
        token = cfg.tushare_token.get_secret_value()
        require(bool(token), "LIVE_TOKEN_MISSING")
        for q in queries(t):
            if base.now() + timedelta(seconds=35) >= market_data.deadline(u):
                errors.append("INSUFFICIENT_TIME_BEFORE_DEADLINE")
                break
            if list((folder / "responses").glob("*.json")):
                last = max(
                    datetime.fromisoformat(base.read(p)["received_at"]) for p in (folder / "responses").glob("*.json")
                )
                pause = 7.1 - (base.now() - last).total_seconds()
                if pause > 0:
                    time.sleep(pause)
            api = q["api"]
            request = {"at": base.now().isoformat(), "query": q, "plan_hash": ph, "max_retries": 0}
            base.save(folder / "requests" / f"{api}.json", request)
            try:
                began = monotonic()
                with httpx.Client(timeout=httpx.Timeout(20, connect=5), follow_redirects=False) as client:
                    with client.stream(
                        "POST",
                        cfg.tushare_api_url,
                        json={
                            "api_name": api,
                            "token": token,
                            "params": q["params"],
                            "fields": ",".join(parser.FIELDS[api]),
                        },
                    ) as response:
                        raw = bytearray()
                        for chunk in response.iter_bytes():
                            raw.extend(chunk)
                            require(
                                len(raw) <= 1048576 and monotonic() - began <= 30, "LIVE_RESPONSE_TOO_LARGE_OR_SLOW"
                            )
                        status = response.status_code
                require(token.encode() not in raw, "LIVE_RESPONSE_CONTAINS_SECRET")
                path = folder / "raw" / f"{api}.json"
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("xb") as f:
                    f.write(raw)
                base.save(
                    folder / "responses" / f"{api}.json",
                    {
                        "received_at": base.now().isoformat(),
                        "request_hash": base.digest(request),
                        "http_status": status,
                        "bytes": len(raw),
                        "sha256": hashlib.sha256(raw).hexdigest(),
                    },
                )
                require(status == 200, "LIVE_HTTP_FAILED")
                parser.parse(raw, api, q["params"]["start_date"], q["params"]["end_date"])
            except Exception as exc:
                error = {
                    "at": base.now().isoformat(),
                    "error": base.error_code(exc),
                    "exception_type": type(exc).__name__,
                }
                base.save(folder / "errors" / f"{api}.json", error)
                errors.append(error["error"])
                break
    at = base.now()
    if not in_window(t, u, at):
        return None
    snapshot = reconstruct_live(t, u, ph, at)
    value = {
        "at": at.isoformat(),
        "t": t,
        "u": u,
        "plan_hash": ph,
        "snapshot": snapshot,
        "available": bool(snapshot["points"]),
        "errors": errors,
        "max_requests": 3,
        "new_cost_cny": 0,
    }
    base.save(target, value)
    return load_live(t, u, ph)
