"""第102轮期限价差联合日内特征；历史和实际采集各自保留完整来源证据。"""

import hashlib
import json
from datetime import datetime, timedelta
from time import monotonic

import httpx
import numpy as np

from app.core.config import get_settings
from app.integrations import tushare_sprint_futures_curve_v2 as parser
from app.services import direction_1d_sprint as base
from app.services import direction_1d_sprint_futures_curve_history as curve
from app.services import direction_1d_sprint_futures_intraday_data as intra
from app.services import direction_1d_sprint_market_only_data as market_data

scope, require = intra.scope, curve.require


def root():
    return base.ROOT / "futures-curve-combination-feasibility-v1"


def combine(day, curves, intraday):
    c, f = curves.get(day), intraday.get(day)
    if c is None or f is None:
        return None
    require(c["date"] == day, "CURVE_DATE_CHANGED")
    return c | f


def reconstruct():
    c, f = curve.history(), intra.history()
    points = {d: combine(d, c["snapshot"]["rows"], f["snapshot"]["rows"]) for d in c["snapshot"]["rows"]}
    require(all(v is not None for v in points.values()), "JOINT_HISTORY_MISSING")
    return {"rows": points, "curve_history_hash": base.digest(c), "intraday_history_hash": base.digest(f)}


def history():
    p, result = base.read(root() / "plan.json"), base.read(root() / "result.json")
    require(curve.sha(root() / "check.py") == p["script_sha256"], "COMBINATION_SCRIPT_CHANGED")
    for name, digest in p["code"].items():
        require(curve.sha(base.PROJECT / name) == digest, "COMBINATION_CODE_CHANGED")
    snapshot = reconstruct()
    require(
        result["plan_hash"] == base.digest(p) and result["snapshot"] == snapshot and len(snapshot["rows"]) == 1383,
        "COMBINATION_HISTORY_CHANGED",
    )
    require(result["history_hash"] == base.digest(snapshot), "COMBINATION_HASH_CHANGED")
    return {
        "at": result["at"],
        "snapshot": {"rows": snapshot["rows"]},
        "source_history_hash": result["history_hash"],
        "feasibility_hash": base.digest(result),
    }


def extend(market, t, u, points):
    f = intra.extend({}, t, u, points)["futures_intraday"]
    if f["available"]:
        require(
            f["date"] == t
            and type(f["annualized_spread_pct"]) in (int, float)
            and np.isfinite(f["annualized_spread_pct"])
            and abs(f["annualized_spread_pct"]) <= 50,
            "CURVE_FEATURE_INVALID",
        )
    return market | {"futures_curve": f}


def original_row(row):
    return row | {"market": {k: v for k, v in row["market"].items() if k != "futures_curve"}}


def parse_actual(raw, t):
    """全CFFEX响应只选当日有效的最近两个具体IF合约，连续别名不能充数。"""
    require(raw and len(raw) <= 524288, "LIVE_BODY_INVALID")
    value = json.loads(raw)
    data = value.get("data") or {}
    require(
        type(value.get("code")) is int
        and value["code"] == 0
        and data.get("fields") == parser.FIELDS
        and isinstance(data.get("items"), list)
        and 1 <= len(data["items"]) <= 1000,
        "LIVE_SCHEMA_INVALID",
    )
    _, original = curve.samples()
    day = t.replace("-", "")
    meta = sorted(
        [v for v in original["basic"] if v["list_date"] <= day <= v["delist_date"]], key=lambda v: v["delist_date"]
    )
    require(len(meta) == 4, "LIVE_CONTRACT_REGISTRY_INVALID")
    selected = {}
    wanted = {v["ts_code"] for v in meta[:2]}
    for row in data["items"]:
        require(isinstance(row, list) and len(row) == len(parser.FIELDS), "LIVE_ROW_INVALID")
        if row[0] not in wanted:
            continue
        require(row[0] not in selected, "LIVE_CONTRACT_DUPLICATE")
        m = next(v for v in meta if v["ts_code"] == row[0])
        body = json.dumps({"code": 0, "data": {"fields": parser.FIELDS, "items": [row]}}).encode()
        expiry = str(datetime.strptime(m["delist_date"], "%Y%m%d").date())
        selected[row[0]] = parser.parse(body, row[0], [t], expiry)[t]
    require(set(selected) == wanted, "LIVE_CONTRACT_MISSING")
    a, b = meta[:2]
    return {t: parser.point(t, a, b, selected[a["ts_code"]], selected[b["ts_code"]])}


def live(t, u, ph, parent_plan_hash):
    c, f = load_live(t, u, ph), intra.live(t, u, parent_plan_hash)
    points = {}
    if c["available"] and f["available"]:
        points[t] = combine(t, c["snapshot"]["points"], f["snapshot"]["points"])
    return {
        "at": max(c["at"], f["at"]),
        "t": t,
        "u": u,
        "curve_source_hash": base.digest(c),
        "intraday_source_hash": base.digest(f),
        "snapshot": {"points": points},
        "available": bool(points),
        "new_source_requests": 0,
    }


def live_root():
    return base.ROOT / "round-102" / "live-curve"


def in_window(t, u, at):
    return (
        t == "2026-09-16"
        and u == "2026-09-17"
        and at.date().isoformat() == u
        and at.hour == 8
        and at < market_data.deadline(u)
    )


def query(t):
    return {
        "api": "fut_daily",
        "params": {"exchange": "CFFEX", "trade_date": t.replace("-", "")},
        "fields": parser.FIELDS,
    }


def reconstruct_live(t, u, ph, at):
    folder = live_root() / u
    req = base.read(folder / "request.json")
    require(req["query"] == query(t) and req["plan_hash"] == ph and req["max_retries"] == 0, "LIVE_REQUEST_CHANGED")
    requested = datetime.fromisoformat(req["at"])
    require(in_window(t, u, requested) and requested <= at, "LIVE_REQUEST_TIME_INVALID")
    snapshot = {"request_hash": base.digest(req), "response_hash": None, "points": {}}
    if not (folder / "response.json").exists():
        return snapshot
    meta, raw = base.read(folder / "response.json"), (folder / "raw.json").read_bytes()
    require(
        meta["request_hash"] == base.digest(req)
        and meta["bytes"] == len(raw)
        and meta["sha256"] == hashlib.sha256(raw).hexdigest(),
        "LIVE_RAW_CHANGED",
    )
    received = datetime.fromisoformat(meta["received_at"])
    require(requested <= received <= at and received < market_data.deadline(u), "LIVE_RESPONSE_LATE")
    snapshot["response_hash"] = base.digest(meta)
    if meta["http_status"] == 200:
        try:
            snapshot["points"] = parse_actual(raw, t)
        except (ValueError, TypeError, KeyError):
            pass
    return snapshot


def load_live(t, u, ph):
    v = base.read(live_root() / u / "snapshot.json")
    at = datetime.fromisoformat(v["at"])
    require(
        v["t"] == t and v["u"] == u and v["plan_hash"] == ph and in_window(t, u, at),
        "LIVE_SNAPSHOT_SCOPE_OR_TIME_INVALID",
    )
    require(
        v["snapshot"] == reconstruct_live(t, u, ph, at) and v["available"] == bool(v["snapshot"]["points"]),
        "LIVE_SNAPSHOT_CHANGED",
    )
    return v


def capture(t, u, ph):
    folder = live_root() / u
    if (folder / "snapshot.json").exists():
        return load_live(t, u, ph)
    if not in_window(t, u, base.now()):
        return None
    errors = []
    if (folder / "request.json").exists():
        errors.append("RESERVED_REQUEST_NOT_RETRIED")
    else:
        if base.now() + timedelta(seconds=35) >= market_data.deadline(u):
            return None
        cfg = get_settings()
        require(cfg.tushare_api_url == "https://api.tushare.pro", "ENDPOINT_CHANGED")
        token = cfg.tushare_token.get_secret_value()
        require(bool(token), "TOKEN_MISSING")
        q = query(t)
        req = {"at": base.now().isoformat(), "query": q, "plan_hash": ph, "max_retries": 0}
        base.save(folder / "request.json", req)
        try:
            began = monotonic()
            with httpx.Client(timeout=httpx.Timeout(20, connect=5), follow_redirects=False) as client:
                with client.stream(
                    "POST",
                    cfg.tushare_api_url,
                    json={
                        "api_name": q["api"],
                        "token": token,
                        "params": q["params"],
                        "fields": ",".join(parser.FIELDS),
                    },
                ) as response:
                    raw = bytearray()
                    for chunk in response.iter_bytes():
                        raw.extend(chunk)
                        require(len(raw) <= 524288 and monotonic() - began <= 30, "RESPONSE_TOO_LARGE_OR_SLOW")
                    status = response.status_code
            require(token.encode() not in raw, "SECRET_IN_RESPONSE")
            with (folder / "raw.json").open("xb") as f:
                f.write(raw)
            base.save(
                folder / "response.json",
                {
                    "received_at": base.now().isoformat(),
                    "request_hash": base.digest(req),
                    "http_status": status,
                    "bytes": len(raw),
                    "sha256": hashlib.sha256(raw).hexdigest(),
                },
            )
            require(status == 200, "HTTP_FAILED")
            parse_actual(raw, t)
        except Exception as exc:
            e = {"at": base.now().isoformat(), "error": base.error_code(exc), "exception_type": type(exc).__name__}
            base.save(folder / "error.json", e)
            errors.append(e["error"])
    at = base.now()
    if not in_window(t, u, at):
        return None
    snap = reconstruct_live(t, u, ph, at)
    v = {
        "at": at.isoformat(),
        "t": t,
        "u": u,
        "plan_hash": ph,
        "snapshot": snap,
        "available": bool(snap["points"]),
        "errors": errors,
        "max_requests": 1,
        "new_cost_cny": 0,
    }
    base.save(folder / "snapshot.json", v)
    return load_live(t, u, ph)
