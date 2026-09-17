"""第97轮IO成交/持仓比；复核全历史原文，并独立记录实际T日来源。"""

import hashlib
import json
import re
from datetime import datetime, timedelta
from time import monotonic

import httpx
import numpy as np

from app.core.config import get_settings
from app.integrations import tushare_sprint_index_options as parser
from app.services import direction_1d_sprint as base
from app.services import direction_1d_sprint_market_fxi_interval as original
from app.services import direction_1d_sprint_market_only_data as market_data

scope = original.data.scope


def root():
    return base.ROOT / "china-index-option-history-v1"


def require(condition, reason):
    if not condition:
        raise ValueError("IO_POSITION_" + reason)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def samples():
    """复核两个交易日与四个实际合约样本；只证明样本和返回合约配对，不宣称全名册完整。"""
    folder = base.ROOT / "china-index-option-feasibility-v1"
    qualification = base.read(folder / "qualification-result.json")
    specs = [("recent", "recent-plan.json", "recent_probe.py")]
    for pf, script in (
        ("qualification-probe-plan.json", "qualification_probe.py"),
        ("contract-probe-plan.json", "contract_probes.py"),
    ):
        specs.extend((q["key"], pf, script) for q in base.read(folder / pf)["queries"])
    results, raw_by_key = {}, {}
    for key, pf, script in specs:
        p = base.read(folder / pf)
        require(p["script_sha256"] == sha(folder / script), "SAMPLE_SCRIPT_CHANGED")
        if key == "recent":
            req, meta, result = (
                base.read(folder / f"recent-{name}.json") for name in ("request", "response", "result")
            )
            raw = (folder / "recent-raw.json").read_bytes()
            fields = p["fields"]
            require(req["params"] == p["params"] and req["fields"] == fields, "RECENT_REQUEST_CHANGED")
        else:
            req, meta, result = (
                base.read(folder / sub / (key + ".json")) for sub in ("requests", "responses", "results")
            )
            raw = (folder / "raw" / (key + ".json")).read_bytes()
            q = next(q for q in p["queries"] if q["key"] == key)
            require(req["query"] == q, "SAMPLE_REQUEST_CHANGED")
            fields = q["fields"]
        require(req["plan_hash"] == base.digest(p) and meta["request_hash"] == base.digest(req), "SAMPLE_PLAN_CHANGED")
        require(
            meta["http_status"] == 200
            and meta["bytes"] == len(raw)
            and meta["sha256"] == hashlib.sha256(raw).hexdigest(),
            "SAMPLE_RAW_CHANGED",
        )
        require(
            datetime.fromisoformat(req["at"])
            <= datetime.fromisoformat(meta["received_at"])
            <= datetime.fromisoformat(result["at"]),
            "SAMPLE_TIME_CHANGED",
        )
        v = json.loads(raw)
        require(type(v["code"]) is int and v["code"] == 0 and v["data"]["fields"] == fields, "SAMPLE_SCHEMA_CHANGED")
        require(
            result["rows"] == [dict(zip(fields, r, strict=True)) for r in v["data"]["items"]], "SAMPLE_PARSED_CHANGED"
        )
        require(
            result["response_hash"] == base.digest(meta) == qualification["receipts"][key]
            and base.digest(result) == qualification["source_result_hashes"][key],
            "SAMPLE_RESULT_CHANGED",
        )
        results[key], raw_by_key[key] = result, raw
    for key in ("observed_contract", "current_put", "early_call", "early_put"):
        x = results[key]["rows"][0]
        m = re.fullmatch(r"IO(\d{4})-([CP])-(\d+(?:\.\d+)?)\.CFX", x["ts_code"])
        require(m is not None, "SAMPLE_CONTRACT_INVALID")
        month, side, strike = m.groups()
        day = "20210104" if key.startswith("early") else "20260915"
        require(
            x["opt_code"] == "OP000300.SH"
            and x["call_put"] == side
            and x["exchange"] == "CFFEX"
            and x["exercise_price"] == float(strike),
            "SAMPLE_ROLE_OR_UNDERLYING_INVALID",
        )
        require(
            x["list_date"] <= day <= x["delist_date"] == x["maturity_date"] and x["maturity_date"][2:6] == month,
            "SAMPLE_CONTRACT_TIME_INVALID",
        )
        require(x["per_unit"] == x["opt_multiplier"] == 100 and x["quote_unit"] == "指数点", "SAMPLE_UNIT_INVALID")
    expected = {
        "2026-09-15": parser.parse(raw_by_key["recent"], ["2026-09-15"]),
        "2021-01-04": parser.parse(raw_by_key["early_day"], ["2021-01-04"]),
    }
    require(qualification["samples"] == expected and qualification["actual_requests"] == 6, "SAMPLE_AGGREGATE_CHANGED")
    return qualification


def reconstruct():
    p, sample = base.read(root() / "plan.json"), samples()
    require(
        p["script_sha256"] == sha(root() / "acquire.py")
        and p["design_sha256"] == sha(root() / "design-before-acquisition.md"),
        "ACQUISITION_CODE_CHANGED",
    )
    for name, digest in p["code"].items():
        require(sha(base.PROJECT / name) == digest, "PARSER_CHANGED")
    require(
        p["sample_qualification_hash"] == base.digest(sample) and p["calendar_hash"] == base.calendar()[1],
        "SOURCE_CONTEXT_CHANGED",
    )
    require(len(p["queries"]) == p["max_requests"] == 277 and p["max_retries"] == 0, "SOURCE_BUDGET_CHANGED")
    rows, receipts = {}, {}
    for value in sample["samples"].values():
        rows.update(value["rows"])
    for q in p["queries"]:
        key = q["key"]
        req, meta, saved = (base.read(root() / sub / (key + ".json")) for sub in ("requests", "responses", "parsed"))
        raw = (root() / "raw" / (key + ".json")).read_bytes()
        require(
            req["query"] == q and req["plan_hash"] == base.digest(p) and meta["request_hash"] == base.digest(req),
            "HISTORY_REQUEST_CHANGED",
        )
        require(
            meta["http_status"] == 200
            and meta["bytes"] == len(raw)
            and meta["sha256"] == hashlib.sha256(raw).hexdigest(),
            "HISTORY_RAW_CHANGED",
        )
        require(
            datetime.fromisoformat(req["at"])
            <= datetime.fromisoformat(meta["received_at"])
            <= datetime.fromisoformat(saved["at"]),
            "HISTORY_TIME_CHANGED",
        )
        parsed = parser.parse(raw, q["expected_days"])
        require(parsed == saved["values"] and saved["response_hash"] == base.digest(meta), "HISTORY_PARSED_CHANGED")
        require(not set(rows) & set(parsed["rows"]), "HISTORY_DATE_OVERLAP")
        rows.update(parsed["rows"])
        receipts[key] = base.digest(meta)
    days = [str(d) for d in base.calendar()[0] if "2021-01-01" <= str(d) <= "2026-09-15"]
    require(set(rows) == set(days) == set(p["expected_days"]) and len(rows) == 1383, "HISTORY_CALENDAR_CHANGED")
    stored = base.read(root() / "history.json")
    require(
        stored["rows"] == rows and stored["receipts"] == receipts and stored["plan_hash"] == base.digest(p),
        "ACQUISITION_HISTORY_CHANGED",
    )
    return {
        "rows": dict(sorted(rows.items())),
        "acquisition_history_hash": base.digest(stored),
        "sample_qualification_hash": base.digest(sample),
        "calendar_hash": base.calendar()[1],
        "historical_first_publication_verified": False,
        "full_active_contract_registry_independently_verified": False,
    }


def history():
    value, q, p = (
        base.read(root() / name)
        for name in ("qualified-history.json", "qualification-result.json", "qualification-plan.json")
    )
    for name, digest in p["code"].items():
        require(sha(base.PROJECT / name) == digest, "QUALIFICATION_CODE_CHANGED")
    require(
        value["qualification_plan_hash"] == base.digest(p) and value["snapshot"] == reconstruct(),
        "QUALIFIED_HISTORY_CHANGED",
    )
    require(
        q["history_hash"] == base.digest(value) and q["status"] == "QUALIFIED_PROVIDER_IO_HISTORY_WITH_LIMITS",
        "QUALIFICATION_CHANGED",
    )
    return value


def extend(market, t, u, points):
    days = list(map(str, base.calendar()[0]))
    require(t in days and u in days and days.index(u) == days.index(t) + 1, "ADJACENCY_INVALID")
    row = points.get(t)
    feature = {"date": t, "available": row is not None}
    if row is not None:
        require(
            all(
                type(row[k]) in (int, float) and np.isfinite(row[k]) for k in ("log_put_call_volume", "log_put_call_oi")
            ),
            "FEATURE_INVALID",
        )
        feature.update(row | {k: float(np.clip(row[k], -20, 20)) for k in ("log_put_call_volume", "log_put_call_oi")})
    return market | {"option_position": feature}


def original_row(row):
    return row | {"market": {k: v for k, v in row["market"].items() if k != "option_position"}}


def live_root():
    return base.ROOT / "round-97" / "live-option-position"


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
        "api": "opt_daily",
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
            snapshot["points"] = parser.parse(raw, [t])["rows"]
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
                        require(len(raw) <= 2097152 and monotonic() - began <= 30, "RESPONSE_TOO_LARGE_OR_SLOW")
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
            parser.parse(raw, [t])
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
