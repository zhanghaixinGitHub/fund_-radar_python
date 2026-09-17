"""具体IF合约期限价差历史：逐条复核原响应及元信息，不把采集成功当作训练资格。"""

import hashlib
import runpy
from datetime import datetime

from app.integrations import tushare_sprint_futures_curve_v2 as parser
from app.services import direction_1d_sprint as base

require = parser.require


def root():
    return base.ROOT / "futures-curve-history-v2"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def response_chain(request, meta, result, raw):
    """校验请求、响应和解析结果的摘要与时间先后，避免只信缓存的派生数组。"""
    require(
        meta["request_hash"] == base.digest(request)
        and meta["http_status"] == 200
        and meta["bytes"] == len(raw)
        and meta["raw_sha256"] == hashlib.sha256(raw).hexdigest(),
        "RAW_RESPONSE_CHANGED",
    )
    require(result["response_hash"] == base.digest(meta), "PARSED_RESPONSE_CHANGED")
    require(
        datetime.fromisoformat(request["at"])
        <= datetime.fromisoformat(meta["received_at"])
        <= datetime.fromisoformat(result["at"]),
        "RESPONSE_TIME_INVALID",
    )


def samples():
    folder = base.ROOT / "stock-index-futures-feasibility-v1"
    p = base.read(folder / "plan.json")
    require(sha(folder / "probe.py") == p["script_sha256"], "SAMPLE_CODE_CHANGED")
    validate = runpy.run_path(str(folder / "probe.py"))["validate"]
    saved = base.read(root() / "sample-qualification.json")
    parsed = {}
    for key in ("basic", "recent", "history"):
        req, meta, result = (base.read(folder / sub / f"{key}.json") for sub in ("requests", "responses", "results"))
        raw = (folder / "raw" / f"{key}.json").read_bytes()
        response_chain(req, meta, result, raw)
        require(
            req["plan_hash"] == base.digest(p) and req["query"] == next(v for v in p["queries"] if v["key"] == key),
            "SAMPLE_QUERY_CHANGED",
        )
        value = validate(raw, req["query"])
        require(value["rows"] == result["rows"], "SAMPLE_ROWS_CHANGED")
        actual = {
            "request_hash": base.digest(req),
            "response_hash": base.digest(meta),
            "raw_sha256": sha(folder / "raw" / f"{key}.json"),
            "parsed_hash": base.digest(result),
        }
        require(actual == saved["source_evidence"][key], "SAMPLE_PROOF_CHANGED")
        parsed[key] = value["rows"]
    require(saved["metadata_hash"] == base.digest(parsed["basic"]), "METADATA_CHANGED")
    return saved, parsed


def reconstruct():
    p = base.read(root() / "plan.json")
    for name, digest in p["code"].items():
        require(sha(base.PROJECT / name) == digest, "PARSER_CHANGED")
    require(
        sha(root() / "acquire.py") == p["script_sha256"]
        and sha(root() / "design-before-analysis.md") == p["design_sha256"],
        "ACQUISITION_CODE_CHANGED",
    )
    sample, original = samples()
    require(
        base.digest(sample) == p["sample_qualification_hash"] and sample["calendar_hash"] == base.calendar()[1],
        "SAMPLE_CONTEXT_CHANGED",
    )
    old = base.ROOT / "futures-curve-history-v1"
    old_plan = base.read(old / "plan.json")
    require(
        base.digest(old_plan) == p["v1_plan_hash"] and sha(old / "acquire.py") == old_plan["script_sha256"],
        "V1_ORIGIN_CHANGED",
    )
    for name, digest in old_plan["code"].items():
        require(sha(base.PROJECT / name) == digest, "V1_PARSER_CHANGED")
    require(len(p["queries"]) == 70 and p["max_requests"] == 69 and p["retries"] == 0, "BUDGET_CHANGED")
    require(len(list((root() / "requests").glob("*.json"))) == 69, "REQUEST_COUNT_CHANGED")
    contracts, receipts = {}, {}
    for query in p["queries"]:
        key = query["key"]
        origin = old if query.get("reuse_v1") else root()
        req, meta = (base.read(origin / sub / f"{key}.json") for sub in ("requests", "responses"))
        result = base.read(root() / "results" / f"{key}.json")
        raw = (origin / "raw" / f"{key}.json").read_bytes()
        response_chain(req, meta, result, raw)
        expected = {k: v for k, v in query.items() if k not in ("expiry_date", "reuse_v1")} if origin == old else query
        require(
            req["query"] == expected and req["plan_hash"] == base.digest(old_plan if origin == old else p),
            "HISTORY_QUERY_CHANGED",
        )
        if origin == old:
            require(
                base.digest(req) == p["reused_request_hash"] and base.digest(meta) == p["reused_response_hash"],
                "REUSED_ORIGIN_CHANGED",
            )
        meta_contract = sample["contract_metadata"][query["contract"]]
        require(query["expiry_date"].replace("-", "") == meta_contract["delist_date"], "EXPIRY_CHANGED")
        rows = parser.parse(raw, query["contract"], query["expected_dates"], query["expiry_date"])
        require(rows == result["rows"], "PARSED_ROWS_CHANGED")
        contracts[query["contract"]] = rows
        receipts[key] = base.digest(meta)
    acquired = base.read(root() / "history.json")
    require(acquired["contracts"] == contracts and acquired["plan_hash"] == base.digest(p), "ACQUIRED_HISTORY_CHANGED")
    days = [str(d) for d in base.calendar()[0] if "2021-01-04" <= str(d) <= "2026-09-15"]
    points = {}
    for day in days:
        compact = day.replace("-", "")
        active = sorted(
            [m for m in original["basic"] if m["list_date"] <= compact <= m["delist_date"]],
            key=lambda m: m["delist_date"],
        )
        require(len(active) == 4, "ACTIVE_CONTRACTS_MISSING")
        a, b = active[:2]
        require([a["ts_code"], b["ts_code"]] == sample["pairs"][day], "NEAREST_PAIR_CHANGED")
        require(all(sample["contract_metadata"][m["ts_code"]] == m for m in (a, b)), "PAIR_METADATA_CHANGED")
        points[day] = parser.point(day, a, b, contracts[a["ts_code"]][day], contracts[b["ts_code"]][day])
    require(
        len(points) == 1383 and all(points[d] == value for d, value in sample["samples"].items()),
        "HISTORY_SAMPLE_OR_COVERAGE_CHANGED",
    )
    return {
        "rows": points,
        "acquisition_history_hash": base.digest(acquired),
        "receipts": receipts,
        "sample_hash": base.digest(sample),
        "calendar_hash": base.calendar()[1],
        "historical_first_publication_verified": False,
        "independent_contract_registry_verified": False,
        "zero_oi_contract_days_on_expiry": sum(row["oi"] == 0 for rows in contracts.values() for row in rows.values()),
    }


def history():
    p = base.read(root() / "qualification-plan.json")
    for name, digest in p["code"].items():
        require(sha(base.PROJECT / name) == digest, "QUALIFICATION_CODE_CHANGED")
    require(sha(root() / "qualify.py") == p["script_sha256"], "QUALIFICATION_SCRIPT_CHANGED")
    value, result = (base.read(root() / name) for name in ("qualified-history.json", "qualification-result.json"))
    require(
        value["qualification_plan_hash"] == base.digest(p) and value["snapshot"] == reconstruct(),
        "QUALIFIED_HISTORY_CHANGED",
    )
    require(
        result["history_hash"] == base.digest(value) and result["status"] == "QUALIFIED_SPECIFIC_IF_CURVE_WITH_LIMITS",
        "QUALIFICATION_CHANGED",
    )
    return value
