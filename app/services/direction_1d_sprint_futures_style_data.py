"""IH/IC相对IF日内走势：全历史映射核验，未来只复用R102实际全CFFEX原响应。"""

import hashlib
import json
import re
import runpy
from datetime import datetime

import numpy as np

from app.integrations import tushare_sprint_futures_style as parser
from app.services import direction_1d_sprint as base
from app.services import direction_1d_sprint_futures_curve_data as current
from app.services import direction_1d_sprint_futures_intraday_data as intra

scope, require = intra.scope, parser.require


def root():
    return base.ROOT / "china-futures-style-history-v1"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def chain(folder, key, sub="parsed"):
    req, meta, saved = (base.read(folder / d / f"{key}.json") for d in ("requests", "responses", sub))
    raw = (folder / "raw" / f"{key}.json").read_bytes()
    digest = meta.get("sha256", meta.get("raw_sha256"))
    require(
        meta["request_hash"] == base.digest(req)
        and meta["http_status"] == 200
        and meta["bytes"] == len(raw)
        and hashlib.sha256(raw).hexdigest() == digest,
        "RAW_CHANGED",
    )
    require(saved["response_hash"] == base.digest(meta), "PARSED_CHANGED")
    require(
        datetime.fromisoformat(req["at"])
        <= datetime.fromisoformat(meta["received_at"])
        <= datetime.fromisoformat(saved["at"]),
        "SOURCE_TIME_CHANGED",
    )
    return req, meta, saved, raw


def samples():
    folder = base.ROOT / "china-futures-style-feasibility-v1"
    p, q, audit = (base.read(folder / f) for f in ("plan.json", "qualification-result.json", "saved-sample-audit.json"))
    require(
        sha(folder / "probe.py") == p["script_sha256"]
        and sha(folder / "design-before-analysis.md") == p["design_sha256"],
        "SAMPLE_CODE_CHANGED",
    )
    require(
        base.digest(p) == q["plan_hash"]
        and base.digest(audit) == p["saved_sample_audit_hash"] == q["sample_audit_hash"],
        "SAMPLE_CONTEXT_CHANGED",
    )
    validate = runpy.run_path(str(folder / "probe.py"))["validate"]
    metadata, mappings = {}, {}
    for spec in p["queries"]:
        req, meta, saved, raw = chain(folder, spec["key"], "results")
        require(req["query"] == spec and req["plan_hash"] == base.digest(p), "SAMPLE_REQUEST_CHANGED")
        require(
            validate(raw, spec)["rows"] == saved["rows"]
            and base.digest(saved) == q["source_result_hashes"][spec["key"]],
            "SAMPLE_ROWS_CHANGED",
        )
        require(base.digest(meta) == q["receipts"][spec["key"]], "SAMPLE_RECEIPT_CHANGED")
        if spec["api"] == "fut_basic":
            metadata[spec["family"]] = {m["ts_code"]: m for m in saved["rows"]}
        else:
            require(len(saved["rows"]) == 1, "SAMPLE_MAPPING_COUNT_CHANGED")
            mappings[spec["family"]] = saved["rows"][0]
    old = base.ROOT / "stock-index-futures-feasibility-v1"
    quotes = {}
    for key, expected in audit["receipts"].items():
        req, meta, _, raw = chain(old, key, "results")
        require(
            base.digest(req) == expected["request_hash"] and base.digest(meta) == expected["response_hash"],
            "OLD_SAMPLE_CHANGED",
        )
        data = json.loads(raw)["data"]
        rows = [dict(zip(data["fields"], row, strict=True)) for row in data["items"]]
        chosen = [r for r in rows if r["ts_code"] in ("IF.CFX", "IH.CFX", "IC.CFX")]
        require(chosen == audit["samples"][key], "OLD_QUOTES_CHANGED")
        quotes[key] = {r["ts_code"]: r for r in rows}
    for family in ("IH", "IC"):
        code = mappings[family]["mapping_ts_code"]
        require(mappings[family]["trade_date"] == "20260915" and code in metadata[family], "CURRENT_MAPPING_CHANGED")
        require(
            all(
                quotes["recent"][family + ".CFX"][k] == quotes["recent"][code][k]
                for k in ("open", "high", "low", "close", "vol", "oi")
            ),
            "CURRENT_MAPPING_PRICE_CHANGED",
        )
    return q, metadata, quotes, mappings


def relative(day, quotes, if_features):
    """开收到收盘涨幅之差，单位百分点；只刻画同T日不同品种相对表现，固定±40截断。"""
    result = {"date": day} | if_features
    for family in ("IC", "IH"):
        value = quotes[family]
        require(value["trade_date"] == day.replace("-", ""), "FEATURE_DATE_CHANGED")
        f = intra.features(value)
        result[family.lower() + "_relative_intraday_pct"] = float(
            np.clip(f["intraday_return_pct"] - if_features["intraday_return_pct"], -40, 40)
        )
    return result


def reconstruct():
    p = base.read(root() / "plan.json")
    require(sha(root() / "acquire.py") == p["script_sha256"], "ACQUISITION_CODE_CHANGED")
    for name, digest in p["code"].items():
        require(sha(base.PROJECT / name) == digest, "PARSER_CHANGED")
    qualification, metadata, samples_by_day, mappings = samples()
    require(
        base.digest(qualification) == p["sample_qualification_hash"] and p["calendar_hash"] == base.calendar()[1],
        "HISTORY_CONTEXT_CHANGED",
    )
    require(
        len(p["queries"]) == p["max_requests"] == 24 and len(list((root() / "requests").glob("*.json"))) == 24,
        "HISTORY_BUDGET_CHANGED",
    )
    rows = {family: {api: {} for api in parser.FIELDS} for family in ("IC", "IH")}
    receipts = {}
    for spec in p["queries"]:
        req, meta, saved, raw = chain(root(), spec["key"])
        require(req["query"] == spec and req["plan_hash"] == base.digest(p), "HISTORY_REQUEST_CHANGED")
        actual = parser.parse(raw, spec["api"], spec["family"], spec["expected_dates"])
        require(actual == saved["rows"], "HISTORY_PARSE_CHANGED")
        dest = rows[spec["family"]][spec["api"]]
        require(not set(dest) & set(actual), "HISTORY_OVERLAP")
        dest.update(actual)
        receipts[spec["key"]] = base.digest(meta)
    reference = intra.history()
    days = set(reference["snapshot"]["rows"])
    for family, data in rows.items():
        require(set(data["fut_daily"]) == set(data["fut_mapping"]) == days, "HISTORY_DATES_CHANGED")
        for day, mapping in data["fut_mapping"].items():
            m = metadata[family][mapping["mapping_ts_code"]]
            require(
                m["list_date"] <= day.replace("-", "") <= m["delist_date"]
                and m["fut_code"] == family
                and m["quote_unit"] == "指数点"
                and m["multiplier"] == qualification["summary"][family]["multiplier"],
                "HISTORY_CONTRACT_INVALID",
            )
        require(data["fut_mapping"]["2026-09-15"] == mappings[family], "LATEST_MAPPING_CHANGED")
        for key, day in (("recent", "2026-09-15"), ("history", "2025-01-02")):
            require(
                data["fut_daily"][day]
                == {k: samples_by_day[key][family + ".CFX"][k] for k in parser.FIELDS["fut_daily"]},
                "SAVED_DAY_MISMATCH",
            )
    points = {
        d: relative(d, {f: rows[f]["fut_daily"][d] for f in rows}, reference["snapshot"]["rows"][d])
        for d in sorted(days)
    }
    require(len(points) == 1383, "FEATURE_DATES_CHANGED")
    return {
        "rows": points,
        "receipts": receipts,
        "intraday_history_hash": base.digest(reference),
        "sample_qualification_hash": base.digest(qualification),
        "calendar_hash": base.calendar()[1],
        "historical_first_publication_verified": False,
    }


def history():
    p = base.read(root() / "qualification-plan.json")
    for name, digest in p["code"].items():
        require(sha(base.PROJECT / name) == digest, "QUALIFICATION_CODE_CHANGED")
    require(sha(root() / "qualify.py") == p["script_sha256"], "QUALIFICATION_SCRIPT_CHANGED")
    value, result = (base.read(root() / f) for f in ("qualified-history.json", "qualification-result.json"))
    require(
        value["qualification_plan_hash"] == base.digest(p) and value["snapshot"] == reconstruct(),
        "QUALIFIED_HISTORY_CHANGED",
    )
    require(
        result["history_hash"] == base.digest(value)
        and result["status"] == "QUALIFIED_IH_IC_RELATIVE_INTRADAY_HISTORY",
        "QUALIFICATION_CHANGED",
    )
    return value


def extend(market, t, u, points):
    f = intra.extend({}, t, u, points)["futures_intraday"]
    if f["available"]:
        require(
            f["date"] == t
            and all(
                type(f[k]) in (int, float) and np.isfinite(f[k]) and abs(f[k]) <= 40
                for k in ("ic_relative_intraday_pct", "ih_relative_intraday_pct")
            ),
            "RELATIVE_FEATURE_INVALID",
        )
    return market | {"futures_style": f}


def original_row(row):
    return row | {"market": {k: v for k, v in row["market"].items() if k != "futures_style"}}


def parse_live(raw, t, metadata, if_features):
    """从实收全行情中验证连续行与唯一具体合约逐字段一致；此为推断映射，不冒称实收mapping接口。"""
    require(raw and len(raw) <= 524288, "LIVE_SIZE_INVALID")
    value = json.loads(raw)
    data = value.get("data") or {}
    require(
        type(value.get("code")) is int
        and value["code"] == 0
        and data.get("fields") == current.parser.FIELDS
        and isinstance(data.get("items"), list)
        and 1 <= len(data["items"]) <= 1000,
        "LIVE_SCHEMA_INVALID",
    )
    records = {}
    for item in data["items"]:
        require(isinstance(item, list) and len(item) == len(data["fields"]), "LIVE_ROW_INVALID")
        row = dict(zip(data["fields"], item, strict=True))
        require(
            row["ts_code"] not in records and row["trade_date"] == t.replace("-", ""), "LIVE_DUPLICATE_OR_DATE_INVALID"
        )
        records[row["ts_code"]] = row
    require(intra.features(records["IF.CFX"]) == if_features, "LIVE_IF_SOURCE_DISAGREEMENT")
    quotes, inferred = {}, {}
    for family in ("IH", "IC"):
        row = records[family + ".CFX"]
        require(
            all(type(row[k]) in (int, float) and np.isfinite(row[k]) and row[k] > 0 for k in ("vol", "oi")),
            "LIVE_LIQUIDITY_INVALID",
        )
        intra.features(row)
        matches = [
            code
            for code, v in records.items()
            if re.fullmatch(family + r"\d{4}\.CFX", code)
            and all(v[k] == row[k] for k in ("open", "high", "low", "close", "vol", "oi"))
        ]
        require(len(matches) == 1 and matches[0] in metadata[family], "LIVE_MAPPING_AMBIGUOUS_OR_UNKNOWN")
        m = metadata[family][matches[0]]
        require(
            m["list_date"] <= t.replace("-", "") <= m["delist_date"] and m["quote_unit"] == "指数点",
            "LIVE_CONTRACT_INACTIVE",
        )
        quotes[family], inferred[family] = row, matches[0]
    return relative(t, quotes, if_features) | {"inferred_contracts": inferred}


def live(t, u, curve_plan_hash, intraday_plan_hash):
    """没有新请求：分别绑定R102完整原响应和R95已实收IF日内原文。"""
    captured, reference = current.load_live(t, u, curve_plan_hash), intra.live(t, u, intraday_plan_hash)
    _, metadata, _, _ = samples()
    points, errors = {}, []
    response = current.live_root() / u / "response.json"
    valid_http = response.exists() and base.read(response)["http_status"] == 200
    if captured["snapshot"]["response_hash"] is not None and valid_http and reference["available"]:
        raw = (current.live_root() / u / "raw.json").read_bytes()
        try:
            points[t] = parse_live(raw, t, metadata, reference["snapshot"]["points"][t])
        except (ValueError, KeyError, TypeError) as exc:
            errors.append(str(exc)[:160])
    return {
        "at": max(captured["at"], reference["at"]),
        "t": t,
        "u": u,
        "curve_source_hash": base.digest(captured),
        "intraday_source_hash": base.digest(reference),
        "snapshot": {"points": points},
        "available": bool(points),
        "errors": errors,
        "new_source_requests": 0,
    }
