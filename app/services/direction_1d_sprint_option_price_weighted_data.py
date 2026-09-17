"""IO收盘价加权的认沽/认购比例；复用已验原文，不把估算权重称为实际成交额。"""

import hashlib
import json
import math

from app.services import direction_1d_sprint as base
from app.services import direction_1d_sprint_option_position_data as old

KEYS = ("log_put_call_close_volume", "log_put_call_close_oi")
scope, require = old.scope, old.require


def root():
    return base.ROOT / "option-price-weighted-feasibility-v1"


def parse(raw, days):
    """同一完整返回集合内求收盘价×手数。单位为指数点·手，固定+1平滑真实零值。

    收盘价不是逐笔成交价格；第二项是收盘估值持仓权重，也不代表资金流入。
    不删除低价、零成交或难预测日期；旧解析器负责日期、重复、数值和C/P配对。
    """
    verified = old.parser.parse(raw, days)
    data = json.loads(raw)["data"]
    grouped = {d: [] for d in days}
    for values in data["items"]:
        row = dict(zip(data["fields"], values, strict=True))
        if row["ts_code"].startswith("IO"):
            d = row["trade_date"]
            grouped[f"{d[:4]}-{d[4:6]}-{d[6:]}"].append(row)
    result = {}
    for day, rows in grouped.items():
        sides = {s: [r for r in rows if f"-{s}-" in r["ts_code"]] for s in ("C", "P")}
        totals = {}
        for side, members in sides.items():
            for quantity in ("vol", "oi"):
                products = [r["close"] * r[quantity] for r in members]
                require(all(math.isfinite(x) and x >= 0 for x in products), "PRICE_WEIGHT_OVERFLOW")
                value = math.fsum(products)
                require(math.isfinite(value), "PRICE_WEIGHT_SUM_INVALID")
                totals[f"{side}_{quantity}"] = value
        # log1p差比先除后取log更稳定，数值口径等于log((P+1)/(C+1))。
        logs = [math.log1p(totals[f"P_{q}"]) - math.log1p(totals[f"C_{q}"]) for q in ("vol", "oi")]
        result[day] = {
            "date": day,
            "available": True,
            **dict(zip(KEYS, [max(-20.0, min(20.0, x)) for x in logs], strict=True)),
            "close_weighted_totals": totals,
            "contract_set_sha256": verified["rows"][day]["contract_set_sha256"],
            "io_contracts": len(rows),
            "zero_close_contracts": sum(r["close"] == 0 for r in rows),
            "zero_close_with_volume": sum(r["close"] == 0 and r["vol"] > 0 for r in rows),
            "quantity_unit": "index_points_times_contracts",
            "actual_turnover": False,
        }
    return result


def reconstruct():
    """先复验旧1383日全部采集证据，再逐原文重建价格权重；不请求供应商。"""
    source = old.history()
    sample = base.ROOT / "china-index-option-feasibility-v1"
    specs = [
        (sample / "recent-raw.json", sample / "recent-response.json", ["2026-09-15"]),
        (sample / "raw/early_day.json", sample / "responses/early_day.json", ["2021-01-04"]),
    ]
    p = base.read(old.root() / "plan.json")
    specs.extend(
        (old.root() / "raw" / (q["key"] + ".json"), old.root() / "responses" / (q["key"] + ".json"), q["expected_days"])
        for q in p["queries"]
    )
    points, raw_hashes = {}, {}
    for path, meta_path, days in specs:
        raw, meta = path.read_bytes(), base.read(meta_path)
        require(hashlib.sha256(raw).hexdigest() == meta["sha256"] and len(raw) == meta["bytes"], "PRICE_RAW_CHANGED")
        values = parse(raw, days)
        require(not set(values) & set(points), "PRICE_DATE_OVERLAP")
        for d, v in values.items():
            native = source["snapshot"]["rows"][d]
            require(v["contract_set_sha256"] == native["contract_set_sha256"], "PRICE_CONTRACT_SET_CHANGED")
            require(
                v["io_contracts"] == native["call_contracts"] + native["put_contracts"], "PRICE_CONTRACT_COUNT_CHANGED"
            )
        points.update(values)
        raw_hashes[str(path.relative_to(base.ROOT))] = meta["sha256"]
    require(set(points) == set(source["snapshot"]["rows"]) and len(points) == 1383, "PRICE_HISTORY_COVERAGE_CHANGED")
    return dict(sorted(points.items())), {"source_history_hash": base.digest(source), "raw_hashes": raw_hashes}


def history():
    p, result = base.read(root() / "plan.json"), base.read(root() / "result.json")
    for name, expected in p["code"].items():
        require(old.sha(base.PROJECT / name) == expected, "PRICE_CODE_CHANGED")
    require(old.sha(root() / "design-before-analysis.md") == p["design_sha256"], "PRICE_DESIGN_CHANGED")
    require(result["plan_hash"] == base.digest(p), "PRICE_PLAN_CHANGED")
    points, proof = reconstruct()
    require(points == result["features"] and proof == result["source_proof"], "PRICE_FEATURES_CHANGED")
    return {
        "at": result["at"],
        "snapshot": {"rows": points},
        "source_history_hash": proof["source_history_hash"],
        "feasibility_hash": base.digest(result),
    }


def extend(market, t, u, points):
    days = list(map(str, base.calendar()[0]))
    require(t in days and u in days and days.index(u) == days.index(t) + 1, "PRICE_ADJACENCY_INVALID")
    row = points.get(t)
    feature = {"date": t, "available": row is not None}
    if row is not None:
        require(row["date"] == t and row["available"] is True, "PRICE_FEATURE_DATE_INVALID")
        require(
            all(type(row[k]) in (int, float) and math.isfinite(row[k]) and abs(row[k]) <= 20 for k in KEYS),
            "PRICE_FEATURE_INVALID",
        )
        feature.update(row)
    return market | {"option_price_weighted": feature}


def original_row(row):
    return row | {"market": {k: v for k, v in row["market"].items() if k != "option_price_weighted"}}


def live(t, u, parent_plan_hash):
    """仅复用R97实际收到并已校验的T日响应，保留同一实际时间和来源摘要。"""
    parent = old.load_live(t, u, parent_plan_hash)
    points = {}
    if parent["available"]:
        raw = (old.live_root() / u / "raw.json").read_bytes()
        meta = base.read(old.live_root() / u / "response.json")
        require(
            meta["sha256"] == hashlib.sha256(raw).hexdigest() and meta["bytes"] == len(raw), "PRICE_LIVE_RAW_CHANGED"
        )
        require(base.digest(meta) == parent["snapshot"]["response_hash"], "PRICE_LIVE_RECEIPT_CHANGED")
        points = parse(raw, [t])
        require(
            points[t]["contract_set_sha256"] == parent["snapshot"]["points"][t]["contract_set_sha256"],
            "PRICE_LIVE_CONTRACTS_CHANGED",
        )
    return {
        "at": parent["at"],
        "t": t,
        "u": u,
        "parent_plan_hash": parent_plan_hash,
        "parent_source_hash": base.digest(parent),
        "snapshot": {"points": points},
        "available": bool(points),
        "new_source_requests": 0,
    }
