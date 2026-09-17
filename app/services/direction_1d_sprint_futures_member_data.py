"""公开IF会员排名特征的完整原文资格与T/U对齐，保持基金原题身份。"""

import hashlib
from datetime import datetime

import numpy as np

from app.integrations import tushare_sprint_futures_holdings as parser
from app.services import direction_1d_sprint as base
from app.services import direction_1d_sprint_futures_basis_data as old

scope, original = old.scope, old.original
KEYS = ("level_imbalance", "change_imbalance")


def root():
    return base.ROOT / "futures-member-holdings-history-v1"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, reason):
    if not condition:
        raise ValueError("FUTURES_MEMBER_DATA_" + reason)


def reconstruct():
    """每个原文重新解析；独立排名合计不能超过同日主力合约单边总持仓。"""
    plan = base.read(root() / "plan.json")
    require(base.calendar()[1] == plan["calendar_hash"], "CALENDAR_CHANGED")
    require(sha(root() / "design.md") == plan["design_sha256"], "DESIGN_CHANGED")
    for name, expected in plan["code"].items():
        require(sha(base.PROJECT / name) == expected, "ACQUISITION_CODE_CHANGED")
    mapping = old.history()
    require(base.digest(mapping) == plan["mapping_hash"], "MAPPING_CHANGED")
    sample = base.read(base.ROOT / "futures-member-holdings-feasibility-v1/qualification-result.json")
    require(base.digest(sample) == plan["sample_qualification_hash"], "SEED_QUALIFICATION_CHANGED")
    collected = base.read(root() / "history.json")
    require(collected["plan_hash"] == base.digest(plan), "COLLECTION_PLAN_CHANGED")
    points, receipts = {}, {}
    for seed in plan["seeds"]:
        req = base.read(base.ROOT / seed["request"])
        meta = base.read(base.ROOT / seed["response"])
        raw = (base.ROOT / seed["raw"]).read_bytes()
        require(
            base.digest(req) == seed["request_hash"] and base.digest(meta) == seed["response_hash"],
            "SEED_RECEIPT_CHANGED",
        )
        require(meta["request_hash"] == base.digest(req) and meta["http_status"] == 200, "SEED_REQUEST_CHANGED")
        require(meta["raw_sha256"] == hashlib.sha256(raw).hexdigest() and meta["bytes"] == len(raw), "SEED_RAW_CHANGED")
        require(datetime.fromisoformat(req["at"]) <= datetime.fromisoformat(meta["received_at"]), "SEED_TIME_INVALID")
        rows = parser.parse(raw, seed["symbol"], [seed["date"]])
        require(not set(points) & set(rows), "SEED_DUPLICATE")
        points.update(rows)
        receipts[seed["date"]] = base.digest(meta)
    for q in plan["queries"]:
        key = q["key"]
        req = base.read(root() / "requests" / f"{key}.json")
        meta = base.read(root() / "responses" / f"{key}.json")
        parsed = base.read(root() / "parsed" / f"{key}.json")
        raw = (root() / "raw" / f"{key}.json").read_bytes()
        require(req["plan_hash"] == base.digest(plan) and req["query"] == q and req["attempt"] == 1, "REQUEST_CHANGED")
        require(meta["request_hash"] == base.digest(req) and meta["http_status"] == 200, "RESPONSE_CHANGED")
        require(meta["sha256"] == hashlib.sha256(raw).hexdigest() and meta["bytes"] == len(raw), "RAW_CHANGED")
        require(
            datetime.fromisoformat(req["at"])
            <= datetime.fromisoformat(meta["received_at"])
            <= datetime.fromisoformat(parsed["at"]),
            "RECEIPT_TIME_INVALID",
        )
        rows = parser.parse(raw, q["params"]["symbol"], q["dates"])
        require(parsed["response_hash"] == base.digest(meta) and parsed["rows"] == rows, "PARSED_CHANGED")
        require(not set(points) & set(rows), "DUPLICATE_DATE")
        points.update(rows)
        receipts.update({day: base.digest(meta) for day in rows})
    require(points == collected["rows"] and receipts == collected["receipts"], "COLLECTION_CHANGED")
    require(
        set(points) == set(mapping["snapshot"]["rows"]) == set(plan["expected_dates"]) and len(points) == 1383,
        "COVERAGE_CHANGED",
    )
    quotes = {}
    for q in base.read(old.root() / "plan.json")["queries"]:
        if q["api"] == "fut_daily":
            quotes.update(base.read(old.root() / "parsed" / (q["key"] + ".json"))["rows"])
    require(set(quotes) == set(points), "QUOTE_COVERAGE_CHANGED")
    features = {}
    for day, point in points.items():
        require(point["main_contract"] == mapping["snapshot"]["rows"][day]["main_contract"], "MAIN_CONTRACT_MISMATCH")
        oi = quotes[day]["oi"]
        require(type(oi) in (int, float) and np.isfinite(oi) and oi > 0, "TOTAL_OI_INVALID")
        require(
            0 < point["ranked_long_contracts"] <= oi and 0 < point["ranked_short_contracts"] <= oi,
            "RANKED_TOTAL_EXCEEDS_OI",
        )
        features[day] = {k: v for k, v in point.items() if k != "rows"} | {"total_open_interest": oi}
    return dict(sorted(features.items())), {
        "source_history_hash": base.digest(collected),
        "mapping_history_hash": base.digest(mapping),
        "receipt_hashes": receipts,
    }


def history():
    plan = base.read(root() / "qualification-plan.json")
    for name, expected in plan["code"].items():
        require(sha(base.PROJECT / name) == expected, "QUALIFICATION_CODE_CHANGED")
    result = base.read(root() / "qualification-result.json")
    require(
        result["plan_hash"] == base.digest(plan) and result["status"] == "QUALIFIED_IF_MEMBER_HISTORY_WITH_LIMITS",
        "QUALIFICATION_CHANGED",
    )
    points, proof = reconstruct()
    require(result["features"] == points and result["source_proof"] == proof, "QUALIFIED_HISTORY_CHANGED")
    return {
        "at": result["at"],
        "snapshot": {"rows": points},
        "source_history_hash": proof["source_history_hash"],
        "qualification_hash": base.digest(result),
    }


def aligned(t, u, points, calendar):
    """预测U只用T日排名；缺源显式不可用，不能向后取最近日期或补零。"""
    require(t in calendar and u in calendar and calendar.index(u) == calendar.index(t) + 1, "ADJACENCY_INVALID")
    point = points.get(t)
    value = {"date": t, "available": point is not None}
    if point is not None:
        require(
            all(
                type(point[k]) in (int, float) and np.isfinite(point[k]) and abs(point[k]) <= 1
                for k in ("level_imbalance", "change_imbalance")
            ),
            "FEATURE_INVALID",
        )
        require(point["long_members"] == point["short_members"] == 20, "MEMBERS_INVALID")
        value.update(point)
    return value


def extend(market, t, u, points):
    return market | {"futures_member": aligned(t, u, points, list(map(str, base.calendar()[0])))}


def original_row(row):
    return row | {"market": {k: v for k, v in row["market"].items() if k != "futures_member"}}
