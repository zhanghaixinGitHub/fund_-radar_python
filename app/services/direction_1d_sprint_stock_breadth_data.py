"""沪深返回股票分布的原文重建与精确T日对齐；仅研究数据，不修改净值或生产表。"""

import hashlib
from datetime import datetime

from app.integrations import tushare_sprint_stock_breadth_v2 as parser
from app.services import direction_1d_sprint as base
from app.services import direction_1d_sprint_market_fxi_interval as original

KEYS = ("breadth", "median_pct", "iqr_pct")
scope = original.data.scope


def root():
    return base.ROOT / "china-stock-breadth-history-v2"


def require(condition, reason):
    if not condition:
        raise ValueError("STOCK_BREADTH_DATA_" + reason)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def reconstruct():
    """完整历史到齐后逐响应重建；不以后台进度、摘要文件或请求成功数代替数据核验。"""
    p = base.read(root() / "plan.json")
    for name, expected in p["code"].items():
        require(sha(base.PROJECT / name) == expected, "ACQUISITION_CODE_CHANGED")
    require(sha(root() / "design-before-acquisition.md") == p["design_sha256"], "ACQUISITION_DESIGN_CHANGED")
    require(
        p["calendar_hash"] == base.calendar()[1] and len(p["queries"]) == p["max_new_requests"] == 1380,
        "ACQUISITION_SCOPE_CHANGED",
    )
    seed = base.read(root() / "reused-history.json")
    require(base.digest(seed) == p["reused_history_hash"], "REUSED_HISTORY_CHANGED")
    points, receipts = {}, {}
    for day, spec in seed["sources"].items():
        req, meta = base.read(base.ROOT / spec["request_path"]), base.read(base.ROOT / spec["response_path"])
        raw = (base.ROOT / spec["raw_path"]).read_bytes()
        require(
            base.digest(base.read(base.ROOT / spec["plan_path"])) == spec["plan_hash"] == req["plan_hash"],
            "REUSED_PLAN_CHANGED",
        )
        require(
            req["query"] == spec["expected_query"] and base.digest(req) == spec["request_hash"],
            "REUSED_REQUEST_CHANGED",
        )
        require(
            meta["request_hash"] == base.digest(req) and base.digest(meta) == spec["response_hash"],
            "REUSED_RECEIPT_CHANGED",
        )
        require(
            meta["http_status"] == 200
            and len(raw) == meta["bytes"]
            and hashlib.sha256(raw).hexdigest() == meta["sha256"],
            "REUSED_RAW_CHANGED",
        )
        require(
            datetime.fromisoformat(req["at"])
            <= datetime.fromisoformat(meta["received_at"])
            <= datetime.fromisoformat(seed["at"]),
            "REUSED_TIME_CHANGED",
        )
        points[day], receipts[day] = parser.parse(raw, day), base.digest(meta)
        require(points[day] == seed["rows"][day], "REUSED_PARSED_CHANGED")
    for q in p["queries"]:
        day = q["date"]
        req, meta, saved = (base.read(root() / sub / (day + ".json")) for sub in ("requests", "responses", "parsed"))
        raw = (root() / "raw" / (day + ".json")).read_bytes()
        require(req["query"] == q and req["plan_hash"] == base.digest(p) and req["attempt"] == 1, "REQUEST_CHANGED")
        require(meta["request_hash"] == base.digest(req) and meta["http_status"] == 200, "RESPONSE_CHANGED")
        require(len(raw) == meta["bytes"] and hashlib.sha256(raw).hexdigest() == meta["sha256"], "RAW_CHANGED")
        require(
            datetime.fromisoformat(req["at"])
            <= datetime.fromisoformat(meta["received_at"])
            <= datetime.fromisoformat(saved["at"]),
            "TIME_CHANGED",
        )
        value = parser.parse(raw, day)
        require(
            saved["value"] == value and saved["response_hash"] == base.digest(meta) and day not in points,
            "PARSED_OR_DUPLICATE_CHANGED",
        )
        points[day], receipts[day] = value, base.digest(meta)
    stored = base.read(root() / "history.json")
    require(len(points) == 1383 and set(points) == set(p["expected_days"]), "HISTORY_INCOMPLETE")
    require(
        stored["rows"] == points and stored["receipts"] == receipts and stored["plan_hash"] == base.digest(p),
        "HISTORY_CHANGED",
    )
    return {
        "rows": dict(sorted(points.items())),
        "acquisition_history_hash": base.digest(stored),
        "source_plan_hash": base.digest(p),
        "scope": "PROVIDER_RETURNED_SH_SZ_DAILY_QUOTES",
        "full_registry_verified": False,
        "historical_first_publication_verified": False,
    }


def history():
    """资格文件须在完整来源核验后单独生成，未完成时明确失败，不读取半份输入。"""
    p = base.read(root() / "qualification-plan.json")
    result = base.read(root() / "qualification-result.json")
    value = base.read(root() / "qualified-history.json")
    for name, expected in p["code"].items():
        require(sha(base.PROJECT / name) == expected, "QUALIFICATION_CODE_CHANGED")
    require(value["qualification_plan_hash"] == base.digest(p), "QUALIFICATION_PLAN_CHANGED")
    require(
        result["history_hash"] == base.digest(value)
        and result["status"] == "QUALIFIED_SH_SZ_DAILY_DISTRIBUTION_WITH_LIMITS",
        "QUALIFICATION_CHANGED",
    )
    require(value["snapshot"] == reconstruct(), "QUALIFIED_HISTORY_CHANGED")
    return value


def extend(market, t, u, points):
    """只取目标日相邻前一交易日T，不跨日补值，不把缺日当作平盘。"""
    days = list(map(str, base.calendar()[0]))
    require(t in days and u in days and days.index(u) == days.index(t) + 1, "ADJACENCY_INVALID")
    row = points.get(t)
    feature = {"date": t, "available": row is not None}
    if row is not None:
        import math

        require(
            row["date"] == t and row["available"] is True and row["scope"] == "PROVIDER_RETURNED_SH_SZ_DAILY_QUOTES",
            "FEATURE_IDENTITY_CHANGED",
        )
        require(all(type(row[k]) in (int, float) and math.isfinite(row[k]) for k in KEYS), "FEATURE_NONFINITE")
        require(
            abs(row["breadth"]) <= 1 and abs(row["median_pct"]) <= 20 and 0 <= row["iqr_pct"] <= 40,
            "FEATURE_RANGE_INVALID",
        )
        feature.update(row)
    return market | {"stock_breadth": feature}


def original_row(row):
    return row | {"market": {k: v for k, v in row["market"].items() if k != "stock_breadth"}}
