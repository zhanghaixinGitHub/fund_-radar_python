"""成交额分布的完整原文核验与T日对齐；复用股票日行情，不增加历史或未来请求。"""

import math

from app.services import direction_1d_sprint as base
from app.services import direction_1d_sprint_stock_breadth_data as old
from app.services import direction_1d_sprint_stock_breadth_live as actual_source
from app.services import direction_1d_sprint_stock_turnover_features as features

KEYS = features.KEYS
scope = old.scope
original = old.original


def root():
    return base.ROOT / "stock-turnover-distribution-feasibility-v1"


def require(condition, reason):
    if not condition:
        raise ValueError("STOCK_TURNOVER_DATA_" + reason)


def reconstruct():
    """依赖原V2完整资格结果，不以部分到齐数据训练；每一份新特征重新核对原股票集合。"""
    require(base.read(old.root() / "progress.json")["status"] == "COMPLETED", "SOURCE_NOT_COMPLETE")
    source = old.history()
    seed = base.read(old.root() / "reused-history.json")
    points, hashes = {}, {}
    for day, expected in source["snapshot"]["rows"].items():
        if day in seed["sources"]:
            raw_path = base.ROOT / seed["sources"][day]["raw_path"]
        else:
            raw_path = old.root() / "raw" / (day + ".json")
        value = features.parse(raw_path.read_bytes(), day)
        require(value["source_distribution_hash"] == base.digest(expected), "SOURCE_DISTRIBUTION_CHANGED")
        require(value["included_set_sha256"] == expected["included_set_sha256"], "SOURCE_UNIVERSE_CHANGED")
        points[day], hashes[day] = value, old.sha(raw_path)
    require(len(points) == 1383, "SOURCE_COVERAGE_CHANGED")
    return dict(sorted(points.items())), {"source_history_hash": base.digest(source), "raw_hashes": hashes}


def history():
    p, result = base.read(root() / "qualification-plan.json"), base.read(root() / "qualification-result.json")
    for name, expected in p["code"].items():
        require(old.sha(base.PROJECT / name) == expected, "QUALIFICATION_CODE_CHANGED")
    require(old.sha(root() / "design-before-analysis.md") == p["design_sha256"], "DESIGN_CHANGED")
    require(base.digest(base.read(root() / "sample-result.json")) == p["sample_result_hash"], "SAMPLE_RESULT_CHANGED")
    require(
        base.digest(base.read(old.root() / "qualification-plan.json")) == p["parent_qualification_plan_hash"],
        "PARENT_QUALIFICATION_CHANGED",
    )
    require(result["plan_hash"] == base.digest(p), "QUALIFICATION_PLAN_CHANGED")
    points, proof = reconstruct()
    require(points == result["features"] and proof == result["source_proof"], "QUALIFIED_FEATURES_CHANGED")
    return {
        "at": result["at"],
        "snapshot": {"rows": points},
        "source_history_hash": proof["source_history_hash"],
        "qualification_hash": base.digest(result),
    }


def extend(market, t, u, points):
    """只用紧邻目标日前T日；金额全零和缺日都显式不可用，不沿用前一天数值。"""
    days = list(map(str, base.calendar()[0]))
    require(t in days and u in days and days.index(u) == days.index(t) + 1, "ADJACENCY_INVALID")
    row = points.get(t)
    feature = {"date": t, "available": False}
    if row is not None:
        require(
            row["date"] == t
            and row["scope"] == "PROVIDER_RETURNED_SH_SZ_DAILY_QUOTES"
            and type(row["available"]) is bool,
            "FEATURE_IDENTITY_CHANGED",
        )
        if row["available"]:
            require(
                all(type(row[k]) in (int, float) and math.isfinite(row[k]) for k in KEYS)
                and abs(row[KEYS[0]]) <= 1
                and abs(row[KEYS[1]]) <= 20
                and type(row["total_amount"]) in (int, float)
                and math.isfinite(row["total_amount"])
                and row["total_amount"] > 0,
                "FEATURE_RANGE_INVALID",
            )
        else:
            require(
                row["unavailable_reason"] == "ZERO_TOTAL_RETURNED_AMOUNT"
                and row["total_amount"] == 0
                and all(row[k] is None for k in KEYS),
                "UNAVAILABLE_FEATURE_CHANGED",
            )
        feature.update(row)
    return market | {"stock_turnover": feature}


def original_row(row):
    return row | {"market": {k: v for k, v in row["market"].items() if k != "stock_turnover"}}


def live(t, u, parent_plan_hash):
    """只派生R109实际接收的原文字节，使用父收据时间；不能用历史补数充当未来输入。"""
    parent = actual_source.load_live(t, u, parent_plan_hash)
    points = {}
    if parent["available"]:
        # load_live已经重验请求、原文摘要、接收时间及目标日，这里再绑定两种分布来源相同。
        raw = (actual_source.root() / u / "raw.bin").read_bytes()
        feature = features.parse(raw, t)
        require(feature["source_distribution_hash"] == base.digest(parent["points"][t]), "LIVE_SOURCE_CHANGED")
        points[t] = feature
    market = extend({}, t, u, points)
    return {
        "at": parent["at"],
        "t": t,
        "u": u,
        "parent_plan_hash": parent_plan_hash,
        "parent_source_hash": base.digest(parent),
        "snapshot": {"points": points},
        "available": market["stock_turnover"]["available"],
        "new_source_requests": 0,
    }
