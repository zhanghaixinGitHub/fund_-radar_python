"""逐份重验已冻结沪深日线并派生交易所相对强弱；无供应商请求、无基金评分。"""

from app.services import direction_1d_sprint as base
from app.services import direction_1d_sprint_stock_breadth_data as old
from app.services import direction_1d_sprint_stock_exchange_spread_features as features

KEYS, scope, original = features.KEYS, old.scope, old.original


def root():
    return base.ROOT / "stock-exchange-spread-feasibility-v1"


def require(condition, reason):
    if not condition:
        raise ValueError("STOCK_EXCHANGE_SPREAD_DATA_" + reason)


def reconstruct():
    """完整重验父历史收据与原文，再核对派生分布身份；不读取部分采集进度。"""
    source = old.history()
    seed = base.read(old.root() / "reused-history.json")
    points, hashes = {}, {}
    for day, expected in source["snapshot"]["rows"].items():
        path = (
            base.ROOT / seed["sources"][day]["raw_path"]
            if day in seed["sources"]
            else old.root() / "raw" / (day + ".json")
        )
        value = features.parse(path.read_bytes(), day)
        require(value["source_distribution_hash"] == base.digest(expected), "PARENT_DISTRIBUTION_CHANGED")
        require(value["included_set_sha256"] == expected["included_set_sha256"], "SOURCE_UNIVERSE_CHANGED")
        points[day], hashes[day] = value, old.sha(path)
    require(len(points) == 1383, "SOURCE_COVERAGE_CHANGED")
    return dict(sorted(points.items())), {"source_history_hash": base.digest(source), "raw_hashes": hashes}


def history():
    """核验资格及全部来源后才返回完整历史；来源不完整时拒绝假造合格状态。"""
    p, result = base.read(root() / "qualification-plan.json"), base.read(root() / "qualification-result.json")
    require(base.calendar()[1] == p["calendar_hash"], "CALENDAR_CHANGED")
    for name, expected in p["code"].items():
        require(old.sha(base.PROJECT / name) == expected, "QUALIFICATION_CODE_CHANGED")
    require(old.sha(root() / "design-before-analysis.md") == p["design_sha256"], "DESIGN_CHANGED")
    require(base.digest(base.read(root() / "sample-result.json")) == p["sample_result_hash"], "SAMPLE_RESULT_CHANGED")
    require(
        base.digest(base.read(old.root() / "qualification-result.json")) == p["parent_qualification_result_hash"],
        "PARENT_QUALIFICATION_CHANGED",
    )
    require(
        result["plan_hash"] == base.digest(p) and result["status"] == "QUALIFIED_STOCK_EXCHANGE_SPREAD_WITH_LIMITS",
        "QUALIFICATION_CHANGED",
    )
    points, proof = reconstruct()
    require(points == result["features"] and proof == result["source_proof"], "QUALIFIED_FEATURES_CHANGED")
    return {
        "at": result["at"],
        "snapshot": {"rows": points},
        "source_history_hash": proof["source_history_hash"],
        "qualification_hash": base.digest(result),
    }


def extend(market, t, u, points):
    return market | {"stock_exchange_spread": features.aligned(t, u, points, list(map(str, base.calendar()[0])))}


def original_row(row):
    return row | {"market": {k: v for k, v in row["market"].items() if k != "stock_exchange_spread"}}
