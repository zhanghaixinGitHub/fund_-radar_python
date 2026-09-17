"""已完整验收的订单分类历史：逐原文/收据重验身份，复用冻结解析结果，不重复发请求。"""

import hashlib
import math
from datetime import datetime

from app.services import direction_1d_sprint as base
from app.services import direction_1d_sprint_stock_breadth_data as quotes

KEYS = ("large_imbalance", "net_fraction")
scope, original = quotes.scope, quotes.original


def root():
    return base.ROOT / "china-stock-moneyflow-history-v2"


def require(condition, reason):
    if not condition:
        raise ValueError("STOCK_MONEYFLOW_DATA_" + reason)


def validate_feature(value, day):
    """金额尺度和派生关系再验一次；缺分母明确不可用，不把缺失当作零净额。"""
    require(value["date"] == day and value["scope"] == "PROVIDER_RETURNED_SH_SZ_MONEYFLOW", "FEATURE_IDENTITY")
    require(type(value["available"]) is bool and value["amount_unit"] == "TEN_THOUSAND_CNY", "FEATURE_UNITS")
    if not value["available"]:
        require(value["unavailable_reason"] == "ZERO_CLASSIFIED_DENOMINATOR", "UNAVAILABLE_REASON")
        require(all(value[k] is None for k in KEYS), "UNAVAILABLE_FEATURE_NOT_NULL")
        return
    require(
        all(type(value[k]) in (int, float) and math.isfinite(value[k]) and abs(value[k]) <= 1 for k in KEYS),
        "FEATURE_RANGE",
    )
    gross, large = value["classified_gross_amount"], value["large_classified_gross_amount"]
    require(all(type(x) in (int, float) and math.isfinite(x) and x > 0 for x in [gross, large]), "DENOMINATOR")
    totals = value["totals"]
    buy = totals["buy_lg_amount"] + totals["buy_elg_amount"]
    sell = totals["sell_lg_amount"] + totals["sell_elg_amount"]
    net = totals["net_mf_amount"] / gross
    require(
        value["large_imbalance"] == (buy - sell) / large
        and large == buy + sell
        and gross == math.fsum(v for k, v in totals.items() if k.endswith("_amount") and k != "net_mf_amount") / 2
        and value["raw_net_fraction"] == net
        and value["net_fraction"] == max(-1.0, min(1.0, net))
        and value["net_fraction_clipped"] == (abs(net) > 1),
        "FEATURE_FORMULA",
    )


def history():
    """全量验收必须已完成；读取时验证同一资格、全部原始字节及收据，没有宽松缓存。

    qualify_all已对每份原文重新解析并完成报价交叉覆盖与窗口校验。这里将不可变
    的验收产物与相同代码、原文字节、请求/响应摘要逐项绑定；只省去重复数值解析，
    不省去任何原文哈希核验。模型计划另行绑定该资格结果哈希，不能重写资格冒充原模型。
    """
    p, result, qualified = (
        base.read(root() / name)
        for name in ["qualification-plan.json", "qualification-result.json", "qualified-history.json"]
    )
    require(base.calendar()[1] == p["calendar_hash"], "CALENDAR_CHANGED")
    for name, expected in p["code"].items():
        require(quotes.sha(base.PROJECT / name) == expected, "QUALIFICATION_CODE_CHANGED")
    require(result["plan_hash"] == base.digest(p), "QUALIFICATION_PLAN_CHANGED")
    require(result["status"] == "QUALIFIED_SH_SZ_MONEYFLOW_WITH_LIMITS", "QUALIFICATION_INCOMPLETE")
    require(result["history_hash"] == base.digest(qualified), "QUALIFIED_HISTORY_CHANGED")
    require(qualified["qualification_plan_hash"] == base.digest(p), "QUALIFIED_HISTORY_PLAN_CHANGED")
    acquisition, stored, seed = (
        base.read(root() / name) for name in ["plan.json", "history.json", "reused-history.json"]
    )
    require(base.digest(acquisition) == p["acquisition_plan_hash"], "ACQUISITION_PLAN_CHANGED")
    require(
        quotes.sha(root() / "design-before-acquisition.md") == acquisition["design_sha256"],
        "ACQUISITION_DESIGN_CHANGED",
    )
    require(base.digest(seed) == acquisition["reused_history_hash"], "REUSED_SOURCE_CHANGED")
    snapshot = qualified["snapshot"]
    require(stored["plan_hash"] == base.digest(acquisition), "ACQUISITION_HISTORY_PLAN_CHANGED")
    require(base.digest(stored) == snapshot["acquisition_history_hash"], "ACQUISITION_HISTORY_CHANGED")
    points = snapshot["rows"]
    require(points == stored["rows"] and snapshot["receipts"] == stored["receipts"], "PARSED_POINTS_CHANGED")
    expected_days = set(acquisition["expected_days"])
    require(len(points) == 1383 and set(points) == expected_days, "HISTORY_COVERAGE_CHANGED")
    require(set(snapshot["raw_hashes"]) == set(snapshot["coverage"]) == expected_days, "PROOF_COVERAGE_CHANGED")
    query_by_day = {q["date"]: q for q in acquisition["queries"]}
    require(
        set(query_by_day).isdisjoint(seed["sources"]) and set(query_by_day) | set(seed["sources"]) == expected_days,
        "REQUEST_SCOPE_CHANGED",
    )
    quote_history = base.read(quotes.root() / "qualified-history.json")
    require(
        base.digest(quote_history) == p["quote_history_hash"] == snapshot["quote_history_hash"], "QUOTE_HISTORY_CHANGED"
    )
    quote_seed = base.read(quotes.root() / "reused-history.json")
    for day, value in points.items():
        if day in seed["sources"]:
            spec = seed["sources"][day]
            request = base.read(base.ROOT / spec["request_path"])
            receipt = base.read(base.ROOT / spec["response_path"])
            raw = (base.ROOT / spec["raw_path"]).read_bytes()
            require(
                base.digest(base.read(base.ROOT / spec["plan_path"])) == spec["plan_hash"] == request["plan_hash"],
                "REUSED_PLAN_CHANGED",
            )
            require(
                request["query"] == spec["expected_query"] and base.digest(request) == spec["request_hash"],
                "REUSED_REQUEST_CHANGED",
            )
            require(base.digest(receipt) == spec["response_hash"], "REUSED_RECEIPT_CHANGED")
            require(value == seed["rows"][day], "REUSED_FEATURE_CHANGED")
        else:
            request, receipt, parsed = (
                base.read(root() / sub / (day + ".json")) for sub in ["requests", "responses", "parsed"]
            )
            raw = (root() / "raw" / (day + ".json")).read_bytes()
            require(
                request["query"] == query_by_day[day] and request["plan_hash"] == base.digest(acquisition),
                "REQUEST_CHANGED",
            )
            require(
                parsed["response_hash"] == base.digest(receipt) and parsed["value"] == value, "PARSED_SOURCE_CHANGED"
            )
            require(
                datetime.fromisoformat(receipt["received_at"]) <= datetime.fromisoformat(parsed["at"]),
                "PARSED_TIME_CHANGED",
            )
        require(request["attempt"] == 1 and receipt["request_hash"] == base.digest(request), "RECEIPT_REQUEST_CHANGED")
        require(receipt["http_status"] == 200 and receipt["bytes"] == len(raw), "RESPONSE_STATUS_OR_SIZE")
        require(hashlib.sha256(raw).hexdigest() == receipt["sha256"] == snapshot["raw_hashes"][day], "RAW_CHANGED")
        require(base.digest(receipt) == snapshot["receipts"][day], "RECEIPT_CHANGED")
        require(
            datetime.fromisoformat(request["at"])
            <= datetime.fromisoformat(receipt["received_at"])
            <= datetime.fromisoformat(qualified["at"]),
            "SOURCE_TIME_CHANGED",
        )
        coverage = snapshot["coverage"][day]
        require(
            coverage["quote_count_coverage"] >= p["minimum_quote_count_coverage"]
            and coverage["quote_amount_coverage"] >= p["minimum_quote_amount_coverage"],
            "QUOTE_COVERAGE_INSUFFICIENT",
        )
        qpath = (
            base.ROOT / quote_seed["sources"][day]["raw_path"]
            if day in quote_seed["sources"]
            else quotes.root() / "raw" / (day + ".json")
        )
        require(quotes.sha(qpath) == coverage["quote_raw_sha256"], "QUOTE_RAW_CHANGED")
        validate_feature(value, day)
    return qualified


def extend(market, t, u, points):
    """日期必须相邻，仅取T；完整历史不足或当天缺失时保持缺失，不跨日填充。"""
    days = list(map(str, base.calendar()[0]))
    require(t in days and u in days and days.index(u) == days.index(t) + 1, "ADJACENCY_INVALID")
    row = points.get(t)
    if row is None:
        value = {"date": t, "available": False}
    else:
        validate_feature(row, t)
        value = dict(row)
    return market | {"stock_moneyflow": value}


def original_row(row):
    return row | {"market": {k: v for k, v in row["market"].items() if k != "stock_moneyflow"}}
