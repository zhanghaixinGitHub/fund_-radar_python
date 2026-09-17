"""逐基金IF反应的历史重建与未来冻结上下文；复用R96输入和R95实际响应。"""

from collections import defaultdict
from datetime import datetime

from app.services import direction_1d_sprint as b
from app.services import direction_1d_sprint_fund_if_response_context as helper
from app.services import direction_1d_sprint_market_futures_intraday as source

scope = source.data.scope


def root():
    return b.ROOT / "fund-if-response-feasibility-v1"


def require(condition, reason):
    if not condition:
        raise ValueError("FUND_IF_DATA_" + reason)


def snapshot():
    p, result, saved = (b.read(root() / n) for n in ("plan.json", "result.json", "features.json"))
    require(result["plan_hash"] == b.digest(p) and result["features_hash"] == b.digest(saved), "PROTOTYPE_CHANGED")
    require(result["status"] == "FUND_IF_RESPONSE_INPUT_FEASIBLE_NOT_TRAINED", "PROTOTYPE_STATUS_CHANGED")
    require(
        p["parameters"] == {"window": 126, "minimum": 63, "ridge": 0.25, "shrink": 126, "beta_clip": 3},
        "RECIPE_CHANGED",
    )
    for name, expected in p["code"].items():
        require(source.core.sha(b.PROJECT / name) == expected, "PROTOTYPE_CODE_CHANGED")
    require(source.core.sha(root() / "design-before-analysis.md") == p["design_sha256"], "DESIGN_CHANGED")
    require(b.digest(b.read(source.root() / "plan.json")) == p["source_plan_hash"], "SOURCE_PLAN_CHANGED")
    require(saved["current_cutoff"] == "2026-09-16", "CURRENT_CUTOFF_CHANGED")
    return p, result, saved


def original_row(row):
    """原题摘要去除新增反应及已知IF字段，恢复R82的原样本与权重核验口径。"""
    return row | {
        "market": {k: v for k, v in row["market"].items() if k not in ("futures_intraday", "fund_if_response")}
    }


def reconstruct():
    """逐基金按时间重算所有历史上下文，与已冻结原型逐项比较，不借用验证期答案。"""
    _, result, saved = snapshot()
    rows, proof = source.reconstruct_dataset()
    source.verify_identity(rows, proof, b.read(source.root() / "source-identity.json")["identity"])
    require(source.question_identity(rows, proof) == saved["source_identity"], "SOURCE_IDENTITY_CHANGED")
    groups = defaultdict(list)
    for row in rows:
        groups[row["code"]].append(row)
    features, contexts = {}, {}
    for code, values in sorted(groups.items()):
        past = []
        for row in sorted(values, key=lambda x: x["u"]):
            context = helper.prior(past, row["u"])
            value = helper.feature(row["market"]["futures_intraday"], context, row["u"])
            features[code + ":" + row["u"]] = value
            past.append(row)
        contexts[code] = helper.prior(values, "2026-09-16")
    require(features == saved["features"] and contexts == saved["current_contexts"], "CONTEXT_REBUILD_CHANGED")
    output = [r | {"market": r["market"] | {"fund_if_response": features[r["code"] + ":" + r["u"]]}} for r in rows]
    require(len(output) == 38979 and b.digest(output) == saved["extended_rows_hash"], "EXTENDED_ROWS_CHANGED")
    return output, proof | {"fund_if_response_features_hash": result["features_hash"]}


def current():
    """推理只读训练前冻结的小快照，不在未来预测时滚入新标签或扫描整个历史。"""
    proposal = b.read(b.ROOT / "round-108/proposal-before-implementation.json")
    value = b.read(b.ROOT / "round-108/current-contexts.json")
    result = b.read(root() / "result.json")
    require(b.digest(value) == proposal["current_context_hash"], "CURRENT_MANIFEST_CHANGED")
    require(
        value["source_features_hash"] == result["features_hash"] and value["source_plan_hash"] == result["plan_hash"],
        "CURRENT_SOURCE_CHANGED",
    )
    require(
        value["cutoff"] == "2026-09-16" and set(value["contexts"]) == {r["code"] for r in scope()},
        "CURRENT_SCOPE_CHANGED",
    )
    for context in value["contexts"].values():
        require(context["cutoff"] == value["cutoff"] and context["count"] == 126, "CURRENT_WINDOW_CHANGED")
        helper.feature({"date": "2026-09-15", "available": False}, context, value["cutoff"])
    return value


def extend(market, t, u, points, code, contexts):
    require(code in contexts, "FUND_CONTEXT_MISSING")
    market = source.data.extend(market, t, u, points)
    return market | {"fund_if_response": helper.feature(market["futures_intraday"], contexts[code], u)}


def in_window(t, u, at):
    return source.data.current.in_window(t, u, at)


def live(t, u, parent_plan_hash):
    """R96原始实际日内快照与更早保存的个体上下文分别绑定，不发新增请求。"""
    require(parent_plan_hash == b.digest(b.read(source.root() / "plan.json")), "LIVE_PARENT_PLAN_CHANGED")
    actual = source.data.live(t, u, b.digest(b.read(b.ROOT / "round-95/plan.json")))
    frozen = current()
    require(
        datetime.fromisoformat(frozen["at"]) <= datetime.fromisoformat(actual["at"]) and frozen["cutoff"] <= t,
        "CONTEXT_AFTER_ACTUAL_CAPTURE",
    )
    return {
        "at": actual["at"],
        "t": t,
        "u": u,
        "parent_plan_hash": parent_plan_hash,
        "parent_source_hash": b.digest(actual),
        "context_manifest_hash": b.digest(frozen),
        "contexts": frozen["contexts"],
        "snapshot": actual["snapshot"],
        "available": actual["available"],
        "new_source_requests": 0,
    }
