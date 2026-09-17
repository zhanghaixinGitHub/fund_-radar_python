"""R122个体上下文绑定：历史逐题重建，前向仅用截止16日已冻结的成熟统计。"""

from collections import defaultdict

from app.services import direction_1d_sprint as b
from app.services import direction_1d_sprint_core_forward as core
from app.services import direction_1d_sprint_fund_moneyflow_context as context
from app.services import direction_1d_sprint_market_stock_moneyflow_v2 as parent

scope = parent.data.scope
KEYS = parent.data.KEYS
CONTEXT_PLAN_HASH = "4971f2d530f72db757c9b1974a59b70fec90c48a1dc4331e2493a6831a1e8c8f"


def root():
    return b.ROOT / "round-122"


def require(ok, reason):
    if not ok:
        raise ValueError("FUND_MONEYFLOW_DATA_" + reason)


def reference():
    """验证生成公式、完整资格与小型当前上下文的同一来源身份；不重新读取当前标签。"""
    plan = b.read(root() / "context-plan.json")
    require(b.digest(plan) == CONTEXT_PLAN_HASH, "CONTEXT_PLAN_CHANGED")
    for name, expected in plan["code_hashes"].items():
        require(core.sha(b.PROJECT / name) == expected, "CONTEXT_CODE_CHANGED")
    result, binding, current = (
        b.read(root() / name)
        for name in ("context-feasibility.json", "context-reference.json", "current-contexts.json")
    )
    require(result["plan_hash"] == b.digest(plan), "QUALIFICATION_PLAN_CHANGED")
    require(binding["feasibility_hash"] == b.digest(result), "QUALIFICATION_CHANGED")
    require(binding["context_snapshot_hash"] == result["snapshot_hash"], "SNAPSHOT_BINDING_CHANGED")
    require(binding["current_contexts_hash"] == b.digest(current), "CURRENT_CONTEXTS_CHANGED")
    require(current["context_snapshot_hash"] == result["snapshot_hash"], "CURRENT_SNAPSHOT_CHANGED")
    require(current["cutoff"] == plan["current_context_cutoff"] == "2026-09-16", "CONTEXT_CUTOFF_CHANGED")
    require(set(current["contexts"]) == {f["code"] for f in scope()}, "CONTEXT_SCOPE_CHANGED")
    for prior in current["contexts"].values():
        context.validate(prior, current["cutoff"])
    return plan, result, binding, current


def original_row(row):
    """同时移除资金流与个体派生项，恢复R82原题字节口径以核验相同训练样本和权重。"""
    return parent.data.original_row(row | {"market": {k: v for k, v in row["market"].items() if k != "fund_moneyflow"}})


def feature(code, u, prior, point):
    """来源缺失保持显式不可用；零历史是有意义状态，不等于丢弃基金。"""
    extra = context.extra_features(prior, u, point)
    return {"code": code, "cutoff": u, "available": extra is not None, "prior": prior, "extra": extra}


def validate_feature(value, point):
    require(isinstance(value["code"], str) and bool(value["code"]), "FUND_CODE_MISSING")
    context.validate(value["prior"], value["cutoff"])
    expected = context.extra_features(value["prior"], value["cutoff"], point)
    require(
        type(value["available"]) is bool and value["available"] == (expected is not None),
        "FEATURE_AVAILABILITY_CHANGED",
    )
    require(value["extra"] == expected, "FEATURE_FORMULA_CHANGED")


def reconstruct():
    """从已验收父题逐基金/逐目标重算上下文，与拟合前快照逐项比较，当前标签不读入特征。"""
    _, qualified, binding, current = reference()
    rows, identity = parent.dataset()
    require(identity == qualified["parent_row_identity"], "PARENT_ROWS_CHANGED")
    snapshot = b.read(root() / "context-snapshot.json")
    require(b.digest(snapshot) == binding["context_snapshot_hash"], "CONTEXT_SNAPSHOT_CHANGED")
    points, grouped = {}, defaultdict(list)
    for row in rows:
        value = row["market"]["stock_moneyflow"]
        require(row["t"] not in points or points[row["t"]] == value, "SHARED_SOURCE_CHANGED")
        points[row["t"]] = value
        grouped[row["code"]].append(row)
    derived, output = [], []
    for row in rows:
        prior = context.context(grouped[row["code"]], row["u"], points)
        point = points.get(row["t"])
        extra = context.extra_features(prior, row["u"], point)
        derived.append({"code": row["code"], "u": row["u"], "context": prior, "extra": extra})
        output.append(
            row | {"market": row["market"] | {"fund_moneyflow": feature(row["code"], row["u"], prior, point)}}
        )
    require(derived == snapshot["rows"], "HISTORICAL_CONTEXT_RECONSTRUCTION_CHANGED")
    recomputed = {code: context.context(values, current["cutoff"], points) for code, values in grouped.items()}
    require(recomputed == current["contexts"] == snapshot["current_contexts"], "CURRENT_RECONSTRUCTION_CHANGED")
    require(
        [original_row(row) for row in output] == [parent.data.original_row(row) for row in rows],
        "ORIGINAL_ROWS_CHANGED",
    )
    proof = {
        "at": b.now().isoformat(),
        "parent_identity": identity,
        "context_reference_hash": b.digest(binding),
        "context_snapshot_hash": b.digest(snapshot),
        "current_contexts_hash": b.digest(current),
    }
    return output, proof


def history():
    """仅供离线试推理；股票资金流仍按原历史证据验证，当前上下文由独立小文件绑定。"""
    _, _, binding, current = reference()
    result = parent.data.history()
    return result | {"context_reference_hash": b.digest(binding), "current_contexts_hash": b.digest(current)}


def extend(market, t, u, points, *, code, frozen_current=None):
    """实际前向通过明确基金代码选择冻结统计；当天资金流只能由调用方的实际来源传入。

    frozen_current是本次已经reference验证的完整小快照；若不传则现场验证。
    截止16日统计可用于16日离线试推理和17日真实预测，但不得回用于更早日期。
    """
    current = reference()[3] if frozen_current is None else frozen_current
    binding = b.read(root() / "context-reference.json")
    require(b.digest(current) == binding["current_contexts_hash"], "CURRENT_CONTEXTS_CHANGED")
    require(current["cutoff"] == "2026-09-16" and current["cutoff"] <= u, "CURRENT_CONTEXT_USED_TOO_EARLY")
    require(code in current["contexts"], "UNKNOWN_FUND")
    point_market = parent.data.extend(market, t, u, points)
    prior = current["contexts"][code]
    context.validate(prior, current["cutoff"])
    return point_market | {"fund_moneyflow": feature(code, u, prior, point_market["stock_moneyflow"])}
