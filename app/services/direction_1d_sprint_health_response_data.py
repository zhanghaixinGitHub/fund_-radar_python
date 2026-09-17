"""个体医疗反应的来源重建与未来冻结上下文；旧标签只用于过去统计，不改变预测标签。"""

import hashlib
from collections import defaultdict
from datetime import datetime

from app.services import direction_1d_sprint as b
from app.services import direction_1d_sprint_china_health_live as live_source
from app.services import direction_1d_sprint_health_response_context as helper
from app.services import direction_1d_sprint_market_china_health as source


def root():
    return b.ROOT / "china-health-response-feasibility-v1"


def require(condition, reason):
    if not condition:
        raise ValueError("HEALTH_RESPONSE_DATA_" + reason)


def snapshot():
    """核验固定收缩方案、来源身份、完整原型摘要；不访问网络。"""
    p, result, saved = (b.read(root() / n) for n in ("plan.json", "result.json", "features.json"))
    require(
        result["plan_hash"] == saved["plan_hash"] == b.digest(p)
        and result["features_hash"] == b.digest(saved)
        and result["status"] == "INPUT_FEASIBLE_NOT_TRAINED",
        "PROTOTYPE_CHANGED",
    )
    require(
        p["lookback"] == 126
        and p["min_observations"] == 63
        and p["variance_ridge"] == 0.25
        and p["shrinkage_pseudocount"] == 126
        and p["beta_clip"] == 3
        and p["current_context_cutoff"] == "2026-09-16",
        "SPEC_CHANGED",
    )
    for name, expected in p["code_hashes"].items():
        require(hashlib.sha256((b.PROJECT / name).read_bytes()).hexdigest() == expected, "PROTOTYPE_CODE_CHANGED")
    require(
        hashlib.sha256((root() / "design-before-analysis.md").read_bytes()).hexdigest() == p["design_sha256"],
        "DESIGN_CHANGED",
    )
    require(
        b.digest(b.read(source.root() / "plan.json")) == p["source_model_plan_hash"]
        and b.digest(b.read(source.root() / "source-identity.json")) == p["source_identity_hash"]
        and b.digest(b.read(source.data.root() / "qualification-result.json")) == p["source_qualification_hash"],
        "SOURCE_CHANGED",
    )
    return p, result, saved


def extend(market, t, u, points, code, contexts):
    """逐基金选择自己的冻结上下文；同一天同组基金也可有不同输入。"""
    require(code in contexts, "FUND_CONTEXT_MISSING")
    prior = contexts[code]
    health = source.data.features(t, u, points)
    value = helper.feature(health, prior, u)
    return market | {
        "health_response": value
        | {
            "new_us_dates": health["new_us_dates"],
            "max_prior_mature": prior["max_mature"],
            "context_cutoff": prior["cutoff"],
        }
    }


def original_row(row):
    return row | {"market": {k: v for k, v in row["market"].items() if k != "health_response"}}


def reconstruct():
    """从旧模型原始数据逐题重算过去统计，对齐冻结原型，当前或未来标签不得参与。"""
    _, result, saved = snapshot()
    rows, proof = source.reconstruct_dataset()
    source.verify_identity(rows, proof, b.read(source.root() / "source-identity.json")["identity"])
    require(source.question_identity(rows, proof) == saved["source_identity"], "ORIGINAL_QUESTION_CHANGED")
    by_code = defaultdict(list)
    for row in rows:
        by_code[row["code"]].append(row)
    output = []
    for row, frozen in zip(rows, saved["rows"], strict=True):
        prior = helper.context(by_code[row["code"]], row["u"])
        value = helper.feature(row["market"]["china_health"], prior, row["u"])
        require(
            frozen
            == {k: row[k] for k in ("code", "group", "t", "u")}
            | {"extra": value, "context": prior, "base_available": row["market"]["available"]},
            "CONTEXT_CHANGED",
        )
        original = source.data.original_row(row)
        output.append(
            original
            | {
                "market": original["market"]
                | {
                    "health_response": value
                    | {
                        "new_us_dates": row["market"]["china_health"]["new_us_dates"],
                        "max_prior_mature": prior["max_mature"],
                        "context_cutoff": prior["cutoff"],
                    }
                }
            }
        )
    current = {code: helper.context(values, "2026-09-16") for code, values in by_code.items()}
    require(current == saved["current_contexts"], "CURRENT_CONTEXT_CHANGED")
    require(len(output) == 38979 and len(current) == 30, "UNIVERSE_CHANGED")
    return output, proof | {"health_response_features_hash": result["features_hash"]}


def current():
    """未来只读本轮训练前保存的30个小快照，绑定大原型摘要，不用未来标签动态更新。"""
    proposal = b.read(b.ROOT / "round-105/proposal-before-implementation.json")
    value = b.read(b.ROOT / "round-105/current-contexts.json")
    require(b.digest(value) == proposal["current_context_hash"], "CURRENT_MANIFEST_CHANGED")
    result = b.read(root() / "result.json")
    require(
        value["source_features_hash"] == result["features_hash"]
        and value["source_plan_hash"] == result["plan_hash"]
        and value["cutoff"] == "2026-09-16",
        "CURRENT_SOURCE_CHANGED",
    )
    require(set(value["contexts"]) == {x["code"] for x in source.original.data.scope()}, "CURRENT_SCOPE_CHANGED")
    for context in value["contexts"].values():
        helper.feature({"available": False, "intraday_log_pct": 0}, context, value["cutoff"])
        require(context["cutoff"] == value["cutoff"] and context["count"] == 126, "CURRENT_WINDOW_CHANGED")
    return value


def live(t, u, parent_plan_hash):
    """复用R104实际KURE响应，附加更早冻结的个体上下文，新增供应商请求数为0。"""
    actual = live_source.load_live(t, u, parent_plan_hash)
    frozen = current()
    require(
        datetime.fromisoformat(frozen["at"]) <= datetime.fromisoformat(actual["at"]) and frozen["cutoff"] <= t,
        "CURRENT_CONTEXT_AFTER_CAPTURE",
    )
    return {
        "at": actual["at"],
        "t": t,
        "u": u,
        "health_source_hash": b.digest(actual),
        "context_manifest_hash": b.digest(frozen),
        "contexts": frozen["contexts"],
        "points": actual["points"],
        "available": actual["available"],
        "new_source_requests": 0,
    }
