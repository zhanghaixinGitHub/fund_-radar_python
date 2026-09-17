"""第77轮：冻结树的反对称推断实验，无新增训练或供应商请求。

负向输入只用于固定的结构约束，不伪称新行情或新的真实标签。
"""

import hashlib
import shutil
from collections import defaultdict

import joblib
import numpy as np

from app.services import direction_1d_sprint as base
from app.services import direction_1d_sprint_market_date_tree as original

CANDIDATES = ("ODD_ROW_ETR3_504", "ODD_DATE_ETR3_504")
CONTROLS = (
    "FROZEN_R76_ROW_ETR3",
    "FROZEN_R76_DATE_ETR3",
    "FROZEN_R70_MARKET_SIGN_LR3",
    "MARKET_MAJORITY3",
    "SPX_SIGN",
    "ALWAYS_UP",
)
BRANCHES = CANDIDATES + CONTROLS
FIRST_TARGET = "2026-09-16"
PROPOSAL_HASH = "0f072ec216072e53c64fd19ae76caea7757a09150c04cd1336de0770e77273d1"
data = original.data
active = original.active


def root():
    return base.ROOT / "round-77"


def fingerprint():
    value = original.fingerprint()
    for name in (
        "app/services/direction_1d_sprint_market_odd_tree.py",
        "scripts/direction_1d_sprint_market_odd_tree.py",
        "tests/test_direction_1d_sprint_market_odd_tree.py",
    ):
        value["code"][name] = hashlib.sha256((base.PROJECT / name).read_bytes()).hexdigest()
    return value


def validate_spec(proposal):
    if (
        proposal["candidates"] != list(CANDIDATES)
        or proposal["controls"] != list(CONTROLS)
        or proposal["branches"] != len(BRANCHES)
        or len(set(BRANCHES)) != len(BRANCHES)
        or proposal["preflight_checks"] != 240
        or proposal["development_fits"] != 0
        or proposal["current_fits"] != 0
        or proposal["reproductions"] != 1
        or proposal["current_cutoff"] != "2026-09-15"
        or proposal["first_target"] != FIRST_TARGET
    ):
        raise ValueError("ODD_TREE_DECLARED_SPEC_CHANGED")


def plan():
    proposal = base.read(root() / "proposal-before-implementation.json")
    validate_spec(proposal)
    if (
        base.digest(proposal) != PROPOSAL_HASH
        or proposal["design_sha256"]
        != hashlib.sha256((root() / "design-before-implementation.md").read_bytes()).hexdigest()
    ):
        raise ValueError("ODD_TREE_PROPOSAL_CHANGED")
    source, _ = original.models()
    if base.digest(source) != proposal["source_result_hash"]:
        raise ValueError("ODD_TREE_PARENT_CHANGED")
    path = root() / "plan.json"
    if path.exists():
        p = base.read(path)
        if (
            p["fingerprint"] != fingerprint()
            or p["calendar_hash"] != base.calendar()[1]
            or p["source_result_hash"] != base.digest(source)
            or p["scope_hash"] != base.digest(data.scope())
        ):
            raise ValueError("ODD_TREE_FROZEN_CONTEXT_CHANGED")
        return p
    active()
    p = {
        "at": base.now().isoformat(),
        "round": 77,
        "proposal_hash": PROPOSAL_HASH,
        "fingerprint": fingerprint(),
        "calendar_hash": base.calendar()[1],
        "scope_hash": base.digest(data.scope()),
        "source_result_hash": base.digest(source),
        "source_model_sha256": source["model_sha256"],
        "candidates": list(CANDIDATES),
        "controls": list(CONTROLS),
        "branch_count": len(BRANCHES),
        "preflight_branch_checks": 240,
        "development_fits": 0,
        "current_fits": 0,
        "reproductions": 1,
        "expected_questions": 7290,
        "expected_dates": 243,
        "current_fit_cutoff": "2026-09-15",
        "first_forward_target": FIRST_TARGET,
        "formula": "0.5 + 0.5 * (g(x) - g(-x))",
        "label": "Original RAW NAV T->adjacentCN U UP/NON_UP;flat remainsNON_UP",
        "threshold": 0.5,
        "model_format": "VERIFIED_ORIGINAL_MODELS_PLUS_FROZEN_ODD_RULE",
        "new_2026_scores": False,
        "new_cost_cny": 0,
        "new_provider_requests": 0,
        "status": "MODEL_NOT_RELEASED",
    }
    base.save(path, p)
    for name in p["fingerprint"]["code"]:
        dest = root() / "code" / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(base.PROJECT / name, dest)
    return p


def original_name(name):
    """明确映射每个新分支，避免以列表切片误把固定规则当成模型头。"""
    mapping = {
        CANDIDATES[0]: original.CANDIDATES[0],
        CANDIDATES[1]: original.CANDIDATES[1],
        CONTROLS[0]: original.CANDIDATES[0],
        CONTROLS[1]: original.CANDIDATES[1],
        CONTROLS[2]: original.CONTROLS[1],
        **{n: n for n in CONTROLS[3:]},
    }
    if name not in mapping:
        raise ValueError("ODD_TREE_BRANCH_INVALID")
    return mapping[name]


def historical_head(name, group, quarter, cutoff):
    native = original_name(name)
    if native == original.CONTROLS[1]:
        return original.frozen_head(native, group, str(quarter), cutoff)
    if native not in original.CANDIDATES:
        return None
    path = original.root() / "checkpoints" / f"{quarter}-{group}-{native}.joblib"
    meta = base.read(path.with_suffix(".json"))
    if (
        meta["name"] != native
        or meta["cutoff"] != cutoff
        or meta["sha256"] != hashlib.sha256(path.read_bytes()).hexdigest()
    ):
        raise ValueError("ODD_TREE_CHECKPOINT_CHANGED")
    head = joblib.load(path)
    if head["cutoff"] != cutoff:
        raise ValueError("ODD_TREE_CHECKPOINT_CUTOFF_CHANGED")
    return head


def odd_scores(model, x):
    """固定反对称规则；零输入得到0.5，分数不是已经校准的真实上涨概率。"""
    x = np.asarray(x, dtype=float)
    if x.ndim != 2 or x.shape[1] != 3 or not np.isfinite(x).all():
        raise ValueError("ODD_TREE_INPUT_INVALID")
    positive, negative = original.tree_scores(model, x), original.tree_scores(model, -x)
    for scores in (positive, negative):
        if not np.isfinite(scores).all() or min(scores) < 0 or max(scores) > 1:
            raise ValueError("ODD_TREE_PARENT_SCORE_INVALID")
    return 0.5 + 0.5 * (positive - negative)


def batch_answers(values, name, head=None):
    native = original_name(name)
    if name in CONTROLS:
        return original.batch_answers(values, native, head)
    if not values:
        return []
    x = np.asarray([original.market_vector(v) for v in values])
    out = [{"prediction": int(row[0] >= 0), "kind": "FIXED_DIRECTION", "route": "SPX_SIGN_FALLBACK"} for row in x]
    indices = [i for i, v in enumerate(values) if v["available"]]
    if indices:
        if (
            head is None
            or head["training_unit"] != original.TRAINING_UNITS[native]
            or head["model"].n_features_in_ != 3
        ):
            raise ValueError("ODD_TREE_HEAD_SCHEMA_CHANGED")
        scores = odd_scores(head["model"], x[indices])
        for i, score in zip(indices, scores, strict=True):
            out[i] = {
                "prediction": int(score >= 0.5),
                "research_score": float(score),
                "kind": "UNCALIBRATED_UP_SCORE",
                "route": name,
            }
    return out


def answers(market, group, bundle):
    return {name: batch_answers([market], name, bundle.get(original_name(name), {}).get(group))[0] for name in BRANCHES}


def manifest_value(p, source):
    return {
        "plan_hash": base.digest(p),
        "source_result_hash": base.digest(source),
        "source_model_sha256": source["model_sha256"],
        "model_format": p["model_format"],
        "new_fits": 0,
        "formula": p["formula"],
        "threshold": p["threshold"],
    }


def prepare():
    """只生成两项新规则的历史开发答案；所有模型均复用原始拟合产物。"""
    p = plan()
    if (root() / "result.json").exists():
        return models()[0]
    rows, proof = data.dataset()
    base.save(root() / "question-proof.json", proof)
    output = defaultdict(list)
    for quarter in range(1, 5):
        start = f"2025-{quarter * 3 - 2:02d}-01"
        end = "2026-01-01" if quarter == 4 else f"2025-{quarter * 3 + 1:02d}-01"
        for group in sorted({r["group"] for r in rows}):
            active()
            exam = [r for r in rows if r["group"] == group and start <= r["u"] < end]
            for name in BRANCHES:
                head = historical_head(name, group, quarter, start)
                choices = batch_answers([r["market"] for r in exam], name, head)
                scored = [
                    {k: r[k] for k in ("code", "family", "group", "t", "u", "y", "actual_direction", "old_question")}
                    | answer
                    for r, answer in zip(exam, choices, strict=True)
                ]
                if name in CONTROLS:
                    old = base.read(original.root() / "folds" / f"{quarter}-{group}-{original_name(name)}.json")
                    if scored != old:
                        raise ValueError("ODD_TREE_CONTROL_ANSWERS_CHANGED")
                base.save(root() / "folds" / f"{quarter}-{group}-{name}.json", scored)
                output[name].extend(scored)
        base.save(root() / "progress.json", {"at": base.now().isoformat(), "quarter": quarter}, replace=True)
    keys = sorted((r["code"], r["u"], r["y"]) for r in output["SPX_SIGN"])
    if (
        set(output) != set(BRANCHES)
        or len(keys) != 7290
        or len({u for _, u, _ in keys}) != 243
        or any(sorted((r["code"], r["u"], r["y"]) for r in values) != keys for values in output.values())
    ):
        raise ValueError("ODD_TREE_COMMON_EXAM_CHANGED")
    source, _ = original.models()
    base.save(root() / "models-manifest.json", manifest_value(p, source))
    metrics = {
        subset: {
            name: base.metrics([r for r in rows if subset == "all" or r["old_question"] == (subset == "old")])
            for name, rows in output.items()
        }
        for subset in ("all", "old", "added")
    }
    result = {
        "at": base.now().isoformat(),
        "plan_hash": base.digest(p),
        "fingerprint": fingerprint(),
        "model_sha256": hashlib.sha256((root() / "models-manifest.json").read_bytes()).hexdigest(),
        "model_format": p["model_format"],
        "metrics": metrics,
        "monthly_metrics": {
            n: {
                f"2025-{m:02d}": base.metrics([r for r in v if r["u"].startswith(f"2025-{m:02d}")])
                for m in range(1, 13)
            }
            for n, v in output.items()
        },
        "group_metrics": {
            n: {g: base.metrics([r for r in v if r["group"] == g]) for g in sorted({r["group"] for r in rows})}
            for n, v in output.items()
        },
        "development_fits": 0,
        "current_fits": 0,
        "winner": max(CANDIDATES, key=lambda n: metrics["all"][n]["accuracy"]),
        "new_2026_scores": False,
        "new_cost_cny": 0,
        "kind": "HISTORICAL_DEVELOPMENT_ONLY",
        "source_limit": "Odd symmetry is an explicit model prior, not a claim about physically inverted markets",
        "status": "MODEL_NOT_RELEASED",
    }
    base.save(root() / "result.json", result)
    return result


def models():
    p = plan()
    result = base.read(root() / "result.json")
    source, bundle = original.models()
    path = root() / "models-manifest.json"
    if (
        result["plan_hash"] != base.digest(p)
        or result["fingerprint"] != fingerprint()
        or result["model_sha256"] != hashlib.sha256(path.read_bytes()).hexdigest()
        or base.read(path) != manifest_value(p, source)
    ):
        raise ValueError("ODD_TREE_MANIFEST_CHANGED")
    return result, bundle


def preflight():
    _, bundle = models()
    rows, _ = data.dataset()
    market = next(r["market"] for r in reversed(rows) if r["market"]["available"])
    for fund in data.scope():
        if set(answers(market, fund["group"], bundle)) != set(BRANCHES):
            raise ValueError("ODD_TREE_PREFLIGHT_BRANCH_MISSING")
    value = {
        "at": base.now().isoformat(),
        "kind": "DRY_RUN_NOT_FORWARD",
        "branch_checks": len(data.scope()) * len(BRANCHES),
    }
    base.save(root() / "preflight.json", value)
    return value


def tick():
    from app.services import direction_1d_sprint_market_child_forward_v2 as runtime
    from app.services import direction_1d_sprint_market_odd_tree as service

    return runtime.tick(service)


def report():
    from app.services import direction_1d_sprint_market_child_forward_v2 as runtime
    from app.services import direction_1d_sprint_market_odd_tree as service

    return runtime.report(service)


live_market = original.live_market
live_answers = answers
