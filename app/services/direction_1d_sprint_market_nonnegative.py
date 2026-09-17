"""第81轮：股票/混合组市场幅度模型的非负权重约束。

一个无截距非负逻辑回归候选与五个对照使用同题、同一SPX回退；只用当时成熟的标签。
"""

import hashlib
import shutil
from collections import defaultdict

import joblib
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from threadpoolctl import threadpool_limits

from app.services import direction_1d_sprint as base
from app.services import direction_1d_sprint_market_current_refresh as current
from app.services import direction_1d_sprint_market_nasdaq_relative as previous
from app.services import direction_1d_sprint_market_only as baseline
from app.services import direction_1d_sprint_return_target as regression
from app.services import direction_1d_sprint_sequence as sequence

CANDIDATES = ("NONNEG_EQUITY_MIXED_RAW3_LR504",)
CONSTRAINED_GROUPS = ("CN_EQUITY", "CN_MIXED")
data = current
CONTROLS = (
    "CURRENT_MATCHED_MARKET_LR3",
    "CURRENT_MATCHED_MARKET_SIGN_LR3",
    "MARKET_MAJORITY3",
    "SPX_SIGN",
    "ALWAYS_UP",
)
BRANCHES = CANDIDATES + CONTROLS

FIRST_TARGET = "2026-09-16"
PROPOSAL_HASH = "e0dcefd85fad7c656768e00de40d1f28fef4053a77ff8452e551f11e035fa606"
active = regression.active


def root():
    return base.ROOT / "round-81"


def fingerprint():
    value = previous.fingerprint()
    for name in (
        "app/services/direction_1d_sprint_market_nonnegative.py",
        "scripts/direction_1d_sprint_market_nonnegative.py",
        "tests/test_direction_1d_sprint_market_nonnegative.py",
    ):
        value["code"][name] = hashlib.sha256((base.PROJECT / name).read_bytes()).hexdigest()
    return value


def control_path(name, group, label):
    if name not in CONTROLS[:2]:
        raise ValueError("MARKET_NASDAQ_RELATIVE_CONTROL_INVALID")
    service = current if label == "current" else baseline
    original = service.CANDIDATES[CONTROLS.index(name)]
    return service.root() / "checkpoints" / f"{label}-{group}-{original}.joblib", original


def validate_spec(proposal, scope_size):
    if (
        proposal["candidates"] != list(CANDIDATES)
        or proposal["controls"] != list(CONTROLS)
        or proposal["branch_count"] != len(BRANCHES)
        or len(set(BRANCHES)) != len(BRANCHES)
        or proposal["preflight_branch_checks"] != len(BRANCHES) * scope_size
        or proposal["max_development_fits"] != 8
        or proposal["max_current_fits"] != 2
        or proposal["reproductions"] != 1
        or proposal["current_cutoff"] != "2026-09-16"
        or proposal["constrained_groups"] != list(CONSTRAINED_GROUPS)
        or proposal["bond_policy"] != "REUSE_SAME_CUTOFF_RAW_CONTROL"
    ):
        raise ValueError("MARKET_NASDAQ_RELATIVE_DECLARED_BRANCHES_OR_BUDGET_CHANGED")


def verify_inputs(p):
    for path, digest in p["input_hashes"].items():
        if base.digest(base.read(base.ROOT / path)) != digest:
            raise ValueError("MARKET_ONLY_FROZEN_INPUT_CHANGED")
    if base.digest(current.data.scope()) != p["scope_hash"]:
        raise ValueError("MARKET_ONLY_SCOPE_CHANGED")


def plan():
    path = root() / "plan.json"
    proposal = base.read(root() / "proposal-before-implementation.json")
    validate_spec(proposal, len(current.data.scope()))
    if (
        base.digest(proposal) != PROPOSAL_HASH
        or hashlib.sha256((root() / "design-before-implementation.md").read_bytes()).hexdigest()
        != proposal["design_sha256"]
    ):
        raise ValueError("MARKET_ONLY_PROPOSAL_CHANGED")
    if path.exists():
        p = base.read(path)
        if p["fingerprint"] != fingerprint() or p["calendar_hash"] != base.calendar()[1]:
            raise ValueError("MARKET_ONLY_CODE_OR_CALENDAR_CHANGED")
        verify_inputs(p)
        return p
    active()
    if base.digest(previous.models()[0]) != proposal["r80_result_hash"]:
        raise ValueError("MARKET_GAP_DELTA_PARENT_RESULT_CHANGED")
    if (
        base.digest(baseline.models()[0]) != proposal["r70_result_hash"]
        or base.digest(current.models()[0]) != proposal["r79_result_hash"]
        or base.digest(base.read(base.ROOT / "fresh-current-market-feasibility-v1/rows.json"))
        != proposal["fresh_rows_hash"]
        or base.digest(base.read(root() / "coefficient-diagnostic-before-proposal.json")) != proposal["diagnostic_hash"]
    ):
        raise ValueError("NONNEG_PROPOSED_SOURCE_CHANGED")
    groups = sorted({f["group"] for f in current.data.scope()})
    paths = [
        "history.json",
        "round-81/proposal-before-implementation.json",
        "round-81/coefficient-diagnostic-before-proposal.json",
        "fresh-current-market-feasibility-v1/plan.json",
        "fresh-current-market-feasibility-v1/result.json",
        "fresh-current-market-feasibility-v1/rows.json",
        "round-70/result.json",
        "round-79/result.json",
        "round-80/result.json",
    ]
    for name in CONTROLS[:2]:
        for group in groups:
            for label in ("1", "2", "3", "4", "current"):
                paths.append(control_path(name, group, label)[0].with_suffix(".json").relative_to(base.ROOT).as_posix())
    p = {
        "at": base.now().isoformat(),
        "round": 81,
        "fingerprint": fingerprint(),
        "calendar_hash": base.calendar()[1],
        "scope_hash": base.digest(current.data.scope()),
        "input_hashes": {name: base.digest(base.read(base.ROOT / name)) for name in paths},
        "proposal_hash": PROPOSAL_HASH,
        "candidates": list(CANDIDATES),
        "controls": list(CONTROLS),
        "current_fit_cutoff": proposal["current_cutoff"],
        "first_forward_target": FIRST_TARGET,
        "development_fits": 8,
        "current_fits": 2,
        "reproductions": 1,
        "expected_2025_questions": 7290,
        "branch_count": len(BRANCHES),
        "preflight_branch_checks": len(BRANCHES) * len(current.data.scope()),
        "expected_2025_dates": 243,
        "old_questions": 5670,
        "training": ("Same504mature marketdates/group; natural family/date weights; LR C.1/nointercept/max1500/seed17"),
        "transform": "raw3/std; nonnegative coefficients for equity/mixed only, bonds reuse rawbaseline; threshold.5",
        "fallback": "SPX_SIGN when fullmarket unavailable; ALWAYS_UP remains fixedUP on every row",
        "label_maturity": "max(nextCNafterU,Tann,Uann) strictlybeforecutoff",
        "source_timing": (
            "Historical first publication unverified; future requires saved input and readback before U08:30"
        ),
        "new_2026_scores": False,
        "new_cost_cny": 0,
        "status": "MODEL_NOT_RELEASED",
    }
    base.save(path, p)
    for name in p["fingerprint"]["code"]:
        dest = root() / "code" / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(base.PROJECT / name, dest)
    return p


market_vector = baseline.market_vector
training_rows = baseline.training_rows


class NonnegativeLogit:
    """可复核的三维研究头；不输出正式校准概率，系数在股票/混合组受非负约束。"""

    fit_intercept = False
    n_features_in_ = 3

    def __init__(self, coefficient):
        self.coef_ = np.asarray(coefficient, dtype=float).reshape(1, 3)
        self.classes_ = np.asarray([0, 1])

    def predict_proba(self, x):
        positive = expit(np.asarray(x) @ self.coef_[0])
        return np.column_stack((1 - positive, positive))


def objective(coef, x, y, weights):
    """与原C0.1配方相同的加权逻辑损失/L2惩罚；除权重和只改善数值尺度。"""
    logits = x @ coef
    total = weights.sum()
    value = (np.dot(weights, np.logaddexp(0, logits) - y * logits) + np.dot(coef, coef) / (2 * 0.1)) / total
    gradient = (x.T @ (weights * (expit(logits) - y)) + coef / 0.1) / total
    return float(value), gradient


def solve(x, y, weights):
    """单次确定性有界求解，并验证非负约束下的一阶最优条件，不自动换参数重试。"""
    with threadpool_limits(limits=2):
        result = minimize(
            objective,
            np.zeros(3),
            args=(x, y, weights),
            method="L-BFGS-B",
            jac=True,
            bounds=[(0, None)] * 3,
            options={"maxiter": 2000, "ftol": 1e-14, "gtol": 1e-9},
        )
    coef = result.x
    loss, gradient = objective(coef, x, y, weights)
    projected = np.where((coef <= 1e-10) & (gradient > 0), 0, gradient)
    residual = float(np.max(np.abs(projected)))
    if not result.success or not np.isfinite(coef).all() or min(coef) < 0 or not np.isfinite(loss) or residual > 1e-6:
        raise ValueError("NONNEG_OPTIMIZATION_NOT_VERIFIED")
    return coef, {
        "success": bool(result.success),
        "iterations": int(result.nit),
        "loss": loss,
        "projected_kkt_residual": residual,
    }


def fit(rows, name, cutoff):
    """只拟合股票和混合组；债券头在调用处完整复用，不消耗拟合预算。"""
    if name not in CANDIDATES or {r["group"] for r in rows} not in [{g} for g in CONSTRAINED_GROUPS]:
        raise ValueError("NONNEG_RECIPE_OR_GROUP_INVALID")
    chosen = training_rows(rows, cutoff)
    x = np.asarray([market_vector(r["market"]) for r in chosen])
    y, weights = np.asarray([r["y"] for r in chosen]), regression.weights(chosen)
    _, mean, scale = sequence.normalize_training(x, weights)
    active()
    coefficient, convergence = solve(x / np.asarray(scale), y, weights)
    active()
    return {
        "model": NonnegativeLogit(coefficient),
        "mean": [0.0] * 3,
        "scale": scale,
        "signed": False,
        "constraint": "NONNEGATIVE",
        "convergence": convergence,
        "fit_hash": base.digest(chosen),
        "weight_hash": base.digest(weights.tolist()),
        "fit_rows": len(chosen),
        "fit_dates": len({r["u"] for r in chosen}),
        "fit_end": max(r["u"] for r in chosen),
        "max_mature_date": max(r["mature"] for r in chosen),
        "cutoff": cutoff,
        "training_feature_mean_not_subtracted": mean,
        "weighted_up_rate": float(np.average(y, weights=weights)),
    }


def candidate_head(rows, name, cutoff, label, group, ph):
    if group == "CN_BOND":
        return frozen_head(CONTROLS[0], group, label, cutoff) | {"constraint": "BOND_REUSED_RAW_CONTROL"}
    return fit_checkpoint(rows, name, cutoff, f"{label}-{group}", ph)


def fit_checkpoint(rows, name, cutoff, label, plan_hash):
    path = root() / "checkpoints" / f"{label}-{name}.joblib"
    receipt, attempt = path.with_suffix(".json"), path.with_suffix(".attempt.json")
    if receipt.exists():
        meta = base.read(receipt)
        if (
            meta["plan_hash"] != plan_hash
            or meta["cutoff"] != cutoff
            or meta["name"] != name
            or meta["sha256"] != hashlib.sha256(path.read_bytes()).hexdigest()
        ):
            raise ValueError("MARKET_ONLY_CHECKPOINT_CHANGED")
        return joblib.load(path)
    if attempt.exists():
        raise ValueError("MARKET_ONLY_FIT_ALREADY_ATTEMPTED")
    base.save(attempt, {"at": base.now().isoformat(), "name": name, "cutoff": cutoff, "plan_hash": plan_hash})
    head = fit(rows, name, cutoff)
    joblib.dump(head, path)
    base.save(
        receipt,
        {
            "at": base.now().isoformat(),
            "name": name,
            "cutoff": cutoff,
            "plan_hash": plan_hash,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        },
    )
    base.save(root() / "training" / f"{label}-{name}.json", {k: v for k, v in head.items() if k != "model"})
    return head


def frozen_head(name, group, label, cutoff):
    """历史对照加载第70轮，当前对照加载第79轮；保持与候选相同的训练数据截止。"""
    path, original = control_path(name, group, label)
    meta = base.read(path.with_suffix(".json"))
    if (
        meta["name"] != original
        or meta["cutoff"] != cutoff
        or meta["sha256"] != hashlib.sha256(path.read_bytes()).hexdigest()
    ):
        raise ValueError("MARKET_NASDAQ_RELATIVE_FROZEN_CONTROL_CHANGED")
    head = joblib.load(path)
    if head["cutoff"] != cutoff or head["mean"] != [0.0] * 3:
        raise ValueError("MARKET_NASDAQ_RELATIVE_FROZEN_CONTROL_SCHEMA_INVALID")
    return head


def batch_answers(values, name, head=None):
    if name not in BRANCHES:
        raise ValueError("NONNEG_BRANCH_INVALID")
    if name in CANDIDATES:
        original = baseline.CANDIDATES[0]
        if any(v["available"] for v in values):
            if head is None or head.get("constraint") not in ("NONNEGATIVE", "BOND_REUSED_RAW_CONTROL"):
                raise ValueError("NONNEG_HEAD_INVALID")
            if head["constraint"] == "NONNEGATIVE" and (
                np.min(head["model"].coef_) < 0 or not np.isfinite(head["model"].coef_).all()
            ):
                raise ValueError("NONNEG_COEFFICIENT_INVALID")
    else:
        original = baseline.CANDIDATES[CONTROLS.index(name)] if name in CONTROLS[:2] else name
    return baseline.batch_answers(values, original, head)


def answers(market, group, bundle):
    return {n: batch_answers([market], n, bundle.get(n, {}).get(group))[0] for n in BRANCHES}


def train():
    p = plan()
    if (root() / "result.json").exists():
        return models()[0]
    rows, proof = data.dataset()
    if not (root() / "training-question-proof.json").exists():
        base.save(root() / "training-question-proof.json", proof)
    groups, output = sorted({r["group"] for r in rows}), defaultdict(list)
    ph = base.digest(p)
    for q in range(1, 5):
        active()
        start, end = f"2025-{q * 3 - 2:02d}-01", "2026-01-01" if q == 4 else f"2025-{q * 3 + 1:02d}-01"
        for group in groups:
            group_rows = [r for r in rows if r["group"] == group]
            exam = [r for r in group_rows if start <= r["u"] < end]
            for name in BRANCHES:
                head = (
                    candidate_head(group_rows, name, start, str(q), group, ph)
                    if name in CANDIDATES
                    else frozen_head(name, group, str(q), start)
                    if name in CONTROLS[:2]
                    else None
                )
                scored = [
                    {k: r[k] for k in ("code", "family", "group", "t", "u", "y", "actual_direction", "old_question")}
                    | a
                    for r, a in zip(exam, batch_answers([r["market"] for r in exam], name, head), strict=True)
                ]
                path = root() / "folds" / f"{q}-{group}-{name}.json"
                if path.exists():
                    if base.read(path) != scored:
                        raise ValueError("MARKET_ONLY_RECOMPUTED_ANSWERS_CHANGED")
                else:
                    base.save(path, scored)
                output[name].extend(scored)
        base.save(root() / "progress.json", {"at": base.now().isoformat(), "quarter": q}, replace=True)
    if set(output) != set(BRANCHES):
        raise ValueError("MARKET_NASDAQ_RELATIVE_COMPLETED_BRANCHES_MISSING")
    expected = sorted((r["code"], r["u"], r["y"]) for r in output["SPX_SIGN"])
    if (
        len(expected) != 7290
        or len({d for _, d, _ in expected}) != 243
        or any(sorted((r["code"], r["u"], r["y"]) for r in v) != expected for v in output.values())
    ):
        raise ValueError("MARKET_ONLY_COMMON_EXAM_CHANGED")
    metrics = {
        subset: {
            n: base.metrics([r for r in v if subset == "all" or r["old_question"] == (subset == "old")])
            for n, v in output.items()
        }
        for subset in ("all", "old", "added")
    }
    bundle = {}
    for name in CANDIDATES + CONTROLS[:2]:
        bundle[name] = {}
        for group in groups:
            active()
            bundle[name][group] = (
                candidate_head(
                    [r for r in rows if r["group"] == group], name, p["current_fit_cutoff"], "current", group, ph
                )
                if name in CANDIDATES
                else frozen_head(name, group, "current", p["current_fit_cutoff"])
            )
    path = root() / "models.joblib"
    joblib.dump(bundle, path)
    result = {
        "at": base.now().isoformat(),
        "metrics": metrics,
        "plan_hash": ph,
        "model_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "fingerprint": fingerprint(),
        "calendar_hash": p["calendar_hash"],
        "development_fits": 8,
        "current_fits": 2,
        "winner": max(CANDIDATES, key=lambda n: metrics["all"][n]["accuracy"]),
        "monthly_metrics": {
            n: {
                f"2025-{m:02d}": base.metrics([r for r in v if r["u"].startswith(f"2025-{m:02d}")])
                for m in range(1, 13)
            }
            for n, v in output.items()
        },
        "group_metrics": {
            n: {g: base.metrics([r for r in v if r["group"] == g]) for g in groups} for n, v in output.items()
        },
        "kind": "HISTORICAL_DEVELOPMENT_ONLY",
        "new_2026_scores": False,
        "new_cost_cny": 0,
        "source_limit": (
            "Historical first publication unverified; "
            "old59.6387 uses different HKfallback, not a same-question comparator"
        ),
        "status": "MODEL_NOT_RELEASED",
    }
    base.save(root() / "result.json", result)
    return result


def models():
    p, result = plan(), base.read(root() / "result.json")
    path = root() / "models.joblib"
    if (
        result["plan_hash"] != base.digest(p)
        or result["fingerprint"] != fingerprint()
        or result["model_sha256"] != hashlib.sha256(path.read_bytes()).hexdigest()
    ):
        raise ValueError("MARKET_ONLY_MODEL_OR_PLAN_CHANGED")
    return result, joblib.load(path)


def preflight():
    _, bundle = models()
    rows, _ = data.dataset()
    market = next(r["market"] for r in reversed(rows) if r["market"]["available"])
    for fund in current.data.scope():
        answers(market, fund["group"], bundle)
    value = {
        "at": base.now().isoformat(),
        "kind": "DRY_RUN_NOT_FORWARD",
        "branch_checks": len(current.data.scope()) * len(BRANCHES),
    }
    base.save(root() / "preflight.json", value)
    return value


def tick():
    from app.services import direction_1d_sprint_market_child_forward_v2 as forward
    from app.services import direction_1d_sprint_market_nonnegative as service

    return forward.tick(service)


def report():
    from app.services import direction_1d_sprint_market_child_forward_v2 as forward
    from app.services import direction_1d_sprint_market_nonnegative as service

    return forward.report(service)


# 该约束实验只使用原三项市场来源，不等待纳指输入。
live_market = current.live_market
live_answers = answers
