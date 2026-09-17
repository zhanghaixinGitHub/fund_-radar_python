"""第91轮：固定八个训练时间块的平滑最差损失，目标仍为下一交易日RAW净值方向。"""

import hashlib
import shutil
from collections import defaultdict

import joblib
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logsumexp, softmax
from threadpoolctl import threadpool_limits

from app.services import direction_1d_sprint as base
from app.services import direction_1d_sprint_core_forward as core
from app.services import direction_1d_sprint_market_kernel_direction as previous
from app.services import direction_1d_sprint_market_only as baseline
from app.services import direction_1d_sprint_return_target as regression
from app.services import direction_1d_sprint_sequence as sequence

original, data, active = previous.original, previous.data, previous.active
reference = previous.reference
CANDIDATES = ("INTERVAL_RAW3_BLOCK_ROBUST504", "INTERVAL_SIGN3_BLOCK_ROBUST504")
CONTROLS = previous.CONTROLS
BRANCHES = CANDIDATES + CONTROLS
FIRST_TARGET = "2026-09-17"
PROPOSAL_HASH = "9fbe0a9ebac7e7aaffa9748e30568f161a81507e4881f0b86ef4ebf8c20f12c9"


def root():
    return base.ROOT / "round-91"


def require(condition, reason):
    if not condition:
        raise ValueError("BLOCK_ROBUST_" + reason)


def fingerprint():
    # 对全部前版本冻结文件核对一次，再把本轮四个新文件纳入独立冻结清单。
    previous_spec = base.read(core.root() / "plan.json")
    code = dict(previous_spec["code"])
    for name, expected in code.items():
        require(core.sha(base.PROJECT / name) == expected, "OLD_CODE_CHANGED")
    for name in (
        "app/services/direction_1d_sprint_market_block_robust.py",
        "app/services/direction_1d_sprint_block_robust_forward.py",
        "scripts/direction_1d_sprint_market_block_robust.py",
        "tests/test_direction_1d_sprint_market_block_robust.py",
    ):
        code[name] = core.sha(base.PROJECT / name)
    return {"code": code, "core_plan_hash": base.digest(previous_spec)}


def plan():
    proposal = base.read(root() / "proposal-before-implementation.json")
    require(base.digest(proposal) == PROPOSAL_HASH, "PROPOSAL_CHANGED")
    require(proposal["design_sha256"] == core.sha(root() / "design-before-implementation.md"), "DESIGN_CHANGED")
    require(proposal["r90_result_hash"] == base.digest(base.read(previous.root() / "result.json")), "SOURCE_CHANGED")
    require(proposal["candidates"] == list(CANDIDATES) and proposal["controls"] == list(CONTROLS), "BRANCHES_CHANGED")
    path = root() / "plan.json"
    context = {
        "fingerprint": fingerprint(),
        "calendar_hash": base.calendar()[1],
        "scope_hash": base.digest(data.scope()),
    }
    if path.exists():
        p = base.read(path)
        require(
            all(p[k] == v for k, v in context.items()) and p["proposal_hash"] == PROPOSAL_HASH, "FROZEN_CONTEXT_CHANGED"
        )
        return p
    active()
    p = (
        proposal
        | context
        | {"at": base.now().isoformat(), "proposal_hash": PROPOSAL_HASH, "status": "MODEL_NOT_RELEASED"}
    )
    base.save(path, p)
    for name in p["fingerprint"]["code"]:
        dest = root() / "code" / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(base.PROJECT / name, dest)
    return p


def time_blocks(rows):
    """按真正进入训练的504个成熟交易日切块；同日基金不能分到不同块。"""
    dates = sorted({r["u"] for r in rows})
    require(len(dates) == 504, "DATES_NOT_504")
    mapping = {day: i // 63 for i, day in enumerate(dates)}
    return np.asarray([mapping[r["u"]] for r in rows], dtype=int)


def loss_gradient(coef, x, y, weights, blocks):
    """块内保持自然权重，块间用固定tau平滑最大损失；返回标量、解析梯度和块诊断。"""
    coef, x, y, weights, blocks = map(np.asarray, (coef, x, y, weights, blocks))
    require(x.ndim == 2 and x.shape[1] == 3 and coef.shape == (3,), "OBJECTIVE_SHAPE")
    require(y.shape == weights.shape == blocks.shape == (len(x),), "OBJECTIVE_ROWS")
    require(
        np.isfinite(x).all()
        and np.isfinite(weights).all()
        and (weights > 0).all()
        and set(y.tolist()) <= {0, 1}
        and set(blocks.tolist()) == set(range(8)),
        "OBJECTIVE_VALUES",
    )
    z = x @ coef
    losses = np.logaddexp(0, z) - y * z
    residual = expit(z) - y
    totals = np.bincount(blocks, weights=weights, minlength=8)
    means = np.bincount(blocks, weights=weights * losses, minlength=8) / totals
    gradients = np.stack(
        [np.bincount(blocks, weights=weights * residual * x[:, j], minlength=8) / totals for j in range(3)], axis=1
    )
    adversary = softmax(means / 0.05)
    penalty = 10.0 / weights.sum()
    value = 0.05 * (logsumexp(means / 0.05) - np.log(8)) + penalty * (coef @ coef) / 2
    gradient = adversary @ gradients + penalty * coef
    require(np.isfinite(value) and np.isfinite(gradient).all(), "OBJECTIVE_NONFINITE")
    return float(value), gradient, means, adversary


def fit(rows, name, cutoff):
    require(name in CANDIDATES, "UNKNOWN_RECIPE")
    chosen = original.training_rows(rows, cutoff)
    require(all(r["mature"] < cutoff and r["u"] < cutoff for r in chosen), "TRAINING_MATURITY_INVALID")
    x = np.asarray([baseline.market_vector(r["market"]) for r in chosen])
    y = np.asarray([r["y"] for r in chosen])
    weights = regression.weights(chosen)
    _, mean, scale = sequence.normalize_training(x, weights)
    signed = name == CANDIDATES[1]
    x = baseline.transformed(x, scale, signed)
    blocks = time_blocks(chosen)
    active()
    with threadpool_limits(limits=2):
        solved = minimize(
            lambda coef: loss_gradient(coef, x, y, weights, blocks)[:2],
            np.zeros(3),
            method="L-BFGS-B",
            jac=True,
            options={"maxiter": 1000, "gtol": 1e-9, "ftol": 1e-12},
        )
    active()
    value, gradient, means, adversary = loss_gradient(solved.x, x, y, weights, blocks)
    require(solved.success and np.max(np.abs(gradient)) < 1e-6, "OPTIMIZER_FAILED")
    return {
        "coef": solved.x.tolist(),
        "mean": [0.0] * 3,
        "scale": scale,
        "signed": signed,
        "tau": 0.05,
        "l2_numerator": 10.0,
        "blocks": 8,
        "dates_per_block": 63,
        "threshold": 0.0,
        "optimizer_success": bool(solved.success),
        "iterations": int(solved.nit),
        "objective": value,
        "gradient_inf": float(np.max(np.abs(gradient))),
        "block_losses": means.tolist(),
        "block_weights": adversary.tolist(),
        "cutoff": cutoff,
        "fit_hash": base.digest(chosen),
        "weight_hash": base.digest(weights.tolist()),
        "block_assignment_hash": base.digest(blocks.tolist()),
        "fit_rows": len(chosen),
        "fit_dates": 504,
        "fit_end": max(r["u"] for r in chosen),
        "max_mature_date": max(r["mature"] for r in chosen),
        "training_feature_mean_not_subtracted": mean,
        "weighted_up_rate": float(np.average(y, weights=weights)),
    }


def validate_head(head, name, cutoff=None):
    require(name in CANDIDATES and head is not None, "HEAD_MISSING")
    coef, scale = np.asarray(head["coef"]), np.asarray(head["scale"])
    require(
        coef.shape == scale.shape == (3,) and np.isfinite(coef).all() and np.isfinite(scale).all() and min(scale) > 0,
        "HEAD_VALUES_INVALID",
    )
    require(
        head["mean"] == [0.0] * 3
        and head["signed"] is (name == CANDIDATES[1])
        and head["tau"] == 0.05
        and head["l2_numerator"] == 10.0
        and head["blocks"] == 8
        and head["dates_per_block"] == 63
        and head["threshold"] == 0.0,
        "HEAD_RECIPE_INVALID",
    )
    losses, weights = np.asarray(head["block_losses"]), np.asarray(head["block_weights"])
    require(
        losses.shape == weights.shape == (8,)
        and np.isfinite(losses).all()
        and min(losses) >= 0
        and np.allclose(weights, softmax(losses / 0.05), atol=1e-12, rtol=1e-12),
        "HEAD_BLOCKS_INVALID",
    )
    require(
        head["optimizer_success"]
        and 0 <= head["gradient_inf"] < 1e-6
        and head["fit_dates"] == 504
        and head["fit_end"] < head["cutoff"]
        and head["max_mature_date"] < head["cutoff"],
        "HEAD_TIMING_OR_OPTIMIZER_INVALID",
    )
    if cutoff is not None:
        require(head["cutoff"] == cutoff, "HEAD_CUTOFF_CHANGED")
    return coef


def candidate_answers(values, name, head):
    require(name in CANDIDATES, "UNKNOWN_BRANCH")
    if not values:
        return []
    x = np.asarray([baseline.market_vector(v) for v in values])
    original.batch_answers(values, "SPX_SIGN")
    output = [{"prediction": int(z[0] >= 0), "kind": "FIXED_DIRECTION", "route": "SPX_SIGN_FALLBACK"} for z in x]
    indices = [i for i, v in enumerate(values) if v["available"]]
    if indices:
        coef = validate_head(head, name)
        matrix = baseline.transformed(x[indices], head["scale"], head["signed"])
        scores = expit(matrix @ coef)
        for i, score in zip(indices, scores, strict=True):
            output[i] = {
                "prediction": int(score >= 0.5),
                "research_score": float(score),
                "kind": "UNCALIBRATED_UP_SCORE",
                "route": name,
            }
    return output


def batch(values, heads, control_heads):
    controls = reference.batch(values, control_heads)
    output = {n: controls[n] for n in CONTROLS}
    output.update({n: candidate_answers(values, n, heads.get(n)) for n in CANDIDATES})
    return output


def answers(market, group, bundle):
    heads = {n: bundle[n][group] for n in CANDIDATES}
    controls = [bundle[n][group] for n in CONTROLS[:2]]
    return {n: choices[0] for n, choices in batch([market], heads, controls).items()}


def checkpoint(rows, name, cutoff, group, ph):
    """一份截止日/组/候选只允许一次拟合，失败不得自动重试并伪装为同次实验。"""
    path = root() / "checkpoints" / f"{cutoff}-{group}-{name}.joblib"
    receipt, attempt = path.with_suffix(".json"), path.with_suffix(".attempt.json")
    if receipt.exists():
        meta = base.read(receipt)
        require(
            meta["plan_hash"] == ph
            and meta["name"] == name
            and meta["cutoff"] == cutoff
            and meta["group"] == group
            and meta["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest(),
            "CHECKPOINT_CHANGED",
        )
        head = joblib.load(path)
        validate_head(head, name, cutoff)
        return head
    require(not attempt.exists(), "FIT_ALREADY_ATTEMPTED")
    base.save(attempt, {"at": base.now().isoformat(), "name": name, "cutoff": cutoff, "group": group, "plan_hash": ph})
    head = fit(rows, name, cutoff)
    validate_head(head, name, cutoff)
    joblib.dump(head, path)
    base.save(
        receipt,
        {
            "at": base.now().isoformat(),
            "name": name,
            "cutoff": cutoff,
            "group": group,
            "plan_hash": ph,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        },
    )
    base.save(root() / "training" / (path.stem + ".json"), head)
    return head


def prepare():
    p = plan()
    if (root() / "result.json").exists():
        return models()[0]
    rows, proof = data.dataset()
    proof_path = root() / "question-proof.json"
    if proof_path.exists():
        require(base.read(proof_path) == proof, "QUESTION_SOURCE_CHANGED")
    else:
        base.save(proof_path, proof)
    groups = sorted({r["group"] for r in rows})
    output = {str(year): defaultdict(list) for year in p["years"]}
    checks = []
    for year in p["years"]:
        for q in range(1, 5):
            start = f"{year}-{q * 3 - 2:02d}-01"
            end = f"{year + 1}-01-01" if q == 4 else f"{year}-{q * 3 + 1:02d}-01"
            for group in groups:
                active()
                group_rows = [r for r in rows if r["group"] == group]
                exam = [r for r in group_rows if start <= r["u"] < end]
                control_heads = [reference.historical_head(year, q, group, i) for i in range(2)]
                heads = {n: checkpoint(group_rows, n, start, group, base.digest(p)) for n in CANDIDATES}
                require(
                    all(
                        h["fit_hash"] == control_heads[0]["fit_hash"]
                        and h["weight_hash"] == control_heads[0]["weight_hash"]
                        for h in heads.values()
                    ),
                    "TRAINING_COMPARISON_CHANGED",
                )
                values = batch([r["market"] for r in exam], heads, control_heads)
                for name, choices in values.items():
                    keys = ("code", "family", "group", "t", "u", "y", "actual_direction") + (
                        ("old_question",) if year == 2025 else ()
                    )
                    scored = [{k: r[k] for k in keys} | answer for r, answer in zip(exam, choices, strict=True)]
                    if name in CONTROLS:
                        require(scored == reference.source_fold(year, q, group, name), "CONTROL_CHANGED")
                    path = root() / "folds" / str(year) / f"{q}-{group}-{name}.json"
                    if path.exists():
                        require(base.read(path) == scored, "SAVED_ANSWERS_CHANGED")
                    else:
                        base.save(path, scored)
                    output[str(year)][name].extend(scored)
                checks.append(
                    {
                        "year": year,
                        "quarter": q,
                        "group": group,
                        "same_fit_rows_and_weights": True,
                        "questions": len(exam),
                    }
                )
            base.save(
                root() / "progress.json", {"at": base.now().isoformat(), "year": year, "quarter": q}, replace=True
            )
        common = sorted((r["code"], r["u"], r["y"]) for r in output[str(year)]["SPX_SIGN"])
        require(
            len(common) == p["expected_questions"][str(year)]
            and len({u for _, u, _ in common}) == p["expected_dates"][str(year)]
            and all(sorted((r["code"], r["u"], r["y"]) for r in v) == common for v in output[str(year)].values()),
            "COMMON_EXAM_CHANGED",
        )
    _, parent_bundle = original.models()
    bundle = {n: {} for n in CANDIDATES + CONTROLS[:2]}
    for group in groups:
        for name in CANDIDATES:
            head = checkpoint(
                [r for r in rows if r["group"] == group], name, p["current_cutoff"], group, base.digest(p)
            )
            native = parent_bundle[original.CANDIDATES[0]][group]
            require(
                head["fit_hash"] == native["fit_hash"] and head["weight_hash"] == native["weight_hash"],
                "CURRENT_FIT_COMPARISON_CHANGED",
            )
            bundle[name][group] = head
        for i, name in enumerate(CONTROLS[:2]):
            bundle[name][group] = parent_bundle[original.CANDIDATES[i]][group]
    model_path = root() / "models.joblib"
    require(not model_path.exists(), "MODEL_BUNDLE_ALREADY_SAVED")
    joblib.dump(bundle, model_path)
    result = {
        "at": base.now().isoformat(),
        "plan_hash": base.digest(p),
        "fingerprint": fingerprint(),
        "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
        "metrics": {year: {n: base.metrics(v) for n, v in branches.items()} for year, branches in output.items()},
        "group_metrics": {
            year: {n: {g: base.metrics([r for r in v if r["group"] == g]) for g in groups} for n, v in branches.items()}
            for year, branches in output.items()
        },
        "training_checks": checks,
        "development_fits": 48,
        "current_fits": 6,
        "new_2026_scores": False,
        "new_cost_cny": 0,
        "new_source_requests": 0,
        "kind": "HISTORICAL_DEVELOPMENT_ONLY",
        "source_limit": (
            "Both years previously inspected; no pristine holdout; scores are not calibrated probabilities; "
            "historical first publication unverified"
        ),
        "status": "MODEL_NOT_RELEASED",
    }
    base.save(root() / "result.json", result)
    return result


def models():
    p, result = plan(), base.read(root() / "result.json")
    path = root() / "models.joblib"
    require(
        result["plan_hash"] == base.digest(p)
        and result["fingerprint"] == fingerprint()
        and result["model_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest(),
        "MODEL_CHANGED",
    )
    bundle = joblib.load(path)
    groups = {r["group"] for r in data.scope()}
    require(
        set(bundle) == set(CANDIDATES + CONTROLS[:2]) and all(set(v) == groups for v in bundle.values()),
        "MODEL_GROUP_MISSING",
    )
    for name in CANDIDATES:
        for head in bundle[name].values():
            validate_head(head, name, p["current_cutoff"])
    return result, bundle


def preflight():
    from app.services import direction_1d_sprint_block_robust_forward as forward

    return forward.preflight()


def tick():
    from app.services import direction_1d_sprint_block_robust_forward as forward

    return forward.run()


def report():
    from app.services import direction_1d_sprint_block_robust_forward as forward

    return forward.report()
