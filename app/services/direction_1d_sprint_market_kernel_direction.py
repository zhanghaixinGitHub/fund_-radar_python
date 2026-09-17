"""第90轮：用高斯核学习三市场信号的平滑联合关系，目标仍为一日原始净值方向。"""

import hashlib
import shutil
from collections import defaultdict

import joblib
import numpy as np
from sklearn.metrics.pairwise import rbf_kernel
from threadpoolctl import threadpool_limits

from app.services import direction_1d_sprint as base
from app.services import direction_1d_sprint_market_fund_bias as previous
from app.services import direction_1d_sprint_market_only as baseline
from app.services import direction_1d_sprint_return_target as regression
from app.services import direction_1d_sprint_sequence as sequence

original, data, active = previous.original, previous.data, previous.active
reference = previous.reference
CANDIDATES = ("INTERVAL_RAW3_RBF_RIDGE504",)
CONTROLS = previous.CONTROLS
BRANCHES = CANDIDATES + CONTROLS
FIRST_TARGET = "2026-09-16"
PROPOSAL_HASH = "4b9c95657aca3a323afe2f04e99fe9201c2235070f8c887e50c989029025138d"


def root():
    return base.ROOT / "round-90"


def require(condition, reason):
    if not condition:
        raise ValueError("KERNEL_DIRECTION_" + reason)


def fingerprint():
    value = previous.fingerprint()
    for name in (
        "app/services/direction_1d_sprint_market_kernel_direction.py",
        "scripts/direction_1d_sprint_market_kernel_direction.py",
        "tests/test_direction_1d_sprint_market_kernel_direction.py",
    ):
        value["code"][name] = hashlib.sha256((base.PROJECT / name).read_bytes()).hexdigest()
    return value


def plan():
    proposal = base.read(root() / "proposal-before-implementation.json")
    require(base.digest(proposal) == PROPOSAL_HASH, "PROPOSAL_CHANGED")
    require(
        proposal["design_sha256"]
        == hashlib.sha256((root() / "design-before-implementation.md").read_bytes()).hexdigest(),
        "DESIGN_CHANGED",
    )
    source, _ = previous.models()
    require(proposal["r89_result_hash"] == base.digest(source), "SOURCE_CHANGED")
    require(proposal["candidates"] == list(CANDIDATES) and proposal["controls"] == list(CONTROLS), "BRANCHES_CHANGED")
    require(
        proposal["alpha"] == 10.0
        and proposal["gamma"] == 1 / 3
        and proposal["threshold"] == 0.0
        and proposal["target"] == "2*y-1",
        "RECIPE_CHANGED",
    )
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
        destination = root() / "code" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(base.PROJECT / name, destination)
    return p


def aggregate_squared_target(x, y, weights):
    """相同向量汇总总权重和±1目标均值；平方损失只差常数，保留每个基金原自然权重。"""
    x, y, weights = np.asarray(x), np.asarray(y), np.asarray(weights)
    require(x.ndim == 2 and x.shape[1] == 3 and len(x) == len(y) == len(weights), "AGGREGATE_SHAPE")
    require(
        np.isfinite(x).all() and np.isfinite(weights).all() and (weights > 0).all() and set(y.tolist()) <= {0, 1},
        "AGGREGATE_VALUES",
    )
    centers, inverse = np.unique(x, axis=0, return_inverse=True)
    totals = np.bincount(inverse, weights=weights, minlength=len(centers))
    targets = np.bincount(inverse, weights=weights * (2 * y - 1), minlength=len(centers)) / totals
    return centers, totals, targets


def solve_kernel(x, y, weights):
    """对称正定形式避免直接求逆；alpha10与自然权重总量固定，不重新平均样本。"""
    centers, totals, targets = aggregate_squared_target(x, y, weights)
    sw = np.sqrt(totals)
    gram = rbf_kernel(centers, centers, gamma=1 / 3) * sw[:, None] * sw[None, :] + 10.0 * np.eye(len(centers))
    rhs = sw * targets
    with threadpool_limits(limits=2):
        solved = np.linalg.solve(gram, rhs)
    residual = float(np.linalg.norm(gram @ solved - rhs) / max(1.0, np.linalg.norm(rhs)))
    require(np.isfinite(solved).all() and residual < 1e-10, "NORMAL_EQUATION_FAILED")
    return centers, sw * solved, totals, targets, residual


def fit(rows, name, cutoff):
    """全部504日训练标签已成熟；同日市场向量汇总只节省计算，不改变标签和统计权重。"""
    require(name in CANDIDATES, "UNKNOWN_RECIPE")
    chosen = original.training_rows(rows, cutoff)
    require(
        len({r["u"] for r in chosen}) == 504 and all(r["mature"] < cutoff and r["u"] < cutoff for r in chosen),
        "TRAINING_MATURITY_INVALID",
    )
    x = np.asarray([baseline.market_vector(r["market"]) for r in chosen])
    y = np.asarray([r["y"] for r in chosen])
    require(set(y.tolist()) == {0, 1}, "LABEL_INVALID")
    weights = regression.weights(chosen)
    _, training_mean, scale = sequence.normalize_training(x, weights)
    x = baseline.transformed(x, scale, False)
    active()
    centers, dual, totals, targets, residual = solve_kernel(x, y, weights)
    require(1 <= len(centers) <= 504, "UNBOUNDED_CENTERS")
    active()
    return {
        "centers": centers.tolist(),
        "dual": dual.tolist(),
        "aggregate_weights": totals.tolist(),
        "aggregate_targets": targets.tolist(),
        "mean": [0.0] * 3,
        "scale": scale,
        "signed": False,
        "alpha": 10.0,
        "gamma": 1 / 3,
        "target": "2*y-1",
        "threshold": 0.0,
        "normal_equation_residual": residual,
        "cutoff": cutoff,
        "fit_hash": base.digest(chosen),
        "weight_hash": base.digest(weights.tolist()),
        "fit_rows": len(chosen),
        "fit_dates": len({r["u"] for r in chosen}),
        "fit_end": max(r["u"] for r in chosen),
        "max_mature_date": max(r["mature"] for r in chosen),
        "training_feature_mean_not_subtracted": training_mean,
        "weighted_up_rate": float(np.average(y, weights=weights)),
    }


def validate_head(head, name, cutoff=None):
    require(name in CANDIDATES and head is not None, "HEAD_MISSING")
    centers, dual, scale = np.asarray(head["centers"]), np.asarray(head["dual"]), np.asarray(head["scale"])
    require(
        centers.ndim == 2
        and centers.shape[1] == 3
        and 1 <= len(centers) <= 504
        and dual.shape == (len(centers),)
        and scale.shape == (3,),
        "HEAD_SHAPE_INVALID",
    )
    require(
        np.isfinite(centers).all() and np.isfinite(dual).all() and np.isfinite(scale).all() and min(scale) > 0,
        "HEAD_VALUES_INVALID",
    )
    weights, targets = np.asarray(head["aggregate_weights"]), np.asarray(head["aggregate_targets"])
    require(
        weights.shape == targets.shape == dual.shape
        and np.isfinite(weights).all()
        and np.isfinite(targets).all()
        and min(weights) > 0
        and np.max(np.abs(targets)) <= 1 + 1e-12,
        "HEAD_AGGREGATE_INVALID",
    )
    require(
        head["mean"] == [0.0] * 3
        and head["signed"] is False
        and head["alpha"] == 10.0
        and head["gamma"] == 1 / 3
        and head["threshold"] == 0.0
        and head["target"] == "2*y-1",
        "HEAD_RECIPE_INVALID",
    )
    require(
        0 <= head["normal_equation_residual"] < 1e-10
        and head["fit_dates"] == 504
        and head["fit_end"] < head["cutoff"]
        and head["max_mature_date"] < head["cutoff"],
        "HEAD_TIMING_INVALID",
    )
    if cutoff is not None:
        require(head["cutoff"] == cutoff, "HEAD_CUTOFF_CHANGED")
    return centers, dual


def candidate_answers(values, name, head):
    """缺源完整保留SPX回退；方向分数可超出[-1,1]，绝不解释为概率。"""
    require(name in CANDIDATES, "UNKNOWN_BRANCH")
    if not values:
        return []
    x = np.asarray([baseline.market_vector(v) for v in values])
    # 同时检查完整区间口径没有更改SPX/CNYA，沿用已冻结输入校验。
    original.batch_answers(values, "SPX_SIGN")
    output = [{"prediction": int(z[0] >= 0), "kind": "FIXED_DIRECTION", "route": "SPX_SIGN_FALLBACK"} for z in x]
    indices = [i for i, v in enumerate(values) if v["available"]]
    if indices:
        centers, dual = validate_head(head, name)
        matrix = baseline.transformed(x[indices], head["scale"], False)
        margins = rbf_kernel(matrix, centers, gamma=head["gamma"]) @ dual
        require(np.isfinite(margins).all(), "MARGIN_INVALID")
        for i, margin in zip(indices, margins, strict=True):
            output[i] = {
                "prediction": int(margin >= 0),
                "research_score": float(margin),
                "kind": "UNCALIBRATED_DIRECTION_MARGIN",
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
        "development_fits": 24,
        "current_fits": 3,
        "new_2026_scores": False,
        "new_cost_cny": 0,
        "new_source_requests": 0,
        "kind": "HISTORICAL_DEVELOPMENT_ONLY",
        "source_limit": (
            "Both years previously inspected; no pristine holdout; margins are not probabilities; "
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
    _, bundle = models()
    rows, _ = data.dataset()
    market = next(r["market"] for r in reversed(rows) if r["market"]["available"])
    for fund in data.scope():
        require(set(answers(market, fund["group"], bundle)) == set(BRANCHES), "PREFLIGHT_BRANCH_MISSING")
    value = {"at": base.now().isoformat(), "kind": "DRY_RUN_NOT_FORWARD", "branch_checks": 180}
    base.save(root() / "preflight.json", value)
    return value


def tick():
    from app.services import direction_1d_sprint_market_child_forward_v2 as forward
    from app.services import direction_1d_sprint_market_kernel_direction as service

    return forward.tick(service)


def report():
    from app.services import direction_1d_sprint_market_child_forward_v2 as forward
    from app.services import direction_1d_sprint_market_kernel_direction as service

    return forward.report(service)


live_market = data.live_market
live_answers = answers
