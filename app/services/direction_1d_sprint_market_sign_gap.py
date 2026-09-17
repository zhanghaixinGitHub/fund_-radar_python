"""第75轮：三项市场方向与开盘缺口幅度，预测仍为相邻交易日原始净值方向。

一个混合四输入候选与五个冻结/固定对照使用同题、同一SPX回退；不读取2026考试成绩选模。
"""

import hashlib
import shutil
import warnings
from collections import defaultdict

import joblib
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

from app.services import direction_1d_sprint as base
from app.services import direction_1d_sprint_adaptive as adaptive
from app.services import direction_1d_sprint_market_gap_ablation as opening
from app.services import direction_1d_sprint_market_gap_ablation_v2 as previous
from app.services import direction_1d_sprint_market_gap_delta_data as data
from app.services import direction_1d_sprint_market_only as baseline
from app.services import direction_1d_sprint_return_target as regression
from app.services import direction_1d_sprint_sequence as sequence

CANDIDATES = ("EXPANDED_SIGN3_OPEN_GAP_LR4_504",)
CONTROLS = ("FROZEN_R70_MARKET_SIGN_LR3", "FROZEN_R74_OPEN_GAP_LR4", "MARKET_MAJORITY3", "SPX_SIGN", "ALWAYS_UP")
BRANCHES = CANDIDATES + CONTROLS
FEATURE_COLUMNS = {CANDIDATES[0]: (0, 1, 2, 3), CONTROLS[0]: (0, 1, 2), CONTROLS[1]: (0, 1, 2, 3)}
TRANSFORMS = {CANDIDATES[0]: "SIGN3_RAW_GAP", CONTROLS[0]: "SIGN3", CONTROLS[1]: "SCALED"}
FIRST_TARGET = "2026-09-16"
PROPOSAL_HASH = "dda0b8c636ffdd1cf902b50a8dcefac1765358646b4f3cd72a25ed529caf9f28"
active = regression.active


def root():
    return base.ROOT / "round-75"


def fingerprint():
    value = previous.fingerprint()
    for name in (
        "app/services/direction_1d_sprint_market_sign_gap.py",
        "scripts/direction_1d_sprint_market_sign_gap.py",
        "tests/test_direction_1d_sprint_market_sign_gap.py",
    ):
        value["code"][name] = hashlib.sha256((base.PROJECT / name).read_bytes()).hexdigest()
    return value


def control_path(name, group, label):
    routes = {CONTROLS[0]: (baseline, baseline.CANDIDATES[1]), CONTROLS[1]: (opening, opening.CANDIDATES[0])}
    if name not in routes:
        raise ValueError("MARKET_SIGN_GAP_CONTROL_INVALID")
    service, original = routes[name]
    return service.root() / "checkpoints" / f"{label}-{group}-{original}.joblib", original


def validate_spec(proposal, scope_size):
    if (
        proposal["candidates"] != list(CANDIDATES)
        or proposal["controls"] != list(CONTROLS)
        or proposal["branch_count"] != len(BRANCHES)
        or len(set(BRANCHES)) != len(BRANCHES)
        or proposal["preflight_branch_checks"] != len(BRANCHES) * scope_size
        or proposal["max_development_fits"] != 12
        or proposal["max_current_fits"] != 3
        or set(FEATURE_COLUMNS) != set(CANDIDATES + CONTROLS[:2])
        or set(TRANSFORMS) != set(FEATURE_COLUMNS)
    ):
        raise ValueError("MARKET_SIGN_GAP_DECLARED_BRANCHES_OR_BUDGET_CHANGED")


def verify_inputs(p):
    for path, digest in p["input_hashes"].items():
        if base.digest(base.read(base.ROOT / path)) != digest:
            raise ValueError("MARKET_ONLY_FROZEN_INPUT_CHANGED")
    if base.digest(data.scope()) != p["scope_hash"]:
        raise ValueError("MARKET_ONLY_SCOPE_CHANGED")


def plan():
    path = root() / "plan.json"
    proposal = base.read(root() / "proposal-before-implementation.json")
    validate_spec(proposal, len(data.scope()))
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
    if base.digest(previous.models()[0]) != proposal["r74_v2_result_hash"]:
        raise ValueError("MARKET_GAP_DELTA_PARENT_RESULT_CHANGED")
    feature_folder = base.ROOT / "market-gap-delta-feasibility-v1"
    if (
        base.digest(base.read(feature_folder / "result.json")) != proposal["feature_feasibility_hash"]
        or base.digest(base.read(feature_folder / "rows.json")) != proposal["expanded5_rows_hash"]
    ):
        raise ValueError("MARKET_GAP_DELTA_PROPOSED_SOURCE_CHANGED")
    groups = sorted({f["group"] for f in data.scope()})
    paths = [
        "history.json",
        "round-75/proposal-before-implementation.json",
        "nav-independent-market-feasibility-v1/plan.json",
        "nav-independent-market-feasibility-v1/result.json",
        "nav-independent-market-feasibility-v1/rows.json",
        "round-70/result.json",
        "round-74/result.json",
        "round-74-v2/result.json",
        "market-gap-delta-feasibility-v1/plan.json",
        "market-gap-delta-feasibility-v1/result.json",
        "market-gap-delta-feasibility-v1/rows.json",
    ]
    for name in CONTROLS[:2]:
        for group in groups:
            for label in ("1", "2", "3", "4", "current"):
                paths.append(control_path(name, group, label)[0].with_suffix(".json").relative_to(base.ROOT).as_posix())
    p = {
        "at": base.now().isoformat(),
        "round": 75,
        "fingerprint": fingerprint(),
        "calendar_hash": base.calendar()[1],
        "scope_hash": base.digest(data.scope()),
        "input_hashes": {name: base.digest(base.read(base.ROOT / name)) for name in paths},
        "proposal_hash": PROPOSAL_HASH,
        "candidates": list(CANDIDATES),
        "controls": list(CONTROLS),
        "current_fit_cutoff": proposal["current_cutoff"],
        "first_forward_target": FIRST_TARGET,
        "development_fits": 12,
        "current_fits": 3,
        "reproductions": 1,
        "expected_2025_questions": 7290,
        "branch_count": len(BRANCHES),
        "preflight_branch_checks": len(BRANCHES) * len(data.scope()),
        "expected_2025_dates": 243,
        "old_questions": 5670,
        "training": (
            "504mature fullmarket dates pergroup; natural date/family weights; mixedLR4 C.1/nointercept/seed17/max1500"
        ),
        "transform": "candidate signfirst3 and scaled openinggap; fixed control transformations; threshold.5",
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


def market_vector(value):
    x = np.asarray(value["features"], dtype=float)
    if x.shape != (5,) or not np.isfinite(x).all() or type(value["available"]) is not bool:
        raise ValueError("MARKET_ONLY_FEATURES_INVALID")
    if value["available"] != bool(value["etf_available"] and value["cnya_available"]):
        raise ValueError("MARKET_ONLY_SOURCE_FLAGS_INVALID")
    return x


def transformed(x, scale, kind):
    """混合模型只把前三项取符号，第四项保留标准化幅度；对照沿各自冻结转换。"""
    x, scale = np.asarray(x, dtype=float), np.asarray(scale, dtype=float)
    width = {"SIGN3_RAW_GAP": 4, "SIGN3": 3, "SCALED": 4}.get(kind)
    if (
        width is None
        or x.ndim != 2
        or x.shape[1] != width
        or scale.shape != (width,)
        or not np.isfinite(x).all()
        or not np.isfinite(scale).all()
        or min(scale) <= 0
    ):
        raise ValueError("MARKET_SIGN_GAP_TRANSFORM_INVALID")
    value = x / scale
    if kind in ("SIGN3_RAW_GAP", "SIGN3"):
        value[:, :3] = np.where(x[:, :3] >= 0, 1.0, -1.0)
    return value


def training_rows(rows, cutoff):
    """T和U的净值只作标签；两者均已公告且标签成熟后才可进入训练。"""
    return adaptive.training_rows([r for r in rows if r["market"]["available"]], cutoff, "MONTHLY_BAL504")


def fit(rows, name, cutoff):
    if name not in CANDIDATES:
        raise ValueError("MARKET_ONLY_RECIPE_INVALID")
    chosen = training_rows(rows, cutoff)
    x = np.asarray([market_vector(r["market"]) for r in chosen])[:, FEATURE_COLUMNS[name]]
    y, weights = np.asarray([r["y"] for r in chosen]), regression.weights(chosen)
    _, mean, scale = sequence.normalize_training(x, weights)
    kind = TRANSFORMS[name]
    active()
    with warnings.catch_warnings(), threadpool_limits(limits=2):
        warnings.simplefilter("error", ConvergenceWarning)
        model = LogisticRegression(C=0.1, max_iter=1500, random_state=17, fit_intercept=False)
        model.fit(transformed(x, scale, kind), y, sample_weight=weights)
    active()
    return {
        "model": model,
        "mean": [0.0] * 4,
        "scale": scale,
        "transform": kind,
        "feature_indices": FEATURE_COLUMNS[name],
        "cutoff": cutoff,
        "fit_hash": base.digest(chosen),
        "weight_hash": base.digest(weights.tolist()),
        "fit_rows": len(chosen),
        "fit_dates": len({r["u"] for r in chosen}),
        "fit_end": max(r["u"] for r in chosen),
        "max_mature_date": max(r["mature"] for r in chosen),
        "training_feature_mean_not_subtracted": mean,
        "weighted_up_rate": float(np.average(y, weights=weights)),
    }


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
    """只复用已冻结的三输入头；R70原SPX回退保持一致；旧模型和答案保持不变。"""
    path, original = control_path(name, group, label)
    meta = base.read(path.with_suffix(".json"))
    if (
        meta["name"] != original
        or meta["cutoff"] != cutoff
        or meta["sha256"] != hashlib.sha256(path.read_bytes()).hexdigest()
    ):
        raise ValueError("MARKET_ONLY_FROZEN_CONTROL_CHANGED")
    old = joblib.load(path)
    if old["cutoff"] != cutoff or old["mean"] != [0.0] * len(FEATURE_COLUMNS[name]):
        raise ValueError("MARKET_ONLY_FROZEN_CONTROL_SCHEMA_INVALID")
    return old | {"transform": TRANSFORMS[name], "original_sha256": meta["sha256"]}


def batch_answers(values, name, head=None):
    if name not in BRANCHES:
        raise ValueError("MARKET_ONLY_BRANCH_INVALID")
    if not values:
        return []
    x = np.asarray([market_vector(v) for v in values])
    out = [{"prediction": int(z[0] >= 0), "kind": "FIXED_DIRECTION", "route": "SPX_SIGN_FALLBACK"} for z in x]
    if name == "ALWAYS_UP":
        return [{"prediction": 1, "kind": "FIXED_DIRECTION", "route": "ALWAYS_UP"} for _ in values]
    if name == "SPX_SIGN":
        return [v | {"route": "SPX_SIGN"} for v in out]
    indices = [i for i, v in enumerate(values) if v["available"]]
    if name == "MARKET_MAJORITY3":
        for i in indices:
            out[i] = {
                "prediction": int(sum(x[i, :3] >= 0) >= 2),
                "kind": "FIXED_DIRECTION",
                "route": "MARKET_MAJORITY3",
            }
    elif indices:
        columns, kind = FEATURE_COLUMNS[name], TRANSFORMS[name]
        if head is None or head["transform"] != kind or head["mean"] != [0.0] * len(columns):
            raise ValueError("MARKET_SIGN_GAP_HEAD_SCHEMA_INVALID")
        scores = head["model"].predict_proba(transformed(x[indices][:, columns], head["scale"], kind))[:, 1]
        if not np.isfinite(scores).all() or min(scores) < 0 or max(scores) > 1:
            raise ValueError("MARKET_SIGN_GAP_SCORE_INVALID")
        for i, score in zip(indices, scores, strict=True):
            out[i] = {
                "prediction": int(score >= 0.5),
                "research_score": float(score),
                "kind": "UNCALIBRATED_UP_SCORE",
                "route": "MARKET_SIGN3_RAW_GAP" if name in CANDIDATES else "MARKET_FROZEN_CONTROL",
            }
    return out


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
                    fit_checkpoint(group_rows, name, start, f"{q}-{group}", ph)
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
        raise ValueError("MARKET_SIGN_GAP_COMPLETED_BRANCHES_MISSING")
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
                fit_checkpoint(
                    [r for r in rows if r["group"] == group], name, p["current_fit_cutoff"], f"current-{group}", ph
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
        "development_fits": 12,
        "current_fits": 3,
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
    for fund in data.scope():
        answers(market, fund["group"], bundle)
    value = {
        "at": base.now().isoformat(),
        "kind": "DRY_RUN_NOT_FORWARD",
        "branch_checks": len(data.scope()) * len(BRANCHES),
    }
    base.save(root() / "preflight.json", value)
    return value


def tick():
    from app.services import direction_1d_sprint_market_child_forward_v2 as forward
    from app.services import direction_1d_sprint_market_sign_gap as service

    return forward.tick(service)


def report():
    from app.services import direction_1d_sprint_market_child_forward_v2 as forward
    from app.services import direction_1d_sprint_market_sign_gap as service

    return forward.report(service)


# 扩展运行器先验证并缓存目标日原始来源，再将同一五维向量用于全部基金。
live_market = data.live_market
live_answers = answers
