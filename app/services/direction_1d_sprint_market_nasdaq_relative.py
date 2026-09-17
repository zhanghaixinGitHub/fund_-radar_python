"""第80轮：纳指相对标普信息在原三市场条件下的增量训练。

两个无截距逻辑回归候选与五个对照使用同题、同一SPX回退；只用当时成熟的标签。
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
from app.services import direction_1d_sprint_market_current_refresh as previous
from app.services import direction_1d_sprint_market_nasdaq_relative_data as data
from app.services import direction_1d_sprint_market_only as baseline
from app.services import direction_1d_sprint_return_target as regression
from app.services import direction_1d_sprint_sequence as sequence

CANDIDATES = ("EXPANDED_NASDAQ_REL_LR4_504", "EXPANDED_SIGN3_NASDAQ_REL_LR4_504")
CONTROLS = (
    "CURRENT_MATCHED_MARKET_LR3",
    "CURRENT_MATCHED_MARKET_SIGN_LR3",
    "MARKET_MAJORITY3",
    "SPX_SIGN",
    "ALWAYS_UP",
)
BRANCHES = CANDIDATES + CONTROLS

FIRST_TARGET = "2026-09-16"
PROPOSAL_HASH = "7335c45712ef892cba25743d0a5c038b87a721ff424b5190a3d96c1e525fee1d"
active = regression.active


def root():
    return base.ROOT / "round-80"


def fingerprint():
    value = previous.fingerprint()
    for name in (
        "app/services/direction_1d_sprint_market_nasdaq_relative.py",
        "scripts/direction_1d_sprint_market_nasdaq_relative.py",
        "tests/test_direction_1d_sprint_market_nasdaq_relative.py",
        "app/services/direction_1d_sprint_market_nasdaq_relative_data.py",
        "tests/test_direction_1d_sprint_market_nasdaq_relative_data.py",
    ):
        value["code"][name] = hashlib.sha256((base.PROJECT / name).read_bytes()).hexdigest()
    return value


def control_path(name, group, label):
    if name not in CONTROLS[:2]:
        raise ValueError("MARKET_NASDAQ_RELATIVE_CONTROL_INVALID")
    service = previous if label == "current" else baseline
    original = service.CANDIDATES[CONTROLS.index(name)]
    return service.root() / "checkpoints" / f"{label}-{group}-{original}.joblib", original


def validate_spec(proposal, scope_size):
    if (
        proposal["candidates"] != list(CANDIDATES)
        or proposal["controls"] != list(CONTROLS)
        or proposal["branch_count"] != len(BRANCHES)
        or len(set(BRANCHES)) != len(BRANCHES)
        or proposal["preflight_branch_checks"] != len(BRANCHES) * scope_size
        or proposal["max_development_fits"] != 24
        or proposal["max_current_fits"] != 6
        or proposal["reproductions"] != 1
        or proposal["current_cutoff"] != "2026-09-16"
    ):
        raise ValueError("MARKET_NASDAQ_RELATIVE_DECLARED_BRANCHES_OR_BUDGET_CHANGED")


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
    if base.digest(previous.models()[0]) != proposal["r79_result_hash"]:
        raise ValueError("MARKET_GAP_DELTA_PARENT_RESULT_CHANGED")
    if (
        base.digest(baseline.models()[0]) != proposal["r70_result_hash"]
        or base.digest(base.read(base.ROOT / "nasdaq-relative-market-feasibility-v1/result.json"))
        != proposal["feasibility_result_hash"]
        or base.digest(base.read(base.ROOT / "nasdaq-relative-market-feasibility-v1/rows.json"))
        != proposal["expanded_rows_hash"]
    ):
        raise ValueError("MARKET_NASDAQ_RELATIVE_PROPOSED_SOURCE_CHANGED")
    groups = sorted({f["group"] for f in data.scope()})
    paths = [
        "history.json",
        "round-80/proposal-before-implementation.json",
        "nasdaq-relative-market-feasibility-v1/plan.json",
        "nasdaq-relative-market-feasibility-v1/result.json",
        "nasdaq-relative-market-feasibility-v1/rows.json",
        "round-70/result.json",
        "round-79/result.json",
    ]
    for name in CONTROLS[:2]:
        for group in groups:
            for label in ("1", "2", "3", "4", "current"):
                paths.append(control_path(name, group, label)[0].with_suffix(".json").relative_to(base.ROOT).as_posix())
    p = {
        "at": base.now().isoformat(),
        "round": 80,
        "fingerprint": fingerprint(),
        "calendar_hash": base.calendar()[1],
        "scope_hash": base.digest(data.scope()),
        "input_hashes": {name: base.digest(base.read(base.ROOT / name)) for name in paths},
        "proposal_hash": PROPOSAL_HASH,
        "candidates": list(CANDIDATES),
        "controls": list(CONTROLS),
        "current_fit_cutoff": proposal["current_cutoff"],
        "first_forward_target": FIRST_TARGET,
        "development_fits": 24,
        "current_fits": 6,
        "reproductions": 1,
        "expected_2025_questions": 7290,
        "branch_count": len(BRANCHES),
        "preflight_branch_checks": len(BRANCHES) * len(data.scope()),
        "expected_2025_dates": 243,
        "old_questions": 5670,
        "training": ("Same504mature marketdates/group; natural family/date weights; LR C.1/nointercept/max1500/seed17"),
        "transform": "raw4/std or signfirst3+relative/std; no mean; no intercept; fixedthreshold.5",
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
    if x.shape != (4,) or not np.isfinite(x).all() or type(value["available"]) is not bool:
        raise ValueError("NASDAQ_RELATIVE_VECTOR_INVALID")
    if value["available"] != bool(value["etf_available"] and value["cnya_available"]):
        raise ValueError("NASDAQ_RELATIVE_FLAGS_INVALID")
    return x


training_rows = baseline.training_rows


def transformed(x, scale, signed_three):
    """原三列可取方向，新增相对差值始终保留幅度；所有尺度来自过去成熟训练行。"""
    x, scale = np.asarray(x, dtype=float), np.asarray(scale, dtype=float)
    if (
        x.ndim != 2
        or x.shape[1] != 4
        or scale.shape != (4,)
        or not np.isfinite(x).all()
        or not np.isfinite(scale).all()
        or min(scale) <= 0
    ):
        raise ValueError("NASDAQ_RELATIVE_TRANSFORM_INVALID")
    z = x / scale
    if signed_three:
        z[:, :3] = np.where(x[:, :3] >= 0, 1.0, -1.0)
    return z


def fit(rows, name, cutoff):
    """两个四输入候选均无截距；数据、日期及权重对齐原三项头，只增加一项信息。"""
    if name not in CANDIDATES:
        raise ValueError("NASDAQ_RELATIVE_RECIPE_INVALID")
    chosen = training_rows(rows, cutoff)
    x = np.asarray([market_vector(r["market"]) for r in chosen])
    y, weights = np.asarray([r["y"] for r in chosen]), regression.weights(chosen)
    _, mean, scale = sequence.normalize_training(x, weights)
    signed_three = name == CANDIDATES[1]
    active()
    with warnings.catch_warnings(), threadpool_limits(limits=2):
        warnings.simplefilter("error", ConvergenceWarning)
        model = LogisticRegression(C=0.1, fit_intercept=False, max_iter=1500, random_state=17)
        model.fit(transformed(x, scale, signed_three), y, sample_weight=weights)
    active()
    return {
        "model": model,
        "mean": [0.0] * 4,
        "scale": scale,
        "signed_three": signed_three,
        "fit_hash": base.digest(chosen),
        "original_fit_hash": base.digest([data.original_row(r) for r in chosen]),
        "weight_hash": base.digest(weights.tolist()),
        "fit_rows": len(chosen),
        "fit_dates": len({r["u"] for r in chosen}),
        "fit_end": max(r["u"] for r in chosen),
        "max_mature_date": max(r["mature"] for r in chosen),
        "cutoff": cutoff,
        "weighted_up_rate": float(np.average(y, weights=weights)),
        "training_feature_mean_not_subtracted": mean,
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
        raise ValueError("NASDAQ_RELATIVE_BRANCH_INVALID")
    if not values:
        return []
    x = np.asarray([market_vector(v) for v in values])
    if name in CONTROLS:
        original = baseline.CANDIDATES[CONTROLS.index(name)] if name in CONTROLS[:2] else name
        return baseline.batch_answers([v | {"features": v["features"][:3]} for v in values], original, head)
    out = [{"prediction": int(z[0] >= 0), "kind": "FIXED_DIRECTION", "route": "SPX_SIGN_FALLBACK"} for z in x]
    indices = [i for i, v in enumerate(values) if v["available"]]
    if indices:
        signed = name == CANDIDATES[1]
        if (
            head is None
            or head["signed_three"] != signed
            or head["mean"] != [0.0] * 4
            or head["model"].fit_intercept
            or head["model"].n_features_in_ != 4
        ):
            raise ValueError("NASDAQ_RELATIVE_HEAD_SCHEMA_INVALID")
        scores = head["model"].predict_proba(transformed(x[indices], head["scale"], signed))[:, 1]
        if not np.isfinite(scores).all() or min(scores) < 0 or max(scores) > 1:
            raise ValueError("NASDAQ_RELATIVE_SCORE_INVALID")
        for i, score in zip(indices, scores, strict=True):
            out[i] = {
                "prediction": int(score >= 0.5),
                "research_score": float(score),
                "kind": "UNCALIBRATED_UP_SCORE",
                "route": name,
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
        "development_fits": 24,
        "current_fits": 6,
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
    from app.services import direction_1d_sprint_market_nasdaq_relative as service

    if data.capture(base.now()) is None:
        return forward.report(service)
    return forward.tick(service)


def report():
    from app.services import direction_1d_sprint_market_child_forward_v2 as forward
    from app.services import direction_1d_sprint_market_nasdaq_relative as service

    return forward.report(service)


# 第70轮父输入与第6轮纳指响应均重验；独立输入摘要随预测保存。
live_market = data.live_market
live_answers = answers
