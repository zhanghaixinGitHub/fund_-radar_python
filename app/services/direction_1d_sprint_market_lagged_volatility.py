"""第84轮：按此前20个完整市场观察的波动调整一日模型输入。

一个无截距逻辑回归候选与五个对照使用同题、同一SPX回退；只用当时成熟的标签。
"""

import hashlib
import shutil
from collections import defaultdict

import joblib
import numpy as np

from app.services import direction_1d_sprint as base
from app.services import direction_1d_sprint_market_expanding_interval as prior_chain
from app.services import direction_1d_sprint_market_fxi_interval as previous
from app.services import direction_1d_sprint_market_lagged_volatility_data as data
from app.services import direction_1d_sprint_market_only as baseline
from app.services import direction_1d_sprint_return_target as regression

CANDIDATES = ("LAGGED_VOL20_INTERVAL_RAW3_LR504",)
CONTROLS = (
    "FROZEN_R82_INTERVAL_RAW3",
    "FROZEN_R82_INTERVAL_SIGN3",
    "MARKET_MAJORITY3",
    "SPX_SIGN",
    "ALWAYS_UP",
)
BRANCHES = CANDIDATES + CONTROLS

FIRST_TARGET = "2026-09-16"
PROPOSAL_HASH = "2eaca624e54ad605b016262b8f99dd6f54e54d5655fc31c2f5a6b39c76176b7e"
active = regression.active


def root():
    return base.ROOT / "round-84"


def fingerprint():
    value = prior_chain.fingerprint()
    for name in (
        "app/services/direction_1d_sprint_market_lagged_volatility.py",
        "scripts/direction_1d_sprint_market_lagged_volatility.py",
        "tests/test_direction_1d_sprint_market_lagged_volatility.py",
        "app/services/direction_1d_sprint_market_lagged_volatility_data.py",
        "tests/test_direction_1d_sprint_market_lagged_volatility_data.py",
    ):
        value["code"][name] = hashlib.sha256((base.PROJECT / name).read_bytes()).hexdigest()
    return value


def control_path(name, group, label):
    if name not in CONTROLS[:2]:
        raise ValueError("LAGGED_VOLATILITY_CONTROL_INVALID")
    service = previous
    original = service.CANDIDATES[CONTROLS.index(name)]
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
        or proposal["reproductions"] != 1
        or proposal["current_cutoff"] != "2026-09-16"
        or proposal["lookback_complete_dates"] != 20
        or proposal["scale_floor_percent"] != 1e-6
        or proposal["future_scan_bound_cn_dates"] != 80
        or proposal["control_input_policy"] != "R82_LEARNED_INTERVAL_FIXED_RULES_ORIGINAL_CANDIDATE_NORMALIZED"
    ):
        raise ValueError("LAGGED_VOLATILITY_DECLARED_BRANCHES_OR_BUDGET_CHANGED")


def verify_inputs(p):
    folder = base.ROOT / "lagged-market-volatility-feasibility-v1"
    spec = base.read(folder / "plan.json")
    if spec["script_sha256"] != hashlib.sha256((folder / "check.py").read_bytes()).hexdigest():
        raise ValueError("LAGGED_VOLATILITY_FEASIBILITY_SCRIPT_CHANGED")
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
    if base.digest(previous.models()[0]) != proposal["r82_result_hash"]:
        raise ValueError("MARKET_GAP_DELTA_PARENT_RESULT_CHANGED")
    if base.digest(prior_chain.models()[0]) != proposal["r83_result_hash"]:
        raise ValueError("LAGGED_VOLATILITY_PRIOR_CHAIN_CHANGED")
    if (
        base.digest(base.read(base.ROOT / "lagged-market-volatility-feasibility-v1/result.json"))
        != proposal["feasibility_result_hash"]
        or base.digest(base.read(base.ROOT / "fxi-full-interval-feasibility-v1/rows.json"))
        != proposal["interval_rows_hash"]
        or base.digest(base.read(base.ROOT / "lagged-market-volatility-feasibility-v1/contexts.json"))
        != proposal["contexts_hash"]
    ):
        raise ValueError("LAGGED_VOLATILITY_PROPOSED_SOURCE_CHANGED")
    groups = sorted({f["group"] for f in data.scope()})
    paths = [
        "history.json",
        "round-84/proposal-before-implementation.json",
        "round-82/result.json",
        "round-83/result.json",
        "lagged-market-volatility-feasibility-v1/contexts.json",
        "lagged-market-volatility-feasibility-v1/plan.json",
        "lagged-market-volatility-feasibility-v1/result.json",
        "fxi-full-interval-feasibility-v1/plan.json",
        "fxi-full-interval-feasibility-v1/result.json",
        "fxi-full-interval-feasibility-v1/rows.json",
    ]
    for name in CONTROLS[:2]:
        for group in groups:
            for label in ("1", "2", "3", "4", "current"):
                paths.append(control_path(name, group, label)[0].with_suffix(".json").relative_to(base.ROOT).as_posix())
    p = {
        "at": base.now().isoformat(),
        "round": 84,
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
            "Last504 mature complete marketdates/group; natural family/date weights; LR C.1/nointercept/max1500/seed17"
        ),
        "transform": (
            "raw3/strictly-prior20-date std floor1e-6 then training std; learned controls use intervalFXI; "
            "fixedrules original; no mean/intercept; threshold.5"
        ),
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


def training_rows(rows, cutoff):
    """原504日样本全部保留；上下文缺失不能偷偷缩小训练范围。"""
    chosen = baseline.training_rows(rows, cutoff)
    if any(not r["market"]["volatility_context"]["available"] for r in chosen):
        raise ValueError("LAGGED_VOLATILITY_TRAIN_CONTEXT_MISSING")
    return chosen


def fit(rows, name, cutoff):
    """保留固定LR配方与样本权重，唯一变化是逐日期的历史市场尺度。"""
    if name not in CANDIDATES:
        raise ValueError("LAGGED_VOLATILITY_RECIPE_INVALID")
    chosen = training_rows(rows, cutoff)
    original = previous.training_rows([data.original_row(r) for r in rows], cutoff)
    if [data.original_row(r) for r in chosen] != original:
        raise ValueError("LAGGED_VOLATILITY_TRAIN_ROWS_CHANGED")
    for row in chosen:
        validate_market(row["market"])
    trained = baseline.fit(rows, baseline.CANDIDATES[0], cutoff)
    return trained | {"original_fit_hash": base.digest(original)}


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
    """历史与当前对照均加载第82轮同截止完整区间头，样本与权重不变。"""
    path, original = control_path(name, group, label)
    meta = base.read(path.with_suffix(".json"))
    if (
        meta["name"] != original
        or meta["cutoff"] != cutoff
        or meta["sha256"] != hashlib.sha256(path.read_bytes()).hexdigest()
    ):
        raise ValueError("LAGGED_VOLATILITY_FROZEN_CONTROL_CHANGED")
    head = joblib.load(path)
    if head["cutoff"] != cutoff or head["mean"] != [0.0] * 3:
        raise ValueError("LAGGED_VOLATILITY_FROZEN_CONTROL_SCHEMA_INVALID")
    return head


def validate_market(value):
    """预测时再次检查三种输入定义，防止对照错误使用动态缩放后的输入。"""
    raw, captured = value["interval_market"], value["volatility_context"]
    market_vector(value)
    market_vector(raw)
    if value["available"] != raw["available"]:
        raise ValueError("LAGGED_VOLATILITY_AVAILABILITY_CHANGED")
    if value["available"]:
        scale = np.asarray(captured["scale"], dtype=float)
        if (
            not captured["available"]
            or scale.shape != (3,)
            or not np.isfinite(scale).all()
            or np.any(scale < data.FLOOR)
        ):
            raise ValueError("LAGGED_VOLATILITY_SCALE_INVALID")
        expected = (np.asarray(raw["features"]) / scale).tolist()
        if value["features"] != expected or captured["normalized_features"] != expected:
            raise ValueError("LAGGED_VOLATILITY_VECTOR_CHANGED")
    elif value["features"] != raw["features"]:
        raise ValueError("LAGGED_VOLATILITY_FALLBACK_CHANGED")
    if value["original_market"] != raw["original_market"]:
        raise ValueError("LAGGED_VOLATILITY_ORIGINAL_RULE_INPUT_CHANGED")


def batch_answers(values, name, head=None):
    if name not in BRANCHES:
        raise ValueError("LAGGED_VOLATILITY_BRANCH_INVALID")
    for value in values:
        validate_market(value)
    if name in CANDIDATES:
        original, selected = baseline.CANDIDATES[0], values
    elif name in CONTROLS[:2]:
        original = baseline.CANDIDATES[CONTROLS.index(name)]
        selected = [v["interval_market"] for v in values]
    else:
        original, selected = name, [v["original_market"] for v in values]
    return baseline.batch_answers(selected, original, head)


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
        raise ValueError("LAGGED_VOLATILITY_COMPLETED_BRANCHES_MISSING")
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
    from app.services import direction_1d_sprint_market_lagged_volatility as service

    if data.capture(base.now()) is None:
        return forward.report(service)
    return forward.tick(service)


def report():
    from app.services import direction_1d_sprint_market_child_forward_v2 as forward
    from app.services import direction_1d_sprint_market_lagged_volatility as service

    return forward.report(service)


# 未来上下文及源摘要独立冻结；缺20个观察时不写新答案。
live_market = data.live_market
live_answers = answers
