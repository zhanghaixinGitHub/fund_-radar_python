"""第82轮：FXI完整海外交易区间与原最新单日输入的对照。

两个无截距逻辑回归候选与五个对照使用同题、同一SPX回退；只用当时成熟的标签。
"""

import hashlib
import shutil
from collections import defaultdict

import joblib

from app.services import direction_1d_sprint as base
from app.services import direction_1d_sprint_market_current_refresh as current
from app.services import direction_1d_sprint_market_fxi_interval_data as data
from app.services import direction_1d_sprint_market_nonnegative as previous
from app.services import direction_1d_sprint_market_only as baseline
from app.services import direction_1d_sprint_return_target as regression

CANDIDATES = ("FULL_INTERVAL_FXI_RAW3_LR504", "FULL_INTERVAL_FXI_SIGN3_LR504")
CONTROLS = (
    "CURRENT_MATCHED_MARKET_LR3",
    "CURRENT_MATCHED_MARKET_SIGN_LR3",
    "MARKET_MAJORITY3",
    "SPX_SIGN",
    "ALWAYS_UP",
)
BRANCHES = CANDIDATES + CONTROLS

FIRST_TARGET = "2026-09-16"
PROPOSAL_HASH = "410d59a5f115c0ad58411dde5e56b3fc0b57f1e4ff04487b81f4eaa00e0146c6"
active = regression.active


def root():
    return base.ROOT / "round-82"


def fingerprint():
    value = previous.fingerprint()
    for name in (
        "app/services/direction_1d_sprint_market_fxi_interval.py",
        "scripts/direction_1d_sprint_market_fxi_interval.py",
        "tests/test_direction_1d_sprint_market_fxi_interval.py",
        "app/services/direction_1d_sprint_market_fxi_interval_data.py",
        "tests/test_direction_1d_sprint_market_fxi_interval_data.py",
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
        or proposal["max_development_fits"] != 24
        or proposal["max_current_fits"] != 6
        or proposal["reproductions"] != 1
        or proposal["current_cutoff"] != "2026-09-16"
        or proposal["control_input_policy"] != "ALL_CONTROLS_USE_ORIGINAL_FXI_FEATURE"
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
    if base.digest(previous.models()[0]) != proposal["r81_result_hash"]:
        raise ValueError("MARKET_GAP_DELTA_PARENT_RESULT_CHANGED")
    if (
        base.digest(baseline.models()[0]) != proposal["r70_result_hash"]
        or base.digest(current.models()[0]) != proposal["r79_result_hash"]
        or base.digest(base.read(base.ROOT / "fxi-full-interval-feasibility-v1/result.json"))
        != proposal["feasibility_result_hash"]
        or base.digest(base.read(base.ROOT / "fxi-full-interval-feasibility-v1/rows.json"))
        != proposal["interval_rows_hash"]
    ):
        raise ValueError("MARKET_NASDAQ_RELATIVE_PROPOSED_SOURCE_CHANGED")
    groups = sorted({f["group"] for f in data.scope()})
    paths = [
        "history.json",
        "round-82/proposal-before-implementation.json",
        "fxi-full-interval-feasibility-v1/plan.json",
        "fxi-full-interval-feasibility-v1/result.json",
        "fxi-full-interval-feasibility-v1/rows.json",
        "round-70/result.json",
        "round-79/result.json",
        "round-81/result.json",
    ]
    for name in CONTROLS[:2]:
        for group in groups:
            for label in ("1", "2", "3", "4", "current"):
                paths.append(control_path(name, group, label)[0].with_suffix(".json").relative_to(base.ROOT).as_posix())
    p = {
        "at": base.now().isoformat(),
        "round": 82,
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
        "transform": (
            "raw3/std or sign3 with full-intervalFXI; controls use originalFXI; no mean/intercept; threshold.5"
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
training_rows = baseline.training_rows


def fit(rows, name, cutoff):
    """仅替换候选的FXI输入；原LR配方直接复用，以便把差异归于输入区间。"""
    if name not in CANDIDATES:
        raise ValueError("FXI_INTERVAL_RECIPE_INVALID")
    trained = baseline.fit(rows, baseline.CANDIDATES[CANDIDATES.index(name)], cutoff)
    chosen = training_rows(rows, cutoff)
    return trained | {"original_fit_hash": base.digest([data.original_row(r) for r in chosen])}


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
        raise ValueError("FXI_INTERVAL_BRANCH_INVALID")
    for value in values:
        market_vector(value)
        market_vector(value["original_market"])
        if (
            value["features"][0] != value["original_market"]["features"][0]
            or value["features"][2] != value["original_market"]["features"][2]
        ):
            raise ValueError("FXI_INTERVAL_OTHER_FEATURE_CHANGED")
    if name in CANDIDATES:
        original = baseline.CANDIDATES[CANDIDATES.index(name)]
        selected = values
    else:
        original = baseline.CANDIDATES[CONTROLS.index(name)] if name in CONTROLS[:2] else name
        selected = [v["original_market"] for v in values]
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
    from app.services import direction_1d_sprint_market_fxi_interval as service

    return forward.tick(service)


def report():
    from app.services import direction_1d_sprint_market_child_forward_v2 as forward
    from app.services import direction_1d_sprint_market_fxi_interval as service

    return forward.report(service)


# 重验父ETF原始字节并解析完整区间；预测同时保留控制需要的原始市场输入。
live_market = data.live_market
live_answers = answers
