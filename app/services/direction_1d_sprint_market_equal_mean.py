"""第87轮：完整区间幅度/方向模型的固定等权组合，不重新拟合任何成员。"""

import hashlib
import shutil
from collections import defaultdict

import joblib
import numpy as np

from app.services import direction_1d_sprint as base
from app.services import direction_1d_sprint_market_fxi_interval as original
from app.services import direction_1d_sprint_market_volatility_regime as previous

CANDIDATES = ("MEAN_RAW_SIGN_INTERVAL_LR504",)
CONTROLS = (
    "FROZEN_R82_INTERVAL_RAW3",
    "FROZEN_R82_INTERVAL_SIGN3",
    "MARKET_MAJORITY3",
    "SPX_SIGN",
    "ALWAYS_UP",
)
BRANCHES = CANDIDATES + CONTROLS
PROPOSAL_HASH = "e3dc0d0bf3e79078f68f0300af4de41c1f1b168430e264ad46ffd950564b566b"
FIRST_TARGET = "2026-09-16"
data, active = original.data, original.active


def root():
    return base.ROOT / "round-87"


def require(condition, reason):
    if not condition:
        raise ValueError("EQUAL_MEAN_" + reason)


def fingerprint():
    value = previous.fingerprint()
    for name in (
        "app/services/direction_1d_sprint_market_equal_mean.py",
        "scripts/direction_1d_sprint_market_equal_mean.py",
        "tests/test_direction_1d_sprint_market_equal_mean.py",
    ):
        value["code"][name] = hashlib.sha256((base.PROJECT / name).read_bytes()).hexdigest()
    return value


def validate_spec(p):
    require(
        p["candidates"] == list(CANDIDATES)
        and p["controls"] == list(CONTROLS)
        and p["weights"] == [0.5, 0.5]
        and p["threshold"] == 0.5
        and p["max_development_fits"] == p["max_current_fits"] == 0
        and p["reproductions"] == 1
        and p["branch_count"] == 6
        and p["preflight_branch_checks"] == 180
        and p["current_cutoff"] == "2026-09-16"
        and p["years"] == [2024, 2025]
        and p["expected_questions"] == {"2024": 7260, "2025": 7290}
        and p["expected_dates"] == {"2024": 242, "2025": 243},
        "DECLARED_RECIPE_CHANGED",
    )


def parent_results():
    """当前成员与现有任务版本分别验证；2024新头保留在第86轮，只用于该年复核。"""
    r82, _ = original.models()
    r85, _ = previous.models()
    p86, r86 = (base.read(base.ROOT / "round-86" / n) for n in ("plan.json", "result.json"))
    require(r86["plan_hash"] == base.digest(p86), "REPLICATION_PLAN_CHANGED")
    require(p86["fingerprint"] == previous.fingerprint(), "REPLICATION_CODE_CHANGED")
    require(
        p86["script_sha256"] == hashlib.sha256((base.ROOT / "round-86/replicate.py").read_bytes()).hexdigest(),
        "REPLICATION_SCRIPT_CHANGED",
    )
    return {
        "r82_result_hash": base.digest(r82),
        "r85_result_hash": base.digest(r85),
        "r86_result_hash": base.digest(r86),
    }


def plan():
    proposal = base.read(root() / "proposal-before-implementation.json")
    validate_spec(proposal)
    require(base.digest(proposal) == PROPOSAL_HASH, "PROPOSAL_CHANGED")
    require(
        proposal["design_sha256"]
        == hashlib.sha256((root() / "design-before-implementation.md").read_bytes()).hexdigest(),
        "DESIGN_CHANGED",
    )
    sources = parent_results()
    require(all(proposal[k] == v for k, v in sources.items()), "SOURCE_RESULTS_CHANGED")
    path = root() / "plan.json"
    if path.exists():
        p = base.read(path)
        require(
            p["fingerprint"] == fingerprint()
            and p["calendar_hash"] == base.calendar()[1]
            and p["scope_hash"] == base.digest(data.scope())
            and p["sources"] == sources,
            "FROZEN_CONTEXT_CHANGED",
        )
        return p
    active()
    p = {
        "at": base.now().isoformat(),
        "round": 87,
        "proposal_hash": PROPOSAL_HASH,
        "fingerprint": fingerprint(),
        "calendar_hash": base.calendar()[1],
        "scope_hash": base.digest(data.scope()),
        "sources": sources,
        "candidates": list(CANDIDATES),
        "controls": list(CONTROLS),
        "weights": [0.5, 0.5],
        "threshold": 0.5,
        "formula": "0.5*R82_RAW_UP_SCORE+0.5*R82_SIGN_UP_SCORE",
        "current_fit_cutoff": "2026-09-16",
        "first_forward_target": FIRST_TARGET,
        "years": [2024, 2025],
        "expected_questions": proposal["expected_questions"],
        "expected_dates": proposal["expected_dates"],
        "development_fits": 0,
        "current_fits": 0,
        "reproductions": 1,
        "branch_count": 6,
        "preflight_branch_checks": 180,
        "model_format": "VERIFIED_R82_CURRENT_MEMBERS_PLUS_FROZEN_EQUAL_MEAN",
        "new_2026_scores": False,
        "new_cost_cny": 0,
        "new_provider_requests": 0,
        "status": "MODEL_NOT_RELEASED",
    }
    base.save(path, p)
    for name in p["fingerprint"]["code"]:
        destination = root() / "code" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(base.PROJECT / name, destination)
    return p


def blend(a, z):
    """只组合相同一日问题的UP分数；缺一个成员或混入纠错分数都拒绝。"""
    scored = ["research_score" in v for v in (a, z)]
    require(scored[0] == scored[1], "PARTIAL_SCORE_MISSING")
    if not scored[0]:
        require(
            a == z
            and a["kind"] == "FIXED_DIRECTION"
            and a["route"] == "SPX_SIGN_FALLBACK"
            and type(a["prediction"]) is int
            and a["prediction"] in (0, 1),
            "MEMBER_FALLBACK_DIFFERS",
        )
        return dict(a)
    for v in (a, z):
        score = v["research_score"]
        require(
            type(score) in (int, float)
            and np.isfinite(score)
            and 0 <= score <= 1
            and v["kind"] == "UNCALIBRATED_UP_SCORE"
            and v["route"] == "MARKET_LR3"
            and type(v["prediction"]) is int
            and v["prediction"] == int(score >= 0.5),
            "MEMBER_SCORE_INVALID",
        )
    score = 0.5 * a["research_score"] + 0.5 * z["research_score"]
    return {
        "prediction": int(score >= 0.5),
        "research_score": float(score),
        "kind": "UNCALIBRATED_UP_SCORE",
        "route": CANDIDATES[0],
    }


def batch(values, heads):
    """两成员都用完整区间，固定三票仍沿原FXI口径；一次计算供六个分支共享。"""
    members = [
        original.batch_answers(values, native, head) for native, head in zip(original.CANDIDATES, heads, strict=True)
    ]
    result = {CONTROLS[i]: choices for i, choices in enumerate(members)}
    result[CANDIDATES[0]] = [blend(a, z) for a, z in zip(*members, strict=True)]
    result.update({name: original.batch_answers(values, name) for name in CONTROLS[2:]})
    return result


def answers(market, group, bundle):
    heads = [bundle[name][group] for name in original.CANDIDATES]
    return {name: choices[0] for name, choices in batch([market], heads).items()}


def head_path(year, quarter, group, member):
    if year == 2025:
        return original.root() / "checkpoints" / f"{quarter}-{group}-{original.CANDIDATES[member]}.joblib"
    require(year == 2024, "UNDECLARED_EXAM_YEAR")
    cutoff = f"2024-{quarter * 3 - 2:02d}-01"
    name = ("INTERVAL_RAW3", "INTERVAL_SIGN3")[member]
    return base.ROOT / "round-86/checkpoints" / f"{cutoff}-{group}-{name}.joblib"


def historical_head(year, quarter, group, member):
    path = head_path(year, quarter, group, member)
    meta = base.read(path.with_suffix(".json"))
    cutoff = f"{year}-{quarter * 3 - 2:02d}-01"
    native = original.CANDIDATES[member] if year == 2025 else ("INTERVAL_RAW3", "INTERVAL_SIGN3")[member]
    parent_plan = original.plan() if year == 2025 else base.read(base.ROOT / "round-86/plan.json")
    require(
        meta["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
        and meta["cutoff"] == cutoff
        and meta["name"] == native
        and meta["plan_hash"] == base.digest(parent_plan),
        "HISTORICAL_MEMBER_CHANGED",
    )
    head = joblib.load(path)
    require(
        head["cutoff"] == cutoff
        and head["fit_end"] < cutoff
        and head["max_mature_date"] < cutoff
        and head["fit_dates"] == 504
        and head["mean"] == [0.0] * 3
        and head["signed"] == bool(member),
        "HISTORICAL_MEMBER_SCOPE_INVALID",
    )
    return head


def source_fold(year, quarter, group, name):
    require(name in CONTROLS, "SOURCE_CONTROL_INVALID")
    if name in CONTROLS[:2]:
        index = CONTROLS.index(name)
        native = original.CANDIDATES[index] if year == 2025 else ("INTERVAL_RAW3", "INTERVAL_SIGN3")[index]
    else:
        native = name
    folder = original.root() if year == 2025 else base.ROOT / "round-86"
    return base.read(folder / "folds" / f"{quarter}-{group}-{native}.json")


def manifest_value(p, source):
    return {
        "plan_hash": base.digest(p),
        "source_result_hash": base.digest(source),
        "source_model_sha256": source["model_sha256"],
        "members": list(original.CANDIDATES),
        "current_cutoff": p["current_fit_cutoff"],
        "weights": [0.5, 0.5],
        "threshold": 0.5,
        "model_format": p["model_format"],
        "new_fits": 0,
    }


def prepare():
    """复算所有父控制并验证原题后保存组合答案；不会训练或搜索任何参数。"""
    p = plan()
    if (root() / "result.json").exists():
        return models()[0]
    rows, proof = data.dataset()
    base.save(root() / "question-proof.json", proof)
    output = {str(year): defaultdict(list) for year in p["years"]}
    groups = sorted({r["group"] for r in rows})
    checks = []
    for year in p["years"]:
        for q in range(1, 5):
            start = f"{year}-{q * 3 - 2:02d}-01"
            end = f"{year + 1}-01-01" if q == 4 else f"{year}-{q * 3 + 1:02d}-01"
            for group in groups:
                active()
                exam = [r for r in rows if r["group"] == group and start <= r["u"] < end]
                heads = [historical_head(year, q, group, i) for i in range(2)]
                chosen = original.training_rows([r for r in rows if r["group"] == group], start)
                require(all(h["fit_hash"] == base.digest(chosen) for h in heads), "MEMBER_TRAINING_ROWS_CHANGED")
                require(heads[0]["weight_hash"] == heads[1]["weight_hash"], "MEMBER_WEIGHTS_CHANGED")
                values = batch([r["market"] for r in exam], heads)
                for name, choices in values.items():
                    keys = ("code", "family", "group", "t", "u", "y", "actual_direction") + (
                        ("old_question",) if year == 2025 else ()
                    )
                    scored = [{k: r[k] for k in keys} | answer for r, answer in zip(exam, choices, strict=True)]
                    if name in CONTROLS:
                        require(scored == source_fold(year, q, group, name), "CONTROL_ANSWERS_CHANGED")
                    path = root() / "folds" / str(year) / f"{q}-{group}-{name}.json"
                    if path.exists():
                        require(base.read(path) == scored, "PREVIOUS_COMBINATION_ANSWERS_CHANGED")
                    else:
                        base.save(path, scored)
                    output[str(year)][name].extend(scored)
                checks.append(
                    {
                        "year": year,
                        "quarter": q,
                        "group": group,
                        "members": 2,
                        "same_training_rows": True,
                        "questions": len(exam),
                    }
                )
            base.save(
                root() / "progress.json", {"at": base.now().isoformat(), "year": year, "quarter": q}, replace=True
            )
        keys = sorted((r["code"], r["u"], r["y"]) for r in output[str(year)]["SPX_SIGN"])
        require(
            len(keys) == p["expected_questions"][str(year)]
            and len({u for _, u, _ in keys}) == p["expected_dates"][str(year)]
            and all(
                sorted((r["code"], r["u"], r["y"]) for r in choices) == keys for choices in output[str(year)].values()
            ),
            "COMMON_EXAM_CHANGED",
        )
    source, _ = original.models()
    base.save(root() / "models-manifest.json", manifest_value(p, source))
    result = {
        "at": base.now().isoformat(),
        "plan_hash": base.digest(p),
        "fingerprint": fingerprint(),
        "model_sha256": hashlib.sha256((root() / "models-manifest.json").read_bytes()).hexdigest(),
        "model_format": p["model_format"],
        "metrics": {year: {n: base.metrics(v) for n, v in branches.items()} for year, branches in output.items()},
        "group_metrics": {
            year: {n: {g: base.metrics([r for r in v if r["group"] == g]) for g in groups} for n, v in branches.items()}
            for year, branches in output.items()
        },
        "member_checks": checks,
        "development_fits": 0,
        "current_fits": 0,
        "new_2026_scores": False,
        "new_cost_cny": 0,
        "kind": "HISTORICAL_DEVELOPMENT_ONLY",
        "source_limit": (
            "Both years previously inspected; not pristine holdout; "
            "equal uncalibrated scores are not formal probability"
        ),
        "status": "MODEL_NOT_RELEASED",
    }
    base.save(root() / "result.json", result)
    return result


def models():
    p, result = plan(), base.read(root() / "result.json")
    source, bundle = original.models()
    path = root() / "models-manifest.json"
    require(
        result["plan_hash"] == base.digest(p)
        and result["fingerprint"] == fingerprint()
        and result["model_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
        and base.read(path) == manifest_value(p, source),
        "MODEL_MANIFEST_CHANGED",
    )
    for native in original.CANDIDATES:
        require(set(bundle[native]) == {r["group"] for r in data.scope()}, "CURRENT_GROUP_MISSING")
        require(all(h["cutoff"] == p["current_fit_cutoff"] for h in bundle[native].values()), "CURRENT_CUTOFF_CHANGED")
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
    from app.services import direction_1d_sprint_market_equal_mean as service

    return forward.tick(service)


def report():
    from app.services import direction_1d_sprint_market_child_forward_v2 as forward
    from app.services import direction_1d_sprint_market_equal_mean as service

    return forward.report(service)


live_market = data.live_market
live_answers = answers
