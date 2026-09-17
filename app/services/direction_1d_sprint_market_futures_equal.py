"""第101轮：固定平均两套冻结期货模型的研究分数，不拟合、不搜索组合权重。"""

import math
import shutil
from collections import defaultdict

from app.services import direction_1d_sprint as base
from app.services import direction_1d_sprint_core_forward as core
from app.services import direction_1d_sprint_market_futures_basis_v2 as basis
from app.services import direction_1d_sprint_market_futures_intraday as intraday
from app.services import direction_1d_sprint_market_option_futures as previous

original, active, reference = previous.original, previous.active, previous.reference
CANDIDATES = ("EQUAL_R94_SIGN_R96_INTRADAY",)
MEMBERS = ("INTERVAL_SIGN3_IF_BASIS_LR4_504", "SIGN3_IF_INTRADAY5_LR504")
CONTROLS = intraday.CONTROLS + ("FROZEN_R94_LAGGED_SIGN4", "FROZEN_R96_INTRADAY5")
BRANCHES = CANDIDATES + CONTROLS
FIRST_TARGET = "2026-09-17"
PROPOSAL_HASH = "dabd2fc6ea2dca95ecdf48f7a41dfac60cbca131d1c6dee1a5985db4076db3ce"


def root():
    return base.ROOT / "round-101"


def require(condition, reason):
    if not condition:
        raise ValueError("FUTURES_EQUAL_" + reason)


def fingerprint():
    code = dict(base.read(previous.root() / "plan.json")["fingerprint"]["code"])
    for name, expected in code.items():
        require(core.sha(base.PROJECT / name) == expected, "OLD_CODE_CHANGED")
    for name in (
        "app/services/direction_1d_sprint_market_futures_equal.py",
        "app/services/direction_1d_sprint_futures_equal_forward.py",
        "scripts/direction_1d_sprint_market_futures_equal.py",
        "tests/test_direction_1d_sprint_market_futures_equal.py",
    ):
        code[name] = core.sha(base.PROJECT / name)
    return {"code": code}


def plan():
    proposal = base.read(root() / "proposal-before-implementation.json")
    require(base.digest(proposal) == PROPOSAL_HASH, "PROPOSAL_CHANGED")
    require(core.sha(root() / "design-before-implementation.md") == proposal["design_sha256"], "DESIGN_CHANGED")
    directory = base.ROOT / "futures-equal-ensemble-feasibility-v1"
    for file, key in (("result.json", "feasibility_hash"), ("plan.json", "feasibility_plan_hash")):
        require(base.digest(base.read(directory / file)) == proposal[key], "FEASIBILITY_CHANGED")
    for file, key in (
        ("check.py", "feasibility_script_sha256"),
        ("design-before-analysis.md", "feasibility_design_sha256"),
    ):
        require(core.sha(directory / file) == proposal[key], "FEASIBILITY_CODE_CHANGED")
    for module in (basis, intraday):
        meta = proposal["sources"][module.root().name]
        manifest = base.read(module.root() / "result.json")
        require(base.digest(manifest) == meta["result_hash"], "SOURCE_RESULT_CHANGED")
        require(base.digest(base.read(module.root() / "plan.json")) == meta["plan_hash"], "SOURCE_PLAN_CHANGED")
        require(core.sha(module.root() / "models.joblib") == meta["model_sha256"], "SOURCE_MODEL_CHANGED")
    require(proposal["candidates"] == list(CANDIDATES) and proposal["controls"] == list(CONTROLS), "BRANCHES_CHANGED")
    require(
        proposal["weights"] == [0.5, 0.5] and proposal["threshold"] == 0.5 and proposal["new_fits"] == 0,
        "RECIPE_CHANGED",
    )
    context = {
        "fingerprint": fingerprint(),
        "calendar_hash": base.calendar()[1],
        "scope_hash": base.digest(basis.data.scope()),
    }
    path = root() / "plan.json"
    if path.exists():
        p = base.read(path)
        require(
            all(p[k] == v for k, v in context.items()) and p["proposal_hash"] == PROPOSAL_HASH, "FROZEN_CONTEXT_CHANGED"
        )
        return p
    active()
    p = proposal | context | {"at": base.now().isoformat(), "proposal_hash": PROPOSAL_HASH}
    base.save(path, p)
    for name in p["fingerprint"]["code"]:
        dest = root() / "code" / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(base.PROJECT / name, dest)
    return p


def combine(a, b, fallback):
    """只合并两套指定模型；缺源保留原SIGN整份答案，非法分数直接报错。"""
    require(type(fallback["prediction"]) is int and fallback["prediction"] in (0, 1), "FALLBACK_INVALID")
    if a["route"] != MEMBERS[0] or b["route"] != MEMBERS[1]:
        return dict(fallback)
    for value in (a, b):
        score = value["research_score"]
        require(
            value["kind"] == "UNCALIBRATED_UP_SCORE"
            and type(score) in (int, float)
            and math.isfinite(score)
            and 0 <= score <= 1,
            "MEMBER_SCORE_INVALID",
        )
        require(value["prediction"] == int(score >= 0.5), "MEMBER_DIRECTION_INVALID")
    score = 0.5 * a["research_score"] + 0.5 * b["research_score"]
    return {
        "prediction": int(score >= 0.5),
        "research_score": float(score),
        "kind": "UNCALIBRATED_UP_SCORE",
        "route": CANDIDATES[0],
    }


def source_bundles(p):
    """加载原模型自身校验过的头，并确认同组训练行、权重和成熟截止日一致。"""
    manifests, bundles = zip(*(module.models() for module in (basis, intraday)), strict=True)
    groups = set(bundles[0][MEMBERS[0]])
    require(groups == set(bundles[1][MEMBERS[1]]), "MODEL_GROUP_CHANGED")
    for module, manifest in zip((basis, intraday), manifests, strict=True):
        require(base.digest(manifest) == p["sources"][module.root().name]["result_hash"], "MODEL_RESULT_CHANGED")
    for group in groups:
        a, b = (bundles[i][MEMBERS[i]][group] for i in range(2))
        require(
            a["cutoff"] == b["cutoff"] == p["current_cutoff"] and a["fit_dates"] == b["fit_dates"] == 504,
            "CURRENT_CUTOFF_CHANGED",
        )
        require(a["fit_hash"] == b["fit_hash"] and a["weight_hash"] == b["weight_hash"], "CURRENT_TRAINING_CHANGED")
    return bundles


def descriptor(p):
    return {
        "sources": p["sources"],
        "weights": p["weights"],
        "threshold": p["threshold"],
        "current_cutoff": p["current_cutoff"],
        "new_fits": 0,
    }


def fold(p, number, year, q, group, name):
    path = base.ROOT / f"round-{number}" / "folds" / str(year) / f"{q}-{group}-{name}.json"
    rows = base.read(path)
    require(base.digest(rows) == p["fold_hashes"][str(path.relative_to(base.ROOT))], "SOURCE_FOLD_CHANGED")
    return rows


def question(row):
    keys = ("code", "family", "group", "t", "u", "y", "actual_direction")
    return {k: row[k] for k in keys} | ({"old_question": row["old_question"]} if "old_question" in row else {})


def prepare():
    p = plan()
    if (root() / "result.json").exists():
        return models()[0]
    bundles = source_bundles(p)
    groups = sorted(bundles[0][MEMBERS[0]])
    output, checks = {}, []
    for year in p["years"]:
        branches = defaultdict(list)
        for q in range(1, 5):
            for group in groups:
                active()
                values = {n: fold(p, 96, year, q, group, n) for n in intraday.CONTROLS}
                values[CONTROLS[-2]] = fold(p, 94, year, q, group, MEMBERS[0])
                values[CONTROLS[-1]] = fold(p, 96, year, q, group, MEMBERS[1])
                fallback = values[CONTROLS[1]]
                identity = [question(r) for r in fallback]
                require(
                    all([question(r) for r in rows] == identity for rows in values.values()), "EXAM_IDENTITY_CHANGED"
                )
                values[CANDIDATES[0]] = [
                    question(f) | combine(a, b, f)
                    for a, b, f in zip(values[CONTROLS[-2]], values[CONTROLS[-1]], fallback, strict=True)
                ]
                for name, rows in values.items():
                    path = root() / "folds" / str(year) / f"{q}-{group}-{name}.json"
                    if path.exists():
                        require(base.read(path) == rows, "SAVED_FOLD_CHANGED")
                    else:
                        base.save(path, rows)
                    branches[name].extend(rows)
                checks.append(
                    {"year": year, "quarter": q, "group": group, "questions": len(fallback), "same_questions": True}
                )
        rows = branches[CANDIDATES[0]]
        require(
            len(rows) == p["expected_questions"][str(year)]
            and len({r["u"] for r in rows}) == p["expected_dates"][str(year)],
            "EXAM_COVERAGE_CHANGED",
        )
        require(sum(r["route"] != CANDIDATES[0] for r in rows) == p["expected_fallback"][str(year)], "FALLBACK_CHANGED")
        output[str(year)] = branches
    base.save(root() / "models.json", descriptor(p))
    result = {
        "at": base.now().isoformat(),
        "plan_hash": base.digest(p),
        "fingerprint": fingerprint(),
        "model_sha256": core.sha(root() / "models.json"),
        "metrics": {y: {n: base.metrics(v) for n, v in branches.items()} for y, branches in output.items()},
        "group_metrics": {
            y: {
                n: {g: base.metrics([r for r in rows if r["group"] == g]) for g in groups}
                for n, rows in branches.items()
            }
            for y, branches in output.items()
        },
        "comparison_checks": checks,
        "development_fits": 0,
        "current_fits": 0,
        "new_2026_scores": False,
        "new_source_requests": 0,
        "new_cost_cny": 0,
        "kind": "HISTORICAL_DEVELOPMENT_ONLY",
        "status": "MODEL_NOT_RELEASED",
        "source_limit": (
            "Previously inspected years, no untouched holdout; "
            "historical first publication unverified; scores uncalibrated"
        ),
    }
    base.save(root() / "result.json", result)
    return result


def models():
    p, result = plan(), base.read(root() / "result.json")
    require(
        result["plan_hash"] == base.digest(p)
        and result["fingerprint"] == fingerprint()
        and result["model_sha256"] == core.sha(root() / "models.json"),
        "MODEL_CHANGED",
    )
    require(base.read(root() / "models.json") == descriptor(p), "DESCRIPTOR_CHANGED")
    return result, source_bundles(p)


def preflight():
    from app.services import direction_1d_sprint_futures_equal_forward as forward

    return forward.preflight()


def tick():
    from app.services import direction_1d_sprint_futures_equal_forward as forward

    return forward.run()


def report():
    from app.services import direction_1d_sprint_futures_equal_forward as forward

    return forward.report()
