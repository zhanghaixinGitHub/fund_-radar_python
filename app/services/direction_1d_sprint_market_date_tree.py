"""第76轮：交易日汇总与逐基金记录的受控树模型对照。

两个三输入树候选与五个对照使用同题、同一SPX回退；训练单位变化不改变业务标签。
"""

import hashlib
import shutil
from collections import defaultdict

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from threadpoolctl import threadpool_limits

from app.services import direction_1d_sprint as base
from app.services import direction_1d_sprint_market_only as baseline
from app.services import direction_1d_sprint_market_only_data as data
from app.services import direction_1d_sprint_market_sign_gap as previous
from app.services import direction_1d_sprint_return_target as regression

CANDIDATES = ("EXPANDED_ROW_ETR3_504", "EXPANDED_DATE_ETR3_504")
CONTROLS = ("FROZEN_R70_MARKET_LR3", "FROZEN_R70_MARKET_SIGN_LR3", "MARKET_MAJORITY3", "SPX_SIGN", "ALWAYS_UP")
BRANCHES = CANDIDATES + CONTROLS
TRAINING_UNITS = {CANDIDATES[0]: "FUND_ROW", CANDIDATES[1]: "FAMILY_WEIGHTED_DATE"}
FIRST_TARGET = "2026-09-16"
PROPOSAL_HASH = "0df43e200bba1d21dd7eea7bc104f763de20e6ef813eb95dba6d7973a50e1428"
active = regression.active


def root():
    return base.ROOT / "round-76"


def fingerprint():
    value = previous.fingerprint()
    for name in (
        "app/services/direction_1d_sprint_market_date_tree.py",
        "scripts/direction_1d_sprint_market_date_tree.py",
        "tests/test_direction_1d_sprint_market_date_tree.py",
    ):
        value["code"][name] = hashlib.sha256((base.PROJECT / name).read_bytes()).hexdigest()
    return value


def control_path(name, group, label):
    if name not in CONTROLS[:2]:
        raise ValueError("MARKET_DATE_TREE_CONTROL_INVALID")
    original = baseline.CANDIDATES[CONTROLS.index(name)]
    return baseline.root() / "checkpoints" / f"{label}-{group}-{original}.joblib", original


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
        or set(TRAINING_UNITS) != set(CANDIDATES)
    ):
        raise ValueError("MARKET_DATE_TREE_DECLARED_BRANCHES_OR_BUDGET_CHANGED")


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
    if base.digest(previous.models()[0]) != proposal["r75_result_hash"]:
        raise ValueError("MARKET_GAP_DELTA_PARENT_RESULT_CHANGED")
    if (
        base.digest(baseline.models()[0]) != proposal["r70_result_hash"]
        or base.digest(base.read(base.ROOT / "nav-independent-market-feasibility-v1/rows.json"))
        != proposal["expanded_rows_hash"]
    ):
        raise ValueError("MARKET_DATE_TREE_PROPOSED_SOURCE_CHANGED")
    groups = sorted({f["group"] for f in data.scope()})
    paths = [
        "history.json",
        "round-76/proposal-before-implementation.json",
        "nav-independent-market-feasibility-v1/plan.json",
        "nav-independent-market-feasibility-v1/result.json",
        "nav-independent-market-feasibility-v1/rows.json",
        "round-70/result.json",
        "round-75/result.json",
    ]
    for name in CONTROLS[:2]:
        for group in groups:
            for label in ("1", "2", "3", "4", "current"):
                paths.append(control_path(name, group, label)[0].with_suffix(".json").relative_to(base.ROOT).as_posix())
    p = {
        "at": base.now().isoformat(),
        "round": 76,
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
        "training": (
            "Same504mature marketdates/group; natural family/date weights; ETR256/depth6/leaf60/seed17/n_jobs2"
        ),
        "transform": "raw3 market inputs; ROW uses0/1, DATE uses family-weighted daily UP fraction; threshold.5",
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


def training_arrays(chosen, unit):
    """同组同日输入相同才允许汇总；家族份额权重和每日期总权重沿用原始样本。

    日期目标是当时训练标签的加权上涨比例，预测业务仍是每只基金下一日是否上涨。
    平方损失汇总后只去掉与模型预测无关的组内标签方差，不改变最优概率的目标。
    """
    if unit not in TRAINING_UNITS.values() or not chosen:
        raise ValueError("MARKET_DATE_TREE_TRAINING_UNIT_INVALID")
    x = np.asarray([market_vector(r["market"]) for r in chosen])
    y = np.asarray([r["y"] for r in chosen], dtype=float)
    weights = regression.weights(chosen)
    if not np.isfinite(y).all() or not np.isin(y, [0.0, 1.0]).all():
        raise ValueError("MARKET_DATE_TREE_NONBINARY_LABEL")
    if unit == "FUND_ROW":
        return x, y, weights, [r["u"] for r in chosen]
    grouped = defaultdict(list)
    for i, row in enumerate(chosen):
        grouped[row["u"]].append(i)
    if len({r["group"] for r in chosen}) != 1:
        raise ValueError("MARKET_DATE_TREE_MIXED_ASSET_GROUPS")
    days, xx, yy, ww = sorted(grouped), [], [], []
    for day in days:
        indices = grouped[day]
        if not np.all(x[indices] == x[indices[0]]):
            raise ValueError("MARKET_DATE_TREE_DIFFERENT_INPUTS_SAME_DATE")
        xx.append(x[indices[0]])
        yy.append(float(np.average(y[indices], weights=weights[indices])))
        ww.append(float(weights[indices].sum()))
    return np.asarray(xx), np.asarray(yy), np.asarray(ww), days


def fit(rows, name, cutoff):
    """叶节点最少60个训练单位；DATE候选的一个单位严格对应一个成熟交易日。"""
    if name not in CANDIDATES:
        raise ValueError("MARKET_DATE_TREE_RECIPE_INVALID")
    chosen = training_rows(rows, cutoff)
    unit = TRAINING_UNITS[name]
    x, y, weights, dates = training_arrays(chosen, unit)
    active()
    with threadpool_limits(limits=2):
        model = ExtraTreesRegressor(
            n_estimators=256,
            max_depth=6,
            min_samples_leaf=60,
            max_features=1.0,
            criterion="squared_error",
            bootstrap=False,
            random_state=17,
            n_jobs=2,
        )
        model.fit(x, y, sample_weight=weights)
    active()
    return {
        "model": model,
        "training_unit": unit,
        "fit_hash": base.digest(chosen),
        "weight_hash": base.digest(regression.weights(chosen).tolist()),
        "training_arrays_hash": base.digest(
            {"x": x.tolist(), "y": y.tolist(), "weights": weights.tolist(), "dates": dates}
        ),
        "fit_rows": len(chosen),
        "fit_dates": len(set(dates)),
        "estimator_training_units": len(x),
        "fit_end": max(r["u"] for r in chosen),
        "max_mature_date": max(r["mature"] for r in chosen),
        "cutoff": cutoff,
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
    """对照直接加载第70轮原始三输入头；检查摘要后沿用其原转换。"""
    path, original = control_path(name, group, label)
    meta = base.read(path.with_suffix(".json"))
    if (
        meta["name"] != original
        or meta["cutoff"] != cutoff
        or meta["sha256"] != hashlib.sha256(path.read_bytes()).hexdigest()
    ):
        raise ValueError("MARKET_DATE_TREE_FROZEN_CONTROL_CHANGED")
    head = joblib.load(path)
    if head["cutoff"] != cutoff or head["mean"] != [0.0] * 3:
        raise ValueError("MARKET_DATE_TREE_FROZEN_CONTROL_SCHEMA_INVALID")
    return head


def batch_answers(values, name, head=None):
    if name not in BRANCHES:
        raise ValueError("MARKET_DATE_TREE_BRANCH_INVALID")
    if name in CONTROLS:
        original = baseline.CANDIDATES[CONTROLS.index(name)] if name in CONTROLS[:2] else name
        return baseline.batch_answers(values, original, head)
    if not values:
        return []
    x = np.asarray([market_vector(v) for v in values])
    out = [{"prediction": int(z[0] >= 0), "kind": "FIXED_DIRECTION", "route": "SPX_SIGN_FALLBACK"} for z in x]
    indices = [i for i, v in enumerate(values) if v["available"]]
    if indices:
        if head is None or head["training_unit"] != TRAINING_UNITS[name] or head["model"].n_features_in_ != 3:
            raise ValueError("MARKET_DATE_TREE_HEAD_SCHEMA_INVALID")
        scores = tree_scores(head["model"], x[indices])
        if not np.isfinite(scores).all() or min(scores) < 0 or max(scores) > 1:
            raise ValueError("MARKET_DATE_TREE_SCORE_INVALID")
        for i, score in zip(indices, scores, strict=True):
            out[i] = {
                "prediction": int(score >= 0.5),
                "research_score": float(score),
                "kind": "UNCALIBRATED_UP_SCORE",
                "route": "MARKET_" + TRAINING_UNITS[name] + "_ETR3",
            }
    return out


def tree_scores(model, x):
    """训练仍并行；推断按冻结树的固定顺序求均值，避免线程求和顺序扰动0.5边界。"""
    return np.mean([tree.predict(x) for tree in model.estimators_], axis=0)


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
        raise ValueError("MARKET_DATE_TREE_COMPLETED_BRANCHES_MISSING")
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
    from app.services import direction_1d_sprint_market_date_tree as service

    return forward.tick(service)


def report():
    from app.services import direction_1d_sprint_market_child_forward_v2 as forward
    from app.services import direction_1d_sprint_market_date_tree as service

    return forward.report(service)


# 复用第70轮独立市场来源；不新增数据请求，前视边界沿用已验证父输入。
def live_market(source):
    if data.load(source["target"]) != source:
        raise ValueError("MARKET_DATE_TREE_LIVE_PARENT_CHANGED")
    return source["market"]


live_answers = answers
