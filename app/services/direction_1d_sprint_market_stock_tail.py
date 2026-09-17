"""第116轮：T日沪深极端涨跌比例，保持原标签、训练样本和对照。"""

import hashlib
import shutil
import warnings
from collections import defaultdict
from datetime import date

import joblib
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

from app.services import direction_1d_sprint as base
from app.services import direction_1d_sprint_core_forward as core
from app.services import direction_1d_sprint_market_only as baseline
from app.services import direction_1d_sprint_market_stock_activity as previous
from app.services import direction_1d_sprint_return_target as regression
from app.services import direction_1d_sprint_sequence as sequence
from app.services import direction_1d_sprint_stock_tail_data as data

original, active, reference = previous.original, previous.active, previous.reference
CANDIDATES = ("SIGN3_STOCK_TAIL5_LR504",)
CONTROLS = previous.CONTROLS
BRANCHES = CANDIDATES + CONTROLS
FIRST_TARGET = "2026-09-17"
PROPOSAL_HASH = "69c88a33be7eb5bca14042c718158809ea517d4711a97eff1986344a5477831b"


def root():
    return base.ROOT / "round-116"


def require(condition, reason):
    if not condition:
        raise ValueError("STOCK_TAIL_" + reason)


def fingerprint():
    code = dict(base.read(base.ROOT / "round-115" / "plan.json")["fingerprint"]["code"])
    for name, expected in code.items():
        require(core.sha(base.PROJECT / name) == expected, "OLD_CODE_CHANGED")
    source_plan = base.read(data.root() / "qualification-plan.json")
    for name, expected in source_plan["code"].items():
        require(core.sha(base.PROJECT / name) == expected, "SOURCE_CODE_CHANGED")
        require(name not in code or code[name] == expected, "SOURCE_CODE_CONFLICT")
        code[name] = expected
    for name in (
        "app/services/direction_1d_sprint_market_stock_tail.py",
        "scripts/direction_1d_sprint_market_stock_tail.py",
        "tests/test_direction_1d_sprint_market_stock_tail.py",
    ):
        code[name] = core.sha(base.PROJECT / name)
    return {"code": code}


def plan():
    proposal = base.read(root() / "proposal-before-implementation.json")
    require(base.digest(proposal) == PROPOSAL_HASH, "PROPOSAL_CHANGED")
    require(proposal["design_sha256"] == core.sha(root() / "design-before-implementation.md"), "DESIGN_CHANGED")
    require(proposal["r115_result_hash"] == base.digest(base.read(previous.root() / "result.json")), "SOURCE_CHANGED")
    require(
        base.digest(base.read(root() / "source-identity.json")) == proposal["source_identity_hash"],
        "SOURCE_IDENTITY_CHANGED",
    )
    require(core.sha(root() / "freeze_identity.py") == proposal["identity_script_sha256"], "IDENTITY_SCRIPT_CHANGED")
    source = base.read(data.root() / "qualification-result.json")
    require(base.digest(source) == proposal["source_qualification_hash"], "TAIL_SOURCE_CHANGED")
    coverage = base.read(root() / "feature-feasibility.json")
    require(
        coverage["source_history_hash"] == source["source_proof"]["source_history_hash"]
        and all(x["tail_available"] == x["questions"] for x in coverage["exam"].values())
        and len(coverage["training_windows"]) == 27
        and all(x["missing"] == 0 and x["dates"] == 504 for x in coverage["training_windows"]),
        "FEATURE_COVERAGE_INVALID",
    )
    require(proposal["candidates"] == list(CANDIDATES) and proposal["controls"] == list(CONTROLS), "BRANCHES_CHANGED")
    context = {
        "fingerprint": fingerprint(),
        "calendar_hash": base.calendar()[1],
        "scope_hash": base.digest(data.scope()),
        "feature_coverage_hash": base.digest(coverage),
    }
    path = root() / "plan.json"
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


def reconstruct_dataset():
    """按共享市场日期缓存新增沪深极端涨跌方向差及尾部占比，避免为同日30基金重复解析整份交易日历。"""
    rows, proof = original.data.dataset()
    h = data.history()
    extra, cache = [], {}
    for row in rows:
        key = row["t"], row["u"]
        if key not in cache:
            cache[key] = data.extend({}, *key, h["snapshot"]["rows"])["stock_tail"]
        extra.append(row | {"market": row["market"] | {"stock_tail": cache[key]}})
    require([data.original_row(r) for r in extra] == rows, "ORIGINAL_ROWS_CHANGED")
    proof = proof | {"stock_tail_history_hash": base.digest(h)}
    return extra, proof


def question_identity(rows, proof):
    """验证时间不是数据身份；只排除at，全部其余证据和两套完整行仍绑定摘要。"""
    require("at" in proof and isinstance(proof["at"], str), "PROOF_TIMESTAMP_MISSING")
    return {
        "stable_proof": {k: v for k, v in proof.items() if k != "at"},
        "rows": len(rows),
        "original_rows_hash": base.digest([data.original_row(r) for r in rows]),
        "extended_rows_hash": base.digest(rows),
    }


def verify_identity(rows, proof, expected):
    actual = question_identity(rows, proof)
    require(actual == expected, "QUESTION_IDENTITY_CHANGED")
    return actual


def dataset():
    rows, proof = reconstruct_dataset()
    frozen = base.read(root() / "source-identity.json")
    identity = verify_identity(rows, proof, frozen["identity"])
    return rows, identity


def vector(value):
    x = baseline.market_vector(value)
    feature = value["stock_tail"]
    require(feature["available"], "STOCK_TAIL_MISSING")
    extra = [feature[k] for k in data.KEYS]
    require(
        all(type(v) in (int, float) and np.isfinite(v) for v in extra) and abs(extra[0]) <= extra[1] <= 1,
        "STOCK_TAIL_INVALID",
    )
    return [*x, *extra]


def transform(x, scale, signed):
    x, scale = np.asarray(x, dtype=float), np.asarray(scale, dtype=float)
    require(
        x.ndim == 2
        and x.shape[1] == 5
        and scale.shape == (5,)
        and np.isfinite(x).all()
        and np.isfinite(scale).all()
        and min(scale) > 0,
        "TRANSFORM_INVALID",
    )
    result = x / scale
    if signed:
        result[:, :3] = np.sign(x[:, :3])
    return result


def fit(rows, name, cutoff):
    require(name in CANDIDATES, "UNKNOWN_RECIPE")
    chosen = original.training_rows(rows, cutoff)
    require(
        len({r["u"] for r in chosen}) == 504
        and all(r["mature"] < cutoff and r["u"] < cutoff and r["market"]["stock_tail"]["available"] for r in chosen),
        "TRAINING_MATURITY_OR_COVERAGE_INVALID",
    )
    x = np.asarray([vector(r["market"]) for r in chosen])
    y = np.asarray([r["y"] for r in chosen])
    weights = regression.weights(chosen)
    _, mean, scale = sequence.normalize_training(x, weights)
    signed = True
    x = transform(x, scale, signed)
    model = LogisticRegression(C=0.1, fit_intercept=False, max_iter=1500, random_state=17)
    active()
    with threadpool_limits(limits=2), warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        model.fit(x, y, sample_weight=weights)
    active()
    return {
        "model": model,
        "mean": [0.0] * 5,
        "scale": scale,
        "signed": signed,
        "cutoff": cutoff,
        "fit_hash": base.digest([data.original_row(r) for r in chosen]),
        "extended_fit_hash": base.digest(chosen),
        "weight_hash": base.digest(weights.tolist()),
        "fit_rows": len(chosen),
        "fit_dates": 504,
        "fit_end": max(r["u"] for r in chosen),
        "max_mature_date": max(r["mature"] for r in chosen),
        "max_feature_date": max(r["market"]["stock_tail"]["date"] for r in chosen),
        "training_feature_mean_not_subtracted": mean,
        "weighted_up_rate": float(np.average(y, weights=weights)),
    }


def validate_head(head, name, cutoff=None):
    require(name in CANDIDATES and head is not None, "HEAD_MISSING")
    model, scale = head["model"], np.asarray(head["scale"])
    require(
        isinstance(model, LogisticRegression)
        and model.n_features_in_ == 5
        and model.coef_.shape == (1, 5)
        and np.isfinite(model.coef_).all()
        and list(model.classes_) == [0, 1],
        "HEAD_MODEL_INVALID",
    )
    params = model.get_params()
    require(
        params["C"] == 0.1
        and params["fit_intercept"] is False
        and params["max_iter"] == 1500
        and params["random_state"] == 17
        and params["solver"] == "lbfgs",
        "HEAD_RECIPE_CHANGED",
    )
    require(
        scale.shape == (5,)
        and np.isfinite(scale).all()
        and min(scale) > 0
        and head["mean"] == [0.0] * 5
        and head["signed"] is True,
        "HEAD_TRANSFORM_INVALID",
    )
    require(
        head["fit_dates"] == 504
        and head["fit_end"] < head["cutoff"]
        and head["max_mature_date"] < head["cutoff"]
        and head["max_feature_date"] < head["fit_end"],
        "HEAD_TIME_INVALID",
    )
    if cutoff is not None:
        require(head["cutoff"] == cutoff, "HEAD_CUTOFF_CHANGED")
    return model


def candidate_answers(values, name, head, fallback):
    """新增信息缺失时保留匹配原模型答案，不能删除该题或把缺失行情当成零。"""
    require(name in CANDIDATES and len(values) == len(fallback), "UNKNOWN_BRANCH_OR_ROWS")
    output = list(fallback)
    indices = [i for i, v in enumerate(values) if v["available"] and v["stock_tail"]["available"]]
    if indices:
        model = validate_head(head, name)
        matrix = transform([vector(values[i]) for i in indices], head["scale"], head["signed"])
        scores = model.predict_proba(matrix)[:, 1]
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
    for name in CANDIDATES:
        output[name] = candidate_answers(values, name, heads.get(name), controls[CONTROLS[1]])
    return output


def answers(market, group, bundle):
    heads = {n: bundle[n][group] for n in CANDIDATES}
    controls = [bundle[n][group] for n in CONTROLS[:2]]
    return {n: v[0] for n, v in batch([market], heads, controls).items()}


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
    base.save(root() / "training" / (path.stem + ".json"), {k: v for k, v in head.items() if k != "model"})
    return head


def prepare():
    p = plan()
    if (root() / "result.json").exists():
        return models()[0]
    rows, proof = dataset()
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
            require(date.fromisoformat(start) < date.fromisoformat(end), "QUARTER_BOUNDARY_INVALID")
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
