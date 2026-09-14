"""一日分数校准研究：仅用较早折外分数学两个系数，保持全部基金和考题，禁止自动发布。"""

import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

from app.services import direction_1d_complexity_study as existing
from app.services.direction_1d_protocol import ZONE, digest
from app.services.direction_1d_training import read, weights, write_new
from app.services.direction_training_artifacts import file_hash

full, activity, tree, sector = existing.full, existing.activity, existing.tree, existing.sector
REFERENCE, CANDIDATE = "ACTIVITY12", "SIGMOID_LOGIT12"
BRANCHES, QUARTERS = (REFERENCE, CANDIDATE), full.QUARTERS
RECIPE = {
    "C": 1.0,
    "solver": "lbfgs",
    "max_iter": 1000,
    "tol": 1e-10,
    "fit_intercept": True,
    "class_weight": None,
    "random_state": 0,
}
PROTOCOL = {
    "formula": "SIGMOID_A_TIMES_LOGIT_P_PLUS_B",
    "clip_epsilon": 1e-6,
    "regularization": "DEFAULT_L2",
    "calibration_data": "PAST_63_MATURE_TARGET_DAYS_3_DISJOINT_21_DAY_OOF_BLOCKS",
    "weights": "ORIGINAL_FAMILY_DATE",
    "threshold": 0.5,
    "direction": "STRICTLY_GREATER_THAN_THRESHOLD",
    "negative_slope": "ALLOWED_REPORTED_NO_POSTHOC_FALLBACK",
    "high_confidence": 0.70,
    "probability_edges": [i / 10 for i in range(11)],
    "confidence_edges": [0.50, 0.55, 0.60, 0.65, 0.70, 0.80, 1.0],
    "question_coverage": 1.0,
}
LIMITS = {
    "kind": "HISTORICAL_CALIBRATION_DEVELOPMENT_ONLY",
    "reference": REFERENCE,
    "candidate": CANDIDATE,
    "recipe": RECIPE,
    "protocol": PROTOCOL,
    "observation": existing.OBSERVATION,
    "max_main_fits": 4,
    "max_replay_fits": 4,
    "new_tree_fits": 0,
    "new_api_calls": 0,
    "database_writes": 0,
    "final_fit": False,
    "model_released": False,
    "historical_first_versions_verified": False,
    "protected_years_excluded": [2025, 2026],
    "bootstrap_seed": 20260913,
    "bootstrap_repetitions": 2000,
    "bootstrap_block_days": 5,
}


def fingerprint() -> dict:
    """绑定实际执行源码、依赖及既有时间隔离逻辑，防止冻结后悄悄改配方。"""
    result = full.fingerprint()
    root = Path(__file__).resolve().parents[2]
    for n in (
        "app/services/direction_1d_complexity_study.py",
        "app/services/direction_1d_calibration_study.py",
        "scripts/direction_1d_calibration_study.py",
        "tests/test_direction_1d_calibration_study.py",
    ):
        result[n] = file_hash(root / n)
    return result


def valid_scores(scores, count: int) -> np.ndarray:
    p = np.asarray(scores, dtype=float)
    if count <= 0 or p.shape != (count,) or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("CALIBRATION_SCORES_INVALID")
    return p


def logit_input(scores) -> np.ndarray:
    """仅对合法0到1分数做数值裁剪，不把NaN、缺失或越界输入修补成可用样本。"""
    p = valid_scores(scores, len(scores))
    p = np.clip(p, PROTOCOL["clip_epsilon"], 1 - PROTOCOL["clip_epsilon"])
    return (np.log(p) - np.log1p(-p)).reshape(-1, 1)


def fit_metadata(rows: list[dict], scores, cutoff: str, q: str, model_hashes: dict) -> dict:
    """拟合前验证较早答案已成熟；调用方只能交入63个折外目标日，不能交随后季度考题。"""
    p = valid_scores(scores, len(rows))
    if len({full.identity(r) for r in rows}) != len(rows) or len({r["u"] for r in rows}) != 63:
        raise ValueError("CALIBRATION_MEMBERS_OR_DAYS_INVALID")
    if any(not "2021-01-01" <= r["t"] < r["u"] <= "2024-12-31" for r in rows):
        raise ValueError("CALIBRATION_PROTECTED_YEAR")
    full.check_available(rows, datetime.fromisoformat(cutoff), fit=True)
    if min(sum(r["y"] == c for r in rows) for c in (0, 1)) < 30 or any(r["y"] not in (0, 1) for r in rows):
        raise ValueError("CALIBRATION_CLASSES_INSUFFICIENT")
    if q not in QUARTERS or set(model_hashes) != {q, *(q + f"-V{k}" for k in (1, 2, 3))}:
        raise ValueError("CALIBRATION_MODEL_LINKS_INVALID")
    return {
        "quarter": q,
        "train_as_of": cutoff,
        "latest_mature_at": max(r["mature_at"] for r in rows),
        "fit_count": len(rows),
        "target_days": 63,
        "classes": {str(c): sum(r["y"] == c for r in rows) for c in (0, 1)},
        "fit_hash": digest(rows),
        "score_hash": digest(p.tolist()),
        "weight_hash": digest(weights(rows).tolist()),
        "control_model_hashes": model_hashes,
    }


def earlier_data(base: Path) -> tuple[dict, dict]:
    """复算原16窗分数和时间关系；每块校准题与生成其分数的树训练题按目标日隔离。"""
    universe, schedule, exam = (read(base / n) for n in ("universe.json", "schedule.json", "exam.json"))
    existing.validate_schedule(universe, schedule, exam)
    controls = read(base / "main/models.json")
    data = {}
    for q in QUARTERS:
        rows, scores, hashes = [], [], {}
        for key in (q, *(q + f"-V{k}" for k in (1, 2, 3))):
            model, info = controls[key][REFERENCE], schedule[key]
            activity.validate_model(model)
            if any(model[k] != info[k] for k in ("train_as_of", "fit_hash", "weight_hash", "fit_count")):
                raise ValueError("CALIBRATION_CONTROL_METADATA_CHANGED")
            hashes[key] = digest(model)
            if key != q:
                checked = [universe[i] for i in info["exam_indexes"]]
                rows.extend(checked)
                scores.extend(activity.predict(model, checked).tolist())
        metadata = fit_metadata(rows, scores, schedule[q]["train_as_of"], q, hashes)
        data[q] = {"rows": rows, "scores": scores, "metadata": metadata}
    return data, controls


def freeze(base: Path, plan: Path, scope: Path, root: Path) -> dict:
    """冻结单一配方与4+4预算，复制输入及旧结果；不拟合、不联网、不覆盖已有研究。"""
    full.verify(base)
    old = full.verify_inputs(base)
    activity.require_replay(base)
    owner = read(scope)
    full.audit.validate_scope(owner)
    if owner["scope_hash"] != old["owner_scope_hash"]:
        raise ValueError("CALIBRATION_OWNER_SCOPE_CHANGED")
    data, _ = earlier_data(base)
    root.mkdir(parents=True, exist_ok=False)
    names = {*old["input_files"], "study.json", "study-receipt.json", "replay-proof.json"}
    names.update(
        p.relative_to(base).as_posix() for mode in ("main", "replay") for p in (base / mode).iterdir() if p.is_file()
    )
    for n in sorted(names):
        p = base / n
        if not p.resolve().is_relative_to(base.resolve()) or p.is_symlink():
            raise ValueError("CALIBRATION_SOURCE_PATH_INVALID")
        tree.copy_file(p, root / "control" / n)
    for n, p in (("plan.md", plan), ("owner-scope.json", scope)):
        tree.copy_file(p, root / n)
    files = {"control/" + n: file_hash(root / "control" / n) for n in names}
    files.update({n: file_hash(root / n) for n in ("plan.md", "owner-scope.json")})
    spec = {
        **LIMITS,
        "created_at": datetime.now(ZONE).isoformat(),
        "fingerprint": fingerprint(),
        "features": activity.FEATURES,
        "feature_recipe": activity.FEATURE_RECIPE,
        "fund_codes": old["fund_codes"],
        "owner_scope_hash": old["owner_scope_hash"],
        "exam_count": old["exam_count"],
        "restored_count": old["restored_count"],
        "source_expires_at": old["source"]["source_expires_at"],
        "input_files": files,
        "calibration_metadata": {q: d["metadata"] for q, d in data.items()},
    }
    spec["cohort_id"] = "D1-CALIBRATION-" + digest({"files": files, "limits": LIMITS})[:24]
    write_new(root / "study.json", spec)
    write_new(root / "study-receipt.json", {"hash": digest(spec), "at": spec["created_at"]})
    return {k: spec[k] for k in ("cohort_id", "created_at", "exam_count", "max_main_fits", "max_replay_fits")}


def verify_inputs(root: Path) -> dict:
    spec = read(root / "study.json")
    if (
        digest(spec) != read(root / "study-receipt.json")["hash"]
        or spec["fingerprint"] != fingerprint()
        or any(spec.get(k) != v for k, v in LIMITS.items())
        or spec["features"] != activity.FEATURES
        or spec["feature_recipe"] != activity.FEATURE_RECIPE
    ):
        raise ValueError("CALIBRATION_SPEC_CODE_OR_BUDGET_CHANGED")
    if datetime.now(ZONE) >= datetime.fromisoformat(spec["source_expires_at"]):
        raise ValueError("CALIBRATION_SOURCE_EXPIRED")
    for n, expected in spec["input_files"].items():
        p = root / n
        if not p.resolve().is_relative_to(root.resolve()) or p.is_symlink() or file_hash(p) != expected:
            raise ValueError("CALIBRATION_INPUT_CHANGED")
    return spec


def validate_model(model: dict) -> None:
    if (
        model.get("candidate") != CANDIDATE
        or model.get("recipe") != RECIPE
        or model.get("protocol") != PROTOCOL
        or model.get("model_released") is not False
        or model.get("target_definition") != "UNIT_NAV_DIRECTION_V1"
        or not np.isfinite([model.get("slope", np.nan), model.get("intercept", np.nan)]).all()
        or not 0 < model.get("iterations", 0) < RECIPE["max_iter"]
        or not 0 <= model.get("restore_max_score_diff", np.inf) <= 1e-12
    ):
        raise ValueError("CALIBRATION_MODEL_INVALID")


def predict(model: dict, scores) -> np.ndarray:
    """推理只接收旧分数与校准系数，接口本身不接收答案，也不生成交易或正式概率结论。"""
    validate_model(model)
    z = model["slope"] * logit_input(scores)[:, 0] + model["intercept"]
    return 1 / (1 + np.exp(-np.clip(z, -700, 700)))


def fit(data: dict, cohort: str) -> dict:
    rows, scores, meta = data["rows"], data["scores"], data["metadata"]
    if fit_metadata(rows, scores, meta["train_as_of"], meta["quarter"], meta["control_model_hashes"]) != meta:
        raise ValueError("CALIBRATION_FIT_INPUT_CHANGED")
    x, w = logit_input(scores), weights(rows)
    with threadpool_limits(limits=1), warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        estimator = LogisticRegression(**RECIPE).fit(x, [r["y"] for r in rows], sample_weight=w)
        model = {
            **meta,
            "cohort_id": cohort,
            "candidate": CANDIDATE,
            "recipe": RECIPE,
            "protocol": PROTOCOL,
            "target_definition": "UNIT_NAV_DIRECTION_V1",
            "model_released": False,
            "slope": float(estimator.coef_[0, 0]),
            "intercept": float(estimator.intercept_[0]),
            "iterations": int(estimator.n_iter_[0]),
            "restore_max_score_diff": 0.0,
        }
        difference = float(np.max(np.abs(predict(model, scores) - estimator.predict_proba(x)[:, 1])))
        model["restore_max_score_diff"] = difference
    validate_model(model)
    return model


def outputs(base: Path, models: dict, controls: dict) -> list[dict]:
    """在同题上应用映射；原12维树每条分数必须与旧封存严格一致，保留全部题目。"""
    exam = read(base / "exam.json")
    old = {full.identity(r): r for r in read(base / "main/predictions.json")}
    result = []
    for q in QUARTERS:
        rows = [r for r in exam if r["quarter"] == q]
        original = activity.predict(controls[q][REFERENCE], rows)
        calibrated = predict(models[q], original)
        for r, a, b in zip(rows, original, calibrated, strict=True):
            previous = old[full.identity(r)]
            if float(a) != previous["scores"][REFERENCE]:
                raise ValueError("CALIBRATION_CONTROL_SCORE_CHANGED")
            scores = {REFERENCE: float(a), CANDIDATE: float(b)}
            result.append({**r, "scores": scores, "directions": {k: int(v > 0.5) for k, v in scores.items()}})
    return sorted(result, key=lambda r: (r["t"], r["family"], r["fund_code"]))


def run(root: Path, *, replay: bool = False) -> dict:
    spec = verify_inputs(root)
    if replay:
        verify(root)
    data, controls = earlier_data(root / "control")
    if {q: d["metadata"] for q, d in data.items()} != spec["calibration_metadata"]:
        raise ValueError("CALIBRATION_EARLIER_INPUT_CHANGED")
    out = root / ("replay" if replay else "main")
    out.mkdir(exist_ok=False)
    write_new(out / "budget-reserved.json", {"at": datetime.now(ZONE).isoformat(), "max_fits": 4})
    models = {}
    for q in QUARTERS:
        write_new(out / (q + "-reserved.json"), {"at": datetime.now(ZONE).isoformat(), "metadata": data[q]["metadata"]})
        models[q] = fit(data[q], spec["cohort_id"])
        write_new(out / (q + ".json"), models[q])
    predictions = outputs(root / "control", models, controls)
    for n, value in (("models", models), ("predictions", predictions)):
        write_new(out / (n + ".json"), value)
    result = {
        "finished_at": datetime.now(ZONE).isoformat(),
        "successful_fits": 4,
        "new_tree_fits": 0,
        "model_released": False,
        "models_hash": digest(models),
        "predictions_hash": digest(predictions),
    }
    write_new(out / "completion.json", result)
    if replay:
        activity.require_replay(root)
        write_new(root / "replay-proof.json", replay_proof(len(predictions)))
    return result


def replay_proof(count: int) -> dict:
    return {
        "models_equal": True,
        "predictions_equal": True,
        "max_score_diff": 0.0,
        "successful_fits": 4,
        "prediction_count": count,
    }


def verify(root: Path) -> dict:
    """零拟合复核：检查预算、时间、数据/系数hash并从系数恢复所有分数，不能覆盖原输出。"""
    spec = verify_inputs(root)
    data, controls = earlier_data(root / "control")
    if {q: d["metadata"] for q, d in data.items()} != spec["calibration_metadata"]:
        raise ValueError("CALIBRATION_EARLIER_INPUT_CHANGED")
    for mode in ("main", "replay"):
        out = root / mode
        if not (out / "completion.json").exists():
            if mode == "main" or out.exists():
                raise ValueError("CALIBRATION_INCOMPLETE_RUN")
            continue
        complete, models = read(out / "completion.json"), read(out / "models.json")
        budget = read(out / "budget-reserved.json")
        if (
            set(models) != set(QUARTERS)
            or digest(models) != complete["models_hash"]
            or complete["successful_fits"] != 4
            or complete["new_tree_fits"] != 0
            or complete["model_released"] is not False
            or budget["max_fits"] != 4
            or {p.name for p in out.glob("*-reserved.json")}
            != {"budget-reserved.json", *(q + "-reserved.json" for q in QUARTERS)}
        ):
            raise ValueError("CALIBRATION_RUN_BUDGET_OR_MODELS")
        for q, model in models.items():
            validate_model(model)
            reserved = read(out / (q + "-reserved.json"))
            if (
                model != read(out / (q + ".json"))
                or model["cohort_id"] != spec["cohort_id"]
                or any(model[k] != v for k, v in data[q]["metadata"].items())
                or reserved["metadata"] != data[q]["metadata"]
                or not spec["created_at"] <= budget["at"] <= reserved["at"] <= complete["finished_at"]
            ):
                raise ValueError("CALIBRATION_MODEL_LINK_OR_RESERVATION_CHANGED")
        predictions = outputs(root / "control", models, controls)
        if (
            len(predictions) != spec["exam_count"]
            or sorted({r["fund_code"] for r in predictions}) != spec["fund_codes"]
            or predictions != read(out / "predictions.json")
            or digest(predictions) != complete["predictions_hash"]
        ):
            raise ValueError("CALIBRATION_OUTPUT_CHANGED")
    if (root / "replay").exists():
        activity.require_replay(root)
        if read(root / "replay-proof.json") != replay_proof(spec["exam_count"]):
            raise ValueError("CALIBRATION_REPLAY_PROOF_CHANGED")
    return {"verified": True, "exam_count": spec["exam_count"], "new_fits": 0, "api_calls": 0}


def bins(rows: list[dict], scores, *, confidence: bool) -> list[dict]:
    """固定分档只用于诊断；空档保留，记录独立目标日数量，不能事后选最好档用于展示。"""
    p = valid_scores(scores, len(rows))
    values, y = (
        (np.maximum(p, 1 - p), (p > 0.5) == [r["y"] for r in rows])
        if confidence
        else (p, np.asarray([r["y"] for r in rows]))
    )
    w = weights(rows)
    edges, result = PROTOCOL["confidence_edges" if confidence else "probability_edges"], []
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        chosen = (values >= lo) & ((values <= hi) if hi == 1 else (values < hi))
        n = int(chosen.sum())
        result.append(
            {
                "lower": lo,
                "upper": hi,
                "count": n,
                "target_days": len({r["u"] for r, selected in zip(rows, chosen, strict=True) if selected}),
                "weight_fraction": float(w[chosen].sum() / w.sum()),
                "mean_score": float(np.average(values[chosen], weights=w[chosen])) if n else None,
                "observed_fraction": float(np.average(y[chosen], weights=w[chosen])) if n else None,
            }
        )
    return result


def summary(rows: list[dict], branch: str) -> dict:
    p = np.asarray([r["scores"][branch] for r in rows])
    result = existing.score_summary(rows, p)
    reliability = bins(rows, p, confidence=False)
    high = np.maximum(p, 1 - p) >= PROTOCOL["high_confidence"]
    correct = (p > 0.5) == [r["y"] for r in rows]
    n, wrong = int(high.sum()), int(np.sum(high & ~correct))
    return {
        **result,
        "probability_bins": reliability,
        "confidence_bins": bins(rows, p, confidence=True),
        "weighted_probability_ece": sum(
            v["weight_fraction"] * abs(v["mean_score"] - v["observed_fraction"]) for v in reliability if v["count"]
        ),
        "high_confidence": {
            "count": n,
            "wrong": wrong,
            "fraction_of_all": n / len(rows),
            "wrong_fraction_of_all": wrong / len(rows),
            "error_rate_within": wrong / n if n else None,
        },
    }


def comparison(rows: list[dict], spec: dict) -> dict:
    overall = {b: summary(rows, b) for b in BRANCHES}
    strata = {
        field: {
            str(v): {b: summary([r for r in rows if r[field] == v], b) for b in BRANCHES}
            for v in sorted({r[field] for r in rows})
        }
        for field in ("quarter", "fund_code", "group", "restored_question")
    }
    months = {
        m: {b: summary([r for r in rows if r["u"][:7] == m], b) for b in BRANCHES}
        for m in sorted({r["u"][:7] for r in rows})
    }
    paired = sector.paired(rows, CANDIDATE, REFERENCE, spec)
    deltas = {
        f: [v[CANDIDATE]["weighted_accuracy"] - v[REFERENCE]["weighted_accuracy"] for v in strata[f].values()]
        for f in ("quarter", "fund_code")
    }
    conditions = {
        "overall_improved": paired["weighted_accuracy_difference"] > 0,
        "interval_positive": paired["block_bootstrap_interval_95"][0] > 0,
        "quarters_improved": sum(d > 0 for d in deltas["quarter"]) >= existing.OBSERVATION["positive_quarters"],
        "funds_improved": sum(d > 0 for d in deltas["fund_code"]) >= existing.OBSERVATION["positive_funds"],
        "fund_drop_controlled": min(deltas["fund_code"]) >= -existing.OBSERVATION["maximum_fund_drop"],
        "both_recalls_preserved": all(
            overall[CANDIDATE][k] >= overall[REFERENCE][k] for k in ("up_recall", "non_up_recall")
        ),
    }
    original_high = [
        r for r in rows if max(r["scores"][REFERENCE], 1 - r["scores"][REFERENCE]) >= PROTOCOL["high_confidence"]
    ]
    return {
        "kind": "DEVELOPMENT_CALIBRATION_NOT_INDEPENDENT_TEST",
        "overall": overall,
        "strata": strata,
        "months": months,
        "paired": paired,
        "conditions": conditions,
        "positive_quarters": sum(d > 0 for d in deltas["quarter"]),
        "positive_funds": sum(d > 0 for d in deltas["fund_code"]),
        "original_high_confidence_same_questions": {b: summary(original_high, b) for b in BRANCHES}
        if original_high
        else None,
        "reliability_metric_differences": {
            k: overall[CANDIDATE][k] - overall[REFERENCE][k]
            for k in ("weighted_log_loss", "weighted_brier", "weighted_probability_ece")
        },
        "status": "DEVELOPMENT_OBSERVATION_CONDITIONS_MET" if all(conditions.values()) else "NO_STABLE_GAIN",
        "prediction_coverage": 1.0,
        "model_released": False,
        "runtime_model_changed": False,
        "limitations": [
            "2021—2024反复使用，分档和区间均为开发诊断",
            "原净值08:00与行情18:00可用假设保留，历史首版本未验证",
            "分数变保守或概率误差下降不等于方向更准",
            "2025/2026答案不解封，真实未来效果待验证",
        ],
    }


def evaluate(root: Path) -> dict:
    verify(root)
    if not (root / "replay-proof.json").exists():
        raise ValueError("CALIBRATION_REPLAY_REQUIRED")
    report = comparison(read(root / "main/predictions.json"), read(root / "study.json"))
    write_new(root / "comparison.json", report)
    write_new(root / "comparison-receipt.json", {"at": datetime.now(ZONE).isoformat(), "hash": digest(report)})
    return report
