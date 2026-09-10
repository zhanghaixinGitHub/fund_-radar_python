"""仅接受已验证队列/阶段矩阵的7列和10列固定逻辑回归、独立校准。"""

import math
import warnings
from collections import Counter

from app.schemas.direction_followup import StudyAnswer, StudyCohort, StudyInput
from app.services.direction_followup_protocol import FEATURE_MARKET
from app.services.direction_training_artifacts import digest
from app.services.historical_nav_evaluation import FEATURE_NAMES


def sigmoid(z):
    return 1 / (1 + math.exp(-z)) if z >= 0 else math.exp(z) / (1 + math.exp(z))


def restore(model):
    if set(model) != {
        "version",
        "feature_names",
        "mean",
        "scale",
        "coefficients",
        "intercept",
        "train_hash",
        "fit_end",
        "train_counts",
        "hash",
    }:
        raise ValueError("MODEL_FIELDS")
    if model["version"] != "DIRECTION_STUDY_LOGISTIC_V1" or model["feature_names"] not in (
        list(FEATURE_NAMES),
        list((*FEATURE_NAMES, *FEATURE_MARKET)),
    ):
        raise ValueError("MODEL_VERSION_OR_FEATURES")
    if model["hash"] != digest({k: v for k, v in model.items() if k != "hash"}):
        raise ValueError("MODEL_HASH")
    dimensions = len(model["feature_names"])
    if any(len(model[k]) != dimensions for k in ("mean", "scale", "coefficients")):
        raise ValueError("MODEL_SHAPE")
    if (
        any(not math.isfinite(x) for k in ("mean", "scale", "coefficients") for x in model[k])
        or not math.isfinite(model["intercept"])
        or any(s <= 0 for s in model["scale"])
    ):
        raise ValueError("MODEL_NUMBERS")
    return model


def logits(model, items):
    model = restore(model)
    if any(len(i.x) != len(model["mean"]) for i in items):
        raise ValueError("PREDICT_DIMENSION")
    return [
        model["intercept"]
        + sum(
            (x - m) / s * c
            for x, m, s, c in zip(i.x, model["mean"], model["scale"], model["coefficients"], strict=True)
        )
        for i in items
    ]


def validated_rows(rows, cohort, train_funds, lower, upper, dimensions, minimum):
    result = []
    if not rows or len(rows) > 10000:
        raise ValueError("ROW_BUDGET")
    keys = set()
    for r in rows:
        if set(r) != {"input", "answer"}:
            raise ValueError("TRAIN_FIELDS")
        item, answer = StudyInput.model_validate(r["input"]), StudyAnswer.model_validate(r["answer"])
        if (
            item.fund not in cohort.funds
            or item.fund not in train_funds
            or item.fund != answer.fund
            or item.cutoff != answer.cutoff
        ):
            raise ValueError("TRAIN_COHORT_OR_IDENTITY")
        if (
            item.key in keys
            or not lower < str(item.cutoff) <= str(answer.available_at) <= upper
            or len(item.x) != dimensions
        ):
            raise ValueError("TRAIN_STAGE_BOUNDARY")
        keys.add(item.key)
        result.append((item, answer))
    counts = Counter(i.fund for i, a in result)
    if (
        set(counts) != set(train_funds)
        or any(n < minimum for n in counts.values())
        or {a.y for i, a in result} != {0, 1}
    ):
        raise ValueError("TRAIN_INSUFFICIENT")
    return result, counts


def execute_job(job):
    import numpy as np
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from threadpoolctl import threadpool_limits

    if set(job) != {"branch", "cohort", "train_funds", "window", "features", "fit", "cal", "exam", "base"}:
        raise ValueError("JOB_FIELDS")
    cohort = StudyCohort.model_validate(job["cohort"])
    if not set(job["train_funds"]).issubset(cohort.funds) or len(set(job["train_funds"])) != len(job["train_funds"]):
        raise ValueError("JOB_COHORT")
    if job["features"] not in (list(FEATURE_NAMES), list((*FEATURE_NAMES, *FEATURE_MARKET))):
        raise ValueError("JOB_FEATURES")
    window = job["window"]
    if not "2021-01-01" <= window["fit_end"] < window["cal_end"] < window["exam_end"] <= "2024-12-31":
        raise ValueError("JOB_PROTECTED_DATES")
    if job["branch"] not in ("ORIGINAL_7", "EXPANDED_7", "MARKET_MATCHED_7", "MARKET_10", "CALIBRATED_6M"):
        raise ValueError("JOB_BRANCH")
    dimensions = len(job["features"])
    fit, counts = validated_rows(
        job["fit"], cohort, job["train_funds"], "2020-12-31", window["fit_end"], dimensions, 252
    )
    items = [StudyInput.model_validate(i) for i in job["exam"]]
    if (
        not items
        or len(items) > 10000
        or len({i.key for i in items}) != len(items)
        or any(
            i.fund not in cohort.funds
            or not window["cal_end"] < str(i.cutoff) <= window["exam_end"]
            or len(i.x) != dimensions
            for i in items
        )
    ):
        raise ValueError("EXAM_INPUT_BOUNDARY")
    if job["branch"] != "CALIBRATED_6M" and (job["base"] is not None or job["cal"]):
        raise ValueError("UNEXPECTED_CALIBRATION_DATA")
    weights = [len(fit) / (len(counts) * counts[i.fund]) for i, a in fit]

    def estimator():
        return LogisticRegression(C=1.0, l1_ratio=0.0, solver="lbfgs", tol=1e-8, max_iter=1000, random_state=0)

    with threadpool_limits(limits=1), warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        warnings.simplefilter("error", RuntimeWarning)
        if job["branch"] == "CALIBRATED_6M":
            base = restore(job["base"])
            if base["train_hash"] != digest(job["fit"]) or base["fit_end"] != window["fit_end"]:
                raise ValueError("CALIBRATION_BASE_MISMATCH")
            cal, cal_counts = validated_rows(
                job["cal"], cohort, job["train_funds"], window["fit_end"], window["cal_end"], dimensions, 60
            )
            z = np.asarray(logits(base, [i for i, a in cal])).reshape(-1, 1)
            fitted = estimator().fit(
                z,
                [a.y for i, a in cal],
                sample_weight=[len(cal) / (len(cal_counts) * cal_counts[i.fund]) for i, a in cal],
            )
            if fitted.classes_.tolist() != [0, 1] or fitted.n_iter_[0] >= 1000:
                raise ValueError("CALIBRATION_NOT_CONVERGED")
            slope, intercept = float(fitted.coef_[0, 0]), float(fitted.intercept_[0])
            if not math.isfinite(slope) or not math.isfinite(intercept):
                raise ValueError("CALIBRATION_NONFINITE")
            scores = [sigmoid(slope * z + intercept) for z in logits(base, items)]
            expected = fitted.predict_proba(np.asarray(logits(base, items)).reshape(-1, 1))[:, 1]
            if any(abs(a - float(b)) > 1e-12 for a, b in zip(scores, expected, strict=True)):
                raise ValueError("CALIBRATION_JSON_REPLAY_MISMATCH")
            artifact = {
                "base": base,
                "slope": slope,
                "intercept": intercept,
                "cal_hash": digest(job["cal"]),
                "cal_counts": dict(cal_counts),
            }
            state = (
                "REVERSED_PENDING_VALIDATION"
                if slope < 0
                else "CONSTANT_PENDING_VALIDATION"
                if slope == 0
                else "FORWARD_PENDING_VALIDATION"
            )
            return {
                "status": "PREDICTED",
                "scores": scores,
                "artifact": artifact,
                "calibration_state": state,
                "base_fitted": False,
                "calibrator_fitted": True,
            }
        x = np.asarray([i.x for i, a in fit], dtype=float)
        scaler = StandardScaler().fit(x, sample_weight=weights)
        fitted = estimator().fit(scaler.transform(x), [a.y for i, a in fit], sample_weight=weights)
        if fitted.classes_.tolist() != [0, 1] or fitted.n_iter_[0] >= 1000:
            raise ValueError("MODEL_NOT_CONVERGED")
        model = {
            "version": "DIRECTION_STUDY_LOGISTIC_V1",
            "feature_names": job["features"],
            "mean": scaler.mean_.tolist(),
            "scale": scaler.scale_.tolist(),
            "coefficients": fitted.coef_[0].tolist(),
            "intercept": float(fitted.intercept_[0]),
            "train_hash": digest(job["fit"]),
            "fit_end": window["fit_end"],
            "train_counts": dict(counts),
        }
        model["hash"] = digest(model)
        scores = [sigmoid(z) for z in logits(model, items)]
        expected = fitted.predict_proba(scaler.transform(np.asarray([i.x for i in items])))[:, 1]
        if any(abs(a - float(b)) > 1e-12 for a, b in zip(scores, expected, strict=True)):
            raise ValueError("JSON_REPLAY_MISMATCH")
        return {
            "status": "PREDICTED",
            "scores": scores,
            "artifact": model,
            "base_fitted": True,
            "calibrator_fitted": False,
        }
