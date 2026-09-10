"""受限线性训练器：FIT 内标准化，固定算法，按规范选择列或划分训练人群。"""

import math
import warnings
from collections import Counter
from datetime import date

from app.schemas.direction_training import DirectionAnswer, DirectionInput
from app.services.direction_linear_protocol import (
    BRANCHES,
    COMBINATION_VERSION,
    COVERAGE_VERSION,
    REGULARIZATION_VERSION,
    fit_boundary,
    planned_dates,
    regularization_c,
    study_rules,
    study_windows,
)
from app.services.direction_training_artifacts import digest
from app.services.direction_training_dataset import FUNDS
from app.services.direction_training_models import recent_lower
from app.services.historical_nav_training import sigmoid
from app.services.trading_calendar import load_calendar


def select_fit(rows, branch, fit_end):
    if branch not in BRANCHES:
        raise ValueError("LINEAR_BRANCH")
    lower = recent_lower(fit_end)
    return [r for r in rows if branch != "RECENT_18M" or date.fromisoformat(r["input"]["cutoff"]) > lower]


def validate_job(payload):
    if set(payload) != {"version", "branch", "window", "fit", "exam"}:
        raise ValueError("LINEAR_JOB_FIELDS")
    branches, _ = study_rules(payload["version"])
    branch, window = payload["branch"], payload["window"]
    allowed = study_windows(payload["version"])
    if branch not in branches or window not in allowed:
        raise ValueError("LINEAR_PROTOCOL_OR_WINDOW")
    fit_end = date.fromisoformat(fit_boundary(payload["version"], branch, window))
    cal_end, exam_end = (date.fromisoformat(window[k]) for k in ("cal_end", "exam_end"))
    if not 1 <= len(payload["fit"]) <= 10000 or not 1 <= len(payload["exam"]) <= 10000:
        raise ValueError("LINEAR_ROW_BUDGET")
    fit, identities = [], set()
    for row in payload["fit"]:
        if set(row) != {"input", "answer"}:
            raise ValueError("LINEAR_TRAIN_FIELDS")
        item, answer = DirectionInput.model_validate(row["input"]), DirectionAnswer.model_validate(row["answer"])
        if item.key != answer.key or item.key in identities or not item.cutoff < answer.available_at <= fit_end:
            raise ValueError("LINEAR_FIT_BOUNDARY")
        if answer.end != load_calendar().future_sessions(item.cutoff)[-1]:
            raise ValueError("LINEAR_LABEL_HORIZON")
        identities.add(item.key)
        fit.append((item, answer))
    exam = [DirectionInput.model_validate(r) for r in payload["exam"]]
    if len({i.key for i in exam}) != len(exam) or any(not cal_end < i.cutoff <= exam_end for i in exam):
        raise ValueError("LINEAR_EXAM_BOUNDARY")
    if payload["version"] == COVERAGE_VERSION:
        allowed_dates = set(planned_dates(payload["version"], window))
        if any(i.cutoff not in allowed_dates for i in exam):
            raise ValueError("LINEAR_EXAM_PROTECTED_LABEL_PERIOD")
    lower = recent_lower(fit_end)
    if branch == "RECENT_18M":
        fit = [(i, a) for i, a in fit if i.cutoff > lower]
    counts = Counter(i.fund for i, a in fit)
    if set(counts) != set(FUNDS) or any(n < 252 for n in counts.values()):
        raise ValueError("LINEAR_FIT_INSUFFICIENT")
    return fit, exam


def restore(model):
    fields = {
        "version",
        "branch",
        "fund",
        "indices",
        "mean",
        "scale",
        "coefficients",
        "intercept",
        "train_hash",
        "train_counts",
        "fit_end",
        "hash",
    }
    if model.get("version") in (REGULARIZATION_VERSION, COMBINATION_VERSION, COVERAGE_VERSION):
        fields |= {"C", "penalty", "solver_iterations"}
    if set(model) != fields:
        raise ValueError("LINEAR_MODEL_FIELDS")
    branches, feature_indices = study_rules(model["version"])
    if model["branch"] not in branches:
        raise ValueError("LINEAR_MODEL_VERSION")
    if model["version"] in (REGULARIZATION_VERSION, COMBINATION_VERSION, COVERAGE_VERSION) and (
        model["C"] != regularization_c(model["version"], model["branch"])
        or model["penalty"] != "L2"
        or type(model["solver_iterations"]) is not int
        or not 1 <= model["solver_iterations"] < 1000
    ):
        raise ValueError("LINEAR_MODEL_REGULARIZATION")
    if model["indices"] != list(feature_indices[model["branch"]]):
        raise ValueError("LINEAR_MODEL_FEATURES")
    if model["fund"] not in (FUNDS if model["branch"] == "PER_FUND" else ("POOLED",)):
        raise ValueError("LINEAR_MODEL_FUND")
    if model["hash"] != digest({k: v for k, v in model.items() if k != "hash"}):
        raise ValueError("LINEAR_MODEL_HASH")
    size = len(model["indices"])
    if any(len(model[k]) != size for k in ("mean", "scale", "coefficients")):
        raise ValueError("LINEAR_MODEL_SHAPE")
    if (
        any(not math.isfinite(v) for k in ("mean", "scale", "coefficients") for v in model[k])
        or not math.isfinite(model["intercept"])
        or any(v <= 0 for v in model["scale"])
    ):
        raise ValueError("LINEAR_MODEL_NUMBERS")
    return model


def predict_model(model, items):
    model = restore(model)
    if any(len(i.x) != 7 or (model["fund"] != "POOLED" and i.fund != model["fund"]) for i in items):
        raise ValueError("LINEAR_PREDICT_SCOPE")
    return [
        sigmoid(
            model["intercept"]
            + sum(
                (i.x[index] - mean) / scale * coefficient
                for index, mean, scale, coefficient in zip(
                    model["indices"], model["mean"], model["scale"], model["coefficients"], strict=True
                )
            )
        )
        for i in items
    ]


def execute_job(payload):
    import numpy as np
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from threadpoolctl import threadpool_limits

    fit, exam = validate_job(payload)
    _, feature_indices = study_rules(payload["version"])
    branch, indices = payload["branch"], feature_indices[payload["branch"]]
    scores, models = {}, {}
    groups = FUNDS if branch == "PER_FUND" else ("POOLED",)
    for group in groups:
        rows = [(i, a) for i, a in fit if group == "POOLED" or i.fund == group]
        inputs = [i for i in exam if group == "POOLED" or i.fund == group]
        counts = Counter(i.fund for i, a in rows)
        weights = [len(rows) / (len(counts) * counts[i.fund]) for i, a in rows]
        x = np.asarray([[i.x[j] for j in indices] for i, a in rows])
        with threadpool_limits(limits=1), warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            warnings.simplefilter("error", RuntimeWarning)
            scaler = StandardScaler().fit(x, sample_weight=weights)
            estimator = LogisticRegression(
                C=regularization_c(payload["version"], branch),
                l1_ratio=0.0,
                solver="lbfgs",
                tol=1e-8,
                max_iter=1000,
                random_state=0,
                fit_intercept=True,
                class_weight=None,
            )
            estimator.fit(scaler.transform(x), [a.y for i, a in rows], sample_weight=weights)
        if estimator.classes_.tolist() != [0, 1] or int(estimator.n_iter_[0]) >= 1000:
            raise ValueError("LINEAR_MODEL_NOT_CONVERGED")
        model = {
            "version": payload["version"],
            "branch": branch,
            "fund": group,
            "indices": list(indices),
            "mean": scaler.mean_.tolist(),
            "scale": scaler.scale_.tolist(),
            "coefficients": estimator.coef_[0].tolist(),
            "intercept": float(estimator.intercept_[0]),
            "train_hash": digest(
                [{"input": i.model_dump(mode="json"), "answer": a.model_dump(mode="json")} for i, a in rows]
            ),
            "train_counts": dict(counts),
            "fit_end": fit_boundary(payload["version"], branch, payload["window"]),
        }
        if payload["version"] in (REGULARIZATION_VERSION, COMBINATION_VERSION, COVERAGE_VERSION):
            model.update(
                C=regularization_c(payload["version"], branch),
                penalty="L2",
                solver_iterations=int(estimator.n_iter_[0]),
            )
        model["hash"] = digest(model)
        models[group] = model
        values = predict_model(model, inputs)
        if inputs:
            expected = estimator.predict_proba(
                scaler.transform(np.asarray([[i.x[j] for j in indices] for i in inputs]))
            )[:, 1]
            if any(abs(a - float(b)) > 1e-12 for a, b in zip(values, expected, strict=True)):
                raise ValueError("LINEAR_JSON_REPLAY")
        scores.update({i.key: v for i, v in zip(inputs, values, strict=True)})
    return {
        "status": "PREDICTED",
        "scores": [scores[i.key] for i in exam],
        "models": models,
        "model_fit_count": len(models),
        "train_counts": dict(Counter(i.fund for i, a in fit)),
    }
