"""月初成熟标签训练器；两种历史规则共享原线性算法，模型可用于当月或整季推理。"""

import math
import warnings
from collections import Counter
from datetime import date, timedelta

from app.schemas.direction_training import DirectionAnswer, DirectionInput
from app.services.direction_linear_protocol import ROLLING_VERSION as VERSION
from app.services.direction_training_artifacts import digest
from app.services.direction_training_dataset import FUNDS
from app.services.historical_nav_training import sigmoid
from app.services.trading_calendar import load_calendar

RECIPES = ("ALL", "252")
MONTHS = tuple(f"{year}-{month:02}" for year in (2023, 2024) for month in range(1, 13))


def fit_end(month):
    if month not in MONTHS:
        raise ValueError("ROLLING_MONTH")
    return date.fromisoformat(month + "-01") - timedelta(days=1)


def model_key(branch, cutoff):
    """季度分支固定取季度首月模型；月度分支按输入日期取当月模型。"""
    if branch not in ("QUARTER_ALL", "QUARTER_252", "MONTH_ALL", "MONTH_252"):
        raise ValueError("ROLLING_BRANCH")
    frequency, recipe = branch.split("_")
    day = date.fromisoformat(str(cutoff))
    month = day.month if frequency == "MONTH" else (day.month - 1) // 3 * 3 + 1
    key = f"{day.year}-{month:02}"
    fit_end(key)
    return f"{key}_{recipe}"


def validate_job(payload):
    if set(payload) != {"version", "month", "recipe", "fit", "exam"} or payload["version"] != VERSION:
        raise ValueError("ROLLING_JOB_FIELDS")
    end = fit_end(payload["month"])
    if (
        payload["recipe"] not in RECIPES
        or not 1 <= len(payload["fit"]) <= 10000
        or not 1 <= len(payload["exam"]) <= 300
    ):
        raise ValueError("ROLLING_JOB_SCOPE")
    calendar = load_calendar()
    fit, identities = [], set()
    for row in payload["fit"]:
        if set(row) != {"input", "answer"}:
            raise ValueError("ROLLING_FIT_FIELDS")
        item, answer = DirectionInput.model_validate(row["input"]), DirectionAnswer.model_validate(row["answer"])
        if item.key != answer.key or item.key in identities or not answer.available_at <= end:
            raise ValueError("ROLLING_LABEL_NOT_MATURE")
        if answer.end != calendar.future_sessions(item.cutoff)[-1]:
            raise ValueError("ROLLING_LABEL_HORIZON")
        if answer.available_at < calendar.future_sessions(item.cutoff, 21)[-1]:
            raise ValueError("ROLLING_LABEL_PUBLICATION")
        identities.add(item.key)
        fit.append((item, answer))
    counts = Counter(i.fund for i, _ in fit)
    if set(counts) != set(FUNDS) or len(set(counts.values())) != 1 or min(counts.values()) < 252:
        raise ValueError("ROLLING_FIT_COUNTS")
    if payload["recipe"] == "252" and set(counts.values()) != {252}:
        raise ValueError("ROLLING_RECENT_COUNT")
    dates = [{i.cutoff for i, _ in fit if i.fund == f} for f in FUNDS]
    if not dates[0] == dates[1] == dates[2]:
        raise ValueError("ROLLING_FIT_DATES")
    exam = [DirectionInput.model_validate(r) for r in payload["exam"]]
    if len({i.key for i in exam}) != len(exam) or any(str(i.cutoff)[:7] != payload["month"] for i in exam):
        raise ValueError("ROLLING_EXAM_MONTH")
    if any(calendar.future_sessions(i.cutoff, 21)[-1] > date(2024, 12, 31) for i in exam):
        raise ValueError("ROLLING_PROTECTED_LABEL_PERIOD")
    for i in [*(i for i, _ in fit), *exam]:
        if i.anchor != calendar.sessions[calendar.at_or_before_index(i.cutoff) - 1]:
            raise ValueError("ROLLING_ANCHOR")
    return fit, exam


def restore(model):
    fields = {
        "version",
        "month",
        "recipe",
        "fit_end",
        "mean",
        "scale",
        "coefficients",
        "intercept",
        "C",
        "penalty",
        "solver_iterations",
        "train_hash",
        "train_counts",
        "train_up_rates",
        "hash",
    }
    if set(model) != fields or model["version"] != VERSION or model["recipe"] not in RECIPES:
        raise ValueError("ROLLING_MODEL_FIELDS")
    if model["fit_end"] != str(fit_end(model["month"])) or model["C"] != 1.0 or model["penalty"] != "L2":
        raise ValueError("ROLLING_MODEL_PROTOCOL")
    if model["hash"] != digest({k: v for k, v in model.items() if k != "hash"}):
        raise ValueError("ROLLING_MODEL_HASH")
    if type(model["solver_iterations"]) is not int or not 1 <= model["solver_iterations"] < 1000:
        raise ValueError("ROLLING_MODEL_CONVERGENCE")
    if (
        any(len(model[k]) != 7 for k in ("mean", "scale", "coefficients"))
        or any(not math.isfinite(v) for k in ("mean", "scale", "coefficients") for v in model[k])
        or not math.isfinite(model["intercept"])
        or any(v <= 0 for v in model["scale"])
    ):
        raise ValueError("ROLLING_MODEL_NUMBERS")
    if set(model["train_counts"]) != set(FUNDS) or set(model["train_up_rates"]) != set(FUNDS):
        raise ValueError("ROLLING_MODEL_FUNDS")
    if (
        any(type(n) is not int or not 252 <= n <= 10000 for n in model["train_counts"].values())
        or len(set(model["train_counts"].values())) != 1
    ):
        raise ValueError("ROLLING_MODEL_COUNTS")
    if model["recipe"] == "252" and set(model["train_counts"].values()) != {252}:
        raise ValueError("ROLLING_MODEL_RECENT_COUNT")
    if any(not math.isfinite(v) or not 0 <= v <= 1 for v in model["train_up_rates"].values()):
        raise ValueError("ROLLING_MODEL_RATES")
    return model


def predict_model(model, items):
    model = restore(model)
    if any(len(i.x) != 7 or i.fund not in FUNDS for i in items):
        raise ValueError("ROLLING_PREDICT_SCOPE")
    return [
        sigmoid(
            model["intercept"]
            + sum(
                (x - mean) / scale * coefficient
                for x, mean, scale, coefficient in zip(
                    i.x, model["mean"], model["scale"], model["coefficients"], strict=True
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
    counts = Counter(i.fund for i, _ in fit)
    weights = [len(fit) / (len(counts) * counts[i.fund]) for i, _ in fit]
    x = np.asarray([i.x for i, _ in fit])
    with threadpool_limits(limits=1), warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        warnings.simplefilter("error", RuntimeWarning)
        scaler = StandardScaler().fit(x, sample_weight=weights)
        estimator = LogisticRegression(
            C=1.0,
            l1_ratio=0.0,
            solver="lbfgs",
            tol=1e-8,
            max_iter=1000,
            random_state=0,
            fit_intercept=True,
            class_weight=None,
        )
        estimator.fit(scaler.transform(x), [a.y for _, a in fit], sample_weight=weights)
    if estimator.classes_.tolist() != [0, 1] or int(estimator.n_iter_[0]) >= 1000:
        raise ValueError("ROLLING_MODEL_NOT_CONVERGED")
    model = {
        "version": VERSION,
        "month": payload["month"],
        "recipe": payload["recipe"],
        "fit_end": str(fit_end(payload["month"])),
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
        "coefficients": estimator.coef_[0].tolist(),
        "intercept": float(estimator.intercept_[0]),
        "C": 1.0,
        "penalty": "L2",
        "solver_iterations": int(estimator.n_iter_[0]),
        "train_hash": digest(
            [{"input": i.model_dump(mode="json"), "answer": a.model_dump(mode="json")} for i, a in fit]
        ),
        "train_counts": dict(counts),
        "train_up_rates": {f: sum(a.y for i, a in fit if i.fund == f) / counts[f] for f in FUNDS},
    }
    model["hash"] = digest(model)
    scores = predict_model(model, exam)
    native = estimator.predict_proba(scaler.transform(np.asarray([i.x for i in exam])))[:, 1]
    if any(abs(a - float(b)) > 1e-12 for a, b in zip(scores, native, strict=True)):
        raise ValueError("ROLLING_NATIVE_PARITY")
    return {"status": "PREDICTED", "model": model, "scores": scores, "model_fit_count": 1}
