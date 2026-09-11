"""季度成熟标签的七/八列线性对照；份额只增加一列，算法设置沿用原研究。"""

import math
import warnings
from collections import Counter

from pydantic import Field, FiniteFloat

from app.schemas.direction_training import DirectionAnswer, DirectionInput
from app.services.direction_etf_share_data import quarter_month
from app.services.direction_linear_protocol import (
    ETF_SHARE_BRANCHES as BRANCHES,
)
from app.services.direction_linear_protocol import (
    ETF_SHARE_VERSION as VERSION,
)
from app.services.direction_linear_protocol import (
    planned_dates,
    study_windows,
)
from app.services.direction_rolling_models import VERSION as ROLLING_VERSION
from app.services.direction_rolling_models import validate_job as validate_time_job
from app.services.direction_training_artifacts import digest
from app.services.direction_training_dataset import FUNDS
from app.services.historical_nav_training import sigmoid
from app.services.trading_calendar import load_calendar


class ShareInput(DirectionInput):
    x: tuple[FiniteFloat, ...] = Field(min_length=7, max_length=8)


def validate_job(payload):
    if (
        set(payload) != {"version", "branch", "window", "fit", "exam"}
        or payload["version"] != VERSION
        or payload["branch"] not in BRANCHES
        or payload["window"] not in study_windows(VERSION)
    ):
        raise ValueError("ETF_SHARE_JOB_SCOPE")
    dimensions = 7 if payload["branch"] == "REFERENCE" else 8
    if not 1 <= len(payload["fit"]) <= 10000 or not 1 <= len(payload["exam"]) <= 300:
        raise ValueError("ETF_SHARE_JOB_SIZE")
    fit = []
    for row in payload["fit"]:
        if set(row) != {"input", "answer"}:
            raise ValueError("ETF_SHARE_FIT_FIELDS")
        fit.append((ShareInput.model_validate(row["input"]), DirectionAnswer.model_validate(row["answer"])))
    exam = [ShareInput.model_validate(i) for i in payload["exam"]]
    if any(len(i.x) != dimensions for i in [*(i for i, _ in fit), *exam]):
        raise ValueError("ETF_SHARE_DIMENSIONS")
    month = quarter_month(payload["window"])

    # 复用已验证的月初成熟边界及等基金、共同日期规则；季度内其余EXAM另按完整计划检查。
    def nav_only(i):
        return {**i.model_dump(mode="json"), "x": list(i.x[:7])}

    validate_time_job(
        {
            "version": ROLLING_VERSION,
            "month": month,
            "recipe": "ALL",
            "fit": [{"input": nav_only(i), "answer": a.model_dump(mode="json")} for i, a in fit],
            "exam": [nav_only(i) for i in exam if str(i.cutoff).startswith(month)],
        }
    )
    allowed = set(planned_dates(VERSION, payload["window"]))
    calendar = load_calendar()
    if len({i.key for i in exam}) != len(exam) or any(
        i.cutoff not in allowed or i.anchor != calendar.sessions[calendar.at_or_before_index(i.cutoff) - 1]
        for i in exam
    ):
        raise ValueError("ETF_SHARE_EXAM_PLAN")
    return fit, exam


def restore(model):
    fields = {
        "version",
        "branch",
        "window",
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
        "hash",
    }
    if set(model) != fields or model["version"] != VERSION or model["branch"] not in BRANCHES:
        raise ValueError("ETF_SHARE_MODEL_FIELDS")
    windows = {w["name"]: w for w in study_windows(VERSION)}
    if (
        model["window"] not in windows
        or model["fit_end"] != windows[model["window"]]["fit_end"]
        or model["C"] != 1.0
        or model["penalty"] != "L2"
    ):
        raise ValueError("ETF_SHARE_MODEL_PROTOCOL")
    if model["hash"] != digest({k: v for k, v in model.items() if k != "hash"}):
        raise ValueError("ETF_SHARE_MODEL_HASH")
    size = 7 if model["branch"] == "REFERENCE" else 8
    if (
        any(
            len(model[k]) != size or any(not math.isfinite(v) for v in model[k])
            for k in ("mean", "scale", "coefficients")
        )
        or any(v <= 0 for v in model["scale"])
        or not math.isfinite(model["intercept"])
    ):
        raise ValueError("ETF_SHARE_MODEL_NUMBERS")
    if type(model["solver_iterations"]) is not int or not 1 <= model["solver_iterations"] < 1000:
        raise ValueError("ETF_SHARE_MODEL_CONVERGENCE")
    counts = model["train_counts"]
    if (
        set(counts) != set(FUNDS)
        or len(set(counts.values())) != 1
        or any(type(n) is not int or not 252 <= n <= 10000 for n in counts.values())
    ):
        raise ValueError("ETF_SHARE_MODEL_COUNTS")
    return model


def predict_model(model, items):
    model = restore(model)
    if any(len(i.x) != len(model["mean"]) or i.fund not in FUNDS for i in items):
        raise ValueError("ETF_SHARE_PREDICT_SCOPE")
    return [
        sigmoid(
            model["intercept"]
            + sum(
                (x - mean) / scale * coef
                for x, mean, scale, coef in zip(i.x, model["mean"], model["scale"], model["coefficients"], strict=True)
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
        raise ValueError("ETF_SHARE_MODEL_NOT_CONVERGED")
    model = {
        "version": VERSION,
        "branch": payload["branch"],
        "window": payload["window"]["name"],
        "fit_end": payload["window"]["fit_end"],
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
    }
    model["hash"] = digest(model)
    scores = predict_model(model, exam)
    native = estimator.predict_proba(scaler.transform(np.asarray([i.x for i in exam])))[:, 1]
    if any(abs(a - float(b)) > 1e-12 for a, b in zip(scores, native, strict=True)):
        raise ValueError("ETF_SHARE_NATIVE_PARITY")
    return {"status": "PREDICTED", "model": model, "scores": scores, "model_fit_count": 1}
