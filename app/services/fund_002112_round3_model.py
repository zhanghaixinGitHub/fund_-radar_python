"""第三轮固定 N7/L20/T20；纯数值模块，无来源采集、登记和自动采用。"""

import warnings
from collections import Counter

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from app.services.fund_002112_round3_data import CLASSES, FEATURES, TIE_ORDER, digest

VARIANTS = {"N7": 7, "L20": 20, "T20": 20}
# 1.9.0 下旧配方的全部显式参数快照；不升级依赖，也不改变原有正则化。
LOGISTIC = {
    "C": 1.0,
    "class_weight": None,
    "dual": False,
    "fit_intercept": True,
    "intercept_scaling": 1,
    "l1_ratio": 0.0,
    "max_iter": 1000,
    "n_jobs": None,
    "penalty": "deprecated",
    "random_state": 0,
    "solver": "lbfgs",
    "tol": 1e-8,
    "verbose": 0,
    "warm_start": False,
}
TREE = {
    "loss": "log_loss",
    "learning_rate": 0.05,
    "max_iter": 100,
    "max_leaf_nodes": 4,
    "max_depth": 2,
    "min_samples_leaf": 30,
    "l2_regularization": 1.0,
    "max_features": 1.0,
    "max_bins": 64,
    "categorical_features": None,
    "monotonic_cst": None,
    "interaction_cst": None,
    "warm_start": False,
    "early_stopping": False,
    "validation_fraction": None,
    "scoring": "loss",
    "n_iter_no_change": 10,
    "tol": 1e-7,
    "verbose": 0,
    "random_state": 0,
    "class_weight": None,
}


def matrix(rows, size):
    x = np.asarray([r["x"][:size] for r in rows], dtype=float)
    if any(len(r["x"]) != 20 for r in rows) or x.shape != (len(rows), size) or not np.isfinite(x).all():
        raise ValueError("FINITE_COMPLETE_20_INPUTS_REQUIRED")
    return x


def directions(scores):
    return [max(TIE_ORDER, key=lambda k: row[list(CLASSES).index(k)]) for row in scores]


def predict(model, rows):
    variant = model["variant"]
    if model["features"] != list(FEATURES[: VARIANTS[variant]]) or model["classes"] != list(CLASSES):
        raise ValueError("MODEL_CONTRACT_CHANGED")
    with threadpool_limits(limits=1):
        scores = model["classifier"].predict_proba(model["scaler"].transform(matrix(rows, VARIANTS[variant])))
    if not np.isfinite(scores).all():
        raise ValueError("NONFINITE_SCORES")
    return [
        {
            "fund_code": r["fund_code"],
            "target": r["target"],
            "actual_direction": r["actual_direction"],
            "input_hash": digest(r),
            "direction": answer,
            "scores": p.tolist(),
        }
        for r, answer, p in zip(rows, directions(scores), scores, strict=True)
    ]


def fit(training, exams, sample_weight, variant):
    """仅一次分类器拟合；调用方须先永久占用槽位。合成测试不进入真实账本。

    加权标准化和树分箱均只见训练集；检查集从不参与早停、迭代选择或任何拟合。
    """
    x = matrix(training, VARIANTS[variant])
    y = np.asarray([r["actual_direction"] for r in training])
    w = np.asarray(sample_weight, dtype=float)
    if set(y) != set(CLASSES) or w.shape != (len(training),) or not np.isfinite(w).all() or min(w) <= 0:
        raise ValueError("CLASSES_OR_WEIGHTS_INVALID")
    if abs(sum(w) - len(w)) > 1e-8:
        raise ValueError("WEIGHT_TOTAL_CHANGED")
    with threadpool_limits(limits=1), warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        scaler = StandardScaler().fit(x, sample_weight=w)
        classifier = HistGradientBoostingClassifier(**TREE) if variant == "T20" else LogisticRegression(**LOGISTIC)
        classifier.fit(scaler.transform(x), y, sample_weight=w)
        if classifier.classes_.tolist() != list(CLASSES):
            raise ValueError("CLASS_ORDER_CHANGED")
    model = {
        "variant": variant,
        "features": list(FEATURES[: VARIANTS[variant]]),
        "classes": list(CLASSES),
        "tie_order": list(TIE_ORDER),
        "recipe": TREE if variant == "T20" else LOGISTIC,
        "training_hash": digest(training),
        "weights_hash": digest(sample_weight),
        "scaler": scaler,
        "classifier": classifier,
    }
    return model, {"train": predict(model, training), "exam": predict(model, exams)}


def state_hash(model):
    """覆盖 scaler、树节点、训练分箱及完整估计器状态；排除运行时间等非预测元数据。"""
    return joblib.hash(model, hash_name="sha1")


def compare_predictions(left, right):
    if len(left) != len(right) or not left:
        raise ValueError("REPLAY_ROWS_MISMATCH")
    for a, b in zip(left, right, strict=True):
        if any(a[k] != b[k] for k in ("fund_code", "target", "actual_direction", "input_hash", "direction")):
            raise ValueError("REPLAY_IDENTITY_OR_DIRECTION_MISMATCH")
    delta = float(np.max(np.abs(np.asarray([r["scores"] for r in left]) - np.asarray([r["scores"] for r in right]))))
    if delta > 1e-12:
        raise ValueError("REPLAY_SCORE_MISMATCH")
    return delta


def metrics(rows):
    actual = Counter(r["actual_direction"] for r in rows)
    confusion = {
        a: {p: sum(r["actual_direction"] == a and r["direction"] == p for r in rows) for p in CLASSES} for a in CLASSES
    }
    correct = {k: confusion[k][k] for k in CLASSES}
    return {
        "days": len(rows),
        "correct": sum(correct.values()),
        "actual": {k: actual[k] for k in CLASSES},
        "class_correct": correct,
        "confusion": confusion,
    }


def comparison(by_variant, *, historical):
    """预定规则只比较同日逐日记录，不能据成绩调整日期或三类分母。"""
    if set(by_variant) != set(VARIANTS):
        raise ValueError("THREE_VARIANTS_REQUIRED")
    expected = [(r["target"], r["actual_direction"], r["input_hash"]) for r in by_variant["L20"]]
    if any(
        [(r["target"], r["actual_direction"], r["input_hash"]) for r in rows] != expected
        for rows in by_variant.values()
    ):
        raise ValueError("COMPARISON_DATES_CHANGED")
    result = {"models": {v: metrics(rows) for v, rows in by_variant.items()}, "pairs": {}, "quarters": {}}
    result["constants"] = {k: metrics([{**r, "direction": k} for r in by_variant["L20"]]) for k in CLASSES}
    for quarter in sorted(
        {r["target"][:4] + "Q" + str((int(r["target"][5:7]) - 1) // 3 + 1) for r in by_variant["L20"]}
    ):
        result["quarters"][quarter] = {
            v: metrics(
                [r for r in rows if r["target"][:4] + "Q" + str((int(r["target"][5:7]) - 1) // 3 + 1) == quarter]
            )
            for v, rows in by_variant.items()
        }
    for baseline in ("N7", "L20"):
        daily = []
        for t, b in zip(by_variant["T20"], by_variant[baseline], strict=True):
            tc, bc = t["direction"] == t["actual_direction"], b["direction"] == b["actual_direction"]
            daily.append(
                {
                    "target": t["target"],
                    "actual": t["actual_direction"],
                    "T20": t["direction"],
                    baseline: b["direction"],
                    "outcome": "both_correct" if tc and bc else "gained" if tc else "lost" if bc else "both_wrong",
                }
            )
        counts = Counter(r["outcome"] for r in daily)
        result["pairs"][baseline] = {
            "gained": counts["gained"],
            "lost": counts["lost"],
            "net": counts["gained"] - counts["lost"],
            "daily": daily,
        }
    t = result["models"]["T20"]
    checks = {
        "total_strictly_above_controls": t["correct"] > max(result["models"][v]["correct"] for v in ("N7", "L20")),
        "total_strictly_above_constants": t["correct"] > max(v["correct"] for v in result["constants"].values()),
    }
    for k in CLASSES:
        checks["class_not_worse_" + k] = all(
            t["class_correct"][k] >= result["models"][v]["class_correct"][k] for v in ("N7", "L20")
        )
    if historical:
        checks["at_least_two_quarters_not_worse_L20"] = (
            sum(q["T20"]["correct"] >= q["L20"]["correct"] for q in result["quarters"].values()) >= 2
        )
    result.update(checks=checks, numerical_passed=all(checks.values()), historical=historical)
    return result
