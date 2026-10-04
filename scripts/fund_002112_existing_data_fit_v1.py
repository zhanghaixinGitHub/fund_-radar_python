"""002112 新实验的固定模型与纯指标；不导入业务服务，不登记或替换模型。

类别固定按持平、下跌、上涨排列，概率相等时也按此顺序决策。
监督拟合只能由 runner 原子预占后进入 worker；人工测试另记调用清单。
"""

# 固定线程配置必须发生在数值库导入前。
# ruff: noqa: E402

from __future__ import annotations

import hashlib
import json
import os
import warnings
from collections import Counter

# 必须早于 NumPy/sklearn 导入，子进程同时继承这些固定设置。
for _variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_variable] = "1"

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

CLASSES = ("FLAT", "DOWN", "UP")
GROUPS = {"A": "N", "B": "NM", "C": "NHM", "D": "NMI", "E": "NHMI", "F": "NHMIF", "G": "NHMIG", "H": "NHMI"}
LR_PARAMS = {"C": 1.0, "solver": "lbfgs", "tol": 1e-8, "max_iter": 1000, "random_state": 0}
H_PARAMS = {
    "learning_rate": 0.05,
    "max_iter": 100,
    "max_depth": 3,
    "max_leaf_nodes": 7,
    "min_samples_leaf": 20,
    "l2_regularization": 1.0,
    "early_stopping": False,
    "random_state": 20260930,
}


def digest(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":"), default=str
        ).encode()
    ).hexdigest()


def dependencies(candidate):
    """仅有协议指定的备用依赖，顺序先 A 后 D 后本体。"""
    return list(dict.fromkeys(["A"] + (["D"] if candidate in "DEFGH" else []) + [candidate]))


def columns(candidate, names):
    return [name for group in GROUPS[candidate] for name in names.get(group, [])]


def usable(row, candidate, names):
    if any(not row["available"].get(g, False) for g in GROUPS[candidate]):
        return False
    cols = columns(candidate, names)
    return bool(cols) and all(row["features"].get(c) is not None and np.isfinite(row["features"][c]) for c in cols)


def route(row, candidate, models, names):
    """路由仅读取 X 与训练成功状态，绝不接收目标标签。"""
    for branch in reversed(dependencies(candidate)):
        if branch in models and usable(row, branch, names):
            reasons = [
                g + ":" + str(row.get("reasons", {}).get(g, "unavailable"))
                for g in GROUPS[candidate]
                if not row["available"].get(g)
            ]
            if candidate not in models:
                reasons.append("requested_model_not_fitted")
            return branch, None if branch == candidate else ";".join(reasons) or "branch_input_not_finite"
    return None, "共同净值输入不可用或 A 未拟合"


def fit_estimator(x, y, candidate, *, retry=False):
    """只见当前训练行；不删除少见真实类别，不引入类别或时间权重。"""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y)
    if x.ndim != 2 or not np.isfinite(x).all() or len(set(y)) < 2:
        raise ValueError("INVALID_TRAIN_MATRIX_OR_CLASSES")
    params = dict(H_PARAMS if candidate == "H" else LR_PARAMS)
    if retry and candidate != "H":
        params["max_iter"] = 4000
    with threadpool_limits(limits=1), warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        scaler = None if candidate == "H" else StandardScaler().fit(x)
        transformed = x if scaler is None else scaler.transform(x)
        model = HistGradientBoostingClassifier(**params) if candidate == "H" else LogisticRegression(**params)
        model.fit(transformed, y)
    convergence = [str(w.message) for w in caught if issubclass(w.category, ConvergenceWarning)]
    return {
        "model": model,
        "scaler": scaler,
        "candidate": candidate,
        "params": params,
        "warnings": [str(w.message) for w in caught],
        "converged": not convergence,
        "observed_classes": model.classes_.tolist(),
        "unlearnable_classes": [c for c in CLASSES if c not in y],
        "zero_variance_columns": np.flatnonzero(np.var(x, axis=0) == 0).tolist(),
    }


def probabilities(bundle, x):
    x = np.asarray(x, dtype=float)
    if bundle["scaler"] is not None:
        x = bundle["scaler"].transform(x)
    with threadpool_limits(limits=1):
        raw = bundle["model"].predict_proba(x)
    out = np.zeros((len(x), 3))
    for i, label in enumerate(bundle["model"].classes_):
        out[:, CLASSES.index(str(label))] = raw[:, i]
    return out


def predict_label(probs):
    return CLASSES[int(np.argmax(probs))]


def model_structure(bundle):
    """独立复现比对参数和树结点，而非只比对序列化文件字节。"""
    model, scaler = bundle["model"], bundle["scaler"]
    result = {"classes": model.classes_.tolist(), "params": bundle["params"]}
    if scaler is not None:
        result.update(
            mean=scaler.mean_.tolist(),
            scale=scaler.scale_.tolist(),
            variance=scaler.var_.tolist(),
            coef=model.coef_.tolist(),
            intercept=model.intercept_.tolist(),
            n_iter=model.n_iter_.tolist(),
        )
    else:
        result.update(
            baseline=model._baseline_prediction.tolist(),
            n_iter=model.n_iter_,
            trees=[[p.nodes.tolist() for p in stage] for stage in model._predictors],
            bins=[x.tolist() for x in model._bin_mapper.bin_thresholds_],
        )
    return result


def compare_models(left, right, x, tolerance=1e-10):
    p, q = probabilities(left, x), probabilities(right, x)
    difference = float(np.max(np.abs(p - q))) if len(p) else 0.0
    # 固定环境要求结构完全一致，概率另有明确的绝对容忍范围。
    same_structure = digest(model_structure(left)) == digest(model_structure(right))
    same_classes = np.array_equal(np.argmax(p, axis=1), np.argmax(q, axis=1))
    return {
        "structure_equal": same_structure,
        "predicted_classes_equal": bool(same_classes),
        "max_probability_difference": difference,
        "tolerance": tolerance,
        "passed": same_structure and bool(same_classes) and difference <= tolerance,
    }


def metrics(records):
    """records 为已到开放阶段的共同日期；真实持平保持独立类别。"""
    matrix = np.zeros((3, 3), dtype=int)
    loss = []
    for record in records:
        actual, predicted = record["actual"], record["predicted"]
        matrix[CLASSES.index(actual), CLASSES.index(predicted)] += 1
        if record.get("probabilities") is not None:
            loss.append(-np.log(max(record["probabilities"][CLASSES.index(actual)], 1e-15)))
    n, correct = int(matrix.sum()), int(np.trace(matrix))
    per_class = {}
    for i, label in enumerate(CLASSES):
        support = int(matrix[i].sum())
        per_class[label] = {
            "n": support,
            "correct": int(matrix[i, i]),
            "recall": float(matrix[i, i] / support) if support else None,
            "evidence": "无样本" if not support else ("证据不足" if support < 30 else "历史样本"),
        }
    return {
        "n": n,
        "correct": correct,
        "accuracy": correct / n if n else None,
        "class_order": CLASSES,
        "confusion_matrix": matrix.tolist(),
        "per_class": per_class,
        "log_loss_clipped_1e15": float(np.mean(loss)) if loss else None,
    }


def paired(base, candidate, calendar, *, bootstrap=True):
    """按原日历留空掩码做 5 日移动块抽样，不能先压缩为仅有预测的日期。"""
    a, b = {r["U"]: r for r in base}, {r["U"]: r for r in candidate}
    days = [d for d in calendar if d in a and d in b]
    cells = Counter({"both_correct": 0, "candidate_only": 0, "baseline_only": 0, "both_wrong": 0})
    diffs = {}
    for d in days:
        if a[d]["actual"] != b[d]["actual"]:
            raise ValueError("LABEL_VERSION_MISMATCH")
        ca, cb = a[d]["predicted"] == a[d]["actual"], b[d]["predicted"] == b[d]["actual"]
        cells["both_correct" if ca and cb else "candidate_only" if cb else "baseline_only" if ca else "both_wrong"] += 1
        diffs[d] = int(cb) - int(ca)
    interval = None
    if bootstrap and days:
        x = np.array([diffs.get(d, np.nan) for d in calendar], dtype=float)
        rng = np.random.default_rng(20260930)
        samples = []
        for _ in range(2000):
            starts = rng.integers(0, max(1, len(x) - 4), size=int(np.ceil(len(x) / 5)))
            indices = np.concatenate([np.arange(s, min(s + 5, len(x))) for s in starts])[: len(x)]
            sample = x[indices]
            if np.isfinite(sample).any():
                samples.append(float(np.nanmean(sample)))
        interval = np.quantile(samples, [0.025, 0.975]).tolist() if samples else None
    return {
        "common_dates": days,
        "n": len(days),
        "baseline_full_n": len(a),
        "candidate_full_n": len(b),
        "baseline": metrics([a[d] for d in days]),
        "candidate": metrics([b[d] for d in days]),
        "paired_cells": dict(cells),
        "net_correct": cells["candidate_only"] - cells["baseline_only"],
        "accuracy_difference": sum(diffs.values()) / len(days) if days else None,
        "block_bootstrap_95pct": interval,
        "bootstrap": {
            "block_sessions": 5,
            "repetitions": 2000,
            "seed": 20260930,
            "calendar_dates": len(calendar),
            "missing_mask_count": len(calendar) - len(days),
        },
    }


def choose_candidate(comparisons, fold_comparisons, input_counts, actual_models):
    """排序完全依据 2025；没有合格者也固定唯一探索候选。"""
    ranked = []
    for c in "BCDEFGH":
        if c not in actual_models or not comparisons[c]["n"]:
            continue
        pair = comparisons[c]
        recall_ok = all(
            pair["candidate"]["per_class"][k]["recall"] is not None
            and pair["baseline"]["per_class"][k]["recall"] is not None
            and pair["candidate"]["per_class"][k]["recall"] >= pair["baseline"]["per_class"][k]["recall"] - 0.05
            for k in ("UP", "DOWN")
        )
        improved_folds = sum(f[c]["net_correct"] > 0 for f in fold_comparisons.values())
        qualified = pair["net_correct"] > 0 and improved_folds >= 2 and recall_ok
        ranked.append(
            {
                "candidate": c,
                "qualified": qualified,
                "net_correct": pair["net_correct"],
                "improved_folds": improved_folds,
                "recall_guard": recall_ok,
                "input_count": input_counts[c],
            }
        )
    ranked.sort(key=lambda r: (-r["net_correct"], r["input_count"], r["candidate"] == "H", r["candidate"]))
    qualified = [r for r in ranked if r["qualified"]]
    chosen = (qualified or ranked)[0] if ranked else None
    return {
        "selected": chosen["candidate"] if chosen else "A",
        "ranked": ranked,
        "status": "VALIDATION_GAIN" if qualified else "NO_VALIDATION_GAIN" if ranked else "NO_ENHANCEMENT",
        "selection_uses_test_labels": False,
    }
