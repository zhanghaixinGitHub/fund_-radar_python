"""固定浅层树与线性对照；树只导出数值节点，复核不加载可执行模型文件。"""

import math
from collections import Counter

from app.services import direction_linear_models as linear
from app.services.direction_linear_protocol import ALGORITHM_BRANCHES, ALGORITHM_VERSION, fit_boundary, study_rules
from app.services.direction_training_artifacts import digest
from app.services.direction_training_dataset import FUNDS
from app.services.direction_training_protocol import TREE
from app.services.historical_nav_training import sigmoid

TREE_BRANCHES = ALGORITHM_BRANCHES[2:]


def restore_tree(model):
    """检查固定参数、特征维度和有界无环树；只支持本实验完整的数值输入。"""
    fields = {
        "version",
        "branch",
        "fund",
        "parameters",
        "dimensions",
        "baseline",
        "trees",
        "train_hash",
        "train_counts",
        "fit_end",
        "hash",
        "format",
    }
    if set(model) != fields or model["format"] != "NUMERIC_HIST_TREE_JSON_V1":
        raise ValueError("TREE_MODEL_FIELDS")
    if model["version"] != ALGORITHM_VERSION or model["branch"] not in TREE_BRANCHES or model["fund"] != "POOLED":
        raise ValueError("TREE_MODEL_SCOPE")
    if (
        model["parameters"] != TREE
        or model["dimensions"] != len(study_rules(ALGORITHM_VERSION)[1][model["branch"]])
        or not math.isfinite(model["baseline"])
        or len(model["trees"]) != TREE["max_iter"]
    ):
        raise ValueError("TREE_MODEL_PARAMETERS")
    if model["hash"] != digest({k: v for k, v in model.items() if k != "hash"}):
        raise ValueError("TREE_MODEL_HASH")
    for nodes in model["trees"]:
        if not 1 <= len(nodes) <= 2 * TREE["max_leaf_nodes"] - 1:
            raise ValueError("TREE_NODE_BUDGET")
        seen, pending, leaves = set(), [(0, 0)], 0
        while pending:
            index, depth = pending.pop()
            if type(index) is not int or not 0 <= index < len(nodes) or index in seen or depth > TREE["max_depth"]:
                raise ValueError("TREE_NODE_STRUCTURE")
            seen.add(index)
            node = nodes[index]
            if set(node) == {"value"}:
                if not math.isfinite(node["value"]):
                    raise ValueError("TREE_LEAF_NUMBER")
                leaves += 1
            elif set(node) == {"feature", "threshold", "left", "right"}:
                if (
                    type(node["feature"]) is not int
                    or not 0 <= node["feature"] < model["dimensions"]
                    or not math.isfinite(node["threshold"])
                ):
                    raise ValueError("TREE_SPLIT_NUMBER")
                pending.extend((node[k], depth + 1) for k in ("left", "right"))
            else:
                raise ValueError("TREE_NODE_FIELDS")
        if len(seen) != len(nodes) or leaves > TREE["max_leaf_nodes"]:
            raise ValueError("TREE_UNREACHABLE_OR_TOO_MANY_LEAVES")
    return model


def predict_model(model, items):
    """按原始数值阈值逐树累加；叶值已含学习率，不再次乘0.05。"""
    if model.get("branch") not in TREE_BRANCHES:
        return linear.predict_model(model, items)
    model = restore_tree(model)
    scores = []
    for item in items:
        if item.fund not in FUNDS or len(item.x) != model["dimensions"] or any(not math.isfinite(x) for x in item.x):
            raise ValueError("TREE_PREDICT_SCOPE")
        value = model["baseline"]
        for nodes in model["trees"]:
            node = nodes[0]
            while "value" not in node:
                node = nodes[node["left"] if item.x[node["feature"]] <= node["threshold"] else node["right"]]
            value += node["value"]
        scores.append(sigmoid(value))
    return scores


def execute_job(payload):
    """每次作业仅拟合一个已冻结分支；训练标签必须在该窗口FIT截止日成熟。"""
    if payload.get("version") != ALGORITHM_VERSION:
        raise ValueError("ALGORITHM_WORKER_VERSION")
    fit, exam = linear.validate_job(payload)
    if payload["branch"] not in TREE_BRANCHES:
        return linear.execute_job(payload)
    import numpy as np
    from sklearn.ensemble import HistGradientBoostingClassifier
    from threadpoolctl import threadpool_limits

    counts = Counter(i.fund for i, _ in fit)
    weights = [len(fit) / (len(counts) * counts[i.fund]) for i, _ in fit]
    with threadpool_limits(limits=1):
        estimator = HistGradientBoostingClassifier(**TREE)
        estimator.fit(np.asarray([i.x for i, _ in fit]), [a.y for _, a in fit], sample_weight=weights)
        if estimator.classes_.tolist() != [0, 1] or estimator.n_iter_ != TREE["max_iter"]:
            raise ValueError("TREE_FIT_SHAPE")
        trees = []
        for iteration in estimator._predictors:
            if len(iteration) != 1:
                raise ValueError("TREE_MULTICLASS_NOT_ALLOWED")
            nodes = iteration[0].nodes
            if any(n["is_categorical"] for n in nodes):
                raise ValueError("TREE_CATEGORICAL_NOT_ALLOWED")
            trees.append(
                [
                    {"value": float(n["value"])}
                    if n["is_leaf"]
                    else {
                        "feature": int(n["feature_idx"]),
                        "threshold": float(n["num_threshold"]),
                        "left": int(n["left"]),
                        "right": int(n["right"]),
                    }
                    for n in nodes
                ]
            )
        model = {
            "format": "NUMERIC_HIST_TREE_JSON_V1",
            "version": ALGORITHM_VERSION,
            "branch": payload["branch"],
            "fund": "POOLED",
            "parameters": dict(TREE),
            "dimensions": len(fit[0][0].x),
            "baseline": float(estimator._baseline_prediction[0, 0]),
            "trees": trees,
            "train_counts": dict(counts),
            "train_hash": digest(
                [{"input": i.model_dump(mode="json"), "answer": a.model_dump(mode="json")} for i, a in fit]
            ),
            "fit_end": fit_boundary(ALGORITHM_VERSION, payload["branch"], payload["window"]),
        }
        model["hash"] = digest(model)
        # 同时检查FIT和EXAM上的数值，避免只有少量考试点恰好走到正确路径。
        items = [i for i, _ in fit] + exam
        scores = predict_model(model, items)
        expected = estimator.predict_proba(np.asarray([i.x for i in items]))[:, 1]
        if any(abs(a - float(b)) > 1e-12 for a, b in zip(scores, expected, strict=True)):
            raise ValueError("TREE_JSON_REPLAY")
    return {
        "status": "PREDICTED",
        "scores": scores[len(fit) :],
        "models": {"POOLED": model},
        "model_fit_count": 1,
        "train_counts": dict(counts),
    }
