"""002112 第二轮的固定实验契约；每个假设只改变一项，禁止按开发成绩搜索参数。"""

import math
import warnings
from datetime import datetime

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from app.services import fund_002112_training_model as old
from app.services import fund_training_package as package
from app.services.direction_1d_protocol import RECIPE, digest
from app.services.direction_1d_three_state import CLASSES, TIE_ORDER
from app.services.direction_1d_training import weights

PROTOCOL = "002112_DIAGNOSED_TWO_HYPOTHESES_V1"
CONFIGS = {"A7": 7, "C20": 20, "W20": 20, "I24": 24}
CANDIDATES = ("W20", "I24")
QUARTERS = (
    ("2023-01-01", "2023-03-31"),
    ("2023-04-01", "2023-06-30"),
    ("2023-07-01", "2023-09-30"),
    ("2023-10-01", "2023-12-31"),
)
FEATURES = package.PLAN["features"]
INTERACTIONS = ["reported_stock_weight_x_" + f for f in FEATURES[16:20]]
PLAN = {
    "version": PROTOCOL,
    "fund_code": "002112",
    "horizon": 1,
    "recipe": RECIPE,
    "quarters": [list(q) for q in QUARTERS],
    "configs": CONFIGS,
    "target_weight": 0.5,
    "other_weight": "EQUAL_PRESENT_PEER_FAMILIES",
    "total_weight": "TRAIN_ROW_COUNT",
    "interaction": "FEATURE_13_TIMES_FEATURES_16_TO_19",
    "no_combined_weight_interaction": True,
    "maximum_fits": 28,
    "fits_per_stage": 2,
    "minimum_class_dates": 30,
    "minimum_dates": 252,
    "minimum_target_dates": 252,
    "historical_selection": {
        "correct_above_A_C_and_best_constant": True,
        "all_class_hits_at_least_A_and_C": True,
        "quarters_at_least_C": 2,
        "ranking": "CORRECT_DESC_THEN_FEWER_FEATURES",
    },
    "development_2024": "ALL_FIXED_CANDIDATES_REPORTED_ONLY_PRESELECTED_CAN_PASS_NO_FALLBACK",
    "final_gate": "CORRECT_ABOVE_OLD_A_C_CONSTANTS_AND_CLASS_HITS_AT_LEAST_OLD_A",
    "protected_years": [2025],
    "reconstructed_2026_labels": False,
    "independent_test": False,
    "automatic_adoption": False,
}


def build_folds(data):
    """自然季度切分全部基金日期，只在切分后计算家族权重；起点不随结果调整。"""
    folds = []
    for i, (start, end) in enumerate(QUARTERS, 1):
        cutoff = datetime.fromisoformat(start + "T08:00:00+08:00")
        train = [r for r in data["train"] if r["target"] < start and datetime.fromisoformat(r["mature_at"]) <= cutoff]
        exam = [r for r in data["train"] if start <= r["target"] <= end and r["fund_code"] == "002112"]
        gate = package.eligibility(train, exam)
        folds.append(
            {
                "name": f"2023Q{i}",
                "start": start,
                "end": end,
                "train": train,
                "exam": exam,
                "gate": gate,
                "eligible": gate["training_ready"],
                "train_hash": digest(train),
                "exam_hash": digest(exam),
            }
        )
    return folds


def names(config):
    if config not in CONFIGS:
        raise ValueError("OPT_CONFIG_INVALID")
    return FEATURES[:7] if config == "A7" else FEATURES + INTERACTIONS if config == "I24" else FEATURES


def vector(row, config):
    x = row["x"]
    if len(x) != 20 or any(type(v) not in (float, int) or not math.isfinite(v) for v in x):
        raise ValueError("OPT_NONFINITE_OR_SHAPE")
    if not 0 <= x[13] <= 1:
        raise ValueError("OPT_STOCK_WEIGHT_RANGE")
    return x[:7] if config == "A7" else x + [x[13] * v for v in x[16:20]] if config == "I24" else list(x)


def sample_weights(rows, config):
    """仅 W20 更改目标/参照之间的总权重；不改日期权重顺序或类别权重。"""
    result = weights(rows)
    if config == "W20":
        own = np.asarray([r["fund_code"] == "002112" for r in rows])
        if not own.any() or own.all():
            raise ValueError("OPT_TARGET_AND_PEERS_REQUIRED")
        result[own] *= len(rows) * PLAN["target_weight"] / result[own].sum()
        result[~own] *= len(rows) * (1 - PLAN["target_weight"]) / result[~own].sum()
    if not np.isfinite(result).all() or (result <= 0).any() or abs(result.sum() - len(rows)) > 1e-8:
        raise ValueError("OPT_WEIGHT_CONTRACT")
    return result


def predict(model, row):
    """使用独立 JSON 契约还原 7/20/24 维模型，不放宽第一轮正式推理的输入限制。"""
    config = model.get("config")
    if (
        config not in CONFIGS
        or model.get("protocol") != PROTOCOL
        or model.get("fund_code") != "002112"
        or model.get("features") != names(config)
        or model.get("recipe") != RECIPE
        or model.get("class_order") != list(CLASSES)
        or model.get("tie_order") != list(TIE_ORDER)
    ):
        raise ValueError("OPT_MODEL_CONTRACT")
    x = vector(row, config)
    size = CONFIGS[config]
    arrays = [model.get("mean", []), model.get("scale", []), *model.get("coef", [])]
    intercept = model.get("intercept", [])
    if (
        len(arrays) != 5
        or any(len(a) != size for a in arrays)
        or len(intercept) != 3
        or any(type(v) not in (int, float) or not math.isfinite(v) for a in [*arrays, intercept] for v in a)
        or any(v <= 0 for v in model["scale"])
    ):
        raise ValueError("OPT_MODEL_VALUE")
    normalized = [(v - mean) / scale for v, mean, scale in zip(x, model["mean"], model["scale"], strict=True)]
    logits = [
        b + sum(w * v for w, v in zip(a, normalized, strict=True))
        for a, b in zip(model["coef"], intercept, strict=True)
    ]
    if not all(math.isfinite(v) for v in logits):
        raise ValueError("OPT_MODEL_NONFINITE")
    exp = [math.exp(v - max(logits)) for v in logits]
    scores = dict(zip(CLASSES, [v / sum(exp) for v in exp], strict=True))
    return {"direction": max(TIE_ORDER, key=scores.__getitem__), "class_scores": scores}


def predict_rows(model, rows):
    return [
        {
            "target": r["target"],
            "fund_code": r["fund_code"],
            "actual_direction": r["actual_direction"],
            "input_hash": digest(r),
            "model_hash": digest(model),
            **predict(model, r),
        }
        for r in rows
    ]


def fit(train, exam, config, stage):
    """单次固定拟合；运行器在调用前永久记一次预算，收敛警告不可自动加迭代或重训。"""
    if not package.eligibility(train, exam)["training_ready"]:
        raise ValueError("OPT_TRAINING_GATE")
    cutoff = (
        "2024-01-01T08:00:00+08:00"
        if stage == "FULL"
        else next(start + "T08:00:00+08:00" for i, (start, _) in enumerate(QUARTERS, 1) if stage == f"2023Q{i}")
    )
    if any(
        r["target"] >= cutoff[:10] or datetime.fromisoformat(r["mature_at"]) > datetime.fromisoformat(cutoff)
        for r in train
    ):
        raise ValueError("OPT_FUTURE_TRAINING_ROW")
    if any(r["target"] < cutoff[:10] or r["fund_code"] != "002112" for r in exam):
        raise ValueError("OPT_EXAM_SCOPE")
    x = np.asarray([vector(r, config) for r in train], dtype=float)
    y = np.asarray([r["actual_direction"] for r in train])
    weight = sample_weights(train, config)
    with threadpool_limits(limits=1), warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        scaler = StandardScaler().fit(x, sample_weight=weight)
        classifier = LogisticRegression(**RECIPE).fit(scaler.transform(x), y, sample_weight=weight)
        if classifier.classes_.tolist() != list(CLASSES):
            raise ValueError("OPT_THREE_CLASSES_REQUIRED")
        model = {
            "protocol": PROTOCOL,
            "fund_code": "002112",
            "config": config,
            "stage": stage,
            "features": names(config),
            "class_order": list(CLASSES),
            "tie_order": list(TIE_ORDER),
            "recipe": RECIPE,
            "mean": scaler.mean_.tolist(),
            "scale": scaler.scale_.tolist(),
            "coef": classifier.coef_.tolist(),
            "intercept": classifier.intercept_.tolist(),
            "train_hash": digest(train),
            "weight_hash": digest(weight.tolist()),
            "train_count": len(train),
            "train_as_of": cutoff,
            "train_end": max(r["target"] for r in train),
        }
        restoration = {}
        for name, rows in (("train", train), ("exam", exam)):
            result = predict_rows(model, rows)
            library = classifier.predict_proba(scaler.transform([vector(r, config) for r in rows]))
            restored = np.asarray([[r["class_scores"][k] for k in CLASSES] for r in result])
            diff = float(np.max(np.abs(library - restored)))
            directions = [max(TIE_ORDER, key=lambda k, p=p: p[list(CLASSES).index(k)]) for p in library]
            if diff > 1e-12 or directions != [r["direction"] for r in result]:
                raise ValueError("OPT_RESTORE_MISMATCH")
            restoration[name] = {"count": len(rows), "max_score_difference": diff, "directions_equal": True}
    return {"model": model, "restore": restoration, "iterations": classifier.n_iter_.tolist()}


def compare(predictions):
    """统一日期和行身份，完整返回三类/常数/正确日；无某类别时为 null。"""
    if not predictions:
        return {"models": {}, "constants": {}, "dates": []}
    first = next(iter(predictions.values()))
    identity = [(r["target"], r["actual_direction"], r["input_hash"]) for r in first]
    if len(identity) != len({r[0] for r in identity}):
        raise ValueError("OPT_DUPLICATE_EXAM_DATE")
    if any(
        [(r["target"], r["actual_direction"], r["input_hash"]) for r in rows] != identity
        for rows in predictions.values()
    ):
        raise ValueError("OPT_COMPARISON_DIFFERENT_ROWS")
    quarters = sorted({r["target"][:4] + "Q" + str((int(r["target"][5:7]) - 1) // 3 + 1) for r in first})
    # 同时保留逐日方向与配对得失，避免总正确数掩盖“新增对了几天、原来对的又丢了几天”。
    differences = {}
    for baseline in ("A7", "C20"):
        if baseline not in predictions:
            continue
        for candidate in CANDIDATES:
            if candidate not in predictions:
                continue
            daily = []
            for old_row, new_row in zip(predictions[baseline], predictions[candidate], strict=True):
                old_correct = old_row["direction"] == old_row["actual_direction"]
                new_correct = new_row["direction"] == new_row["actual_direction"]
                outcome = (
                    "both_correct"
                    if old_correct and new_correct
                    else "baseline_only"
                    if old_correct
                    else "candidate_only"
                    if new_correct
                    else "both_wrong"
                )
                daily.append({"target": old_row["target"], "outcome": outcome})
            differences[f"{candidate}_vs_{baseline}"] = {
                "counts": {
                    k: sum(r["outcome"] == k for r in daily)
                    for k in ("both_correct", "baseline_only", "candidate_only", "both_wrong")
                },
                "daily": daily,
            }
    return {
        "models": {key: old.metrics(rows) for key, rows in predictions.items()},
        "constants": {k: old.metrics([{**r, "direction": k} for r in first]) for k in CLASSES},
        "dates": [r[0] for r in identity],
        "quarters": {
            quarter: {
                key: old.metrics(
                    [r for r in rows if r["target"][:4] + "Q" + str((int(r["target"][5:7]) - 1) // 3 + 1) == quarter]
                )
                for key, rows in predictions.items()
            }
            for quarter in quarters
        },
        "differences": differences,
        "daily": [
            {
                "target": r["target"],
                "actual_direction": r["actual_direction"],
                "directions": {key: rows[i]["direction"] for key, rows in predictions.items()},
            }
            for i, r in enumerate(first)
        ],
    }


def select_historical(overall, folds):
    """只读取历史分段结果；2024 结果不作为本函数参数，从接口隔离事后改选。"""
    scores = overall["models"]
    decisions = {}
    complete = set(scores) == set(CONFIGS) and len(folds) == 3 and all(set(f["models"]) == set(CONFIGS) for f in folds)
    for candidate in CANDIDATES:
        checks = {"all_stages_complete": complete}
        if complete:
            own, a, c = scores[candidate], scores["A7"], scores["C20"]
            checks.update(
                {
                    "correct_above_baselines": own["correct"]
                    > max(a["correct"], c["correct"], *(v["correct"] for v in overall["constants"].values())),
                    "all_classes_at_least_baselines": all(
                        own["class_correct"][k] >= max(a["class_correct"][k], c["class_correct"][k]) for k in CLASSES
                    ),
                    "at_least_two_quarters": sum(
                        f["models"][candidate]["correct"] >= f["models"]["C20"]["correct"] for f in folds
                    )
                    >= 2,
                }
            )
        decisions[candidate] = {"passed": all(checks.values()), "checks": checks}
    passed = [k for k, v in decisions.items() if v["passed"]]
    passed.sort(key=lambda k: (-scores[k]["correct"], CONFIGS[k]))
    return {"selected": passed[0] if passed else None, "candidates": decisions, "2024_used_for_selection": False}


def final_gate(decision, development):
    chosen = decision["selected"]
    checks = {"historical_selected": chosen is not None}
    if chosen and chosen in development["models"]:
        scores = development["models"]
        own, a, c = scores[chosen], scores["A7"], scores["C20"]
        checks.update(
            {
                "correct_above_baselines": own["correct"]
                > max(a["correct"], c["correct"], *(v["correct"] for v in development["constants"].values())),
                "all_classes_at_least_A": all(own["class_correct"][k] >= a["class_correct"][k] for k in CLASSES),
            }
        )
    else:
        checks["selected_model_complete"] = False
    return {
        "selected": chosen if all(checks.values()) else None,
        "checks": checks,
        "fallback_selection_allowed": False,
        "adopted": False,
    }
