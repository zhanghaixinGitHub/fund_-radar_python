"""离线统一模型对照；不注册模型、不联网补数、不改真实预测或原训练包。"""

import hashlib
import warnings
from collections import defaultdict
from datetime import date, datetime, time
from pathlib import Path

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from app.services import direction_1d_training as original
from app.services.direction_1d_protocol import FEATURES, RECIPE, ZONE, digest, input_days

CANDIDATES = ("POOLED_7", "POOLED_7_TYPE")
BASELINE = "GROUPED_7"
BASE_FILES = (
    "protocol.json",
    "history.json",
    "dataset.json",
    "dataset-manifest.json",
    "models.json",
    "metrics.json",
    "predictions.json",
    "completion.json",
)
QUARTERS = ("2024Q1", "2024Q2", "2024Q3", "2024Q4")
SIMPLE = ("ALWAYS_UP", "ALWAYS_NON_UP", "INITIAL_GROUP_MAJORITY", "MOMENTUM")
read, write_new = original.read, original.write_new


def fingerprint() -> dict:
    """训练实现与依赖必须匹配冻结版本，代码漂移时使用代码快照恢复。"""
    result = original.fingerprint()
    result["comparison_code"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return result


def vector(row: dict, candidate: str) -> list[float]:
    """类型提示只调整同一模型的输入，不分别训练各组斜率。"""
    if candidate not in CANDIDATES or row["group"] not in {"CN_EQUITY", "CN_MIXED", "CN_BOND"}:
        raise ValueError("UNKNOWN_CANDIDATE_OR_GROUP")
    x = [float(v) for v in row["x"]]
    if len(x) != 7 or not np.isfinite(x).all():
        raise ValueError("INVALID_FEATURES")
    return x + (
        [float(row["group"] == "CN_BOND"), float(row["group"] == "CN_MIXED")] if candidate.endswith("TYPE") else []
    )


def availability(rows: list[dict], history: dict) -> dict:
    """逐条检查61日输入公告；此历史时刻是假设，不是过去已收取的证据。"""
    lookup = {f["fund_code"]: {r["date"]: r for r in f["rows"]} for f in history["funds"]}
    result = {}
    for row in rows:
        if row["kind"] != "HISTORICAL_RECONSTRUCTION" or row["u"] > "2024-12-31":
            raise ValueError("PROTECTED_OR_FORWARD_DATA_NOT_ALLOWED")
        days = [str(d) for d in input_days(date.fromisoformat(row["t"]))]
        latest = max(lookup[row["fund_code"]][d]["ann_date"] or d for d in days)
        result[row["fund_code"], row["t"]] = datetime.combine(date.fromisoformat(latest), time(8), ZONE)
    return result


def exact_fit(rows: list[dict], models: dict, quarter: str, available: dict) -> list[dict]:
    """严格复用旧各组FIT并集，避免比较时训练日期或基金范围暗中变化。"""
    selected = set()
    for group in sorted({r["group"] for r in rows}):
        model = models[group + "-" + quarter]
        cutoff = datetime.fromisoformat(model["train_as_of"])
        fit_rows = original.select_fit([r for r in rows if r["group"] == group], cutoff)
        if digest(fit_rows) != model["fit_hash"]:
            raise ValueError("BASELINE_FIT_MISMATCH")
        if any(available[r["fund_code"], r["t"]] > cutoff for r in fit_rows):
            raise ValueError("FIT_INPUT_NOT_AVAILABLE")
        selected.update((r["fund_code"], r["t"]) for r in fit_rows)
    return [r for r in rows if (r["fund_code"], r["t"]) in selected]


def common_exam(rows: list[dict], predictions: list[dict], models: dict, available: dict) -> tuple[list[dict], int]:
    by_key = {(r["fund_code"], r["t"]): r for r in rows}
    result, seen = [], set()
    for p in predictions:
        key = (p["fund_code"], p["t"])
        if key in seen:
            raise ValueError("DUPLICATE_BASELINE_QUESTION")
        seen.add(key)
        row = by_key[key]
        if row["y"] != p["y"] or row["u"] != p["u"]:
            raise ValueError("BASELINE_ANSWER_MISMATCH")
        if abs(original.score(models[p["job"]], row["x"]) - p["score"]) > 1e-12:
            raise ValueError("BASELINE_SCORE_MISMATCH")
        cutoff = datetime.combine(date.fromisoformat(row["u"]), time(8), ZONE)
        if available[key] <= cutoff:
            result.append({**row, "quarter": p["job"].rsplit("-", 1)[1], "baseline_score": p["score"]})
    return sorted(result, key=lambda r: (r["t"], r["family"], r["fund_code"])), len(predictions) - len(result)


def freeze(base: Path, output: Path) -> dict:
    original.verify(base)
    rows, models = read(base / "dataset.json"), read(base / "models.json")
    if digest(rows) != read(base / "dataset-manifest.json")["hash"]:
        raise ValueError("DATASET_CHANGED")
    available = availability(rows, read(base / "history.json"))
    exam, excluded = common_exam(rows, read(base / "predictions.json"), models, available)
    fits = {q: exact_fit(rows, models, q, available) for q in (*QUARTERS, "FINAL")}
    output.mkdir(parents=True, exist_ok=False)
    (output / "baseline").mkdir()
    for name in BASE_FILES:
        write_new(output / "baseline" / name, read(base / name))
    spec = {
        "kind": "HISTORICAL_COMPARISON_ONLY",
        "created_at": datetime.now(ZONE).isoformat(),
        "candidates": CANDIDATES,
        "recipe": RECIPE,
        "fingerprint": fingerprint(),
        "max_main_fits": 10,
        "max_replay_fits": 10,
        "threshold": 0.5,
        "base_hashes": {name: digest(read(base / name)) for name in BASE_FILES},
        "common_exam_hash": digest(exam),
        "exam_count": len(exam),
        "excluded_exam_count": excluded,
        "fit_hashes": {q: digest(v) for q, v in fits.items()},
        "fit_counts": {q: len(v) for q, v in fits.items()},
        "primary_metric": "FAMILY_DATE_WEIGHTED_ACCURACY",
        "bootstrap_seed": 20260912,
        "bootstrap_repetitions": 2000,
        "bootstrap_block_days": 5,
        "availability_assumption": "ANN_DATE_AT_08_00_NOT_FIRST_RECEIPT_PROOF",
        "protected_years_excluded": [2025, 2026],
        "model_released": False,
        "cohort_id": "D1-U-" + digest({"dataset": digest(rows), "candidates": CANDIDATES})[:24],
    }
    write_new(output / "study.json", spec)
    write_new(output / "common-exam.json", exam)
    return spec


def verify_inputs(root: Path) -> dict:
    spec = read(root / "study.json")
    if spec["fingerprint"] != fingerprint() or spec["candidates"] != list(CANDIDATES) or spec["recipe"] != RECIPE:
        raise ValueError("COMPARISON_CODE_OR_RECIPE_CHANGED")
    for name, expected in spec["base_hashes"].items():
        if digest(read(root / "baseline" / name)) != expected:
            raise ValueError("COMPARISON_BASE_CHANGED")
    if digest(read(root / "common-exam.json")) != spec["common_exam_hash"]:
        raise ValueError("COMPARISON_EXAM_CHANGED")
    return spec


def predict(model: dict, rows: list[dict]) -> np.ndarray:
    x = np.asarray([vector(r, model["candidate"]) for r in rows])
    z = ((x - np.asarray(model["mean"])) / np.asarray(model["scale"])) @ np.asarray(model["coef"]) + model["intercept"]
    return 1 / (1 + np.exp(-np.clip(z, -700, 700)))


def fit(rows: list[dict], candidate: str, cohort: str, cutoff: str) -> dict:
    if len({r["t"] for r in rows}) < 252 or min(sum(r["y"] == y for r in rows) for y in (0, 1)) < 30:
        raise ValueError("INSUFFICIENT_SAMPLES")
    x, y, weight = (
        np.asarray([vector(r, candidate) for r in rows]),
        np.asarray([r["y"] for r in rows]),
        original.weights(rows),
    )
    with threadpool_limits(limits=1), warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        scaler = StandardScaler().fit(x, sample_weight=weight)
        estimator = LogisticRegression(**RECIPE).fit(scaler.transform(x), y, sample_weight=weight)
    model = {
        "candidate": candidate,
        "cohort_id": cohort,
        "target_definition": "UNIT_NAV_DIRECTION_V1",
        "kind": "RESEARCH_ONLY",
        "model_released": False,
        "train_as_of": cutoff,
        "features": list(FEATURES) + (["is_bond", "is_mixed"] if candidate.endswith("TYPE") else []),
        "recipe": RECIPE,
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
        "coef": estimator.coef_[0].tolist(),
        "intercept": float(estimator.intercept_[0]),
        "fit_hash": digest(rows),
        "weight_hash": digest(weight.tolist()),
        "fit_count": len(rows),
        "distinct_dates": len({r["t"] for r in rows}),
        "weight_total": float(weight.sum()),
        "classes": {str(y): sum(r["y"] == y for r in rows) for y in (0, 1)},
    }
    difference = float(np.max(np.abs(predict(model, rows) - estimator.predict_proba(scaler.transform(x))[:, 1])))
    if difference > 1e-12:
        raise ValueError("MODEL_RESTORE_MISMATCH")
    model["restore_max_score_diff"] = difference
    return model


def run(root: Path, *, replay: bool = False) -> dict:
    spec = verify_inputs(root)
    if replay:
        verify(root)
    out = root / ("replay" if replay else "main")
    out.mkdir(exist_ok=False)
    write_new(out / "fit-budget-reserved.json", {"at": datetime.now(ZONE).isoformat(), "max_fits": 10})
    rows, old_models = read(root / "baseline/dataset.json"), read(root / "baseline/models.json")
    available = availability(rows, read(root / "baseline/history.json"))
    exam, models = read(root / "common-exam.json"), {}
    for quarter in (*QUARTERS, "FINAL"):
        selected = exact_fit(rows, old_models, quarter, available)
        if digest(selected) != spec["fit_hashes"][quarter]:
            raise ValueError("FIT_CHANGED")
        cutoff = spec["created_at"] if quarter == "FINAL" else old_models["CN_EQUITY-" + quarter]["train_as_of"]
        for candidate in CANDIDATES:
            key = candidate + "-" + quarter
            # 拟合前留凭证；失败原样保留，不悄悄再跑一遍凑出较好成绩。
            write_new(
                out / (key + "-reserved.json"), {"at": datetime.now(ZONE).isoformat(), "fit_hash": digest(selected)}
            )
            model = fit(selected, candidate, spec["cohort_id"], cutoff)
            write_new(out / (key + ".json"), model)
            models[key] = model
    predictions = []
    for r in exam:
        initial_majority = old_models[r["group"] + "-2024Q1"]["majority"]
        scores = {BASELINE: r["baseline_score"]}
        scores.update({c: float(predict(models[c + "-" + r["quarter"]], [r])[0]) for c in CANDIDATES})
        predictions.append(
            {
                **{k: r[k] for k in ("fund_code", "group", "family", "t", "u", "y", "actual_direction", "quarter")},
                "kind": "HISTORICAL_RECONSTRUCTION",
                "scores": scores,
                "directions": {
                    **{c: int(s > 0.5) for c, s in scores.items()},
                    "ALWAYS_UP": 1,
                    "ALWAYS_NON_UP": 0,
                    "INITIAL_GROUP_MAJORITY": initial_majority,
                    "MOMENTUM": r["momentum"],
                },
            }
        )
    write_new(out / "models.json", models)
    write_new(out / "predictions.json", predictions)
    result = {
        "successful_fits": len(models),
        "finished_at": datetime.now(ZONE).isoformat(),
        "models_hash": digest(models),
        "predictions_hash": digest(predictions),
        "model_released": False,
    }
    write_new(out / "completion.json", result)
    if replay:
        previous = read(root / "main/predictions.json")
        same_models = read(root / "main/models.json") == models
        difference = max(
            abs(a["scores"][c] - b["scores"][c]) for a, b in zip(previous, predictions, strict=True) for c in CANDIDATES
        )
        if not same_models or previous != predictions:
            raise ValueError("REPLAY_MISMATCH")
        write_new(
            root / "replay-proof.json",
            {
                "models_equal": same_models,
                "max_score_diff": difference,
                "prediction_count": len(predictions),
                "new_fit_count": len(models),
            },
        )
    return result


def verify(root: Path) -> dict:
    verify_inputs(root)
    result = read(root / "main/completion.json")
    for mode in ("main", "replay"):
        if not (root / mode / "completion.json").exists():
            continue
        proof = read(root / mode / "completion.json")
        for name in ("models", "predictions"):
            if digest(read(root / mode / (name + ".json"))) != proof[name + "_hash"]:
                raise ValueError("COMPARISON_ARTIFACT_CHANGED")
        if proof["successful_fits"] != 10 or len(list((root / mode).glob("*-reserved.json"))) != 11:
            raise ValueError("FIT_BUDGET_MISMATCH")
    return result


def metrics(rows: list[dict], branch: str) -> dict:
    y = np.asarray([r["y"] for r in rows])
    predicted = np.asarray([r["directions"][branch] for r in rows])
    weight = original.weights(rows)
    correct = predicted == y
    recalls = [float(np.mean(correct[y == cls])) if np.any(y == cls) else None for cls in (1, 0)]
    return {
        "count": len(rows),
        "correct": int(correct.sum()),
        "accuracy": float(correct.mean()),
        "weighted_accuracy": float(np.average(correct, weights=weight)),
        "up_recall": recalls[0],
        "non_up_recall": recalls[1],
        "balanced_accuracy": sum(recalls) / 2 if None not in recalls else None,
        "predicted_up_count": int(predicted.sum()),
        "flat_count": sum(r["actual_direction"] == "FLAT" for r in rows),
        "target_dates": len({r["u"] for r in rows}),
        "families": len({r["family"] for r in rows}),
    }


def paired(rows: list[dict], branch: str, spec: dict) -> dict:
    """同一目标日全部基金一起重采样；固定5日块，避免把29只基金当29份独立市场。"""
    weight = original.weights(rows)
    by_day = defaultdict(lambda: [0.0, 0.0])
    wins, losses = 0, 0
    for r, w in zip(rows, weight, strict=True):
        a = int(r["directions"][branch] == r["y"])
        b = int(r["directions"][BASELINE] == r["y"])
        by_day[r["u"]][0] += w * (a - b)
        by_day[r["u"]][1] += w
        wins += int(a > b)
        losses += int(a < b)
    totals = np.asarray([by_day[day] for day in sorted(by_day)])
    rng = np.random.default_rng(spec["bootstrap_seed"])
    n, block = len(totals), spec["bootstrap_block_days"]
    estimates = []
    for _ in range(spec["bootstrap_repetitions"]):
        starts = rng.integers(0, n, size=(n + block - 1) // block)
        indices = ((starts[:, None] + np.arange(block)) % n).ravel()[:n]
        sampled = totals[indices].sum(axis=0)
        estimates.append(sampled[0] / sampled[1])
    return {
        "extra_correct": wins - losses,
        "candidate_only_correct": wins,
        "baseline_only_correct": losses,
        "weighted_accuracy_difference": float(totals[:, 0].sum() / totals[:, 1].sum()),
        "block_bootstrap_interval_95": np.quantile(estimates, [0.025, 0.975]).tolist(),
        "kind": "EXPLORATORY_DEVELOPMENT_INTERVAL_NOT_INDEPENDENT_TEST",
    }


def evaluate(root: Path) -> dict:
    verify(root)
    spec, proof = read(root / "study.json"), read(root / "replay-proof.json")
    if not proof["models_equal"] or proof["max_score_diff"] > 1e-12:
        raise ValueError("REPLAY_REQUIRED")
    rows = read(root / "main/predictions.json")
    branches = (BASELINE, *CANDIDATES, *SIMPLE)
    overall = {b: metrics(rows, b) for b in branches}
    strata = {}
    for field in ("quarter", "group", "fund_code"):
        strata[field] = {
            value: {b: metrics([r for r in rows if r[field] == value], b) for b in branches}
            for value in sorted({r[field] for r in rows})
        }
    best = max((BASELINE, *CANDIDATES), key=lambda b: overall[b]["weighted_accuracy"])
    result = {
        "kind": "DEVELOPMENT_COMPARISON",
        "overall": overall,
        "strata": strata,
        "paired": {c: paired(rows, c, spec) for c in CANDIDATES},
        "observed_best": best,
        "model_released": False,
        "runtime_model_changed": False,
        "limitations": [
            "2021—2024已参与过开发，不是独立测试。",
            "日期公告时刻与历史类型采用假设。",
            "两种新候选共享历史问题，区间仅作探索，不证明未来提升。",
            "盘中行情/新闻不在现有授权范围，未启用盘中模型。",
        ],
    }
    write_new(root / "comparison.json", result)
    write_new(root / "comparison-receipt.json", {"hash": digest(result), "created_at": datetime.now(ZONE).isoformat()})
    return result
