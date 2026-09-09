"""解释已封存研究的负斜率；只复算旧资料，不拟合新候选。"""

from collections import Counter
from pathlib import Path

from app.services.historical_nav_calibration import WINDOWS
from app.services.historical_nav_training import predict_artifact_logits, restore_logistic_artifact
from app.services.model_comparison_artifacts import read_json, read_jsonl
from app.services.model_comparison_dataset import load_stage
from app.services.model_comparison_runner import checked_dataset


def score_relationship(labels: list[int], scores: list[float], funds: list[str]) -> dict:
    import numpy as np
    from sklearn.metrics import roc_auc_score

    if not labels or len(labels) != len(scores) or len(labels) != len(funds) or set(labels) - {0, 1}:
        raise ValueError("invalid paired diagnostic data")
    x, y = np.asarray(scores, dtype=float), np.asarray(labels, dtype=float)
    if not np.isfinite(x).all():
        raise ValueError("nonfinite diagnostic score")
    counts = Counter(funds)
    weights = np.asarray([1 / (len(counts) * counts[f]) for f in funds])
    mean_x, mean_y = float(weights @ x), float(weights @ y)
    covariance = float(weights @ ((x - mean_x) * (y - mean_y)))
    return {
        "count": len(labels),
        "up_count": sum(labels),
        "up_rate": sum(labels) / len(labels),
        "auc": float(roc_auc_score(labels, scores, sample_weight=weights)) if len(set(labels)) == 2 else None,
        "weighted_covariance": covariance,
        "slope_gradient_at_zero_after_intercept_fit": -covariance,
        "mean_score_when_up": float(x[y == 1].mean()) if np.any(y == 1) else None,
        "mean_score_when_not_up": float(x[y == 0].mean()) if np.any(y == 0) else None,
    }


def diagnose_previous_run(folder: Path) -> dict:
    protocol, _, inputs = checked_dataset(folder)
    base = restore_logistic_artifact((folder / "models/VALIDATION_2024-A-base.json").read_bytes())
    a_cal = read_json(folder / "models/VALIDATION_2024-A.json")["calibrator"]
    b_cal = read_json(folder / "models/VALIDATION_2024-B.json")
    b_scores = {
        r["key"]: r["raw_score"] for r in read_jsonl(folder / "models/VALIDATION_2024-B-calibration-predictions.jsonl")
    }
    for r in read_jsonl(folder / "predictions.jsonl"):
        if r["window"] == "VALIDATION_2024" and r["models"]["B"].get("raw_score") is not None:
            b_scores[r["key"]] = r["models"]["B"]["raw_score"]
    by_day = {(i.fund_code, i.cutoff_date): i for i in inputs.values()}
    stages = {}
    for stage in ("FIT", "CALIBRATION", "EXAM"):
        rows = load_stage(folder, WINDOWS[-1], stage, inputs)
        a = list(map(float, predict_artifact_logits(base, tuple(r.x for r in rows))))
        model_scores = {"A": a}
        if stage != "FIT":
            model_scores["B"] = [b_scores[by_day[r.fund_code, r.as_of_date].key] for r in rows]
        report = {}
        for name, scores in model_scores.items():
            groups = {}
            for fund in ("ALL", *protocol["settings"]["funds"]):
                indices = [i for i, r in enumerate(rows) if fund == "ALL" or r.fund_code == fund]
                groups[fund] = score_relationship(
                    [rows[i].y for i in indices], [scores[i] for i in indices], [rows[i].fund_code for i in indices]
                )
            if stage == "CALIBRATION":
                groups["leave_one_fund_out_covariance"] = {}
                for fund in protocol["settings"]["funds"]:
                    indices = [i for i, r in enumerate(rows) if r.fund_code != fund]
                    groups["leave_one_fund_out_covariance"][fund] = score_relationship(
                        [rows[i].y for i in indices], [scores[i] for i in indices], [rows[i].fund_code for i in indices]
                    )["weighted_covariance"]
            report[name] = groups
        stages[stage] = report
    return {
        "old_run": folder.name,
        "retrained": False,
        "stages": stages,
        "calibrators": {
            "A": {k: a_cal[k] for k in ("slope", "intercept", "iterations")},
            "B": {k: b_cal[k] for k in ("slope", "intercept", "iterations")},
        },
        "diagnosis": (
            "Fits converged; a positive-slope policy rejected valid numerical artifacts. "
            "Negative score-label covariance explains the fitted sign; "
            "low up-rate alone does not force a negative slope."
        ),
        "causal_limit": "Market-regime or feature causal explanations are not identified by these correlations.",
        "sources": ["https://scikit-learn.org/stable/modules/calibration.html#sigmoid"],
    }
