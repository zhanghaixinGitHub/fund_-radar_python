"""第十三轮：学习SPX基线的错误，在有足够纠错分数时才改变原方向。

误差目标仍只来自截止点前已成熟的真实标签；保持自然错题比例，不把错题强行平衡到一半。
所有基金题目继续作答，阈值0.55提前固定，不按开发成绩扫描阈值或删除难题。
"""

import hashlib
import shutil
import warnings
from collections import Counter, defaultdict
from datetime import date, datetime, time

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

from app.services import direction_1d_sprint as base
from app.services import direction_1d_sprint_adaptive as adaptive
from app.services import direction_1d_sprint_dual_us as dual
from app.services import direction_1d_sprint_fund_response as previous
from app.services import direction_1d_sprint_overnight as overnight
from app.services import direction_1d_sprint_return_target as regression
from app.services import direction_1d_sprint_sequence as previous_round
from app.services import direction_1d_sprint_sparse as sparse

CANDIDATES = ("LR8_ERR504", "TREE8_ERR504", "EXTRA8_ERR504")
LEARNED = CANDIDATES
FIRST_TARGET = "2026-09-16"
FLIP_THRESHOLD = 0.55


def root():
    return base.ROOT / "round-13"


def active():
    regression.active()


def fingerprint():
    value = previous_round.fingerprint()
    for name in (
        "app/services/direction_1d_sprint_correction.py",
        "scripts/direction_1d_sprint_correction.py",
        "tests/test_direction_1d_sprint_correction.py",
    ):
        value["code"][name] = hashlib.sha256((base.PROJECT / name).read_bytes()).hexdigest()
    return value


def vector(x):
    """基准日及以前的基金收益与国内指数，加SPX实际隔夜收益，单位在此固定。"""
    if len(x) != 32 or not np.isfinite(x).all() or x[3] < 0:
        raise ValueError("CORRECTION_INPUT_INVALID")
    vol = max(float(x[3]), 0.0001)
    return [
        float(x[30]) * 100,
        abs(float(x[30])) * 100,
        float(x[31]),
        float(np.clip(x[7] / vol, -5, 5)),
        float(np.clip(x[0] / (vol * np.sqrt(5)), -5, 5)),
        float(np.clip(x[1] / (vol * np.sqrt(20)), -5, 5)),
        float(x[18]) * 100,
        float(x[23]) * 100,
    ]


def dataset():
    rows = [r | {"z": vector(r["x"])} for r in overnight.dataset()]
    return rows, {"at": base.now().isoformat(), "rows": len(rows), "target": "SPX rule wrong iff (SPX>=0) != actual UP"}


def plan():
    path = root() / "plan.json"
    if path.exists():
        value = base.read(path)
        if value["fingerprint"] != fingerprint():
            raise ValueError("ROUND_13_CODE_CHANGED")
        return value
    active()
    value = {
        "at": base.now().isoformat(),
        "fingerprint": fingerprint(),
        "candidates": CANDIDATES,
        "hypothesis": "Predict SPX baseline errors and selectively correct them instead of replacing the entire rule.",
        "motivation": "round-12/spx-error-analysis.json, exploratory2025only; not independent validation",
        "features": [
            "SPX_percent",
            "absSPX_percent",
            "US_sessions",
            "NAV_ret1_over_vol20",
            "NAV_ret5_over_sqrt5_vol20",
            "NAV_ret20_over_sqrt20_vol20",
            "CSI300_ret1_percent",
            "CSI500_ret1_percent",
        ],
        "target": "1 iff SPX>=0 direction differs from actual next NAV UP; 0 otherwise",
        "training_dates": 504,
        "development_year": 2025,
        "quarterly_refit": True,
        "max_development_fits": 36,
        "max_current_fits": 9,
        "max_reproduction_count": 1,
        "weights": "date/family equal, natural error prior, no class rebalancing",
        "flip_threshold": FLIP_THRESHOLD,
        "threshold_search": False,
        "uncalibrated_score": True,
        "logistic_C": 0.1,
        "tree": {
            "iterations": 120,
            "leaves": 7,
            "min_leaf": 80,
            "learning_rate": 0.03,
            "l2": 10,
            "early_stopping": False,
            "seed": 17,
        },
        "extra_trees": {"trees": 256, "depth": 6, "min_leaf": 80, "max_features": 1.0, "bootstrap": False, "seed": 17},
        "controls": ["SPX_SIGN", "LR2_BAL252", "ALWAYS_UP"],
        "selection": "highest common5670 date-family accuracy among3; no retuning",
        "input_hashes": {
            n: base.digest(base.read(base.ROOT / n))
            for n in ("history.json", "round-02/market.json", "round-03/spx.json")
        },
        "first_forward_target": FIRST_TARGET,
        "current_fit_cutoff": str(base.now().date()),
        "new_provider_calls": 0,
        "new_cost_cny": 0,
        "this_round_2026_scores_read": False,
        "historical_availability": "RECONSTRUCTED_NOT_TRUE_FORWARD",
    }
    base.save(path, value)
    for name in value["fingerprint"]["code"]:
        dest = root() / "code" / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(base.PROJECT / name, dest)
    return value


def fit(rows, name, cutoff):
    """基线预测可以由当日输入重建，但纠错标签必须在训练截止前已成熟。"""
    if name not in CANDIDATES:
        raise ValueError("CORRECTION_UNKNOWN_RECIPE")
    chosen = adaptive.training_rows(rows, cutoff, "MONTHLY_BAL504")
    x = np.asarray([r["z"] for r in chosen])
    y = np.asarray([int(int(r["z"][0] >= 0) != r["y"]) for r in chosen])
    if len(Counter(y)) != 2 or min(Counter(y).values()) < 20:
        raise ValueError("CORRECTION_ERROR_LABELS_INSUFFICIENT")
    weights = regression.weights(chosen)
    x, mean, scale = previous_round.normalize_training(x, weights)
    active()
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        if name == "LR8_ERR504":
            model = LogisticRegression(C=0.1, max_iter=1500, random_state=17)
        elif name == "TREE8_ERR504":
            model = HistGradientBoostingClassifier(
                max_iter=120,
                max_leaf_nodes=7,
                min_samples_leaf=80,
                learning_rate=0.03,
                l2_regularization=10,
                early_stopping=False,
                random_state=17,
            )
        else:
            model = ExtraTreesClassifier(
                n_estimators=256,
                max_depth=6,
                min_samples_leaf=80,
                max_features=1.0,
                bootstrap=False,
                random_state=17,
                n_jobs=2,
            )
        model.fit(x, y, sample_weight=weights)
    active()
    return {
        "model": model,
        "mean": mean,
        "scale": scale,
        "error_target_counts": {str(k): int(v) for k, v in Counter(y).items()},
        "natural_error_rate": float(np.average(y, weights=weights)),
        "fit_hash": base.digest(chosen),
        "fit_rows": len(chosen),
        "fit_dates": len({r["u"] for r in chosen}),
        "fit_end": max(r["u"] for r in chosen),
        "max_mature_date": max(r["mature"] for r in chosen),
        "cutoff": cutoff,
    }


def answer(z, name, trained=None):
    if name not in CANDIDATES or len(z) != 8 or not np.isfinite(z).all():
        raise ValueError("CORRECTION_ANSWER_INPUT_INVALID")
    x = (np.asarray(z) - trained["mean"]) / trained["scale"]
    if not np.isfinite(x).all():
        raise ValueError("CORRECTION_NORMALIZED_INPUT_INVALID")
    score = float(trained["model"].predict_proba([x])[0, 1])
    if not np.isfinite(score) or not 0 <= score <= 1:
        raise ValueError("CORRECTION_SCORE_INVALID")
    baseline = int(z[0] >= 0)
    flipped = score > FLIP_THRESHOLD
    return {
        "prediction": 1 - baseline if flipped else baseline,
        "baseline_prediction": baseline,
        "flipped": flipped,
        "research_score": score,
        "kind": "UNCALIBRATED_BASELINE_ERROR_SCORE",
    }


def fit_checkpoint(rows, name, cutoff, label):
    """每个预定模型只启动一次；完成模型可断点复用，中断的拟合明确报错而非偷偷重跑。"""
    path = root() / "checkpoints" / f"{label}.joblib"
    receipt = path.with_suffix(".json")
    attempt = path.with_suffix(".attempt.json")
    if receipt.exists():
        value = base.read(receipt)
        if (
            value["name"] != name
            or value["cutoff"] != cutoff
            or hashlib.sha256(path.read_bytes()).hexdigest() != value["sha256"]
        ):
            raise ValueError("CORRECTION_CHECKPOINT_CHANGED")
        return joblib.load(path)
    if attempt.exists():
        raise ValueError("CORRECTION_PREVIOUS_FIT_INTERRUPTED")
    base.save(attempt, {"at": base.now().isoformat(), "name": name, "cutoff": cutoff})
    trained = fit(rows, name, cutoff)
    joblib.dump(trained, path)
    base.save(
        receipt,
        {
            "at": base.now().isoformat(),
            "name": name,
            "cutoff": cutoff,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        },
    )
    return trained


def train():
    p = plan()
    if (root() / "result.json").exists():
        return models()[0]
    active()
    for name, expected in p["input_hashes"].items():
        if base.digest(base.read(base.ROOT / name)) != expected:
            raise ValueError("ROUND_13_INPUT_CHANGED")
    rows, proofs = dataset()
    base.save(root() / "training-question-proof.json", proofs)
    groups = sorted({r["group"] for r in rows})
    output = defaultdict(list)
    with threadpool_limits(limits=2):
        for q in range(1, 5):
            active()
            start, end = f"2025-{q * 3 - 2:02d}-01", "2026-01-01" if q == 4 else f"2025-{q * 3 + 1:02d}-01"
            for group in groups:
                exam = [r for r in rows if r["group"] == group and start <= r["u"] < end]
                for name in CANDIDATES:
                    path = root() / f"folds/{q}-{group}-{name}.json"
                    if path.exists():
                        scored = base.read(path)
                    else:
                        trained = (
                            fit_checkpoint([r for r in rows if r["group"] == group], name, start, f"{q}-{group}-{name}")
                            if name in LEARNED
                            else None
                        )
                        scored = [
                            {k: r[k] for k in ("code", "family", "group", "u", "y", "actual_direction")}
                            | answer(r["z"], name, trained)
                            for r in exam
                        ]
                        base.save(path, scored)
                        if trained:
                            base.save(
                                root() / f"training/{q}-{group}-{name}.json",
                                {k: v for k, v in trained.items() if k != "model"},
                            )
                    output[name].extend(scored)
                for control in ("SPX_SIGN", "LR2_BAL252"):
                    output[control].extend(base.read(sparse.root() / f"folds/{q}-{group}-{control}.json"))
            base.save(root() / "progress.json", {"at": base.now().isoformat(), "quarter": q}, replace=True)
        output["ALWAYS_UP"] = [r | {"prediction": 1} for r in output["SPX_SIGN"]]
        expected = sorted((r["code"], r["u"], r["y"]) for r in output["SPX_SIGN"])
        if len(expected) != 5670 or any(
            sorted((r["code"], r["u"], r["y"]) for r in v) != expected for v in output.values()
        ):
            raise ValueError("ROUND_13_COMMON_EXAM_CHANGED")
        metrics = {n: base.metrics(v) for n, v in output.items()}
        winner = max(CANDIDATES, key=lambda n: (metrics[n]["accuracy"], -CANDIDATES.index(n)))
        active()
        bundle = {
            n: {
                g: fit_checkpoint([r for r in rows if r["group"] == g], n, p["current_fit_cutoff"], f"current-{g}-{n}")
                for g in groups
            }
            for n in LEARNED
        }
    path = root() / "models.joblib"
    joblib.dump(bundle, path)
    value = {
        "at": base.now().isoformat(),
        "winner": winner,
        "metrics": metrics,
        "fingerprint": fingerprint(),
        "monthly_metrics": {
            n: {
                f"2025-{m:02d}": base.metrics([r for r in v if r["u"].startswith(f"2025-{m:02d}")])
                for m in range(1, 13)
            }
            for n, v in output.items()
        },
        "group_metrics": {
            n: {g: base.metrics([r for r in v if r["group"] == g]) for g in groups} for n, v in output.items()
        },
        "model_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "plan_hash": base.digest(p),
        "development_fits": 36,
        "current_fits": 9,
        "this_round_2026_scores_read": False,
        "kind": "DEVELOPMENT_ONLY_AFTER_RESERVED_AUDIT_CONSUMPTION",
        "new_cost_cny": 0,
        "first_forward_target": FIRST_TARGET,
    }
    base.save(root() / "result.json", value)
    return value


def models():
    result = base.read(root() / "result.json")
    path = root() / "models.joblib"
    if (
        result["fingerprint"] != fingerprint()
        or hashlib.sha256(path.read_bytes()).hexdigest() != result["model_sha256"]
    ):
        raise ValueError("ROUND_13_MODEL_OR_CODE_CHANGED")
    return result, joblib.load(path)


def preflight():
    _, bundle = models()
    selected = {}
    rows, _ = dataset()
    for row in reversed(rows):
        selected.setdefault(row["code"], row)
    with threadpool_limits(limits=2):
        for row in selected.values():
            for name in CANDIDATES:
                answer(row["z"], name, bundle[name][row["group"]] if name in LEARNED else None)
    value = {
        "at": base.now().isoformat(),
        "kind": "DRY_RUN_NOT_FORWARD",
        "branch_checks": len(selected) * len(CANDIDATES),
    }
    base.save(root() / "preflight.json", value)
    return value


def tick():
    at = base.now()
    end = datetime.fromisoformat(base.read(base.ROOT / "protocol.json")["deadline_at"])
    w = base.window(at)
    if at >= end or w["status"] != "OPEN" or not time(7) <= at.time() < time(8, 30):
        return report()
    target = w["target_nav_date"]
    if target < FIRST_TARGET:
        return report()
    paths = [
        p
        for p in (previous.root() / "forward" / target).glob("*.json")
        if not (root() / "forward" / target / p.name).exists()
    ]
    if not paths:
        return report()
    overnight.source()
    manifest, bundle = models()
    deadline = datetime.combine(date.fromisoformat(target), time(8, 30), base.ZONE)
    with threadpool_limits(limits=2):
        for path in paths:
            if base.now() >= deadline:
                break
            p5, _, source, original = dual.read_parent(path)
            z = vector(source["x"])
            choices = {n: answer(z, n, bundle[n][original["group"]]) for n in CANDIDATES}
            value = {
                "at": base.now().isoformat(),
                "u": target,
                "code": original["code"],
                "parent_hash": base.digest(p5),
                "source_hash": base.digest(source),
                "original_hash": base.digest(original),
                "model_hash": manifest["model_sha256"],
                "answers": choices,
                "z": z,
                "status": "MODEL_NOT_RELEASED",
            }
            saved = root() / "forward" / target / path.name
            base.save(saved, value)
            readback = base.now()
            verified = base.read(saved) == value and readback < deadline
            base.save(
                root() / "receipts" / target / path.name,
                {
                    "readback_at": readback.isoformat(),
                    "forecast_hash": base.digest(value),
                    "status": "VERIFIED" if verified else "LATE_OR_INVALID",
                },
            )
    return report()


def report():
    if not (root() / "result.json").exists():
        return {"phase": "NOT_TRAINED"}
    original_report = base.report()
    result = base.read(root() / "result.json")
    paired, good, late, pending, closed = defaultdict(list), 0, 0, 0, 0
    for path in (root() / "forward").glob("*/*.json"):
        value = base.read(path)
        receipt_path = root() / "receipts" / value["u"] / path.name
        receipt = base.read(receipt_path) if receipt_path.exists() else {}
        deadline = datetime.combine(date.fromisoformat(value["u"]), time(8, 30), base.ZONE)
        if (
            value["u"] < FIRST_TARGET
            or receipt.get("status") != "VERIFIED"
            or receipt.get("forecast_hash") != base.digest(value)
            or datetime.fromisoformat(receipt["readback_at"]) >= deadline
            or datetime.fromisoformat(value["at"]) >= deadline
        ):
            late += 1
            continue
        p5, p4, source, original = dual.read_parent(previous.root() / "forward" / value["u"] / path.name)
        if (
            any(
                value[k] != base.digest(v)
                for k, v in (("parent_hash", p5), ("source_hash", source), ("original_hash", original))
            )
            or value["model_hash"] != result["model_sha256"]
        ):
            raise ValueError("ROUND_13_PARENT_OR_MODEL_CHANGED")
        if not np.allclose(value["z"], vector(source["x"]), rtol=0, atol=1e-12):
            raise ValueError("ROUND_13_VECTOR_CHANGED")
        good += 1
        closed += value["u"] in original_report["closed_targets"]
        outcome_path = base.ROOT / "outcomes" / value["u"] / path.name
        if not outcome_path.exists():
            pending += 1
            continue
        outcome = base.read(outcome_path)
        if outcome["forecast_hash"] != base.digest(original):
            raise ValueError("OUTCOME_INPUT_CHANGED")
        for name, choice in (value["answers"] | p5["answers"] | p4["answers"] | original["answers"]).items():
            paired[name].append(original | outcome | {"prediction": choice["prediction"]})
        paired["ALWAYS_UP"].append(original | outcome | {"prediction": 1})
    closed_targets = [d for d in original_report["closed_targets"] if d >= FIRST_TARGET]
    due = len(closed_targets) * original_report["eligible_funds"]
    whole = len(closed_targets) * original_report["watchlist_funds"]
    value = {
        "at": base.now().isoformat(),
        "primary_candidate": result["winner"],
        "verified_forecasts": good,
        "invalid_or_late": late,
        "pending": pending,
        "eligible_coverage": closed / due if due else None,
        "whole_watchlist_coverage": closed / whole if whole else None,
        "missing_due_predictions": due - closed,
        "matched_forward_metrics": {n: base.metrics(v) for n, v in paired.items()},
        "new_cost_cny": 0,
        "model_released": False,
    }
    base.save(root() / "report.json", value, replace=True)
    return value
