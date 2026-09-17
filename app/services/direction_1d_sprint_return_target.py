"""第七轮：用连续净值变化训练回归器，按预测值正负回答原来的一日方向问题。

训练目标按基准日已知20日波动标准化并限幅；模型输出仅是研究分数，不作为概率、
预期收益或交易建议展示。复用第五轮已验证父答案与第三轮隔夜输入，不新增采集。
"""

import hashlib
import importlib.metadata
import shutil
import warnings
from collections import Counter, defaultdict
from datetime import date, datetime, time
from decimal import Decimal

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import HuberRegressor, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from app.services import direction_1d_sprint as base
from app.services import direction_1d_sprint_dual_us as dual
from app.services import direction_1d_sprint_fund_response as previous
from app.services import direction_1d_sprint_overnight as overnight
from app.services import direction_1d_sprint_sparse as sparse

CANDIDATES = ("RIDGE2_RET504", "HUBER2_RET504", "MEDIAN8_RET504")


def root():
    return base.ROOT / "round-07"


def fingerprint():
    code = dual.fingerprint()
    for name in (
        "app/services/direction_1d_sprint_return_target.py",
        "scripts/direction_1d_sprint_return_target.py",
        "tests/test_direction_1d_sprint_return_target.py",
        "tests/test_direction_1d_sprint_dual_us.py",
        "tests/test_direction_1d_sprint_fund_response.py",
    ):
        code[name] = hashlib.sha256((base.PROJECT / name).read_bytes()).hexdigest()
    return {
        "code": code,
        "libraries": {n: importlib.metadata.version(n) for n in ("numpy", "scikit-learn", "joblib", "threadpoolctl")},
    }


def active():
    if base.now() >= datetime.fromisoformat(base.read(base.ROOT / "protocol.json")["deadline_at"]):
        raise ValueError("SPRINT_DEADLINE_REACHED")


def plan():
    path = root() / "plan.json"
    if path.exists():
        p = base.read(path)
        if p["fingerprint"] != fingerprint():
            raise ValueError("ROUND_07_CODE_CHANGED")
        return p
    active()
    p = {
        "at": base.now().isoformat(),
        "fingerprint": fingerprint(),
        "candidates": list(CANDIDATES),
        "training_target": "clip(raw next NAV return / max(prior20d volatility,0.0001),-5,5)",
        "prediction_target": "unchanged raw unit NAV UP iff target NAV > base NAV; flat means NON_UP",
        "score_policy": "normalized return research score only; strictly positive implies UP, not a probability",
        "training_weights": "equal date and family; no class rebalancing; total weight equals family-days",
        "features": {
            "RIDGE2_RET504": [30, 31],
            "HUBER2_RET504": [30, 31],
            "MEDIAN8_RET504": list(sparse.FEATURE_INDICES),
        },
        "recipes": {
            "ridge_alpha": 10,
            "huber_alpha": 0.0001,
            "huber_epsilon": 1.35,
            "huber_max_iter": 1000,
            "median_quantile": 0.5,
            "tree_iter": 60,
            "tree_leaves": 3,
            "tree_min_leaf": 60,
            "tree_learning_rate": 0.05,
            "tree_l2": 10,
        },
        "development_year": 2025,
        "training_dates": 504,
        "max_fits": {"development": 36, "current_forward": 9},
        "max_reproduction_count": 1,
        "selection": "highest 2025 common-question score among three predeclared regression recipes",
        "reserved_2026_already_consumed": True,
        "this_round_2026_scores_read": False,
        "independent_evaluation": "only genuinely saved future answers; no reuse of consumed2026 as new holdout",
        "new_provider_calls": 0,
        "new_cost_cny": 0,
        "input_hashes": {
            n: base.digest(base.read(base.ROOT / n))
            for n in (
                "history.json",
                "round-02/market.json",
                "round-03/spx.json",
                "round-05/result.json",
                "reserved-audit-consumption.json",
            )
        },
    }
    base.save(path, p)
    for name in p["fingerprint"]["code"]:
        dest = root() / "code" / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(base.PROJECT / name, dest)
    base.save(root() / "code/manifest.json", p["fingerprint"])
    return p


def normalized_target(base_nav, target_nav, x):
    """波动来自T日之前20日收益，最小1个基点；只压缩训练极端幅度，不删除任何考试样本。"""
    a, b = Decimal(str(base_nav)), Decimal(str(target_nav))
    if not a.is_finite() or not b.is_finite() or min(a, b) <= 0:
        raise ValueError("REGRESSION_TARGET_NAV_INVALID")
    if len(x) != 32 or not np.isfinite(x).all() or x[3] < 0:
        raise ValueError("REGRESSION_TARGET_INPUT_INVALID")
    value = float(b / a - 1) / max(float(x[3]), 0.0001)
    if not np.isfinite(value):
        raise ValueError("REGRESSION_TARGET_NOT_FINITE")
    return float(np.clip(value, -5.0, 5.0))


def dataset():
    nav = base.read(base.ROOT / "history.json")
    lookup = {f["fund_code"]: {r["date"]: r["nav"] for r in f["rows"]} for f in nav["funds"]}
    result = []
    for row in overnight.dataset():
        target = normalized_target(lookup[row["code"]][row["t"]], lookup[row["code"]][row["u"]], row["x"])
        if int(target > 0) != row["y"]:
            raise ValueError("REGRESSION_DIRECTION_LABEL_CHANGED")
        result.append(row | {"return_target": target})
    return result


def vector(x, name):
    if name not in CANDIDATES or len(x) != 32 or not np.isfinite(x).all():
        raise ValueError("REGRESSION_VECTOR_INVALID")
    return [x[i] for i in (sparse.FEATURE_INDICES if name == "MEDIAN8_RET504" else (30, 31))]


def weights(rows):
    """不平衡涨跌类别，因为回归目标是连续值；重复产品份额不能增加该产品的权重。"""
    counts = Counter((r["u"], r["family"]) for r in rows)
    families = Counter(u for u, _ in counts)
    values = np.asarray([1 / (families[r["u"]] * counts[r["u"], r["family"]]) for r in rows])
    return values * len(counts) / values.sum()


def fit(rows, name, cutoff):
    chosen = previous.selected(rows, cutoff)
    if any(r["u"] >= cutoff for r in chosen):
        raise ValueError("REGRESSION_FUTURE_TRAINING_TARGET")
    x = np.asarray([vector(r["x"], name) for r in chosen])
    y = np.asarray([r["return_target"] for r in chosen])
    if not np.isfinite(y).all() or np.max(np.abs(y)) > 5:
        raise ValueError("REGRESSION_TRAINING_TARGET_INVALID")
    w = weights(chosen)
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        if name == "MEDIAN8_RET504":
            model = HistGradientBoostingRegressor(
                loss="quantile",
                quantile=0.5,
                max_iter=60,
                max_leaf_nodes=3,
                min_samples_leaf=60,
                learning_rate=0.05,
                l2_regularization=10,
                early_stopping=False,
                random_state=0,
            )
            model.fit(x, y, sample_weight=w)
        else:
            estimator = (
                Ridge(alpha=10)
                if name == "RIDGE2_RET504"
                else HuberRegressor(
                    epsilon=1.35,
                    alpha=0.0001,
                    max_iter=1000,
                )
            )
            model = Pipeline([("scale", StandardScaler()), ("estimator", estimator)])
            model.fit(x, y, scale__sample_weight=w, estimator__sample_weight=w)
    return {
        "model": model,
        "fit_hash": base.digest(chosen),
        "fit_rows": len(chosen),
        "fit_dates": len({r["u"] for r in chosen}),
        "fit_end": max(r["u"] for r in chosen),
        "max_mature_date": max(r["mature"] for r in chosen),
    }


def answer(x, name, trained):
    score = float(trained["model"].predict([vector(x, name)])[0])
    if not np.isfinite(score):
        raise ValueError("REGRESSION_PREDICTION_NOT_FINITE")
    return {"prediction": int(score > 0), "research_score": score, "kind": "NORMALIZED_RETURN_RESEARCH_SCORE"}


def train():
    p = plan()
    if (root() / "result.json").exists():
        return models()[0]
    active()
    overnight.source()
    for name, expected in p["input_hashes"].items():
        if base.digest(base.read(base.ROOT / name)) != expected:
            raise ValueError("ROUND_07_INPUT_CHANGED")
    rows = dataset()
    groups = sorted({r["group"] for r in rows})
    output = defaultdict(list)
    with threadpool_limits(limits=2):
        for q in range(1, 5):
            active()
            start, end = f"2025-{q * 3 - 2:02d}-01", "2026-01-01" if q == 4 else f"2025-{q * 3 + 1:02d}-01"
            control = base.read(previous.root() / f"folds/{q}.json")["GROUP_LR2_BAL504"]
            output["GROUP_LR2_BAL504"].extend(control)
            output["ALWAYS_UP"].extend(r | {"prediction": 1} for r in control)
            for group in groups:
                exam = [r for r in rows if r["group"] == group and start <= r["u"] < end]
                parent = base.read(sparse.root() / f"folds/{q}-{group}-SPX_SIGN.json")
                if {(r["code"], r["u"], r["y"]) for r in exam} != {(r["code"], r["u"], r["y"]) for r in parent} or len(
                    exam
                ) != len(parent):
                    raise ValueError("ROUND_07_COMMON_EXAM_MISMATCH")
                output["SPX_SIGN"].extend(parent)
                for name in CANDIDATES:
                    path = root() / f"folds/{q}-{group}-{name}.json"
                    if path.exists():
                        scored = base.read(path)
                    else:
                        model = fit([r for r in rows if r["group"] == group], name, start)
                        scored = [
                            {k: r[k] for k in ("code", "family", "group", "u", "y", "actual_direction")}
                            | answer(r["x"], name, model)
                            for r in exam
                        ]
                        base.save(path, scored)
                        base.save(
                            root() / f"training/{q}-{group}-{name}.json",
                            {k: v for k, v in model.items() if k != "model"},
                        )
                    output[name].extend(scored)
            base.save(root() / "progress.json", {"at": base.now().isoformat(), "quarter": q}, replace=True)
        metrics = {n: base.metrics(v) for n, v in output.items()}
        if len({v["count"] for v in metrics.values()}) != 1 or metrics[CANDIDATES[0]]["count"] != 5670:
            raise ValueError("ROUND_07_TOTAL_QUESTIONS_CHANGED")
        winner = max(CANDIDATES, key=lambda n: (metrics[n]["accuracy"], -CANDIDATES.index(n)))
        active()
        bundle = {
            n: {g: fit([r for r in rows if r["group"] == g], n, str(base.now().date())) for g in groups}
            for n in CANDIDATES
        }
    path = root() / "models.joblib"
    joblib.dump(bundle, path)
    value = {
        "at": base.now().isoformat(),
        "winner": winner,
        "metrics": metrics,
        "fingerprint": fingerprint(),
        "plan_hash": base.digest(p),
        "model_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "kind": "DEVELOPMENT_ONLY_AFTER_RESERVED_AUDIT_CONSUMPTION",
        "this_round_2026_scores_read": False,
        "new_cost_cny": 0,
        "new_provider_calls": 0,
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
        raise ValueError("ROUND_07_MODEL_OR_CODE_CHANGED")
    return result, joblib.load(path)


def preflight():
    _, bundle = models()
    selected = {}
    for row in reversed(dataset()):
        selected.setdefault(row["code"], row)
    with threadpool_limits(limits=2):
        for row in selected.values():
            for name in CANDIDATES:
                answer(row["x"], name, bundle[name][row["group"]])
    value = {"at": base.now().isoformat(), "kind": "DRY_RUN_NOT_FORWARD", "branch_checks": len(selected) * 3}
    base.save(root() / "preflight.json", value)
    return value


def tick():
    at = base.now()
    end = datetime.fromisoformat(base.read(base.ROOT / "protocol.json")["deadline_at"])
    w = base.window(at)
    if at >= end or w["status"] != "OPEN" or not time(7) <= at.time() < time(8, 30):
        return report()
    target = w["target_nav_date"]
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
            choices = {n: answer(source["x"], n, bundle[n][original["group"]]) for n in CANDIDATES}
            value = {
                "at": base.now().isoformat(),
                "u": target,
                "code": original["code"],
                "parent_hash": base.digest(p5),
                "source_hash": base.digest(source),
                "original_hash": base.digest(original),
                "model_hash": manifest["model_sha256"],
                "answers": choices,
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
            receipt.get("status") != "VERIFIED"
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
            raise ValueError("ROUND_07_PARENT_OR_MODEL_CHANGED")
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
    due = original_report["due_eligible_predictions"]
    whole = len(original_report["closed_targets"]) * original_report["watchlist_funds"]
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
