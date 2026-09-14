"""第四轮：减少隔夜模型输入并平衡训练方向，复用已保存的第三轮同题输入。

本轮不增加供应商查询，也不修改前三轮代码。规则分支没有概率；学习器输出仍只是
未校准的研究分数。所有训练、保存和核对沿用本轮三天研究的时间与证据边界。
"""

import hashlib
from collections import Counter, defaultdict
from datetime import date, datetime, time

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from app.services import direction_1d_sprint as base
from app.services import direction_1d_sprint_overnight as parent

CANDIDATES = ("LR2_BAL252", "LR8_BAL252", "TREE8_BAL252", "SPX_SIGN")
FEATURE_INDICES = (30, 31, 7, 0, 1, 3, 18, 23)
FEATURE_NAMES = (
    "SPX_T15_U0830_RETURN",
    "NEW_US_SESSION_COUNT",
    "NAV_RETURN_1D",
    "NAV_RETURN_5D",
    "NAV_RETURN_20D",
    "NAV_VOLATILITY_20D",
    "CSI300_RETURN_1D",
    "CSI500_RETURN_1D",
)


def root():
    return base.ROOT / "round-04"


def fingerprint():
    value = parent.fingerprint()
    for name in ("app/services/direction_1d_sprint_sparse.py", "scripts/direction_1d_sprint_sparse.py"):
        value[name] = hashlib.sha256((base.PROJECT / name).read_bytes()).hexdigest()
    return value


def plan():
    """先固定特征、类权重和容量；所有调整仅使用2025开发诊断作为研究依据。"""
    path = root() / "plan.json"
    if path.exists():
        result = base.read(path)
        if result["code"] != fingerprint():
            raise ValueError("ROUND_04_CODE_CHANGED")
        return result
    result = {
        "at": base.now().isoformat(),
        "code": fingerprint(),
        "candidates": list(CANDIDATES),
        "features": list(FEATURE_NAMES),
        "indices_in_parent_x32": list(FEATURE_INDICES),
        "training_days": 252,
        "training_weights": "equal date and product-family then balance UP/NON_UP using only train labels",
        "development_year": 2025,
        "threshold": 0.5,
        "tree": {"max_iter": 60, "max_leaf_nodes": 3, "min_samples_leaf": 60, "learning_rate": 0.05, "l2": 10},
        "monotonicity": "SPX return nondecreasing for CN_EQUITY/CN_MIXED; unconstrained for CN_BOND",
        "rule": "SPX cumulative return >= 0 predicts UP, otherwise NON_UP; no probability",
        "selection": "highest same-question development accuracy among four frozen branches before future outcomes",
        "maximum_development_fits": 36,
        "maximum_forward_fits": 9,
        "new_provider_requests": 0,
        "new_cost_cny": 0,
        "parent_result_hash": base.digest(base.read(parent.root() / "result.json")),
        "parent_data_hash": base.digest(base.read(parent.root() / "spx.json")),
        "diagnostic_hash": base.digest(base.read(parent.root() / "development-diagnostics.json")),
        "held_out_2026_scores_read": False,
        "historical_availability": "ASSUMED_NOT_TRUE_FORWARD",
    }
    base.save(path, result)
    return result


def vector(x, name):
    """只取父输入的固定列，收益和波动均为小数比例；第二列是交易日个数。"""
    if name not in CANDIDATES or len(x) != 32 or not np.isfinite(x).all():
        raise ValueError("SPARSE_INPUT_INVALID")
    chosen = FEATURE_INDICES[:2] if name in ("LR2_BAL252", "SPX_SIGN") else FEATURE_INDICES
    return [x[i] for i in chosen]


def training_weights(rows):
    """先让日期和产品家族等权，再让两类训练总权重相等，不查看考试标签。"""
    counts = Counter((r["u"], r["family"]) for r in rows)
    families = Counter(day for day, _ in counts)
    weights = np.asarray([1 / (families[r["u"]] * counts[r["u"], r["family"]]) for r in rows])
    labels = np.asarray([r["y"] for r in rows])
    totals = {c: float(weights[labels == c].sum()) for c in (0, 1)}
    if not all(totals.values()):
        raise ValueError("FIT_SINGLE_CLASS")
    balanced = weights * np.asarray([weights.sum() / (2 * totals[y]) for y in labels])
    # 总权重取不同日期/产品组合数，重复份额不增加正则目标中的样本总量。
    return balanced * len(counts) / balanced.sum()


def fit(rows, name, cutoff):
    if name not in CANDIDATES or name == "SPX_SIGN":
        raise ValueError("CANDIDATE_NOT_TRAINABLE")
    chosen = [r for r in rows if r["mature"] < cutoff]
    dates = set(sorted({r["u"] for r in chosen})[-252:])
    chosen = [r for r in chosen if r["u"] in dates]
    groups = {r["group"] for r in chosen}
    classes = Counter(r["y"] for r in chosen)
    if len(groups) != 1 or len(dates) < 120 or len(classes) != 2 or min(classes.values()) < 20:
        raise ValueError("FIT_SAMPLE_INSUFFICIENT")
    weights = training_weights(chosen)
    x = np.asarray([vector(r["x"], name) for r in chosen])
    y = np.asarray([r["y"] for r in chosen])
    if name.startswith("LR"):
        model = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=1500, random_state=0))
        model.fit(x, y, standardscaler__sample_weight=weights, logisticregression__sample_weight=weights)
    else:
        group = next(iter(groups))
        monotonic = [int(group in ("CN_EQUITY", "CN_MIXED"))] + [0] * 7
        model = HistGradientBoostingClassifier(
            max_iter=60,
            max_leaf_nodes=3,
            min_samples_leaf=60,
            learning_rate=0.05,
            l2_regularization=10,
            monotonic_cst=monotonic,
            early_stopping=False,
            random_state=0,
        )
        model.fit(x, y, sample_weight=weights)
    return {
        "model": model,
        "features": list(FEATURE_NAMES[:2] if name == "LR2_BAL252" else FEATURE_NAMES),
        "fit_rows": len(chosen),
        "fit_dates": len(dates),
        "fit_end": max(r["u"] for r in chosen),
        "fit_hash": base.digest(chosen),
        "weight_hash": base.digest(weights.tolist()),
    }


def answer(x, name, trained=None):
    values = vector(x, name)
    if name == "SPX_SIGN":
        return {"prediction": int(values[0] >= 0), "kind": "FIXED_RULE", "research_score": None}
    score = float(trained["model"].predict_proba([values])[0, 1])
    if not np.isfinite(score) or not 0 <= score <= 1:
        raise ValueError("RESEARCH_SCORE_INVALID")
    return {"prediction": int(score > 0.5), "kind": "UNCALIBRATED_MODEL_SCORE", "research_score": score}


def train():
    p = plan()
    parent.source()
    if (root() / "result.json").exists():
        models()
        return base.read(root() / "result.json")
    if p["parent_result_hash"] != base.digest(base.read(parent.root() / "result.json")) or p[
        "parent_data_hash"
    ] != base.digest(base.read(parent.root() / "spx.json")):
        raise ValueError("ROUND_04_PARENT_CHANGED")
    rows = parent.dataset()
    groups = sorted({r["group"] for r in rows})
    predictions = defaultdict(list)
    with threadpool_limits(limits=2):
        for q in range(1, 5):
            start = f"2025-{q * 3 - 2:02d}-01"
            end = "2026-01-01" if q == 4 else f"2025-{q * 3 + 1:02d}-01"
            for group in groups:
                grouped = [r for r in rows if r["group"] == group]
                exam = [r for r in grouped if start <= r["u"] < end]
                control = base.read(parent.root() / f"folds/{q}-{group}-TREE32_252.json")
                if {(r["code"], r["u"]) for r in exam} != {(r["code"], r["u"]) for r in control}:
                    raise ValueError("COMMON_EXAM_MISMATCH")
                predictions["TREE32_252"].extend(control)
                predictions["ALWAYS_UP"].extend(r | {"prediction": 1} for r in control)
                for name in CANDIDATES:
                    path = root() / f"folds/{q}-{group}-{name}.json"
                    if path.exists():
                        scored = base.read(path)
                    else:
                        trained = None if name == "SPX_SIGN" else fit(grouped, name, start)
                        scored = [
                            {k: r[k] for k in ("code", "family", "group", "u", "y", "actual_direction")}
                            | answer(r["x"], name, trained)
                            for r in exam
                        ]
                        base.save(path, scored)
                    predictions[name].extend(scored)
            base.save(root() / "progress.json", {"at": base.now().isoformat(), "quarter": q}, replace=True)
        metrics = {n: base.metrics(v) for n, v in predictions.items()}
        winner = max(CANDIDATES, key=lambda n: (metrics[n]["accuracy"], -CANDIDATES.index(n)))
        bundle = {
            n: {g: fit([r for r in rows if r["group"] == g], n, str(base.now().date())) for g in groups}
            for n in CANDIDATES
            if n != "SPX_SIGN"
        }
    path = root() / "models.joblib"
    joblib.dump(bundle, path)
    result = {
        "at": base.now().isoformat(),
        "metrics": metrics,
        "winner": winner,
        "code": fingerprint(),
        "plan_hash": base.digest(p),
        "model_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "held_out_2026_scores_read": False,
        "kind": "DEVELOPMENT_ASSUMED_AVAILABILITY_NOT_FORWARD",
        "new_provider_requests": 0,
        "new_cost_cny": 0,
    }
    base.save(root() / "result.json", result)
    return result


def models():
    result = base.read(root() / "result.json")
    path = root() / "models.joblib"
    if result["code"] != fingerprint() or hashlib.sha256(path.read_bytes()).hexdigest() != result["model_sha256"]:
        raise ValueError("ROUND_04_MODEL_OR_CODE_CHANGED")
    return result, joblib.load(path)


def preflight():
    _, bundle = models()
    chosen = {}
    for row in reversed(parent.dataset()):
        chosen.setdefault(row["code"], row)
    count = 0
    with threadpool_limits(limits=2):
        for row in chosen.values():
            for name in CANDIDATES:
                trained = None if name == "SPX_SIGN" else bundle[name][row["group"]]
                answer(row["x"], name, trained)
                count += 1
    value = {"at": base.now().isoformat(), "kind": "DRY_RUN_NOT_FORWARD", "branch_checks": count}
    base.save(root() / "preflight.json", value)
    return value


def read_parent(path):
    """父答案及其供应商实际到达记录均须在截止前；不将历史快照当未来输入。"""
    value = base.read(path)
    receipt = base.read(parent.root() / "receipts" / value["u"] / path.name)
    observation = base.read(parent.root() / value["observation_file"])
    deadline = datetime.combine(date.fromisoformat(value["u"]), time(8, 30), base.ZONE)
    if (
        receipt["status"] != "VERIFIED"
        or receipt["forecast_hash"] != base.digest(value)
        or datetime.fromisoformat(receipt["readback_at"]) >= deadline
        or datetime.fromisoformat(value["at"]) >= deadline
        or value["observation_hash"] != base.digest(observation)
        or datetime.fromisoformat(observation["received_at"]) >= deadline
        or observation["parsed"]["status"] != "COMPLETE"
    ):
        raise ValueError("PARENT_NOT_VERIFIED_BEFORE_CUTOFF")
    original = base.read(base.ROOT / "forward" / value["u"] / path.name)
    if value["original_hash"] != base.digest(original):
        raise ValueError("ORIGINAL_INPUT_CHANGED")
    return value, original


def tick():
    at = base.now()
    end = datetime.fromisoformat(base.read(base.ROOT / "protocol.json")["deadline_at"])
    w = base.window(at)
    if at >= end or w["status"] != "OPEN" or not time(7) <= at.time() < time(8, 30):
        return report()
    target = w["target_nav_date"]
    paths = [
        p
        for p in (parent.root() / "forward" / target).glob("*.json")
        if not (root() / "forward" / target / p.name).exists()
    ]
    if not paths:
        return report()
    parent.source()
    manifest, bundle = models()
    deadline = datetime.combine(date.fromisoformat(target), time(8, 30), base.ZONE)
    with threadpool_limits(limits=2):
        for path in paths:
            if base.now() >= deadline:
                break
            previous, original = read_parent(path)
            answers = {
                n: answer(previous["x"], n, None if n == "SPX_SIGN" else bundle[n][original["group"]])
                for n in CANDIDATES
            }
            payload = {
                "at": base.now().isoformat(),
                "u": target,
                "code": original["code"],
                "parent_hash": base.digest(previous),
                "original_hash": base.digest(original),
                "model_hash": manifest["model_sha256"],
                "answers": answers,
                "status": "MODEL_NOT_RELEASED",
            }
            saved = root() / "forward" / target / path.name
            base.save(saved, payload)
            readback = base.now()
            valid = base.read(saved) == payload and readback < deadline
            base.save(
                root() / "receipts" / target / path.name,
                {
                    "readback_at": readback.isoformat(),
                    "forecast_hash": base.digest(payload),
                    "status": "VERIFIED" if valid else "LATE_OR_INVALID",
                },
            )
    return report()


def report():
    if not (root() / "result.json").exists():
        return {"phase": "NOT_TRAINED"}
    original_report = base.report()
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
        previous, original = read_parent(parent.root() / "forward" / value["u"] / path.name)
        if value["parent_hash"] != base.digest(previous) or value["original_hash"] != base.digest(original):
            raise ValueError("ROUND_04_PARENT_CHANGED")
        good += 1
        closed += value["u"] in original_report["closed_targets"]
        outcome_path = base.ROOT / "outcomes" / value["u"] / path.name
        if not outcome_path.exists():
            pending += 1
            continue
        outcome = base.read(outcome_path)
        if outcome["forecast_hash"] != base.digest(original):
            raise ValueError("OUTCOME_INPUT_CHANGED")
        for name, choice in (value["answers"] | previous["answers"] | original["answers"]).items():
            paired[name].append(original | outcome | {"prediction": choice["prediction"]})
        paired["ALWAYS_UP"].append(original | outcome | {"prediction": 1})
    due = original_report["due_eligible_predictions"]
    whole = len(original_report["closed_targets"]) * original_report["watchlist_funds"]
    value = {
        "at": base.now().isoformat(),
        "primary_candidate": base.read(root() / "result.json")["winner"],
        "verified_forecasts": good,
        "invalid_or_late": late,
        "pending": pending,
        "eligible_coverage": closed / due if due else None,
        "whole_watchlist_coverage": closed / whole if whole else None,
        "missing_due_predictions": due - closed,
        "matched_forward_metrics": {k: base.metrics(v) for k, v in paired.items()},
        "new_provider_requests": 0,
        "new_cost_cny": 0,
        "model_released": False,
    }
    base.save(root() / "report.json", value, replace=True)
    return value
