"""第八轮：让训练跟随时间变化，比较更新频率、近期权重和少量行情交互。

不重复使用已消费的2026保留期评分。开发按真实时间推进，每个考试月/周只用
该周期开始前已成熟的标签；预测仍覆盖原30只基金，不靠删题提高命中率。
"""

import hashlib
import shutil
import warnings
from collections import Counter, defaultdict
from datetime import date, datetime, time, timedelta

import joblib
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from app.services import direction_1d_sprint as base
from app.services import direction_1d_sprint_dual_us as dual
from app.services import direction_1d_sprint_fund_response as previous
from app.services import direction_1d_sprint_overnight as overnight
from app.services import direction_1d_sprint_return_target as regression

# 五项月度消融逐项改变窗口、衰减、类别权重，再比较周度刷新及六项稀疏特征。
# 元组依次是刷新周期、训练目标日期数、权重半衰期、是否平衡涨跌、是否加行情特征。
RECIPES = {
    "MONTHLY_BAL504": ("month", 504, None, True, False),
    "MONTHLY_BAL252": ("month", 252, None, True, False),
    "MONTHLY_BAL126": ("month", 126, None, True, False),
    "MONTHLY_DECAY252": ("month", 252, 63, True, False),
    "MONTHLY_NAT252": ("month", 252, 63, False, False),
    "WEEKLY_NAT252": ("week", 252, 63, False, False),
    "WEEKLY_REGIME252": ("week", 252, 63, False, True),
}
CANDIDATES = tuple(RECIPES)
FIRST_TARGET = "2026-09-16"


def root():
    return base.ROOT / "round-08"


def fingerprint():
    value = regression.fingerprint()
    for name in (
        "app/services/direction_1d_sprint_adaptive.py",
        "scripts/direction_1d_sprint_adaptive.py",
        "tests/test_direction_1d_sprint_adaptive.py",
    ):
        value["code"][name] = hashlib.sha256((base.PROJECT / name).read_bytes()).hexdigest()
    return value


def active():
    regression.active()


def anchor(target, name):
    """每个考试周期沿用周期起点模型，不能提前吃到周期内后来才公布的标签。"""
    value = date.fromisoformat(target)
    if RECIPES[name][0] == "month":
        return value.replace(day=1).isoformat()
    return (value - timedelta(days=value.weekday())).isoformat()


def vector(x, name):
    """收益均为小数。归一化只用基准日已知波动，限幅阈值预先固定而非按考试调节。"""
    if name not in RECIPES or len(x) != 32 or not np.isfinite(x).all() or x[3] < 0:
        raise ValueError("ADAPTIVE_INPUT_INVALID")
    move, sessions = float(x[30]), float(x[31])
    if not RECIPES[name][4]:
        return [move, sessions]
    vol = max(float(x[3]), 0.0001)
    momentum = float(np.clip(x[7] / vol, -5, 5))
    relative_vol = float(np.clip(x[11] / vol, 0, 5))
    return [move, sessions, momentum, relative_vol, move * momentum, move * relative_vol]


def training_rows(rows, cutoff, name):
    eligible = [r for r in rows if r["mature"] < cutoff and r["u"] < cutoff]
    dates = set(sorted({r["u"] for r in eligible})[-RECIPES[name][1] :])
    chosen = [r for r in eligible if r["u"] in dates]
    counts = Counter(r["y"] for r in chosen)
    if len(dates) < 120 or len(counts) != 2 or min(counts.values()) < 20:
        raise ValueError("ADAPTIVE_TRAIN_INSUFFICIENT")
    return chosen


def weights(rows, name):
    """先按日期和产品家族等权，再按交易样本日期衰减；重复份额不改变总影响力。"""
    _, _, half_life, balance, _ = RECIPES[name]
    counts = Counter((r["u"], r["family"]) for r in rows)
    families = Counter(u for u, _ in counts)
    ages = {day: age for age, day in enumerate(sorted(families, reverse=True))}
    values = np.asarray(
        [
            (0.5 ** (ages[r["u"]] / half_life) if half_life else 1.0) / (families[r["u"]] * counts[r["u"], r["family"]])
            for r in rows
        ]
    )
    if balance:
        y = np.asarray([r["y"] for r in rows])
        totals = {c: values[y == c].sum() for c in (0, 1)}
        if min(totals.values()) <= 0:
            raise ValueError("ADAPTIVE_SINGLE_CLASS")
        values *= np.asarray([values.sum() / (2 * totals[c]) for c in y])
    return values * len(counts) / values.sum()


def fit(rows, name, cutoff):
    chosen = training_rows(rows, cutoff, name)
    x = np.asarray([vector(r["x"], name) for r in chosen])
    y = np.asarray([r["y"] for r in chosen])
    w = weights(chosen, name)
    model = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=1500, random_state=0))
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        model.fit(x, y, standardscaler__sample_weight=w, logisticregression__sample_weight=w)
    return {
        "model": model,
        "fit_hash": base.digest(chosen),
        "weight_hash": base.digest(w.tolist()),
        "fit_rows": len(chosen),
        "fit_dates": len({r["u"] for r in chosen}),
        "fit_end": max(r["u"] for r in chosen),
        "max_mature_date": max(r["mature"] for r in chosen),
        "cutoff": cutoff,
    }


def answer(x, name, trained):
    score = float(trained["model"].predict_proba([vector(x, name)])[0, 1])
    if not np.isfinite(score) or not 0 <= score <= 1:
        raise ValueError("ADAPTIVE_SCORE_INVALID")
    return {"prediction": int(score > 0.5), "research_score": score, "kind": "UNCALIBRATED_RESEARCH_SCORE"}


def plan():
    path = root() / "plan.json"
    if path.exists():
        value = base.read(path)
        if value["fingerprint"] != fingerprint():
            raise ValueError("ROUND_08_CODE_CHANGED")
        return value
    active()
    rows = overnight.dataset()
    exam = [r for r in rows if r["u"].startswith("2025-")]
    schedule = {n: sorted({anchor(r["u"], n) for r in exam}) for n in CANDIDATES}
    groups = sorted({r["group"] for r in exam})
    value = {
        "at": base.now().isoformat(),
        "fingerprint": fingerprint(),
        "recipes": RECIPES,
        "hypothesis": "Slow updates, stale regimes and artificial class priors may suppress accuracy.",
        "development_year": 2025,
        "historical_availability": "RECONSTRUCTED_NOT_TRUE_FORWARD",
        "expected_questions": 5670,
        "expected_dates": 199,
        "schedule": schedule,
        "max_development_fits": sum(len(v) for v in schedule.values()) * len(groups),
        "max_current_fits": len(CANDIDATES) * len(groups),
        "max_reproduction_count": 1,
        "minimum_train_dates": 120,
        "C": 0.1,
        "threshold": 0.5,
        "feature_policy": (
            "SPX return/session count; regime adds T-day return/vol20, vol5/vol20 and their SPX interactions"
        ),
        "selection": "2025 date-family accuracy, tie follows recipe order; report all candidates and monthly stability",
        "reserved_2026_already_consumed": True,
        "this_round_2026_scores_read": False,
        "new_cost_cny": 0,
        "new_provider_calls": 0,
        "first_forward_target": FIRST_TARGET,
        "current_fit_cutoff": str(base.now().date()),
        "input_hashes": {
            n: base.digest(base.read(base.ROOT / n))
            for n in (
                "history.json",
                "round-02/market.json",
                "round-03/spx.json",
                "round-05/result.json",
            )
        },
    }
    base.save(path, value)
    for name in value["fingerprint"]["code"]:
        dest = root() / "code" / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(base.PROJECT / name, dest)
    return value


def train():
    p = plan()
    if (root() / "result.json").exists():
        return models()[0]
    active()
    for name, expected in p["input_hashes"].items():
        if base.digest(base.read(base.ROOT / name)) != expected:
            raise ValueError("ROUND_08_INPUT_CHANGED")
    rows = overnight.dataset()
    groups = sorted({r["group"] for r in rows})
    output = defaultdict(list)
    fits = 0
    with threadpool_limits(limits=2):
        for name in CANDIDATES:
            for start in p["schedule"][name]:
                active()
                for group in groups:
                    exam = [
                        r
                        for r in rows
                        if r["group"] == group and r["u"].startswith("2025-") and anchor(r["u"], name) == start
                    ]
                    if not exam:
                        continue
                    path = root() / f"folds/{name}-{start}-{group}.json"
                    if path.exists():
                        scored = base.read(path)
                    else:
                        trained = fit([r for r in rows if r["group"] == group], name, start)
                        scores = trained["model"].predict_proba([vector(r["x"], name) for r in exam])[:, 1]
                        scored = [
                            {k: r[k] for k in ("code", "family", "group", "u", "y", "actual_direction")}
                            | {"prediction": int(s > 0.5), "research_score": float(s)}
                            for r, s in zip(exam, scores, strict=True)
                        ]
                        base.save(path, scored)
                        base.save(
                            root() / f"training/{name}-{start}-{group}.json",
                            {k: v for k, v in trained.items() if k != "model"},
                        )
                    fits += 1
                    output[name].extend(scored)
            base.save(
                root() / "progress.json",
                {"at": base.now().isoformat(), "candidate": name, "completed_development_fits": fits},
                replace=True,
            )
        for q in range(1, 5):
            control = base.read(previous.root() / f"folds/{q}.json")["GROUP_LR2_BAL504"]
            output["GROUP_LR2_BAL504"].extend(control)
            output["ALWAYS_UP"].extend(r | {"prediction": 1} for r in control)
            for group in groups:
                output["SPX_SIGN"].extend(base.read(base.ROOT / f"round-04/folds/{q}-{group}-SPX_SIGN.json"))
        expected = sorted((r["code"], r["u"], r["y"]) for r in output["GROUP_LR2_BAL504"])
        if len(expected) != p["expected_questions"] or any(
            sorted((r["code"], r["u"], r["y"]) for r in v) != expected for v in output.values()
        ):
            raise ValueError("ROUND_08_COMMON_EXAM_CHANGED")
        metrics = {n: base.metrics(v) for n, v in output.items()}
        winner = max(CANDIDATES, key=lambda n: (metrics[n]["accuracy"], -CANDIDATES.index(n)))
        active()
        bundle = {
            n: {g: fit([r for r in rows if r["group"] == g], n, p["current_fit_cutoff"]) for g in groups}
            for n in CANDIDATES
        }
    if fits > p["max_development_fits"]:
        raise ValueError("ROUND_08_FIT_BUDGET_EXCEEDED")
    path = root() / "models.joblib"
    joblib.dump(bundle, path)
    value = {
        "at": base.now().isoformat(),
        "winner": winner,
        "metrics": metrics,
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
        "fingerprint": fingerprint(),
        "plan_hash": base.digest(p),
        "model_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "development_fits": fits,
        "current_fits": len(CANDIDATES) * len(groups),
        "kind": "DEVELOPMENT_ONLY_AFTER_RESERVED_AUDIT_CONSUMPTION",
        "this_round_2026_scores_read": False,
        "new_cost_cny": 0,
        "new_provider_calls": 0,
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
        raise ValueError("ROUND_08_MODEL_OR_CODE_CHANGED")
    return result, joblib.load(path)


def preflight():
    _, bundle = models()
    selected = {}
    for row in reversed(overnight.dataset()):
        selected.setdefault(row["code"], row)
    with threadpool_limits(limits=2):
        for row in selected.values():
            for name in CANDIDATES:
                answer(row["x"], name, bundle[name][row["group"]])
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
            raise ValueError("ROUND_08_PARENT_OR_MODEL_CHANGED")
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
