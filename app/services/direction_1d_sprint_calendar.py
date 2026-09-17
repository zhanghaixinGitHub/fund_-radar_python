"""第十六轮：加入已知交易日历，预测期限仍为下一交易日。

逻辑回归与第五轮分组模型对照，纠错随机树与第十三轮同结构对照；
只增加星期、休市间隔及月初月末，不使用未来行情、标签或新增供应商数据。
"""

import hashlib
import shutil
import warnings
from collections import Counter, defaultdict
from datetime import date, datetime, time

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from app.services import direction_1d_sprint as base
from app.services import direction_1d_sprint_adaptive as adaptive
from app.services import direction_1d_sprint_adjusted as previous_round
from app.services import direction_1d_sprint_correction as correction
from app.services import direction_1d_sprint_dual_us as dual
from app.services import direction_1d_sprint_fund_response as previous
from app.services import direction_1d_sprint_overnight as overnight
from app.services import direction_1d_sprint_return_target as regression
from app.services import direction_1d_sprint_sequence as sequence
from app.services import direction_1d_sprint_sparse as sparse

CANDIDATES = ("CAL_LR10_BAL504", "CAL_EXTRA16_ERR504")
LEARNED = CANDIDATES
FIRST_TARGET = "2026-09-16"
FLIP_THRESHOLD = 0.55


def root():
    return base.ROOT / "round-16"


def active():
    regression.active()


def fingerprint():
    value = previous_round.fingerprint()
    for name in (
        "app/services/direction_1d_sprint_calendar.py",
        "scripts/direction_1d_sprint_calendar.py",
        "tests/test_direction_1d_sprint_calendar.py",
    ):
        value["code"][name] = hashlib.sha256((base.PROJECT / name).read_bytes()).hexdigest()
    return value


def calendar_vectors():
    """依据完整交易日历识别月边界；不能把开发题集缺失日期误认为休市日。

    8项依次为周一至周五独热标记、距上个交易日的自然日数/14（上限1）、
    月内首个交易日、月内最后交易日。只读日期，不读当天或未来行情。
    """
    days, _ = base.calendar()
    output = {}
    for i in range(1, len(days) - 1):
        t, u, after = days[i - 1 : i + 2]
        if not t < u < after or u.weekday() > 4:
            raise ValueError("CALENDAR_ORDER_INVALID")
        output[str(t), str(u)] = [float(u.weekday() == n) for n in range(5)] + [
            min((u - t).days, 14) / 14.0,
            float((t.year, t.month) != (u.year, u.month)),
            float((u.year, u.month) != (after.year, after.month)),
        ]
    return output


def vector(x, t, u, calendars=None):
    """保留旧纠错模型8项输入，再追加8项日历；拒绝跨过下一个交易日。"""
    calendars = calendar_vectors() if calendars is None else calendars
    if (t, u) not in calendars:
        raise ValueError("CALENDAR_TARGET_NOT_ADJACENT_OR_UNCOVERED")
    return correction.vector(x) + calendars[t, u]


def selected_features(z, name):
    """逻辑回归保留旧SPX小数收益和交易场次，随机树保留旧8项输入单位。"""
    if name not in CANDIDATES or len(z) != 16 or not np.isfinite(z).all():
        raise ValueError("CALENDAR_FEATURE_INVALID")
    return [z[0] / 100, z[2]] + list(z[8:]) if name == CANDIDATES[0] else list(z)


def dataset():
    calendars = calendar_vectors()
    rows = [r | {"z": vector(r["x"], r["t"], r["u"], calendars)} for r in overnight.dataset()]
    return rows, {
        "at": base.now().isoformat(),
        "rows": len(rows),
        "calendar_hash": base.calendar()[1],
        "target": "raw unit NAV T to next CN trading day",
    }


def plan():
    path = root() / "plan.json"
    if path.exists():
        value = base.read(path)
        if value["fingerprint"] != fingerprint() or value["calendar_hash"] != base.calendar()[1]:
            raise ValueError("ROUND_16_CODE_OR_CALENDAR_CHANGED")
        return value
    active()
    value = {
        "at": base.now().isoformat(),
        "fingerprint": fingerprint(),
        "candidates": CANDIDATES,
        "hypothesis": "Known weekday, holiday gap and month boundaries may change next-day response to SPX.",
        "calendar_features": [
            "Monday",
            "Tuesday",
            "Wednesday",
            "Thursday",
            "Friday",
            "calendar_gap_days_clipped14_div14",
            "first_CN_session_of_month",
            "last_CN_session_of_month",
        ],
        "calendar_hash": base.calendar()[1],
        "calendar_availability": (
            "Known exchange trading calendar; no market values or future labels. "
            "Historical calendar announcement vintages unavailable."
        ),
        "target": "strictly positive raw unit NAV change T to adjacent next CN trading day; flat NON_UP",
        "recipes": {
            CANDIDATES[0]: {
                "base": "GROUP_LR2_BAL504",
                "features": "original SPX+sessions then8calendar",
                "C": 0.1,
                "max_iter": 1500,
                "seed": 0,
                "threshold": 0.5,
                "scale": "same weighted StandardScaler as round05",
                "weights": "date/family then class balanced",
            },
            CANDIDATES[1]: {
                "base": "EXTRA8_ERR504",
                "features": "original8 correction then8calendar",
                "trees": 256,
                "depth": 6,
                "min_leaf": 80,
                "max_features": 1.0,
                "bootstrap": False,
                "seed": 17,
                "threads": 2,
                "flip_threshold": FLIP_THRESHOLD,
                "scale": "same train-only weighted centered variance as round13",
                "weights": "date/family, natural error prior",
            },
        },
        "training_dates": 504,
        "development_year": 2025,
        "quarterly_refit": True,
        "max_development_fits": 24,
        "max_current_fits": 6,
        "max_reproduction_count": 1,
        "expected_questions": 5670,
        "expected_dates": 199,
        "threshold_search": False,
        "controls": ["SPX_SIGN", "GROUP_LR2_BAL504", "EXTRA8_ERR504", "ALWAYS_UP"],
        "selection": "highest common5670 date-family accuracy among2; no retuning",
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
    """只添加日历；两种学习器的窗口、类别权重、归一化、结构和阈值分别沿用对应对照。"""
    if name not in CANDIDATES:
        raise ValueError("CALENDAR_UNKNOWN_RECIPE")
    is_error = name == CANDIDATES[1]
    chosen = adaptive.training_rows(rows, cutoff, "MONTHLY_BAL504") if is_error else previous.selected(rows, cutoff)
    x = np.asarray([selected_features(r["z"], name) for r in chosen])
    y = np.asarray([int(int(r["z"][0] >= 0) != r["y"]) if is_error else r["y"] for r in chosen])
    if len(Counter(y)) != 2 or min(Counter(y).values()) < 20:
        raise ValueError("CALENDAR_LABELS_INSUFFICIENT")
    weights = regression.weights(chosen) if is_error else sparse.training_weights(chosen)
    if is_error:
        x, mean, scale = sequence.normalize_training(x, weights)
    else:
        scaler = StandardScaler().fit(x, sample_weight=weights)
        mean, scale = scaler.mean_.tolist(), scaler.scale_.tolist()
        x = scaler.transform(x)
    if not np.isfinite(x).all():
        raise ValueError("CALENDAR_NORMALIZED_INPUT_INVALID")
    active()
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        model = (
            ExtraTreesClassifier(
                n_estimators=256,
                max_depth=6,
                min_samples_leaf=80,
                max_features=1.0,
                bootstrap=False,
                random_state=17,
                n_jobs=2,
            )
            if is_error
            else LogisticRegression(C=0.1, max_iter=1500, random_state=0)
        )
        model.fit(x, y, sample_weight=weights)
    active()
    return {
        "model": model,
        "mean": mean,
        "scale": scale,
        "training_target": "baseline_error" if is_error else "next_NAV_UP",
        "weighted_target_rate": float(np.average(y, weights=weights)),
        "fit_hash": base.digest(chosen),
        "fit_rows": len(chosen),
        "fit_dates": len({r["u"] for r in chosen}),
        "fit_end": max(r["u"] for r in chosen),
        "max_mature_date": max(r["mature"] for r in chosen),
        "cutoff": cutoff,
    }


def batch_answers(values, name, trained):
    """同一模型批量推理，避免逐题调度线程；与实时单题使用同一阈值实现。"""
    if not values:
        return []
    x = (np.asarray([selected_features(z, name) for z in values]) - trained["mean"]) / trained["scale"]
    if not np.isfinite(x).all():
        raise ValueError("CALENDAR_NORMALIZED_INPUT_INVALID")
    scores = trained["model"].predict_proba(x)[:, 1]
    if not np.isfinite(scores).all() or min(scores) < 0 or max(scores) > 1:
        raise ValueError("CALENDAR_SCORE_INVALID")
    result = []
    for z, score in zip(values, scores, strict=True):
        value = {"research_score": float(score), "kind": "UNCALIBRATED_RESEARCH_SCORE"}
        if name == CANDIDATES[1]:
            baseline, flipped = int(z[0] >= 0), bool(score > FLIP_THRESHOLD)
            value |= {
                "prediction": 1 - baseline if flipped else baseline,
                "baseline_prediction": baseline,
                "flipped": flipped,
                "kind": "UNCALIBRATED_BASELINE_ERROR_SCORE",
            }
        else:
            value["prediction"] = int(score > 0.5)
        result.append(value)
    return result


def answer(z, name, trained=None):
    return batch_answers([z], name, trained)[0]


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
            raise ValueError("CALENDAR_CHECKPOINT_CHANGED")
        return joblib.load(path)
    if attempt.exists():
        raise ValueError("CALENDAR_PREVIOUS_FIT_INTERRUPTED")
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
            raise ValueError("ROUND_16_INPUT_CHANGED")
    rows, proofs = dataset()
    proof_path = root() / "training-question-proof.json"
    if not proof_path.exists():
        base.save(proof_path, proofs)
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
                            {k: r[k] for k in ("code", "family", "group", "u", "y", "actual_direction")} | choice
                            for r, choice in zip(
                                exam, batch_answers([r["z"] for r in exam], name, trained), strict=True
                            )
                        ]
                        base.save(path, scored)
                        if trained:
                            base.save(
                                root() / f"training/{q}-{group}-{name}.json",
                                {k: v for k, v in trained.items() if k != "model"},
                            )
                    output[name].extend(scored)
                for control in ("SPX_SIGN",):
                    output[control].extend(base.read(sparse.root() / f"folds/{q}-{group}-{control}.json"))
                output["EXTRA8_ERR504"].extend(base.read(correction.root() / f"folds/{q}-{group}-EXTRA8_ERR504.json"))
            output["GROUP_LR2_BAL504"].extend(base.read(previous.root() / f"folds/{q}.json")["GROUP_LR2_BAL504"])
            base.save(root() / "progress.json", {"at": base.now().isoformat(), "quarter": q}, replace=True)
        output["ALWAYS_UP"] = [r | {"prediction": 1} for r in output["SPX_SIGN"]]
        expected = sorted((r["code"], r["u"], r["y"]) for r in output["SPX_SIGN"])
        if len(expected) != 5670 or any(
            sorted((r["code"], r["u"], r["y"]) for r in v) != expected for v in output.values()
        ):
            raise ValueError("ROUND_16_COMMON_EXAM_CHANGED")
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
        "calendar_hash": p["calendar_hash"],
        "development_fits": 24,
        "current_fits": 6,
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
        or result["calendar_hash"] != base.calendar()[1]
        or hashlib.sha256(path.read_bytes()).hexdigest() != result["model_sha256"]
    ):
        raise ValueError("ROUND_16_MODEL_OR_CODE_CHANGED")
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
            z = vector(source["x"], original["base"], original["u"])
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
            raise ValueError("ROUND_16_PARENT_OR_MODEL_CHANGED")
        if not np.allclose(value["z"], vector(source["x"], original["base"], original["u"]), rtol=0, atol=1e-12):
            raise ValueError("ROUND_16_VECTOR_CHANGED")
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
