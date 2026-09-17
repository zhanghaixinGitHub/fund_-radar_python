"""第十八轮：用严格过去的样本外分数校准一日纠错分数。

复用第十七轮外层模型；校准器只学习外层截止前两个63日区间的样本外分数。
不扫描考试阈值，不改变基金覆盖、原始净值标签或已有预测。
"""

import hashlib
import shutil
import warnings
from collections import Counter, defaultdict
from datetime import date, datetime, time

import joblib
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

from app.services import direction_1d_sprint as base
from app.services import direction_1d_sprint_adaptive as adaptive
from app.services import direction_1d_sprint_dual_us as dual
from app.services import direction_1d_sprint_fund_response as previous
from app.services import direction_1d_sprint_hk as previous_round
from app.services import direction_1d_sprint_hk_live as hk_live
from app.services import direction_1d_sprint_overnight as overnight
from app.services import direction_1d_sprint_return_target as regression
from app.services import direction_1d_sprint_sparse as sparse

CANDIDATES = ("HK_PLATT_ERR504", "HK_ISO_ERR504")
LEARNED = CANDIDATES
FIRST_TARGET = "2026-09-16"


def root():
    return base.ROOT / "round-18"


def active():
    regression.active()


def fingerprint():
    value = previous_round.fingerprint()
    for name in (
        "app/services/direction_1d_sprint_calibrated.py",
        "scripts/direction_1d_sprint_calibrated.py",
        "tests/test_direction_1d_sprint_calibrated.py",
    ):
        value["code"][name] = hashlib.sha256((base.PROJECT / name).read_bytes()).hexdigest()
    return value


def vector(x, t, u, markets):
    return previous_round.vector(x, t, u, markets)


def dataset():
    return previous_round.dataset()


def plan():
    path = root() / "plan.json"
    if path.exists():
        value = base.read(path)
        if value["fingerprint"] != fingerprint() or value["calendar_hash"] != base.calendar()[1]:
            raise ValueError("ROUND_18_CODE_OR_CALENDAR_CHANGED")
        return value
    active()
    parent, _ = previous_round.models()
    parent_plan = previous_round.plan()
    value = {
        "at": base.now().isoformat(),
        "fingerprint": fingerprint(),
        "candidates": CANDIDATES,
        "hypothesis": (
            "Calibrate natural error scores using past out-of-fold errors, without exam-based threshold selection."
        ),
        "target": "SPX rule wrong against next CN day raw unit NAV direction; flat NON_UP",
        "calibration_dates": 126,
        "inner_blocks": 2,
        "block_dates": 63,
        "inner_base_training_dates": 504,
        "base_recipe": "frozen round17 HK_EXTRA12_ERR504",
        "reuse_outer_base_models": 15,
        "max_inner_base_fits": 30,
        "max_calibrator_fits": 30,
        "max_development_fits": 48,
        "max_current_fits": 12,
        "max_reproduction_count": 1,
        "platt": {
            "input": "logit(raw_score clipped1e-6..1-1e-6)",
            "C": 1.0,
            "max_iter": 1500,
            "seed": 0,
            "nonpositive_slope": "identity fallback recorded, no refit",
        },
        "isotonic": {"increasing": True, "out_of_bounds": "clip", "y_min": 0, "y_max": 1},
        "weights": "date/family natural error prior; no class balancing",
        "flip_threshold": 0.5,
        "threshold_search": False,
        "calibrated_score_is_formal_probability": False,
        "controls": ["HK_EXTRA12_ERR504 threshold0.55", "HK_RAW05_ERR504 threshold0.5", "SPX_SIGN"],
        "expected_questions": 5670,
        "expected_dates": 199,
        "development_year": 2025,
        "calendar_hash": base.calendar()[1],
        "parent_result_hash": base.digest(parent),
        "input_hashes": parent_plan["input_hashes"],
        "current_fit_cutoff": parent_plan["current_fit_cutoff"],
        "selection": "highest2025 common date/family accuracy among two; no retuning",
        "first_forward_target": FIRST_TARGET,
        "new_provider_calls": 0,
        "new_cost_cny": 0,
        "this_round_2026_scores_read": False,
        "reserved_period_note": (
            "Mature2026 OOF scores only build current calibration training data, not reserved-period evaluation."
        ),
        "historical_availability": "RECONSTRUCTED_NOT_TRUE_FORWARD",
    }
    base.save(path, value)
    for name in value["fingerprint"]["code"]:
        dest = root() / "code" / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(base.PROJECT / name, dest)
    return value


def inner_model(rows, cutoff, key):
    """本轮内层模型只创建一次；不同校准方法共用，不能各自重复拟合同一窗口。"""
    path = root() / "inner-models" / f"{key}.joblib"
    receipt, attempt = path.with_suffix(".json"), path.with_suffix(".attempt.json")
    if receipt.exists():
        saved = base.read(receipt)
        if saved["cutoff"] != cutoff or hashlib.sha256(path.read_bytes()).hexdigest() != saved["sha256"]:
            raise ValueError("CALIBRATION_INNER_CHECKPOINT_CHANGED")
        return joblib.load(path), saved["sha256"]
    if attempt.exists():
        raise ValueError("CALIBRATION_INNER_FIT_INTERRUPTED_NO_RETRY")
    if len(list((root() / "inner-models").glob("*.attempt.json"))) >= 30:
        raise ValueError("CALIBRATION_INNER_FIT_BUDGET")
    base.save(attempt, {"at": base.now().isoformat(), "cutoff": cutoff})
    trained = previous_round.fit(rows, "HK_EXTRA12_ERR504", cutoff)
    joblib.dump(trained, path)
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    base.save(receipt, {"at": base.now().isoformat(), "cutoff": cutoff, "sha256": sha})
    return trained, sha


def calibration_rows(rows, cutoff):
    """只用外层截止前成熟标签；每段的原始分数由该段开始前训练的模型产生。

    外层模型的训练内分数不能进入校准，当前日期也不能代替历史截止点。
    """
    groups = {r["group"] for r in rows}
    if len(groups) != 1:
        raise ValueError("CALIBRATION_GROUP_NOT_UNIQUE")
    group = next(iter(groups))
    path = root() / "calibration-inputs" / f"{cutoff}-{group}.json"
    if path.exists():
        return base.read(path)["rows"]
    selected = adaptive.training_rows(rows, cutoff, "MONTHLY_BAL504")
    dates = sorted({r["u"] for r in selected})[-126:]
    if len(dates) != 126:
        raise ValueError("CALIBRATION_DATES_INSUFFICIENT")
    output = []
    for start, end in ((dates[0], dates[63]), (dates[63], cutoff)):
        active()
        trained, sha = inner_model(rows, start, f"{cutoff}-{group}-{start}")
        if trained["max_mature_date"] >= start:
            raise ValueError("CALIBRATION_INNER_FUTURE_LEAKAGE")
        exam = [r for r in selected if start <= r["u"] < end]
        choices = previous_round.batch_answers([r["z"] for r in exam], "HK_EXTRA12_ERR504", trained)
        for row, choice in zip(exam, choices, strict=True):
            output.append(
                {k: row[k] for k in ("code", "family", "group", "u", "mature")}
                | {
                    "error_label": int(choice["baseline_prediction"] != row["y"]),
                    "raw_score": choice["research_score"],
                    "inner_cutoff": start,
                    "inner_model_sha256": sha,
                }
            )
    if len({r["u"] for r in output}) != 126 or any(r["mature"] >= cutoff for r in output):
        raise ValueError("CALIBRATION_TRAINING_DATES_CHANGED")
    base.save(path, {"at": base.now().isoformat(), "outer_cutoff": cutoff, "rows": output})
    return output


def outer_model(group, cutoff):
    """复用已冻结外层模型，核对检查点回执；不触发旧轮次重新训练。"""
    manifest, bundle = previous_round.models()
    p = plan()
    if base.digest(manifest) != p["parent_result_hash"]:
        raise ValueError("CALIBRATION_PARENT_MODEL_CHANGED")
    if cutoff == p["current_fit_cutoff"]:
        trained = bundle["HK_EXTRA12_ERR504"][group]
    else:
        quarters = {"2025-01-01": 1, "2025-04-01": 2, "2025-07-01": 3, "2025-10-01": 4}
        if cutoff not in quarters:
            raise ValueError("CALIBRATION_OUTER_CUTOFF_NOT_FROZEN")
        path = previous_round.root() / "checkpoints" / f"{quarters[cutoff]}-{group}-HK_EXTRA12_ERR504.joblib"
        receipt = base.read(path.with_suffix(".json"))
        if receipt["cutoff"] != cutoff or hashlib.sha256(path.read_bytes()).hexdigest() != receipt["sha256"]:
            raise ValueError("CALIBRATION_OUTER_CHECKPOINT_CHANGED")
        trained = joblib.load(path)
    if trained["cutoff"] != cutoff or trained["max_mature_date"] >= cutoff:
        raise ValueError("CALIBRATION_OUTER_MODEL_TIME_INVALID")
    return trained


def score_input(values):
    clipped = np.clip(np.asarray(values, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(clipped / (1 - clipped)).reshape(-1, 1)


def fit(rows, name, cutoff):
    """校准方法和阈值事前固定；负斜率Platt退回原始分数并留档，不反复调参。"""
    if name not in CANDIDATES:
        raise ValueError("CALIBRATION_UNKNOWN_RECIPE")
    learned = calibration_rows(rows, cutoff)
    if len({r["u"] for r in learned}) != 126 or any(r["mature"] >= cutoff for r in learned):
        raise ValueError("CALIBRATION_TRAINING_DATES_CHANGED")
    scores = np.asarray([r["raw_score"] for r in learned])
    y = np.asarray([r["error_label"] for r in learned])
    if (
        not np.isfinite(scores).all()
        or min(scores) < 0
        or max(scores) > 1
        or min(Counter(y).values()) < 20
        or len(Counter(y)) != 2
    ):
        raise ValueError("CALIBRATION_TRAINING_INPUT_INVALID")
    weights = regression.weights(learned)
    active()
    fallback, slope = False, None
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        if name == CANDIDATES[0]:
            calibrator = LogisticRegression(C=1, max_iter=1500, random_state=0)
            calibrator.fit(score_input(scores), y, sample_weight=weights)
            slope = float(calibrator.coef_[0, 0])
            fallback = slope <= 0
        else:
            calibrator = IsotonicRegression(increasing=True, out_of_bounds="clip", y_min=0, y_max=1)
            calibrator.fit(scores, y, sample_weight=weights)
    active()
    group = learned[0]["group"]
    return {
        "base_model": outer_model(group, cutoff),
        "calibrator": calibrator,
        "identity_fallback": fallback,
        "platt_slope": slope,
        "fit_hash": base.digest(learned),
        "fit_rows": len(learned),
        "fit_dates": len({r["u"] for r in learned}),
        "fit_end": max(r["u"] for r in learned),
        "max_mature_date": max(r["mature"] for r in learned),
        "cutoff": cutoff,
        "weighted_error_rate": float(np.average(y, weights=weights)),
    }


def batch_answers(values, name, trained):
    if name not in CANDIDATES:
        raise ValueError("CALIBRATION_UNKNOWN_RECIPE")
    raw = previous_round.batch_answers(values, "HK_EXTRA12_ERR504", trained["base_model"])
    if not raw:
        return []
    original_scores = np.asarray([r["research_score"] for r in raw])
    if trained["identity_fallback"]:
        scores = original_scores
    elif name == CANDIDATES[0]:
        scores = trained["calibrator"].predict_proba(score_input(original_scores))[:, 1]
    else:
        scores = trained["calibrator"].predict(original_scores)
    if not np.isfinite(scores).all() or min(scores) < 0 or max(scores) > 1:
        raise ValueError("CALIBRATION_SCORE_INVALID")
    output = []
    for raw_answer, score in zip(raw, scores, strict=True):
        baseline = raw_answer["baseline_prediction"]
        flip = bool(score > 0.5)
        output.append(
            {
                "prediction": 1 - baseline if flip else baseline,
                "baseline_prediction": baseline,
                "flipped": flip,
                "raw_error_score": raw_answer["research_score"],
                "research_score": float(score),
                "kind": "PAST_OOF_CALIBRATED_RESEARCH_SCORE_NOT_RELEASED",
            }
        )
    return output


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
            raise ValueError("HK_MODEL_CHECKPOINT_CHANGED")
        return joblib.load(path)
    if attempt.exists():
        raise ValueError("HK_MODEL_PREVIOUS_FIT_INTERRUPTED")
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
            raise ValueError("ROUND_18_INPUT_CHANGED")
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
                                {k: v for k, v in trained.items() if k not in ("base_model", "calibrator")},
                            )
                    output[name].extend(scored)
                for control in ("SPX_SIGN",):
                    output[control].extend(base.read(sparse.root() / f"folds/{q}-{group}-{control}.json"))
                control = base.read(previous_round.root() / f"folds/{q}-{group}-HK_EXTRA12_ERR504.json")
                output["HK_EXTRA12_ERR504"].extend(control)
                output["HK_RAW05_ERR504"].extend(
                    r
                    | {
                        "prediction": 1 - r["baseline_prediction"]
                        if r["research_score"] > 0.5
                        else r["baseline_prediction"]
                    }
                    for r in control
                )
            base.save(root() / "progress.json", {"at": base.now().isoformat(), "quarter": q}, replace=True)
        output["ALWAYS_UP"] = [r | {"prediction": 1} for r in output["SPX_SIGN"]]
        expected = sorted((r["code"], r["u"], r["y"]) for r in output["SPX_SIGN"])
        if len(expected) != 5670 or any(
            sorted((r["code"], r["u"], r["y"]) for r in v) != expected for v in output.values()
        ):
            raise ValueError("ROUND_18_COMMON_EXAM_CHANGED")
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
        "development_fits": 48,
        "current_fits": 12,
        "this_round_2026_scores_read": False,
        "kind": "DEVELOPMENT_ONLY_AFTER_RESERVED_AUDIT_CONSUMPTION",
        "inner_base_fits": len(list((root() / "inner-models").glob("*.joblib"))),
        "calibrator_fits": len(list((root() / "checkpoints").glob("*.joblib"))),
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
        raise ValueError("ROUND_18_MODEL_OR_CODE_CHANGED")
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
    hk_input = hk_live.capture(at)
    deadline = datetime.combine(date.fromisoformat(target), time(8, 30), base.ZONE)
    with threadpool_limits(limits=2):
        for path in paths:
            if base.now() >= deadline:
                break
            p5, _, source, original = dual.read_parent(path)
            z = vector(source["x"], original["base"], original["u"], hk_input["rows"])
            choices = {n: answer(z, n, bundle[n][original["group"]]) for n in CANDIDATES}
            # 固定阈值对照与校准方案共用同一个已冻结外层模型，不新增拟合或选择阈值。
            reference = choices[CANDIDATES[0]]
            for control, threshold in (("HK_EXTRA12_ERR504", 0.55), ("HK_RAW05_ERR504", 0.5)):
                baseline = reference["baseline_prediction"]
                flipped = reference["raw_error_score"] > threshold
                choices[control] = {
                    "prediction": 1 - baseline if flipped else baseline,
                    "research_score": reference["raw_error_score"],
                    "kind": "FROZEN_BASE_FIXED_THRESHOLD_CONTROL",
                }
            value = {
                "at": base.now().isoformat(),
                "u": target,
                "code": original["code"],
                "parent_hash": base.digest(p5),
                "source_hash": base.digest(source),
                "hk_input_hash": base.digest(hk_input),
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
            raise ValueError("ROUND_18_PARENT_OR_MODEL_CHANGED")
        hk_input = hk_live.load(value["u"])
        if value["hk_input_hash"] != base.digest(hk_input):
            raise ValueError("HK_MODEL_LIVE_INPUT_CHANGED")
        if not np.allclose(
            value["z"], vector(source["x"], original["base"], original["u"], hk_input["rows"]), rtol=0, atol=1e-12
        ):
            raise ValueError("ROUND_18_VECTOR_CHANGED")
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
