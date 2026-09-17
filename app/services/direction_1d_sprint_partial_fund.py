"""第四十轮：共享一日线性纠错规律，同时学习受较强约束的基金个体偏移。

原HK12后追加固定30基金身份和基金与SPX变化的交互，不根据目标日收益选择基金。
仅共享输入按训练数据标准化，身份列保持0.25倍尺度，未来代码表与模型和答案绑定。
"""

import hashlib
import shutil
import warnings
from collections import Counter, defaultdict
from datetime import date, datetime, time

import joblib
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

from app.services import direction_1d_sprint as base
from app.services import direction_1d_sprint_adaptive as adaptive
from app.services import direction_1d_sprint_correction as correction
from app.services import direction_1d_sprint_dual_us as dual
from app.services import direction_1d_sprint_fund_response as previous
from app.services import direction_1d_sprint_hk as hk
from app.services import direction_1d_sprint_hk_live as hk_live
from app.services import direction_1d_sprint_overnight as overnight
from app.services import direction_1d_sprint_partial_fund_features as feature_source
from app.services import direction_1d_sprint_return_target as regression
from app.services import direction_1d_sprint_sector_joint as previous_round
from app.services import direction_1d_sprint_sequence as sequence
from app.services import direction_1d_sprint_sparse as sparse

CANDIDATES = ("PARTIAL_FUND_LR72_ERR504",)
LEARNED = CANDIDATES
FIRST_TARGET = "2026-09-16"
FLIP_THRESHOLD = 0.55


def root():
    return base.ROOT / "round-40"


def active():
    regression.active()


def fingerprint():
    value = previous_round.fingerprint()
    for name in (
        "app/services/direction_1d_sprint_partial_fund.py",
        "scripts/direction_1d_sprint_partial_fund.py",
        "tests/test_direction_1d_sprint_partial_fund.py",
        "app/services/direction_1d_sprint_partial_fund_features.py",
        "tests/test_direction_1d_sprint_partial_fund_features.py",
        ".local-runs/direction-1d-sprint-20260914/round-40/feature-feasibility.py",
    ):
        value["code"][name] = hashlib.sha256((base.PROJECT / name).read_bytes()).hexdigest()
    return value


def vector(x, t, u, markets, code, codes):
    z = hk.vector(x, t, u, markets)
    return z + feature_source.extras(z, code, codes)


def live_vector(source, original, markets, codes):
    """基金身份必须取不可变父答案的原始代码，未知代码拒绝，不按相似基金替代。"""
    sequence.live_vector(source, original)
    return vector(source["x"], original["base"], original["u"], markets, original["code"], codes)


def freeze_codes(p):
    """训练前将代码表落盘并与本轮计划绑定，重启不能重新排序后继续使用旧模型。"""
    wanted = {"plan_hash": base.digest(p), "proposal_hash": p["proposal_hash"], "codes": p["fund_codes"]}
    path = root() / "fund-codes.json"
    if path.exists():
        saved = base.read(path)
        if any(saved[k] != v for k, v in wanted.items()):
            raise ValueError("PARTIAL_FUND_SAVED_CODES_CHANGED")
        return saved
    saved = {"at": base.now().isoformat()} | wanted
    base.save(path, saved)
    return saved


def frozen_codes():
    """核对计划、模型结果和实际代码表摘要，拒绝仅重写外层摘要后的列错位。"""
    p = plan()
    result = base.read(root() / "result.json")
    saved = base.read(root() / "fund-codes.json")
    if (
        saved["plan_hash"] != base.digest(p)
        or saved["proposal_hash"] != p["proposal_hash"]
        or saved["codes"] != p["fund_codes"]
        or base.digest(saved) != result["fund_codes_hash"]
    ):
        raise ValueError("PARTIAL_FUND_CODES_MODEL_BINDING_CHANGED")
    feature_source.validate_codes(saved["codes"])
    return saved


def selected_features(z, name):
    if name not in CANDIDATES or len(z) != 72 or not np.isfinite(z).all():
        raise ValueError("PARTIAL_FUND_INPUT_INVALID")
    return z


def dataset():
    return feature_source.dataset()


def training_rows(rows, cutoff):
    """使用原港股模型同一组504个成熟日期，不改变样本或预测标签。"""
    return adaptive.training_rows(rows, cutoff, "MONTHLY_BAL504")


def plan():
    path = root() / "plan.json"
    if path.exists():
        value = base.read(path)
        if value["fingerprint"] != fingerprint() or value["calendar_hash"] != base.calendar()[1]:
            raise ValueError("ROUND_40_CODE_OR_CALENDAR_CHANGED")
        return value
    active()
    if base.read(previous_round.root() / "result.json")["fingerprint"] != previous_round.fingerprint():
        raise ValueError("PARTIAL_FUND_PREVIOUS_FINGERPRINT_CHANGED")
    parent, _ = hk.models()
    value = {
        "at": base.now().isoformat(),
        "fingerprint": fingerprint(),
        "candidates": CANDIDATES,
        "hypothesis": (
            "Shared linear effects with shrunken fund deviations may improve individual direction discrimination"
        ),
        "motivation": (
            "Combine shared HK12 coefficients and fund-specific intercept/SPX slope;compare to round35shared linear"
        ),
        "base_candidate": "HK_EXTRA12_ERR504",
        "parent_result_hash": base.digest(parent),
        "target": "raw unit NAV T to adjacent CN trading day U, strictly up versus NON_UP; flat separate",
        "input_features": "HK12+30fundID*.25+30fundID*.25*SPXpctclip+-5;codeorderfrozen",
        "training_dates": 504,
        "minimum_training": "120distinctdates; originaldirection and baselineerror classes each>=20rows",
        "sample_audit": "Before each fit verify exact504training row digest against round17",
        "logistic_regression": {
            "C": 0.1,
            "max_iter": 1500,
            "seed": 17,
            "solver": "lbfgs",
            "penalty": "L2 default",
            "class_balance": False,
            "convergence_warning": "fail without automatic retry",
        },
        "diagnostic_hashes": {
            str(number): base.digest(base.read(base.ROOT / f"round-{number}/fit-generalization-result.json"))
            for number in (38, 39)
        },
        "proposal_hash": base.digest(base.read(root() / "proposal-before-training.json")),
        "fund_codes": feature_source.reference()[0]["codes"],
        "weights": "date/family natural baseline-error prior; no class balance",
        "normalization": "Global12trainweightedcenteredsquare;extra60mean0scale1 retains fixedfundshrinkage",
        "flip_threshold": FLIP_THRESHOLD,
        "score_semantics": "Uncalibrated SPXbaseline error score; strictflip>0.55",
        "quarterly_refit": True,
        "development_year": 2025,
        "max_development_fits": 12,
        "max_current_fits": 3,
        "max_reproduction_count": 1,
        "expected_questions": 5670,
        "expected_dates": 199,
        "controls": ["HK_EXTRA12_ERR504", "LR8_ERR504", "SPX_SIGN", "ALWAYS_UP"],
        "selection": "one fixed algorithm recipe; no parameter or threshold search",
        "calendar_hash": base.calendar()[1],
        "input_hashes": hk.plan()["input_hashes"]
        | {
            "round-40/" + name: base.digest(base.read(root() / name))
            for name in ("feature-plan.json", "proposal-before-training.json", "input-feasibility.json")
        },
        "current_fit_cutoff": str(base.now().date()),
        "first_forward_target": FIRST_TARGET,
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
    """沿用第35轮逻辑回归配方，只有原12项标准化，追加基金偏移保持0.25缩小尺度。"""
    if name not in CANDIDATES:
        raise ValueError("PARTIAL_FUND_MODEL_RECIPE_INVALID")
    chosen = adaptive.training_rows(rows, cutoff, "MONTHLY_BAL504")
    x = np.asarray([selected_features(r["z"], name) for r in chosen])
    y = np.asarray([int(int(r["z"][0] >= 0) != r["y"]) for r in chosen])
    if len(Counter(y)) != 2 or min(Counter(y).values()) < 20:
        raise ValueError("PARTIAL_FUND_MODEL_ERROR_LABELS_INSUFFICIENT")
    weights = regression.weights(chosen)
    x, mean, scale = feature_source.normalize_training(x, weights)
    active()
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        model = LogisticRegression(C=0.1, max_iter=1500, random_state=17)
        model.fit(x, y, sample_weight=weights)
    active()
    return {
        "model": model,
        "mean": mean,
        "scale": scale,
        "fit_hash": base.digest(chosen),
        "fit_rows": len(chosen),
        "fit_dates": len({r["u"] for r in chosen}),
        "fit_end": max(r["u"] for r in chosen),
        "max_mature_date": max(r["mature"] for r in chosen),
        "cutoff": cutoff,
        "weighted_error_rate": float(np.average(y, weights=weights)),
    }


def batch_answers(values, name, trained):
    """同一模型批量推理，避免逐题调度线程；与实时单题使用同一阈值实现。"""
    if not values:
        return []
    x = (np.asarray([selected_features(z, name) for z in values]) - trained["mean"]) / trained["scale"]
    if not np.isfinite(x).all():
        raise ValueError("PARTIAL_FUND_MODEL_NORMALIZED_INPUT_INVALID")
    with threadpool_limits(limits=2):
        scores = trained["model"].predict_proba(x)[:, 1]
    if not np.isfinite(scores).all() or min(scores) < 0 or max(scores) > 1:
        raise ValueError("PARTIAL_FUND_MODEL_SCORE_INVALID")
    result = []
    for z, score in zip(values, scores, strict=True):
        baseline, flipped = int(z[0] >= 0), bool(score > FLIP_THRESHOLD)
        prediction = 1 - baseline if flipped else baseline
        value = {
            "research_score": float(score),
            "kind": "UNCALIBRATED_BASELINE_ERROR_SCORE",
            "prediction": prediction,
            "baseline_prediction": baseline,
            "flipped": prediction != baseline,
        }
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
            raise ValueError("PARTIAL_FUND_MODEL_CHECKPOINT_CHANGED")
        return joblib.load(path)
    if attempt.exists():
        raise ValueError("PARTIAL_FUND_MODEL_PREVIOUS_FIT_INTERRUPTED")
    base.save(attempt, {"at": base.now().isoformat(), "name": name, "cutoff": cutoff})
    short = adaptive.training_rows(rows, cutoff, "MONTHLY_BAL504")
    group = rows[0]["group"]
    if label.startswith("current-"):
        expected = hk.models()[1]["HK_EXTRA12_ERR504"][group]["fit_hash"]
    else:
        quarter = label.split("-")[0]
        expected = base.read(hk.root() / f"training/{quarter}-{group}-HK_EXTRA12_ERR504.json")["fit_hash"]
    if base.digest([r | {"z": r["z"][:12]} for r in short]) != expected:
        raise ValueError("PARTIAL_FUND_CONTROL_TRAINING_ROWS_CHANGED")
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
            raise ValueError("ROUND_40_INPUT_CHANGED")
    rows, proofs = dataset()
    code_manifest = freeze_codes(p)
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
                output["HK_EXTRA12_ERR504"].extend(base.read(hk.root() / f"folds/{q}-{group}-HK_EXTRA12_ERR504.json"))
                output["LR8_ERR504"].extend(base.read(correction.root() / f"folds/{q}-{group}-LR8_ERR504.json"))
            base.save(root() / "progress.json", {"at": base.now().isoformat(), "quarter": q}, replace=True)
        output["ALWAYS_UP"] = [r | {"prediction": 1} for r in output["SPX_SIGN"]]
        expected = sorted((r["code"], r["u"], r["y"]) for r in output["SPX_SIGN"])
        if len(expected) != 5670 or any(
            sorted((r["code"], r["u"], r["y"]) for r in v) != expected for v in output.values()
        ):
            raise ValueError("ROUND_40_COMMON_EXAM_CHANGED")
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
        "fund_codes_hash": base.digest(code_manifest),
        "plan_hash": base.digest(p),
        "calendar_hash": p["calendar_hash"],
        "development_fits": 12,
        "current_fits": 3,
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
        raise ValueError("ROUND_40_MODEL_OR_CODE_CHANGED")
    return result, joblib.load(path)


def preflight():
    _, bundle = models()
    frozen_codes()
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
    code_manifest = frozen_codes()
    deadline = datetime.combine(date.fromisoformat(target), time(8, 30), base.ZONE)
    with threadpool_limits(limits=2):
        for path in paths:
            if base.now() >= deadline:
                break
            p5, _, source, original = dual.read_parent(path)
            z = live_vector(source, original, hk_input["rows"], code_manifest["codes"])
            choices = {n: answer(z, n, bundle[n][original["group"]]) for n in CANDIDATES}
            value = {
                "at": base.now().isoformat(),
                "u": target,
                "code": original["code"],
                "parent_hash": base.digest(p5),
                "source_hash": base.digest(source),
                "hk_input_hash": base.digest(hk_input),
                "fund_codes_hash": base.digest(code_manifest),
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
    code_manifest = frozen_codes()
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
            or value["fund_codes_hash"] != base.digest(code_manifest)
        ):
            raise ValueError("ROUND_40_PARENT_OR_MODEL_CHANGED")
        hk_input = hk_live.load(value["u"])
        if value["hk_input_hash"] != base.digest(hk_input):
            raise ValueError("PARTIAL_FUND_MODEL_LIVE_INPUT_CHANGED")
        if not np.allclose(
            value["z"], live_vector(source, original, hk_input["rows"], code_manifest["codes"]), rtol=0, atol=1e-12
        ):
            raise ValueError("ROUND_40_VECTOR_CHANGED")
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
