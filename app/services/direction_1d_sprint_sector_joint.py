"""第三十九轮：联合估计基金对五个行业的历史敏感度，预测下一交易日方向。

历史上下文只用当时已成熟的同基金输入；未来上下文固定于模型训练截止日。
保持原HK12训练样本、自然错误权重与0.55纠错门槛，不改写已保存的未来答案。
"""

import hashlib
import shutil
from collections import Counter, defaultdict
from datetime import date, datetime, time

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from threadpoolctl import threadpool_limits

from app.services import direction_1d_sprint as base
from app.services import direction_1d_sprint_adaptive as adaptive
from app.services import direction_1d_sprint_correction as correction
from app.services import direction_1d_sprint_dual_us as dual
from app.services import direction_1d_sprint_fund_response as previous
from app.services import direction_1d_sprint_hk as hk
from app.services import direction_1d_sprint_hk_live as hk_live
from app.services import direction_1d_sprint_overnight as overnight
from app.services import direction_1d_sprint_return_target as regression
from app.services import direction_1d_sprint_sector_data as sector_data
from app.services import direction_1d_sprint_sector_exposure as previous_round
from app.services import direction_1d_sprint_sector_joint_context as context_source
from app.services import direction_1d_sprint_sequence as sequence
from app.services import direction_1d_sprint_sparse as sparse

CANDIDATES = ("JOINT_HK23_ERR504",)
LEARNED = CANDIDATES
FIRST_TARGET = "2026-09-16"
FLIP_THRESHOLD = 0.55


def root():
    return base.ROOT / "round-39"


def active():
    regression.active()


def fingerprint():
    value = previous_round.fingerprint()
    for name in (
        "app/services/direction_1d_sprint_sector_joint.py",
        "scripts/direction_1d_sprint_sector_joint.py",
        "tests/test_direction_1d_sprint_sector_joint.py",
        "app/services/direction_1d_sprint_sector_joint_context.py",
        "tests/test_direction_1d_sprint_sector_joint_context.py",
        ".local-runs/direction-1d-sprint-20260914/round-39/context-feasibility.py",
    ):
        value["code"][name] = hashlib.sha256((base.PROJECT / name).read_bytes()).hexdigest()
    return value


def vector(x, t, u, markets, points, prior):
    z = hk.vector(x, t, u, markets)
    sector_z = z + sector_data.features(x, t, u, points)
    return z + context_source.extra_features({"z": sector_z, "u": u}, prior)


def live_vector(source, original, markets, points, snapshot):
    """按原基金代码查找冻结上下文；拒绝未来截止日或找不到该基金的快照。"""
    sequence.live_vector(source, original)
    if snapshot["cutoff"] > original["u"]:
        raise ValueError("JOINT_CONTEXT_FUTURE_CUTOFF")
    if original["code"] not in snapshot["contexts"]:
        raise ValueError("JOINT_CONTEXT_FUND_MISSING")
    prior = snapshot["contexts"][original["code"]]
    context_source.validate_context(prior, snapshot["cutoff"])
    return vector(source["x"], original["base"], original["u"], markets, points, prior)


def freeze_context(p):
    """上下文已经由dataset重算对齐，在此与本轮方案绑定，断点续跑保持原时间和内容。"""
    proposal, _, reference = context_source.reference()
    if p["current_fit_cutoff"] != proposal["current_context_cutoff"]:
        raise ValueError("JOINT_CONTEXT_CUTOFF_CHANGED")
    wanted = {
        "plan_hash": base.digest(p),
        "cutoff": p["current_fit_cutoff"],
        "reference_hash": base.digest(reference),
        "contexts": reference["contexts"],
    }
    path = root() / "current-context.json"
    if path.exists():
        saved = base.read(path)
        if any(saved[key] != value for key, value in wanted.items()):
            raise ValueError("JOINT_CURRENT_CONTEXT_CHANGED")
        return saved
    saved = {"at": base.now().isoformat()} | wanted
    base.save(path, saved)
    return saved


def current_contexts():
    """同时核验模型结果、训练计划和源快照绑定，不能替换斜率后沿用旧模型。"""
    p = plan()
    result = base.read(root() / "result.json")
    snapshot = base.read(root() / "current-context.json")
    if (
        snapshot["plan_hash"] != base.digest(p)
        or snapshot["cutoff"] != p["current_fit_cutoff"]
        or snapshot["reference_hash"] != p["input_hashes"]["round-39/current-context-feasibility.json"]
        or base.digest(snapshot) != result["context_snapshot_hash"]
    ):
        raise ValueError("JOINT_CURRENT_CONTEXT_CHANGED")
    for prior in snapshot["contexts"].values():
        context_source.validate_context(prior, snapshot["cutoff"])
    return snapshot


def selected_features(z, name):
    if name not in CANDIDATES or len(z) != 23 or not np.isfinite(z).all():
        raise ValueError("JOINT_INPUT_INVALID")
    return z


def dataset():
    return context_source.dataset()


def training_rows(rows, cutoff):
    """使用原港股模型同一组504个成熟日期，不改变样本或预测标签。"""
    return adaptive.training_rows(rows, cutoff, "MONTHLY_BAL504")


def plan():
    path = root() / "plan.json"
    if path.exists():
        value = base.read(path)
        if value["fingerprint"] != fingerprint() or value["calendar_hash"] != base.calendar()[1]:
            raise ValueError("ROUND_39_CODE_OR_CALENDAR_CHANGED")
        return value
    active()
    if base.read(previous_round.root() / "result.json")["fingerprint"] != previous_round.fingerprint():
        raise ValueError("JOINT_PREVIOUS_FINGERPRINT_CHANGED")
    parent, _ = hk.models()
    value = {
        "at": base.now().isoformat(),
        "fingerprint": fingerprint(),
        "candidates": CANDIDATES,
        "hypothesis": "Joint ridge exposure may handle correlated sector factors missing from separate regressions",
        "motivation": "Five jointly estimated sector effects, fixedridge0.25; compare to round38separateeffects",
        "base_candidate": "HK_EXTRA12_ERR504",
        "parent_result_hash": base.digest(parent),
        "target": "raw unit NAV T to adjacent CN trading day U, strictly up versus NON_UP; flat separate",
        "input_features": (
            "HK12+5joint-ridge-betas+5beta*sector-relative-returns+contextcount/126; reuse round37 capture"
        ),
        "training_dates": 504,
        "minimum_training": "120distinctdates; originaldirection and baselineerror classes each>=20rows",
        "sample_audit": "Before each fit verify exact504training row digest against round17",
        "extra_trees": {
            "trees": 256,
            "depth": 6,
            "min_leaf": 80,
            "max_features": 1.0,
            "bootstrap": False,
            "seed": 17,
            "threads": 2,
        },
        "proposal_hash": base.digest(base.read(root() / "proposal-before-training.json")),
        "input_feasibility_hash": base.digest(base.read(root() / "input-feasibility.json")),
        "weights": "date/family natural baseline-error prior; no class balance",
        "normalization": "train-only weighted centered-square scale, same round12 implementation",
        "flip_threshold": FLIP_THRESHOLD,
        "score_semantics": "Uncalibrated SPXbaseline error score; strictflip>0.55",
        "quarterly_refit": True,
        "development_year": 2025,
        "max_development_fits": 12,
        "max_current_fits": 3,
        "max_reproduction_count": 1,
        "expected_questions": 5670,
        "expected_dates": 199,
        "controls": ["HK_EXTRA12_ERR504", "TREE8_ERR504", "SPX_SIGN", "ALWAYS_UP"],
        "selection": "one fixed algorithm recipe; no parameter or threshold search",
        "calendar_hash": base.calendar()[1],
        "input_hashes": previous_round.plan()["input_hashes"]
        | {
            "round-39/" + name: base.digest(base.read(root() / name))
            for name in ("proposal-before-training.json", "input-feasibility.json", "current-context-feasibility.json")
        },
        "current_fit_cutoff": context_source.reference()[0]["current_context_cutoff"],
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
    """严格沿用第17轮树配方；只追加十一项个体行业上下文信息，记录原训练样本摘要。"""
    if name not in CANDIDATES:
        raise ValueError("JOINT_MODEL_RECIPE_INVALID")
    chosen = adaptive.training_rows(rows, cutoff, "MONTHLY_BAL504")
    x = np.asarray([selected_features(r["z"], name) for r in chosen])
    y = np.asarray([int(int(r["z"][0] >= 0) != r["y"]) for r in chosen])
    if len(Counter(y)) != 2 or min(Counter(y).values()) < 20:
        raise ValueError("JOINT_MODEL_ERROR_LABELS_INSUFFICIENT")
    weights = regression.weights(chosen)
    x, mean, scale = sequence.normalize_training(x, weights)
    active()
    model = ExtraTreesClassifier(
        n_estimators=256, max_depth=6, min_samples_leaf=80, max_features=1.0, bootstrap=False, random_state=17, n_jobs=2
    )
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
        raise ValueError("JOINT_MODEL_NORMALIZED_INPUT_INVALID")
    with threadpool_limits(limits=2):
        scores = trained["model"].predict_proba(x)[:, 1]
    if not np.isfinite(scores).all() or min(scores) < 0 or max(scores) > 1:
        raise ValueError("JOINT_MODEL_SCORE_INVALID")
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
            raise ValueError("JOINT_MODEL_CHECKPOINT_CHANGED")
        return joblib.load(path)
    if attempt.exists():
        raise ValueError("JOINT_MODEL_PREVIOUS_FIT_INTERRUPTED")
    base.save(attempt, {"at": base.now().isoformat(), "name": name, "cutoff": cutoff})
    short = adaptive.training_rows(rows, cutoff, "MONTHLY_BAL504")
    group = rows[0]["group"]
    if label.startswith("current-"):
        expected = hk.models()[1]["HK_EXTRA12_ERR504"][group]["fit_hash"]
    else:
        quarter = label.split("-")[0]
        expected = base.read(hk.root() / f"training/{quarter}-{group}-HK_EXTRA12_ERR504.json")["fit_hash"]
    if base.digest([r | {"z": r["z"][:12]} for r in short]) != expected:
        raise ValueError("JOINT_CONTROL_TRAINING_ROWS_CHANGED")
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
            raise ValueError("ROUND_39_INPUT_CHANGED")
    rows, proofs = dataset()
    context_snapshot = freeze_context(p)
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
                output["TREE8_ERR504"].extend(base.read(correction.root() / f"folds/{q}-{group}-TREE8_ERR504.json"))
            base.save(root() / "progress.json", {"at": base.now().isoformat(), "quarter": q}, replace=True)
        output["ALWAYS_UP"] = [r | {"prediction": 1} for r in output["SPX_SIGN"]]
        expected = sorted((r["code"], r["u"], r["y"]) for r in output["SPX_SIGN"])
        if len(expected) != 5670 or any(
            sorted((r["code"], r["u"], r["y"]) for r in v) != expected for v in output.values()
        ):
            raise ValueError("ROUND_39_COMMON_EXAM_CHANGED")
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
        "context_snapshot_hash": base.digest(context_snapshot),
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
        raise ValueError("ROUND_39_MODEL_OR_CODE_CHANGED")
    return result, joblib.load(path)


def preflight():
    _, bundle = models()
    current_contexts()
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
    context_snapshot = current_contexts()
    hk_input = hk_live.capture(at)
    sector_input = sector_data.capture(base.now())
    deadline = datetime.combine(date.fromisoformat(target), time(8, 30), base.ZONE)
    with threadpool_limits(limits=2):
        for path in paths:
            if base.now() >= deadline:
                break
            p5, _, source, original = dual.read_parent(path)
            z = live_vector(source, original, hk_input["rows"], sector_input["rows"], context_snapshot)
            choices = {n: answer(z, n, bundle[n][original["group"]]) for n in CANDIDATES}
            value = {
                "at": base.now().isoformat(),
                "u": target,
                "code": original["code"],
                "parent_hash": base.digest(p5),
                "source_hash": base.digest(source),
                "hk_input_hash": base.digest(hk_input),
                "sector_input_hash": base.digest(sector_input),
                "original_hash": base.digest(original),
                "model_hash": manifest["model_sha256"],
                "context_snapshot_hash": base.digest(context_snapshot),
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
    context_snapshot = current_contexts()
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
            or value["context_snapshot_hash"] != base.digest(context_snapshot)
        ):
            raise ValueError("ROUND_39_PARENT_OR_MODEL_CHANGED")
        hk_input = hk_live.load(value["u"])
        sector_input = sector_data.load(value["u"])
        if value["sector_input_hash"] != base.digest(sector_input):
            raise ValueError("JOINT_SECTOR_INPUT_CHANGED")
        if value["hk_input_hash"] != base.digest(hk_input):
            raise ValueError("JOINT_MODEL_LIVE_INPUT_CHANGED")
        if not np.allclose(
            value["z"],
            live_vector(source, original, hk_input["rows"], sector_input["rows"], context_snapshot),
            rtol=0,
            atol=1e-12,
        ):
            raise ValueError("ROUND_39_VECTOR_CHANGED")
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
