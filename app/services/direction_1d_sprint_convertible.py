"""第四十七轮：核实可转债基金基准，单独检验滞后偏离能否改善下一日方向。

005284使用自己的504成熟日期训练纠错树；另外29只复用原HK模型，仍同题整体评估。
不以单基金成绩替代全部30只成绩，历史开发结果不证明真实未来准确率。
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
from app.services import direction_1d_sprint_convertible_data as convertible_data
from app.services import direction_1d_sprint_dual_us as dual
from app.services import direction_1d_sprint_food as previous_round
from app.services import direction_1d_sprint_fund_response as previous
from app.services import direction_1d_sprint_hk as hk
from app.services import direction_1d_sprint_hk_live as hk_live
from app.services import direction_1d_sprint_overnight as overnight
from app.services import direction_1d_sprint_return_target as regression
from app.services import direction_1d_sprint_sequence as sequence
from app.services import direction_1d_sprint_sparse as sparse

CANDIDATES = ("CB_BENCH_HK16_ERR504", "CB_ONLY_HK12_ERR504")
LEARNED = CANDIDATES
FIRST_TARGET = "2026-09-16"
FLIP_THRESHOLD = 0.55
FUND = "005284"
CONTROL = "HK_EXTRA12_ERR504"


def root():
    return base.ROOT / "round-47"


def active():
    regression.active()


def fingerprint():
    value = previous_round.fingerprint()
    for name in (
        "app/integrations/tushare_sprint_convertible.py",
        "app/services/direction_1d_sprint_convertible_data.py",
        "app/services/direction_1d_sprint_convertible.py",
        "scripts/direction_1d_sprint_convertible.py",
        "tests/test_direction_1d_sprint_convertible.py",
        "tests/test_direction_1d_sprint_convertible_data.py",
        ".local-runs/direction-1d-sprint-20260914/round-47/feature-feasibility.py",
        ".local-runs/direction-1d-sprint-20260914/convertible-benchmark-data-v1/acquire.py",
        ".local-runs/direction-1d-sprint-20260914/convertible-csi800-supplement-v1/acquire.py",
    ):
        value["code"][name] = hashlib.sha256((base.PROJECT / name).read_bytes()).hexdigest()
    return value


def vector(x, code, t, u, markets, points):
    return hk.vector(x, t, u, markets) + convertible_data.features(x, code, t, u, points)


def live_vector(source, original, markets, points):
    """原始基金身份、净值序列和公告时点验证后计算特征；指数只取T及之前五日。"""
    sequence.live_vector(source, original)
    return vector(source["x"], original["code"], original["base"], original["u"], markets, points)


def selected_features(z, name):
    if name not in CANDIDATES or len(z) != 17 or not np.isfinite(z).all() or z[-1] not in (0.0, 1.0):
        raise ValueError("CB_MODEL_INPUT_INVALID")
    if z[-1] == 0 and any(z[12:16]):
        raise ValueError("CB_UNMAPPED_FEATURE_NOT_ZERO")
    return list(z[:16] if name == CANDIDATES[0] else z[:12])


def dataset():
    """保留全部原题，特征仅对事先选定的005284适用；不按结果增删基金或日期。"""
    rows, proof = hk.dataset()
    points = convertible_data.history()
    expanded = [r | {"z": r["z"] + convertible_data.features(r["x"], r["code"], r["t"], r["u"], points)} for r in rows]
    feasible, proposal = (
        base.read(root() / "input-feasibility.json"),
        base.read(root() / "proposal-before-training.json"),
    )
    if (
        feasible["proposal_hash"] != base.digest(proposal)
        or feasible["rows_hash"] != base.digest(expanded)
        or feasible["history_hash"] != base.digest(points)
        or proposal["script_sha256"] != hashlib.sha256((root() / "feature-feasibility.py").read_bytes()).hexdigest()
    ):
        raise ValueError("CB_PROTOTYPE_CHANGED")
    return expanded, proof | {
        "expanded_rows_hash": base.digest(expanded),
        "convertible_rows": sum(r["code"] == FUND for r in rows),
    }


def plan():
    path = root() / "plan.json"
    if path.exists():
        value = base.read(path)
        if value["fingerprint"] != fingerprint() or value["calendar_hash"] != base.calendar()[1]:
            raise ValueError("ROUND_47_CODE_OR_CALENDAR_CHANGED")
        return value
    active()
    parent, _ = hk.models()
    previous_round.models()
    convertible_data.history()
    value = {
        "at": base.now().isoformat(),
        "fingerprint": fingerprint(),
        "candidates": CANDIDATES,
        "hypothesis": "Own benchmark andlagged deviation may inform convertiblefund next-day error",
        "target": "Raw unit NAV T to adjacent CN U, UP vs NON_UP; keep flat separate",
        "feature_order": "HK12+composite1/5day minusCSI500+fund1/5day minus composite+route flag",
        "routing": "Only005284 new model; other29 reuse exact HK model and answers; no whole-scope exclusion",
        "ablation": "Sameconvertible HK12 model distinguishes individualization from four new features",
        "recipe": "ExtraTrees256 depth6 leaf80 max_features1 bootstrapFalse seed17 threads2",
        "training": "005284 last504 mature dates; natural date/family errorweights; train-only scaling",
        "flip_threshold": FLIP_THRESHOLD,
        "training_dates": 504,
        "quarterly_refit": True,
        "development_year": 2025,
        "max_development_fits": 8,
        "max_current_fits": 2,
        "max_reproduction_count": 1,
        "expected_questions": 5670,
        "expected_dates": 199,
        "selection": "Two fixed feature ablations; no parameter or threshold search",
        "parent_result_hash": base.digest(parent),
        "calendar_hash": base.calendar()[1],
        "input_hashes": hk.plan()["input_hashes"]
        | {
            name: base.digest(base.read(base.ROOT / name))
            for name in (
                "convertible-benchmark-data-v1/plan.json",
                "convertible-benchmark-data-v1/identities.json",
                "convertible-benchmark-data-v1/history.json",
                "convertible-benchmark-data-v1/result.json",
                "convertible-csi800-supplement-v1/plan.json",
                "convertible-csi800-supplement-v1/identity.json",
                "convertible-csi800-supplement-v1/history.json",
                "convertible-csi800-supplement-v1/result.json",
                "round-47/proposal-before-training.json",
                "round-47/input-feasibility.json",
            )
        },
        "current_fit_cutoff": str(base.now().date()),
        "first_forward_target": FIRST_TARGET,
        "new_cost_cny": 0,
        "this_round_2026_scores_read": False,
        "source_timing": "Close<=T; historicalfirst-version unproven; live receipt/readback<U08:30",
        "live_budget": "Three indices, three slots each; max9requests/target, no retry",
        "benchmark_boundary": "80/10/10 benchmark daily returns thencompound; not actualholding",
    }
    base.save(path, value)
    for name in value["fingerprint"]["code"]:
        dest = root() / "code" / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(base.PROJECT / name, dest)
    return value


def fit(rows, name, cutoff):
    """仅事先选定基金的新模型发生拟合；未来或未成熟标签不能进入训练。"""
    if name not in CANDIDATES:
        raise ValueError("CB_RECIPE_INVALID")
    chosen = adaptive.training_rows([r for r in rows if r["code"] == FUND], cutoff, "MONTHLY_BAL504")
    if len({r["u"] for r in chosen}) < 120 or any(r["z"][-1] != 1 for r in chosen):
        raise ValueError("CB_TRAINING_SCOPE_OR_DATES_INVALID")
    x = np.asarray([selected_features(r["z"], name) for r in chosen])
    y = np.asarray([int(int(r["z"][0] >= 0) != r["y"]) for r in chosen])
    if len(Counter(y)) != 2 or min(Counter(y).values()) < 20:
        raise ValueError("CB_ERROR_CLASSES_INSUFFICIENT")
    weights = regression.weights(chosen)
    x, mean, scale = sequence.normalize_training(x, weights)
    active()
    with threadpool_limits(limits=2):
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
        "fit_hash": base.digest(chosen),
        "fit_rows": len(chosen),
        "fit_dates": len({r["u"] for r in chosen}),
        "fit_end": max(r["u"] for r in chosen),
        "max_mature_date": max(r["mature"] for r in chosen),
        "cutoff": cutoff,
        "weighted_error_rate": float(np.average(y, weights=weights)),
        "training_target": "SPX_baseline_error",
        "fund_code": FUND,
    }


def batch_answers(values, name, trained):
    """默认逐字复用HK分支的分数与方向，只覆盖具有已核验基金标记的行。"""
    if not values:
        return []
    selected = [selected_features(z, name) for z in values]
    choices = hk.batch_answers([z[:12] for z in values], CONTROL, trained["control"])
    indices = [i for i, z in enumerate(values) if z[-1] == 1]
    if indices:
        convertible = trained["convertible"]
        if convertible is None:
            raise ValueError("CB_MODEL_GROUP_ROUTE_INVALID")
        x = (np.asarray([selected[i] for i in indices]) - convertible["mean"]) / convertible["scale"]
        if not np.isfinite(x).all():
            raise ValueError("CB_NORMALIZED_INPUT_INVALID")
        scores = convertible["model"].predict_proba(x)[:, 1]
        if not np.isfinite(scores).all() or min(scores) < 0 or max(scores) > 1:
            raise ValueError("CB_SCORE_INVALID")
        for i, score in zip(indices, scores, strict=True):
            baseline, flipped = int(values[i][0] >= 0), bool(score > FLIP_THRESHOLD)
            choices[i] = {
                "research_score": float(score),
                "kind": "UNCALIBRATED_BASELINE_ERROR_SCORE",
                "prediction": 1 - baseline if flipped else baseline,
                "baseline_prediction": baseline,
                "flipped": flipped,
            }
    return choices


def answer(z, name, trained=None):
    return batch_answers([z], name, trained)[0]


def fit_checkpoint(rows, name, cutoff, label):
    """旧HK模型只校验、加载；新可转债基金模型每季度和当前版本各拟合一次。"""
    path = root() / "checkpoints" / f"{label}.joblib"
    receipt, attempt = path.with_suffix(".json"), path.with_suffix(".attempt.json")
    if receipt.exists():
        value = base.read(receipt)
        if (
            value["name"] != name
            or value["cutoff"] != cutoff
            or value["sha256"] != hashlib.sha256(path.read_bytes()).hexdigest()
        ):
            raise ValueError("CB_CHECKPOINT_CHANGED")
        return joblib.load(path)
    if attempt.exists():
        raise ValueError("CB_PREVIOUS_FIT_INTERRUPTED")
    group = rows[0]["group"]
    if label.startswith("current-"):
        control = hk.models()[1][CONTROL][group]
    else:
        q = label.split("-")[0]
        src = hk.root() / "checkpoints" / f"{q}-{group}-{CONTROL}.joblib"
        meta = base.read(src.with_suffix(".json"))
        if (
            meta["name"] != CONTROL
            or meta["cutoff"] != cutoff
            or meta["sha256"] != hashlib.sha256(src.read_bytes()).hexdigest()
        ):
            raise ValueError("CB_HK_CHECKPOINT_CHANGED")
        control = joblib.load(src)
    chosen = adaptive.training_rows(rows, cutoff, "MONTHLY_BAL504")
    if control["cutoff"] != cutoff or base.digest([r | {"z": r["z"][:12]} for r in chosen]) != control["fit_hash"]:
        raise ValueError("CB_CONTROL_TRAINING_ROWS_CHANGED")
    base.save(attempt, {"at": base.now().isoformat(), "name": name, "cutoff": cutoff, "new_fit": group == "CN_BOND"})
    convertible = fit(rows, name, cutoff) if group == "CN_BOND" else None
    trained = {
        "control": control,
        "convertible": convertible,
        "group": group,
        "control_fit_hash": control["fit_hash"],
        "new_fit_count": int(convertible is not None),
    }
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
            raise ValueError("ROUND_47_INPUT_CHANGED")
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
                                {
                                    "group": group,
                                    "control_fit_hash": trained["control_fit_hash"],
                                    "new_fit_count": trained["new_fit_count"],
                                    "convertible": {
                                        k: v for k, v in (trained["convertible"] or {}).items() if k != "model"
                                    },
                                },
                            )
                    output[name].extend(scored)
                for control in ("SPX_SIGN",):
                    output[control].extend(base.read(sparse.root() / f"folds/{q}-{group}-{control}.json"))
                output["HK_EXTRA12_ERR504"].extend(base.read(hk.root() / f"folds/{q}-{group}-HK_EXTRA12_ERR504.json"))
            base.save(root() / "progress.json", {"at": base.now().isoformat(), "quarter": q}, replace=True)
        output["ALWAYS_UP"] = [r | {"prediction": 1} for r in output["SPX_SIGN"]]
        expected = sorted((r["code"], r["u"], r["y"]) for r in output["SPX_SIGN"])
        if len(expected) != 5670 or any(
            sorted((r["code"], r["u"], r["y"]) for r in v) != expected for v in output.values()
        ):
            raise ValueError("ROUND_47_COMMON_EXAM_CHANGED")
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
        "development_fits": 8,
        "current_fits": 2,
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
        raise ValueError("ROUND_47_MODEL_OR_CODE_CHANGED")
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
    convertible_input = convertible_data.capture(base.now())
    deadline = datetime.combine(date.fromisoformat(target), time(8, 30), base.ZONE)
    with threadpool_limits(limits=2):
        for path in paths:
            if base.now() >= deadline:
                break
            p5, _, source, original = dual.read_parent(path)
            z = live_vector(source, original, hk_input["rows"], convertible_input["rows"])
            choices = {n: answer(z, n, bundle[n][original["group"]]) for n in CANDIDATES}
            value = {
                "at": base.now().isoformat(),
                "u": target,
                "code": original["code"],
                "parent_hash": base.digest(p5),
                "source_hash": base.digest(source),
                "hk_input_hash": base.digest(hk_input),
                "convertible_input_hash": base.digest(convertible_input),
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
            raise ValueError("ROUND_47_PARENT_OR_MODEL_CHANGED")
        hk_input = hk_live.load(value["u"])
        convertible_input = convertible_data.load(value["u"])
        if value["convertible_input_hash"] != base.digest(convertible_input):
            raise ValueError("CB_MODEL_LIVE_INPUT_CHANGED")
        if value["hk_input_hash"] != base.digest(hk_input):
            raise ValueError("CB_MODEL_LIVE_INPUT_CHANGED")
        if not np.allclose(
            value["z"], live_vector(source, original, hk_input["rows"], convertible_input["rows"]), rtol=0, atol=1e-12
        ):
            raise ValueError("ROUND_47_VECTOR_CHANGED")
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
