"""第四十八轮：对比美股ETF时段内和整日价格变化对下一交易日方向的增量。

两种特征方案使用相同样本与纠错树；无新美股收盘或价格质量异常时复用原HK答案。
保留全部30只基金、5670道开发题，不用历史多轮择优冒充独立未来验证。
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
from app.services import direction_1d_sprint_convertible as previous_round
from app.services import direction_1d_sprint_dual_us as dual
from app.services import direction_1d_sprint_fund_response as previous
from app.services import direction_1d_sprint_hk as hk
from app.services import direction_1d_sprint_hk_live as hk_live
from app.services import direction_1d_sprint_overnight as overnight
from app.services import direction_1d_sprint_return_target as regression
from app.services import direction_1d_sprint_sequence as sequence
from app.services import direction_1d_sprint_sparse as sparse
from app.services import direction_1d_sprint_us_etf_data as us_etf_data

CANDIDATES = ("ETF_SESSION_HK14_ERR504", "ETF_DAY_HK14_ERR504")
LEARNED = CANDIDATES
FIRST_TARGET = "2026-09-16"
FLIP_THRESHOLD = 0.55
CONTROL = "HK_EXTRA12_ERR504"


def root():
    return base.ROOT / "round-48"


def active():
    regression.active()


def fingerprint():
    value = previous_round.fingerprint()
    for name in (
        "app/integrations/sina_sprint_etf.py",
        "app/integrations/sina_sprint_decode.cjs",
        "app/data/decoders/sina_daily_v1.js",
        "app/data/decoders/sina_daily_v1.LICENSE",
        "app/services/direction_1d_sprint_us_etf_data.py",
        "app/services/direction_1d_sprint_us_etf.py",
        "scripts/direction_1d_sprint_us_etf.py",
        "tests/test_direction_1d_sprint_us_etf.py",
        "tests/test_direction_1d_sprint_us_etf_data.py",
        ".local-runs/direction-1d-sprint-20260914/round-48/feature-feasibility.py",
        ".local-runs/direction-1d-sprint-20260914/sina-etf-source-v1/qualify.py",
        ".local-runs/direction-1d-sprint-20260914/sina-etf-source-v1/fetch_public.py",
    ):
        value["code"][name] = hashlib.sha256((base.PROJECT / name).read_bytes()).hexdigest()
    return value


def vector(x, code, t, u, markets, points):
    return hk.vector(x, t, u, markets) + us_etf_data.features(t, u, points)


def live_vector(source, original, markets, points):
    """原始基金身份、净值序列和公告时点验证后计算特征；ETF只取目标08:30前最近已收盘的两日。"""
    sequence.live_vector(source, original)
    return vector(source["x"], original["code"], original["base"], original["u"], markets, points)


def selected_features(z, name):
    if name not in CANDIDATES or len(z) != 17 or not np.isfinite(z).all() or z[-1] not in (0.0, 1.0):
        raise ValueError("US_ETF_MODEL_INPUT_INVALID")
    if z[-1] == 0 and any(z[12:16]):
        raise ValueError("US_ETF_UNMAPPED_FEATURE_NOT_ZERO")
    return list(z[:12]) + ([z[12], z[14]] if name == CANDIDATES[0] else [z[13], z[15]])


def dataset():
    """保留全部原题；两只ETF均有效且有新美国收盘时使用新特征，否则原HK回退。"""
    rows, proof = hk.dataset()
    points = us_etf_data.history()
    expanded = [r | {"z": r["z"] + us_etf_data.features(r["t"], r["u"], points)} for r in rows]
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
        raise ValueError("US_ETF_PROTOTYPE_CHANGED")
    return expanded, proof | {
        "expanded_rows_hash": base.digest(expanded),
        "etf_available_rows": sum(r["z"][-1] == 1 for r in expanded),
    }


def plan():
    path = root() / "plan.json"
    if path.exists():
        value = base.read(path)
        if value["fingerprint"] != fingerprint() or value["calendar_hash"] != base.calendar()[1]:
            raise ValueError("ROUND_48_CODE_OR_CALENDAR_CHANGED")
        return value
    active()
    parent, _ = hk.models()
    previous_round.models()
    us_etf_data.history()
    value = {
        "at": base.now().isoformat(),
        "fingerprint": fingerprint(),
        "candidates": CANDIDATES,
        "hypothesis": "US ETF session-only versus full-day movement for next-day fund direction",
        "target": "Raw unit NAV T to adjacent CN U, UP vs NON_UP; keep flat separate",
        "feature_order": "HK12+FXI(session,day)+CNYA(session,day)+route; each candidate selects two features",
        "routing": "All30 fund samequestions; source unavailable or no newUSclose reuses exactHK predictions",
        "ablation": "Same recipe/group/windows; only session-only versus full-day ETF move differs",
        "recipe": "ExtraTrees256 depth6 leaf80 max_features1 bootstrapFalse seed17 threads2",
        "training": "Last504 mature available dates/group; natural date/family error weights; train-only scaling",
        "flip_threshold": FLIP_THRESHOLD,
        "training_dates": 504,
        "quarterly_refit": True,
        "development_year": 2025,
        "max_development_fits": 24,
        "max_current_fits": 6,
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
                "sina-etf-source-v1/qualification-plan.json",
                "sina-etf-source-v1/qualification-result.json",
                "sina-etf-source-v1/history.json",
                "round-48/proposal-before-training.json",
                "round-48/input-feasibility.json",
            )
        },
        "current_fit_cutoff": str(base.now().date()),
        "first_forward_target": FIRST_TARGET,
        "new_cost_cny": 0,
        "this_round_2026_scores_read": False,
        "source_timing": "T15:00<latest USclose<U08:30; first-version unproven; actual receipt/readback<U08:30",
        "live_budget": "TwoETFquotes once per0700/0730/0800 slot; max6requests/target, no retry",
        "price_boundary": "Provider unadjusted quotes; day movement retains dividend price effects, not total return",
    }
    base.save(path, value)
    for name in value["fingerprint"]["code"]:
        dest = root() / "code" / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(base.PROJECT / name, dest)
    return value


def fit(rows, name, cutoff):
    """只用成熟且价格可用的历史行；无效价格不参加新拟合，原题仍保留HK回退。"""
    if name not in CANDIDATES:
        raise ValueError("US_ETF_RECIPE_INVALID")
    chosen = adaptive.training_rows(
        [r for r in rows if r["u"] < cutoff and r["mature"] < cutoff and r["z"][-1] == 1], cutoff, "MONTHLY_BAL504"
    )
    if len({r["u"] for r in chosen}) < 120 or any(r["z"][-1] != 1 for r in chosen):
        raise ValueError("US_ETF_TRAINING_SCOPE_OR_DATES_INVALID")
    x = np.asarray([selected_features(r["z"], name) for r in chosen])
    y = np.asarray([int(int(r["z"][0] >= 0) != r["y"]) for r in chosen])
    if len(Counter(y)) != 2 or min(Counter(y).values()) < 20:
        raise ValueError("US_ETF_ERROR_CLASSES_INSUFFICIENT")
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
        "source_route": "BOTH_ETF_AVAILABLE_AFTER_CHINA_CLOSE",
    }


def batch_answers(values, name, trained):
    """默认逐字复用HK分支的分数与方向，只覆盖具有两只ETF有效新收盘的行。"""
    if not values:
        return []
    selected = [selected_features(z, name) for z in values]
    choices = hk.batch_answers([z[:12] for z in values], CONTROL, trained["control"])
    indices = [i for i, z in enumerate(values) if z[-1] == 1]
    if indices:
        us_etf = trained["us_etf"]
        if us_etf is None:
            raise ValueError("US_ETF_MODEL_GROUP_ROUTE_INVALID")
        x = (np.asarray([selected[i] for i in indices]) - us_etf["mean"]) / us_etf["scale"]
        if not np.isfinite(x).all():
            raise ValueError("US_ETF_NORMALIZED_INPUT_INVALID")
        scores = us_etf["model"].predict_proba(x)[:, 1]
        if not np.isfinite(scores).all() or min(scores) < 0 or max(scores) > 1:
            raise ValueError("US_ETF_SCORE_INVALID")
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
    """旧HK只校验加载；两种ETF方案按季度及资产组独立拟合一次。"""
    path = root() / "checkpoints" / f"{label}.joblib"
    receipt, attempt = path.with_suffix(".json"), path.with_suffix(".attempt.json")
    if receipt.exists():
        value = base.read(receipt)
        if (
            value["name"] != name
            or value["cutoff"] != cutoff
            or value["sha256"] != hashlib.sha256(path.read_bytes()).hexdigest()
        ):
            raise ValueError("US_ETF_CHECKPOINT_CHANGED")
        return joblib.load(path)
    if attempt.exists():
        raise ValueError("US_ETF_PREVIOUS_FIT_INTERRUPTED")
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
            raise ValueError("US_ETF_HK_CHECKPOINT_CHANGED")
        control = joblib.load(src)
    chosen = adaptive.training_rows(rows, cutoff, "MONTHLY_BAL504")
    if control["cutoff"] != cutoff or base.digest([r | {"z": r["z"][:12]} for r in chosen]) != control["fit_hash"]:
        raise ValueError("US_ETF_CONTROL_TRAINING_ROWS_CHANGED")
    base.save(attempt, {"at": base.now().isoformat(), "name": name, "cutoff": cutoff, "new_fit": True})
    us_etf = fit(rows, name, cutoff)
    trained = {
        "control": control,
        "us_etf": us_etf,
        "group": group,
        "control_fit_hash": control["fit_hash"],
        "new_fit_count": int(us_etf is not None),
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
            raise ValueError("ROUND_48_INPUT_CHANGED")
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
                                    "us_etf": {k: v for k, v in (trained["us_etf"] or {}).items() if k != "model"},
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
            raise ValueError("ROUND_48_COMMON_EXAM_CHANGED")
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
        raise ValueError("ROUND_48_MODEL_OR_CODE_CHANGED")
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
    us_etf_input = us_etf_data.capture(base.now())
    if hk_input is None or us_etf_input is None:
        return report()
    deadline = datetime.combine(date.fromisoformat(target), time(8, 30), base.ZONE)
    with threadpool_limits(limits=2):
        for path in paths:
            if base.now() >= deadline:
                break
            p5, _, source, original = dual.read_parent(path)
            z = live_vector(source, original, hk_input["rows"], us_etf_input["rows"])
            choices = {n: answer(z, n, bundle[n][original["group"]]) for n in CANDIDATES}
            value = {
                "at": base.now().isoformat(),
                "u": target,
                "code": original["code"],
                "parent_hash": base.digest(p5),
                "source_hash": base.digest(source),
                "hk_input_hash": base.digest(hk_input),
                "us_etf_input_hash": base.digest(us_etf_input),
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
    bundle = models()[1] if any((root() / "forward").glob("*/*.json")) else None
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
            raise ValueError("ROUND_48_PARENT_OR_MODEL_CHANGED")
        hk_input = hk_live.load(value["u"])
        us_etf_input = us_etf_data.load(value["u"])
        if value["us_etf_input_hash"] != base.digest(us_etf_input):
            raise ValueError("US_ETF_MODEL_LIVE_INPUT_CHANGED")
        if value["hk_input_hash"] != base.digest(hk_input):
            raise ValueError("US_ETF_MODEL_LIVE_INPUT_CHANGED")
        if not np.allclose(
            value["z"], live_vector(source, original, hk_input["rows"], us_etf_input["rows"]), rtol=0, atol=1e-12
        ):
            raise ValueError("ROUND_48_VECTOR_CHANGED")
        expected_answers = {n: answer(value["z"], n, bundle[n][original["group"]]) for n in CANDIDATES}
        if value["answers"] != expected_answers:
            raise ValueError("ROUND_48_SAVED_ANSWER_CHANGED")
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
