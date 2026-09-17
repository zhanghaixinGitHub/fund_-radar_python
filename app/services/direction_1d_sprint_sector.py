"""第三十七轮：使用五个行业相对收益检验一日HK12纠错模型。

行业因子由所有基金共用，不根据当前名称推断历史持仓或精确跟踪基准。
仅新增五项基准日已结束的行业收益，保留原训练日期、模型配方和改判门槛。
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
from app.services import direction_1d_sprint_dual_us as dual
from app.services import direction_1d_sprint_fund_response as previous
from app.services import direction_1d_sprint_hk as hk
from app.services import direction_1d_sprint_hk_live as hk_live
from app.services import direction_1d_sprint_overnight as overnight
from app.services import direction_1d_sprint_return_target as regression
from app.services import direction_1d_sprint_sector_data as sector_data
from app.services import direction_1d_sprint_sensitivity as previous_round
from app.services import direction_1d_sprint_sequence as sequence
from app.services import direction_1d_sprint_sparse as sparse

CANDIDATES = ("SECTOR_HK17_ERR504",)
LEARNED = CANDIDATES
FIRST_TARGET = "2026-09-16"
FLIP_THRESHOLD = 0.55


def root():
    return base.ROOT / "round-37"


def active():
    regression.active()


def fingerprint():
    value = previous_round.fingerprint()
    for name in (
        "app/services/direction_1d_sprint_sector.py",
        "scripts/direction_1d_sprint_sector.py",
        "tests/test_direction_1d_sprint_sector.py",
        "app/services/direction_1d_sprint_sector_data.py",
        "tests/test_direction_1d_sprint_sector_data.py",
        "app/integrations/tushare_sprint_sector.py",
        ".local-runs/direction-1d-sprint-20260914/sector-data-feasibility-v1/acquire.py",
        ".local-runs/direction-1d-sprint-20260914/sector-data-feasibility-v1/check_coverage.py",
    ):
        value["code"][name] = hashlib.sha256((base.PROJECT / name).read_bytes()).hexdigest()
    return value


def vector(x, t, u, markets, points):
    return hk.vector(x, t, u, markets) + sector_data.features(x, t, u, points)


def live_vector(source, original, markets, points):
    """复用父答案的净值和港股输入；行业收盘日期必须等于基准日及前一国内交易日，真实响应必须在目标日截止前取得。"""
    sequence.live_vector(source, original)
    return vector(source["x"], original["base"], original["u"], markets, points)


def selected_features(z, name):
    if name not in CANDIDATES or len(z) != 17 or not np.isfinite(z).all():
        raise ValueError("SECTOR_MODEL_INPUT_INVALID")
    return z


def dataset():
    """按同一HK12样本重算五个行业因子，必须逐字节对齐训练前可行性摘要。"""
    rows, proof = hk.dataset()
    points = sector_data.history()
    cache, output, features = {}, [], []
    for row in rows:
        key = (row["t"], row["u"], row["x"][23])
        if key not in cache:
            cache[key] = sector_data.features(row["x"], row["t"], row["u"], points)
        extra = cache[key]
        output.append(row | {"z": row["z"] + extra})
        features.append({"code": row["code"], "u": row["u"], "extra": extra})
    proposal = base.read(root() / "proposal-before-training.json")
    if base.digest(features) != proposal["features_hash"]:
        raise ValueError("SECTOR_FEATURES_DIFFER_FROM_PROTOTYPE")
    return output, proof | {
        "at": base.now().isoformat(),
        "sector_rows": {k: len(v) for k, v in points.items()},
        "features_hash": base.digest(features),
        "sector_input_time": "exact baseT andpreviousCN",
    }


def plan():
    path = root() / "plan.json"
    if path.exists():
        value = base.read(path)
        if value["fingerprint"] != fingerprint() or value["calendar_hash"] != base.calendar()[1]:
            raise ValueError("ROUND_37_CODE_OR_CALENDAR_CHANGED")
        return value
    active()
    if base.read(previous_round.root() / "result.json")["fingerprint"] != previous_round.fingerprint():
        raise ValueError("SECTOR_PREVIOUS_FINGERPRINT_CHANGED")
    parent, _ = hk.models()
    sector_data.history()
    value = {
        "at": base.now().isoformat(),
        "fingerprint": fingerprint(),
        "candidates": CANDIDATES,
        "hypothesis": "Common sector relative movements may reveal missing industry exposure beyond broad indices",
        "base_candidate": "HK_EXTRA12_ERR504",
        "parent_result_hash": base.digest(parent),
        "target": "raw unit NAV T to adjacent CN trading day U, strictly up versus NON_UP; flat separate",
        "feature_order": (
            "HK12; coal/bank/chip/nonferrous/consumer-electronics exactT/previousCN return "
            "minus CSI500 x23,percentagepoints,clip+-20"
        ),
        "sector_source": sector_data.SOURCE,
        "sector_url": sector_data.URL,
        "source_timing": (
            "Historical current versions reconstructed,first-release versions unproven;"
            "actual rawreceiveandreadback<U08:30"
        ),
        "future_evidence": "Raw JSON/request/receive/required2dates/hashes; input andpredictionreadback<U08:30",
        "live_budget": {
            "slots": ["0700", "0730", "0800"],
            "max_requests_per_target": 15,
            "shared_across_funds": True,
            "max_bytes": sector_data.MAX_BYTES,
            "redirects": False,
            "automatic_retries": 0,
        },
        "extra_trees": {
            "trees": 256,
            "depth": 6,
            "min_leaf": 80,
            "max_features": 1.0,
            "bootstrap": False,
            "seed": 17,
            "threads": 2,
        },
        "weights": "date/family natural error prior, no class balance",
        "normalization": "train-only weighted centered-square scale, same round12 implementation",
        "flip_threshold": 0.55,
        "training_dates": 504,
        "quarterly_refit": True,
        "development_year": 2025,
        "max_development_fits": 12,
        "max_current_fits": 3,
        "max_reproduction_count": 1,
        "expected_questions": 5670,
        "expected_dates": 199,
        "controls": ["HK_EXTRA12_ERR504", "SPX_SIGN", "ALWAYS_UP"],
        "selection": "one predeclared candidate; no parameter or threshold search",
        "calendar_hash": base.calendar()[1],
        "input_hashes": hk.plan()["input_hashes"]
        | {
            name: base.digest(base.read(base.ROOT / name))
            for name in (
                "sector-data-feasibility-v1/plan.json",
                "sector-data-feasibility-v1/coverage-plan.json",
                "round-37/proposal-before-training.json",
                "sector-data-feasibility-v1/history.json",
                "sector-data-feasibility-v1/coverage-result.json",
            )
        },
        "current_fit_cutoff": str(base.now().date()),
        "first_forward_target": FIRST_TARGET,
        "new_historical_provider_calls": 0,
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
    """严格沿用第17轮树配方；只追加五项行业相对收益输入，记录原训练样本摘要。"""
    if name not in CANDIDATES:
        raise ValueError("SECTOR_MODEL_RECIPE_INVALID")
    chosen = adaptive.training_rows(rows, cutoff, "MONTHLY_BAL504")
    x = np.asarray([selected_features(r["z"], name) for r in chosen])
    y = np.asarray([int(int(r["z"][0] >= 0) != r["y"]) for r in chosen])
    if len(Counter(y)) != 2 or min(Counter(y).values()) < 20:
        raise ValueError("SECTOR_MODEL_ERROR_LABELS_INSUFFICIENT")
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
        raise ValueError("SECTOR_MODEL_NORMALIZED_INPUT_INVALID")
    scores = trained["model"].predict_proba(x)[:, 1]
    if not np.isfinite(scores).all() or min(scores) < 0 or max(scores) > 1:
        raise ValueError("SECTOR_MODEL_SCORE_INVALID")
    result = []
    for z, score in zip(values, scores, strict=True):
        value = {"research_score": float(score), "kind": "UNCALIBRATED_RESEARCH_SCORE"}
        baseline, flipped = int(z[0] >= 0), bool(score > FLIP_THRESHOLD)
        value |= {
            "prediction": 1 - baseline if flipped else baseline,
            "baseline_prediction": baseline,
            "flipped": flipped,
            "kind": "UNCALIBRATED_BASELINE_ERROR_SCORE",
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
            raise ValueError("SECTOR_MODEL_CHECKPOINT_CHANGED")
        return joblib.load(path)
    if attempt.exists():
        raise ValueError("SECTOR_MODEL_PREVIOUS_FIT_INTERRUPTED")
    base.save(attempt, {"at": base.now().isoformat(), "name": name, "cutoff": cutoff})
    chosen = adaptive.training_rows(rows, cutoff, "MONTHLY_BAL504")
    group = rows[0]["group"]
    if label.startswith("current-"):
        expected = hk.models()[1]["HK_EXTRA12_ERR504"][group]["fit_hash"]
    else:
        quarter = label.split("-")[0]
        expected = base.read(hk.root() / f"training/{quarter}-{group}-HK_EXTRA12_ERR504.json")["fit_hash"]
    if base.digest([r | {"z": r["z"][:12]} for r in chosen]) != expected:
        raise ValueError("SECTOR_CONTROL_TRAINING_ROWS_CHANGED")
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
            raise ValueError("ROUND_37_INPUT_CHANGED")
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
                output["HK_EXTRA12_ERR504"].extend(base.read(hk.root() / f"folds/{q}-{group}-HK_EXTRA12_ERR504.json"))
            base.save(root() / "progress.json", {"at": base.now().isoformat(), "quarter": q}, replace=True)
        output["ALWAYS_UP"] = [r | {"prediction": 1} for r in output["SPX_SIGN"]]
        expected = sorted((r["code"], r["u"], r["y"]) for r in output["SPX_SIGN"])
        if len(expected) != 5670 or any(
            sorted((r["code"], r["u"], r["y"]) for r in v) != expected for v in output.values()
        ):
            raise ValueError("ROUND_37_COMMON_EXAM_CHANGED")
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
        raise ValueError("ROUND_37_MODEL_OR_CODE_CHANGED")
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
    sector_input = sector_data.capture(base.now())
    deadline = datetime.combine(date.fromisoformat(target), time(8, 30), base.ZONE)
    with threadpool_limits(limits=2):
        for path in paths:
            if base.now() >= deadline:
                break
            p5, _, source, original = dual.read_parent(path)
            z = live_vector(source, original, hk_input["rows"], sector_input["rows"])
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
            raise ValueError("ROUND_37_PARENT_OR_MODEL_CHANGED")
        hk_input = hk_live.load(value["u"])
        sector_input = sector_data.load(value["u"])
        if value["sector_input_hash"] != base.digest(sector_input):
            raise ValueError("SECTOR_MODEL_LIVE_INPUT_CHANGED")
        if value["hk_input_hash"] != base.digest(hk_input):
            raise ValueError("SECTOR_MODEL_LIVE_INPUT_CHANGED")
        if not np.allclose(
            value["z"], live_vector(source, original, hk_input["rows"], sector_input["rows"]), rtol=0, atol=1e-12
        ):
            raise ValueError("ROUND_37_VECTOR_CHANGED")
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
