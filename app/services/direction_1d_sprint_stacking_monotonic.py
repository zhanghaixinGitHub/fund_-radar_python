"""第五十九轮：用严格按时间留出的幅度预测与三票差额训练一日组合模型。

基础模型未见过被预测题的标签，第二层只学习两个预测信号的关系，不直接使用未来涨跌幅。
保留全部30只基金、5670道开发题，不用历史多轮择优冒充独立未来验证。
"""

import hashlib
import shutil
import warnings
from collections import Counter, defaultdict
from datetime import date, datetime, time

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits

from app.services import direction_1d_sprint as base
from app.services import direction_1d_sprint_cnya_data as cnya_data
from app.services import direction_1d_sprint_dual_us as dual
from app.services import direction_1d_sprint_etf_joint as majority
from app.services import direction_1d_sprint_etf_runtime_v2 as runtime
from app.services import direction_1d_sprint_fund_response as previous
from app.services import direction_1d_sprint_hk as hk
from app.services import direction_1d_sprint_hk_live as hk_live
from app.services import direction_1d_sprint_overnight as overnight
from app.services import direction_1d_sprint_return_target as regression
from app.services import direction_1d_sprint_sector_return as previous_round
from app.services import direction_1d_sprint_sequence as sequence
from app.services import direction_1d_sprint_sparse as sparse
from app.services import direction_1d_sprint_stacking as prior_round
from app.services import direction_1d_sprint_stacking_oof as oof
from app.services import direction_1d_sprint_us_etf_data as us_etf_data
from app.services import direction_1d_sprint_us_sector_etf_data as sectors

CANDIDATES = ("MONOTONIC_STACK_HGB2_504",)
FEATURE_INDICES = (0, 12, 17, 20, 21)
SHRINKAGE = 0.25
CODE_ORDER = (
    "001021",
    "001632",
    "002112",
    "002170",
    "004237",
    "004605",
    "005187",
    "005284",
    "005312",
    "006038",
    "006730",
    "007045",
    "007509",
    "007832",
    "007950",
    "008164",
    "008888",
    "008960",
    "010737",
    "011036",
    "011103",
    "013180",
    "013275",
    "013330",
    "013472",
    "014156",
    "015596",
    "016008",
    "017493",
    "160323",
)
LEARNED = CANDIDATES
FIRST_TARGET = "2026-09-16"
UP_THRESHOLD = 0.5
CONTROL = "HK_EXTRA12_ERR504"


def root():
    return base.ROOT / "round-59"


def active():
    regression.active()


def fingerprint():
    value = prior_round.fingerprint()
    for name in (
        "app/services/direction_1d_sprint_stacking_monotonic.py",
        "scripts/direction_1d_sprint_stacking_monotonic.py",
        "tests/test_direction_1d_sprint_stacking_monotonic.py",
    ):
        value["code"][name] = hashlib.sha256((base.PROJECT / name).read_bytes()).hexdigest()
    return value


def vector(x, code, t, u, markets, etf_points, cn_points, sector_points):
    return previous_round.vector(x, code, t, u, markets, etf_points, cn_points, sector_points)


def live_vector(source, original, markets, etf_points, cn_points, sector_points):
    return previous_round.live_vector(source, original, markets, etf_points, cn_points, sector_points)


def available(z):
    return previous_round.available(z)


def selected_features(z, name):
    if name not in CANDIDATES:
        raise ValueError("STACKING_CANDIDATE_INVALID")
    return previous_round.selected_features(z, previous_round.CANDIDATES[1])


def dataset():
    return previous_round.dataset()


def plan():
    path = root() / "plan.json"
    if path.exists():
        value = base.read(path)
        if value["fingerprint"] != fingerprint() or value["calendar_hash"] != base.calendar()[1]:
            raise ValueError("ROUND_59_CODE_OR_CALENDAR_CHANGED")
        return value
    active()
    prior_round.models()
    prior = prior_round.plan()
    proposal = base.read(root() / "proposal-before-implementation.json")
    _, manifest = oof.load()
    prior_analysis = base.read(prior_round.root() / "analysis-result.json")
    expected = {
        "max_reproduction_count": 1,
        "new_cost_cny": 0,
        "new_inner_base_fits": 0,
        "new_meta_current_fits": 3,
        "new_meta_development_fits": 12,
        "new_provider_requests": 0,
        "reused_inner_checkpoints": 51,
        "total_new_fits": 15,
    }
    if (
        proposal["budget"] != expected
        or proposal["candidate"] != CANDIDATES[0]
        or proposal["oof_manifest_hash"] != base.digest(manifest)
        or proposal["prior_analysis_hash"] != base.digest(prior_analysis)
        or proposal["current_fit_cutoff"] != prior["current_fit_cutoff"]
    ):
        raise ValueError("STACKING_PROPOSAL_CHANGED")
    value = prior | {
        "at": base.now().isoformat(),
        "fingerprint": fingerprint(),
        "candidates": CANDIDATES,
        "hypothesis": proposal["hypothesis"],
        "target": "Raw adjacentCN U NAV_UP/NON_UP;score>=.5 UP,flatseparate",
        "training_target": "raw_NAV_UP",
        "features": proposal["features"],
        "recipe": proposal["recipe"],
        "inner_recipe": prior["inner_recipe"],
        "max_development_fits": 12,
        "max_current_fits": 3,
        "new_inner_base_fits": 0,
        "reused_inner_checkpoints": 51,
        "total_new_fits": 15,
        "input_hashes": prior["input_hashes"]
        | {
            "round-59/proposal-before-implementation.json": base.digest(proposal),
            "round-58/oof/manifest.json": base.digest(manifest),
            "round-58/analysis-result.json": base.digest(prior_analysis),
        },
        "control": "R58LR2, R50majority andR57ridge155;same rawrows andOOFfeatures",
        "live_budget": "Reuse allpreviousquotes/currentbase model,0extraGET",
        "new_2026_audit_metrics": False,
        "limits": "2026OOF currenttrainingfeaturesonly;developmentisnotfutureproof",
    }
    for unused in ("feature_indices", "shrinkage", "normalization"):
        value.pop(unused, None)
    value["normalization"] = "Two training-only scales,no centering;innerbase keepsfrozenR57recipe"
    base.save(path, value)
    for name in value["fingerprint"]["code"]:
        dest = root() / "code" / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(base.PROJECT / name, dest)
    return value


def fit(rows, name, cutoff):
    """在相同成熟题上学习浅层单调组合，基础模型和输入均复用第58轮。"""
    if name not in CANDIDATES:
        raise ValueError("STACKING_RECIPE_INVALID")
    selected = oof.chosen(rows, cutoff)
    x, manifest_hash = oof.training_features(selected)
    y = np.asarray([r["y"] for r in selected])
    weights = regression.weights(selected)
    if len({r["u"] for r in selected}) < 120 or len(Counter(y)) != 2 or min(Counter(y).values()) < 20:
        raise ValueError("STACKING_TRAINING_INSUFFICIENT")
    _, unused_mean, scale = sequence.normalize_training(x, weights)
    active()
    with warnings.catch_warnings(), threadpool_limits(limits=2):
        warnings.simplefilter("error", ConvergenceWarning)
        # 两个输入都保持单调递增：其他条件相同时，更偏涨的输入不能被拟合成更偏跌。
        # 只允许三叶浅树，固定60轮，不按开发期成绩搜索深度、迭代次数或阈值。
        model = HistGradientBoostingClassifier(
            loss="log_loss",
            max_iter=60,
            max_leaf_nodes=3,
            min_samples_leaf=80,
            learning_rate=0.05,
            l2_regularization=10.0,
            monotonic_cst=[1, 1],
            early_stopping=False,
            random_state=17,
            max_bins=255,
        )
        model.fit(x / np.asarray(scale), y, sample_weight=weights)
    active()
    return {
        "model": model,
        "mean": [0.0, 0.0],
        "scale": scale,
        "training_feature_mean_not_subtracted": unused_mean,
        "fit_hash": base.digest(selected),
        "feature_hash": base.digest(x.tolist()),
        "oof_manifest_hash": manifest_hash,
        "fit_rows": len(selected),
        "fit_dates": len({r["u"] for r in selected}),
        "fit_end": max(r["u"] for r in selected),
        "max_mature_date": max(r["mature"] for r in selected),
        "cutoff": cutoff,
        "weighted_up_rate": float(np.average(y, weights=weights)),
        "training_target": "raw_NAV_UP",
    }


def batch_answers(values, name, trained):
    if not values:
        return []
    for z in values:
        selected_features(z, name)
    result = majority.batch_answers([z[:20] for z in values], majority.CANDIDATES[0], trained)
    indices = [i for i, z in enumerate(values) if available(z)]
    if indices:
        head = trained["us_etf"]
        raw = [values[i] for i in indices]
        if head["mean"] != [0.0, 0.0] or head["model"].n_features_in_ != 2:
            raise ValueError("STACKING_META_SCHEMA_CHANGED")
        x = np.asarray(oof.meta_vectors(raw, oof.base_scores(raw, trained["ridge_base"])))
        scores = head["model"].predict_proba(x / np.asarray(head["scale"]))[:, 1]
        if not np.isfinite(scores).all() or min(scores) < 0 or max(scores) > 1:
            raise ValueError("STACKING_SCORE_INVALID")
        for i, score in zip(indices, scores, strict=True):
            baseline, prediction = int(values[i][0] >= 0), int(score >= UP_THRESHOLD)
            result[i] = {
                "research_score": float(score),
                "kind": "UNCALIBRATED_UP_SCORE",
                "prediction": prediction,
                "baseline_prediction": baseline,
                "flipped": prediction != baseline,
            }
    return result


def answer(z, name, trained=None):
    return batch_answers([z], name, trained)[0]


def fit_checkpoint(rows, name, cutoff, label):
    path = root() / "checkpoints" / f"{label}.joblib"
    receipt = path.with_suffix(".json")
    attempt = path.with_suffix(".attempt.json")
    if receipt.exists():
        value = base.read(receipt)
        if (
            value["name"] != name
            or value["cutoff"] != cutoff
            or value["sha256"] != hashlib.sha256(path.read_bytes()).hexdigest()
        ):
            raise ValueError("STACKING_CHECKPOINT_CHANGED")
        return joblib.load(path)
    if attempt.exists():
        raise ValueError("STACKING_FIT_INTERRUPTED_NO_AUTORETRY")
    group = rows[0]["group"]
    base_name = previous_round.CANDIDATES[1]
    old_path = previous_round.root() / "checkpoints" / f"{label.replace(name, base_name)}.joblib"
    old_meta = base.read(old_path.with_suffix(".json"))
    if old_meta["cutoff"] != cutoff or old_meta["sha256"] != hashlib.sha256(old_path.read_bytes()).hexdigest():
        raise ValueError("STACKING_OUTER_BASE_CHANGED")
    old = joblib.load(old_path)
    selected = oof.chosen(rows, cutoff)
    if old["us_etf"]["fit_hash"] != base.digest(selected):
        raise ValueError("STACKING_OUTER_ROWS_CHANGED")
    base.save(attempt, {"at": base.now().isoformat(), "name": name, "cutoff": cutoff, "new_fit": True})
    head = fit(rows, name, cutoff)
    # 与上一轮保持完全相同的训练题和两列时间留出输入，差异仅为组合模型。
    prior_name = prior_round.CANDIDATES[0]
    prior_path = prior_round.root() / "checkpoints" / f"{label.replace(name, prior_name)}.joblib"
    prior_receipt = base.read(prior_path.with_suffix(".json"))
    if prior_receipt["sha256"] != hashlib.sha256(prior_path.read_bytes()).hexdigest():
        raise ValueError("MONOTONIC_PRIOR_CHECKPOINT_CHANGED")
    prior_head = joblib.load(prior_path)["us_etf"]
    for key in ("fit_hash", "feature_hash", "oof_manifest_hash", "cutoff", "scale"):
        if head[key] != prior_head[key]:
            raise ValueError("MONOTONIC_PAIRED_INPUT_CHANGED")
    result = {
        "control": old["control"],
        "us_etf": head,
        "ridge_base": old["us_etf"],
        "group": group,
        "control_fit_hash": old["control_fit_hash"],
        "base_checkpoint_sha256": old_meta["sha256"],
        "new_fit_count": 1,
    }
    joblib.dump(result, path)
    base.save(
        receipt,
        {
            "at": base.now().isoformat(),
            "name": name,
            "cutoff": cutoff,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        },
    )
    return result


def train():
    p = plan()
    if (root() / "result.json").exists():
        return models()[0]
    active()
    for name, expected in p["input_hashes"].items():
        if base.digest(base.read(base.ROOT / name)) != expected:
            raise ValueError("ROUND_59_INPUT_CHANGED")
    rows, proofs = dataset()
    # 复用已冻结的51个内部窗口；本轮禁止缺失时再训练基础模型。
    _, oof_manifest = oof.load()
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
                        trained = fit_checkpoint(
                            [r for r in rows if r["group"] == group], name, start, f"{q}-{group}-{name}"
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
                output[majority.LEARNED[0]].extend(
                    base.read(majority.root() / f"folds/{q}-{group}-{majority.LEARNED[0]}.json")
                )
                parent_candidate = previous_round.CANDIDATES[1]
                output[parent_candidate].extend(
                    base.read(previous_round.root() / f"folds/{q}-{group}-{parent_candidate}.json")
                )
            base.save(root() / "progress.json", {"at": base.now().isoformat(), "quarter": q}, replace=True)
        output["ALWAYS_UP"] = [r | {"prediction": 1} for r in output["SPX_SIGN"]]
        expected = sorted((r["code"], r["u"], r["y"]) for r in output["SPX_SIGN"])
        if len(expected) != 5670 or any(
            sorted((r["code"], r["u"], r["y"]) for r in v) != expected for v in output.values()
        ):
            raise ValueError("ROUND_59_COMMON_EXAM_CHANGED")
        metrics = {n: base.metrics(v) for n, v in output.items()}
        winner = max(CANDIDATES, key=lambda n: (metrics[n]["accuracy"], -CANDIDATES.index(n)))
        active()
        bundle = {
            n: {
                g: fit_checkpoint([r for r in rows if r["group"] == g], n, p["current_fit_cutoff"], f"current-{g}-{n}")
                for g in groups
            }
            for n in CANDIDATES
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
        "inner_base_fits": 0,
        "total_new_fits": 15,
        "oof_manifest_hash": base.digest(oof_manifest),
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
        raise ValueError("ROUND_59_MODEL_OR_CODE_CHANGED")
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
                answer(row["z"], name, bundle[name][row["group"]])
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
    sector_input = sectors.capture(base.now())
    cn_input_path = cnya_data.root() / target / "input.json"
    cn_input = cnya_data.load(target) if cn_input_path.exists() else None
    if hk_input is None or us_etf_input is None or cn_input is None or sector_input is None:
        return report()
    deadline = datetime.combine(date.fromisoformat(target), time(8, 30), base.ZONE)
    with threadpool_limits(limits=2):
        for path in paths:
            if base.now() >= deadline:
                break
            p5, _, source, original = dual.read_parent(path)
            z = live_vector(
                source, original, hk_input["rows"], us_etf_input["rows"], cn_input["rows"], sector_input["rows"]
            )
            choices = {n: answer(z, n, bundle[n][original["group"]]) for n in CANDIDATES}
            value = {
                "at": base.now().isoformat(),
                "u": target,
                "code": original["code"],
                "parent_hash": base.digest(p5),
                "source_hash": base.digest(source),
                "hk_input_hash": base.digest(hk_input),
                "us_etf_input_hash": base.digest(us_etf_input),
                "cnya_input_hash": base.digest(cn_input),
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
            raise ValueError("ROUND_59_PARENT_OR_MODEL_CHANGED")
        hk_input = hk_live.load(value["u"])
        us_etf_input = us_etf_data.load(value["u"])
        cn_input = cnya_data.load(value["u"])
        sector_input = sectors.load(value["u"])
        if value["sector_input_hash"] != base.digest(sector_input):
            raise ValueError("STACKING_INPUT_CHANGED")
        if value["cnya_input_hash"] != base.digest(cn_input):
            raise ValueError("ETF_JOINT_CNYA_INPUT_CHANGED")
        if value["us_etf_input_hash"] != base.digest(us_etf_input):
            raise ValueError("US_ETF_MODEL_LIVE_INPUT_CHANGED")
        if value["hk_input_hash"] != base.digest(hk_input):
            raise ValueError("US_ETF_MODEL_LIVE_INPUT_CHANGED")
        if not np.allclose(
            value["z"],
            live_vector(
                source, original, hk_input["rows"], us_etf_input["rows"], cn_input["rows"], sector_input["rows"]
            ),
            rtol=0,
            atol=1e-12,
        ):
            raise ValueError("ROUND_59_VECTOR_CHANGED")
        expected_answers = {n: answer(value["z"], n, bundle[n][original["group"]]) for n in CANDIDATES}
        if not runtime.answers_match(value["answers"], expected_answers):
            raise ValueError("ROUND_59_SAVED_ANSWER_CHANGED")
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
