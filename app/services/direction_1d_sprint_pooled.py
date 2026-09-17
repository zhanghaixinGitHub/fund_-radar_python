"""第二十轮：合并资产组历史样本训练同一个一日纠错模型。

保持每组原有成熟样本选择，再合并训练；12项历史输入附加已知资产组标记。
只预测相邻交易日的原始单位净值方向，不把多基金样本当作独立目标日期。
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
from app.services import direction_1d_sprint_sequence as sequence
from app.services import direction_1d_sprint_sparse as sparse
from app.services import direction_1d_sprint_technical as previous_round

CANDIDATES = ("POOL_HK15_ERR504",)
LEARNED = CANDIDATES
FIRST_TARGET = "2026-09-16"
FLIP_THRESHOLD = 0.55
GROUPS = ("CN_BOND", "CN_EQUITY", "CN_MIXED")


def root():
    return base.ROOT / "round-20"


def active():
    regression.active()


def fingerprint():
    value = previous_round.fingerprint()
    for name in (
        "app/services/direction_1d_sprint_pooled.py",
        "scripts/direction_1d_sprint_pooled.py",
        "tests/test_direction_1d_sprint_pooled.py",
    ):
        value["code"][name] = hashlib.sha256((base.PROJECT / name).read_bytes()).hexdigest()
    return value


def group_vector(group):
    """顺序固定的资产类型标记；只使用既有基金分类，未知类型拒绝推理。"""
    if group not in GROUPS:
        raise ValueError("POOLED_GROUP_UNKNOWN")
    return [float(group == name) for name in GROUPS]


def vector(x, t, u, markets, group):
    return hk.vector(x, t, u, markets) + group_vector(group)


def live_vector(source, original, markets):
    """检查父答案的61条净值和公告时点，再沿用其已冻结资产组。"""
    sequence.live_vector(source, original)
    return vector(source["x"], original["base"], original["u"], markets, original["group"])


def selected_features(z, name):
    if name not in CANDIDATES or len(z) != 15 or not np.isfinite(z).all():
        raise ValueError("POOLED_INPUT_INVALID")
    if list(z[-3:]) not in [group_vector(group) for group in GROUPS]:
        raise ValueError("POOLED_GROUP_VECTOR_INVALID")
    return z


def dataset():
    rows, proof = hk.dataset()
    return [r | {"z": r["z"] + group_vector(r["group"])} for r in rows], proof | {
        "at": base.now().isoformat(),
        "group_order": list(GROUPS),
        "group_rows": dict(Counter(r["group"] for r in rows)),
        "target": "same rawNAV next CN day labels as round17; group inputs use existing classification",
    }


def pooled_training_rows(rows, cutoff):
    """逐组沿用原504成熟日期选择后取并集，确保只改变共享训练方式。

    每组的成熟日与题目集合不变；合并后自然按日期和产品家族加权，
    不强行平衡基金类型或错误类别。缺少任何既定资产组时明确失败。
    """
    if {r["group"] for r in rows} != set(GROUPS):
        raise ValueError("POOLED_GROUP_COVERAGE_INVALID")
    chosen = []
    for group in GROUPS:
        chosen.extend(adaptive.training_rows([r for r in rows if r["group"] == group], cutoff, "MONTHLY_BAL504"))
    for row in chosen:
        if row["z"][-3:] != group_vector(row["group"]):
            raise ValueError("POOLED_TRAINING_GROUP_CHANGED")
    return sorted(chosen, key=lambda r: (r["u"], r["family"], r["code"]))


def plan():
    path = root() / "plan.json"
    if path.exists():
        value = base.read(path)
        if value["fingerprint"] != fingerprint() or value["calendar_hash"] != base.calendar()[1]:
            raise ValueError("ROUND_20_CODE_OR_CALENDAR_CHANGED")
        return value
    active()
    # 冻结新方案前核对完整上一轮源码链，旧模型不得被改写。
    if base.read(previous_round.root() / "result.json")["fingerprint"] != previous_round.fingerprint():
        raise ValueError("POOLED_PREVIOUS_FINGERPRINT_CHANGED")
    parent, _ = hk.models()
    value = {
        "at": base.now().isoformat(),
        "fingerprint": fingerprint(),
        "candidates": CANDIDATES,
        "hypothesis": (
            "Pool same mature group samples to share error structure across assets; known group flags retained"
        ),
        "base_candidate": "HK_EXTRA12_ERR504",
        "parent_result_hash": base.digest(parent),
        "target": "raw unit NAV T to adjacent CN trading day U, strictly up versus NON_UP; flat separate",
        "feature_order": "original round17 HK12, then group one-hot CN_BOND/CN_EQUITY/CN_MIXED",
        "group_classification": "existing frozen snapshot classification; no additional historical vintage claim",
        "extra_trees": {
            "trees": 256,
            "depth": 6,
            "min_leaf": 80,
            "max_features": 1.0,
            "bootstrap": False,
            "seed": 17,
            "threads": 2,
        },
        "weights": "natural date/family over pooled rows; no class or group balancing",
        "normalization": "weighted centered-square scale from pooled training rows only, same round12 implementation",
        "training_dates": "same per-group last504mature target dates as round17, then union; merged count recorded",
        "group_order": GROUPS,
        "model_sharing": "one pooled model per quarter/currentcutoff versus original3separate group models",
        "flip_threshold": 0.55,
        "quarterly_refit": True,
        "development_year": 2025,
        "max_development_fits": 4,
        "max_current_fits": 1,
        "max_reproduction_count": 1,
        "expected_questions": 5670,
        "expected_dates": 199,
        "controls": ["HK_EXTRA12_ERR504", "SPX_SIGN", "ALWAYS_UP"],
        "selection": "one predeclared candidate; no parameter or threshold search",
        "calendar_hash": base.calendar()[1],
        "input_hashes": hk.plan()["input_hashes"],
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
    """原分组样本并集拟合一个共享模型，记录每组题量与选择摘要。"""
    if name not in CANDIDATES:
        raise ValueError("POOLED_RECIPE_INVALID")
    chosen = pooled_training_rows(rows, cutoff)
    x = np.asarray([selected_features(r["z"], name) for r in chosen])
    y = np.asarray([int(int(r["z"][0] >= 0) != r["y"]) for r in chosen])
    if len(Counter(y)) != 2 or min(Counter(y).values()) < 20:
        raise ValueError("POOLED_ERROR_LABELS_INSUFFICIENT")
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
        "group_rows": dict(Counter(r["group"] for r in chosen)),
        "group_dates": {g: len({r["u"] for r in chosen if r["group"] == g}) for g in GROUPS},
        "group_selection_hashes": {
            g: base.digest([r | {"z": r["z"][:12]} for r in chosen if r["group"] == g]) for g in GROUPS
        },
    }


def batch_answers(values, name, trained):
    """同一模型批量推理，避免逐题调度线程；与实时单题使用同一阈值实现。"""
    if not values:
        return []
    x = (np.asarray([selected_features(z, name) for z in values]) - trained["mean"]) / trained["scale"]
    if not np.isfinite(x).all():
        raise ValueError("POOLED_MODEL_NORMALIZED_INPUT_INVALID")
    scores = trained["model"].predict_proba(x)[:, 1]
    if not np.isfinite(scores).all() or min(scores) < 0 or max(scores) > 1:
        raise ValueError("POOLED_MODEL_SCORE_INVALID")
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
            raise ValueError("POOLED_MODEL_CHECKPOINT_CHANGED")
        return joblib.load(path)
    if attempt.exists():
        raise ValueError("POOLED_MODEL_PREVIOUS_FIT_INTERRUPTED")
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
            raise ValueError("ROUND_20_INPUT_CHANGED")
    rows, proof = dataset()
    if not (root() / "training-question-proof.json").exists():
        base.save(root() / "training-question-proof.json", proof)
    output = defaultdict(list)
    with threadpool_limits(limits=2):
        for q in range(1, 5):
            active()
            start, end = f"2025-{q * 3 - 2:02d}-01", "2026-01-01" if q == 4 else f"2025-{q * 3 + 1:02d}-01"
            exam = [r for r in rows if start <= r["u"] < end]
            for name in CANDIDATES:
                path = root() / f"folds/{q}-ALL-{name}.json"
                if path.exists():
                    scored = base.read(path)
                else:
                    # 核对每组实际训练题目与原第17轮完全相同，不只核对504日数量。
                    for group in GROUPS:
                        old_rows = adaptive.training_rows(
                            [r for r in rows if r["group"] == group], start, "MONTHLY_BAL504"
                        )
                        expected = base.read(hk.root() / f"training/{q}-{group}-HK_EXTRA12_ERR504.json")["fit_hash"]
                        if base.digest([r | {"z": r["z"][:12]} for r in old_rows]) != expected:
                            raise ValueError("POOLED_GROUP_SELECTION_CHANGED")
                    trained = fit_checkpoint(rows, name, start, f"{q}-ALL-{name}")
                    scored = [
                        {k: r[k] for k in ("code", "family", "group", "u", "y", "actual_direction")} | choice
                        for r, choice in zip(exam, batch_answers([r["z"] for r in exam], name, trained), strict=True)
                    ]
                    base.save(path, scored)
                    base.save(
                        root() / f"training/{q}-ALL-{name}.json", {k: v for k, v in trained.items() if k != "model"}
                    )
                output[name].extend(scored)
            for group in GROUPS:
                output["SPX_SIGN"].extend(base.read(sparse.root() / f"folds/{q}-{group}-SPX_SIGN.json"))
                output["HK_EXTRA12_ERR504"].extend(base.read(hk.root() / f"folds/{q}-{group}-HK_EXTRA12_ERR504.json"))
            base.save(root() / "progress.json", {"at": base.now().isoformat(), "quarter": q}, replace=True)
        output["ALWAYS_UP"] = [r | {"prediction": 1} for r in output["SPX_SIGN"]]
        expected = sorted((r["code"], r["u"], r["y"]) for r in output["SPX_SIGN"])
        if len(expected) != 5670 or any(
            sorted((r["code"], r["u"], r["y"]) for r in v) != expected for v in output.values()
        ):
            raise ValueError("ROUND_20_COMMON_EXAM_CHANGED")
        metrics = {n: base.metrics(v) for n, v in output.items()}
        old_bundle = hk.models()[1]["HK_EXTRA12_ERR504"]
        for group in GROUPS:
            old_rows = adaptive.training_rows(
                [r for r in rows if r["group"] == group], p["current_fit_cutoff"], "MONTHLY_BAL504"
            )
            if base.digest([r | {"z": r["z"][:12]} for r in old_rows]) != old_bundle[group]["fit_hash"]:
                raise ValueError("POOLED_CURRENT_GROUP_SELECTION_CHANGED")
        bundle = {n: {"POOLED": fit_checkpoint(rows, n, p["current_fit_cutoff"], f"current-ALL-{n}")} for n in LEARNED}
    path = root() / "models.joblib"
    joblib.dump(bundle, path)
    value = {
        "at": base.now().isoformat(),
        "winner": CANDIDATES[0],
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
            n: {g: base.metrics([r for r in v if r["group"] == g]) for g in GROUPS} for n, v in output.items()
        },
        "model_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "plan_hash": base.digest(p),
        "calendar_hash": p["calendar_hash"],
        "development_fits": 4,
        "current_fits": 1,
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
        raise ValueError("ROUND_20_MODEL_OR_CODE_CHANGED")
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
                answer(row["z"], name, bundle[name]["POOLED"] if name in LEARNED else None)
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
            z = live_vector(source, original, hk_input["rows"])
            choices = {n: answer(z, n, bundle[n]["POOLED"]) for n in CANDIDATES}
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
            raise ValueError("ROUND_20_PARENT_OR_MODEL_CHANGED")
        hk_input = hk_live.load(value["u"])
        if value["hk_input_hash"] != base.digest(hk_input):
            raise ValueError("POOLED_MODEL_LIVE_INPUT_CHANGED")
        if not np.allclose(value["z"], live_vector(source, original, hk_input["rows"]), rtol=0, atol=1e-12):
            raise ValueError("ROUND_20_VECTOR_CHANGED")
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
