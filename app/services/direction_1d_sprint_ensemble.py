"""第二十三轮：三个已冻结纠错模型的固定等权分数组合。

成员为原8项、港股12项和VIX14项纠错树；不重新拟合成员或学习组合权重。
只计算原错误分数的算术平均，沿用0.55改判门槛和原始单位净值的一日目标。
"""

import hashlib
import shutil
from collections import defaultdict
from datetime import date, datetime, time

import numpy as np
from threadpoolctl import threadpool_limits

from app.services import direction_1d_sprint as base
from app.services import direction_1d_sprint_correction as correction
from app.services import direction_1d_sprint_dual_us as dual
from app.services import direction_1d_sprint_fund_response as previous
from app.services import direction_1d_sprint_hk as hk
from app.services import direction_1d_sprint_hk_live as hk_live
from app.services import direction_1d_sprint_overnight as overnight
from app.services import direction_1d_sprint_return_target as regression
from app.services import direction_1d_sprint_sequence as sequence
from app.services import direction_1d_sprint_sparse as sparse
from app.services import direction_1d_sprint_vix as previous_round
from app.services import direction_1d_sprint_vix_data as vix_data

CANDIDATES = ("MEAN3_ERR504",)
LEARNED = CANDIDATES
FIRST_TARGET = "2026-09-16"
FLIP_THRESHOLD = 0.55
MEMBERS = (("EXTRA8_ERR504", 8, correction), ("HK_EXTRA12_ERR504", 12, hk), ("VIX_HK14_ERR504", 14, previous_round))


def root():
    return base.ROOT / "round-23"


def active():
    regression.active()


def fingerprint():
    value = previous_round.fingerprint()
    for name in (
        "app/services/direction_1d_sprint_ensemble.py",
        "scripts/direction_1d_sprint_ensemble.py",
        "tests/test_direction_1d_sprint_ensemble.py",
    ):
        value["code"][name] = hashlib.sha256((base.PROJECT / name).read_bytes()).hexdigest()
    return value


def vector(x, t, u, markets, points):
    return hk.vector(x, t, u, markets) + vix_data.features(t, u, points)


def live_vector(source, original, markets, points):
    """复用父答案的净值和港股输入；VIX锚点只由基准日决定，不追逐最新隔夜值。"""
    sequence.live_vector(source, original)
    return vector(source["x"], original["base"], original["u"], markets, points)


def selected_features(z, name):
    if name not in CANDIDATES or len(z) != 14 or not np.isfinite(z).all():
        raise ValueError("ENSEMBLE_MODEL_INPUT_INVALID")
    return z


def dataset():
    return previous_round.dataset()


def plan():
    path = root() / "plan.json"
    if path.exists():
        value = base.read(path)
        if value["fingerprint"] != fingerprint() or value["calendar_hash"] != base.calendar()[1]:
            raise ValueError("ROUND_23_CODE_OR_CALENDAR_CHANGED")
        return value
    active()
    specs = {}
    cutoffs = set()
    for name, size, module in MEMBERS:
        result, _ = module.models()
        parent_plan = base.read(module.root() / "plan.json")
        cutoffs.add(parent_plan["current_fit_cutoff"])
        specs[name] = {
            "input_size": size,
            "result_hash": base.digest(result),
            "model_sha256": result["model_sha256"],
            "source_round": module.root().name,
        }
    if len(cutoffs) != 1:
        raise ValueError("ENSEMBLE_MEMBER_CUTOFF_MISMATCH")
    value = {
        "at": base.now().isoformat(),
        "fingerprint": fingerprint(),
        "candidates": CANDIDATES,
        "hypothesis": "Fixed mean of three comparable error forests may reduce single-feature-set sensitivity",
        "target": "same raw unit NAV next CN trading day, UP versus NON_UP; no horizon change",
        "members": specs,
        "member_fold_hashes": {
            str(path.relative_to(base.ROOT)).replace("\\", "/"): base.digest(base.read(path))
            for name, _, module in MEMBERS
            for path in sorted((module.root() / "folds").glob(f"*-{name}.json"))
        },
        "member_order": [name for name, _, _ in MEMBERS],
        "weights": [1 / 3] * 3,
        "rule": "arithmetic mean of3uncalibrated error scores, flip commonSPXbaseline iff mean>0.55",
        "training": "no new fitting; reuse36development member models and9currentmembermodels",
        "max_development_fits": 0,
        "max_current_fits": 0,
        "max_reproduction_count": 1,
        "current_fit_cutoff": next(iter(cutoffs)),
        "selection": "one fixed ensemble; no weight learning, member search or threshold search",
        "expected_questions": 5670,
        "expected_dates": 199,
        "controls": ["EXTRA8_ERR504", "HK_EXTRA12_ERR504", "VIX_HK14_ERR504", "SPX_SIGN", "ALWAYS_UP"],
        "calendar_hash": base.calendar()[1],
        "input_hashes": previous_round.plan()["input_hashes"],
        "first_forward_target": FIRST_TARGET,
        "new_provider_calls": 0,
        "new_cost_cny": 0,
        "this_round_2026_scores_read": False,
        "historical_availability": "RECONSTRUCTED_NOT_TRUE_FORWARD",
        "live_inputs": "shared original/HK/VIX verified input; VIX anchor<=T15 and actualreceipt/readback<U08:30",
    }
    base.save(path, value)
    for name in value["fingerprint"]["code"]:
        dest = root() / "code" / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(base.PROJECT / name, dest)
    return value


def blend(scores, baseline):
    """仅接受三个有效分数；不能在某个成员失败时悄悄退回两模型平均。"""
    values = np.asarray(scores, dtype=float)
    if values.shape != (3,) or not np.isfinite(values).all() or min(values) < 0 or max(values) > 1:
        raise ValueError("ENSEMBLE_MEMBER_SCORES_INVALID")
    if baseline not in (0, 1):
        raise ValueError("ENSEMBLE_BASELINE_INVALID")
    score = float(values.mean())
    flipped = score > FLIP_THRESHOLD
    return {
        "prediction": 1 - baseline if flipped else baseline,
        "baseline_prediction": baseline,
        "flipped": flipped,
        "research_score": score,
        "kind": "UNCALIBRATED_FROZEN_ENSEMBLE_SCORE",
        "member_scores": {name: float(v) for (name, _, _), v in zip(MEMBERS, values, strict=True)},
    }


def batch_answers(values, name, trained):
    if not values:
        return []
    z = np.asarray([selected_features(row, name) for row in values])
    member_scores = []
    for member, size, _ in MEMBERS:
        fitted = trained[member]
        x = (z[:, :size] - fitted["mean"]) / fitted["scale"]
        if not np.isfinite(x).all():
            raise ValueError("ENSEMBLE_MEMBER_INPUT_INVALID")
        member_scores.append(fitted["model"].predict_proba(x)[:, 1])
    return [blend(scores, int(row[0] >= 0)) for row, scores in zip(z, np.asarray(member_scores).T, strict=True)]


def answer(z, name, trained=None):
    return batch_answers([z], name, trained)[0]


def train():
    """从固定成员的已保存开发答案生成组合结果，明确记录新增拟合次数为0。"""
    p = plan()
    if (root() / "result.json").exists():
        return models()[0]
    active()
    output = defaultdict(list)
    groups = ("CN_BOND", "CN_EQUITY", "CN_MIXED")
    for q in range(1, 5):
        active()
        for group in groups:
            member_rows = {}
            for name, _, module in MEMBERS:
                source_path = module.root() / f"folds/{q}-{group}-{name}.json"
                raw = base.read(source_path)
                key = str(source_path.relative_to(base.ROOT)).replace("\\", "/")
                if base.digest(raw) != p["member_fold_hashes"][key]:
                    raise ValueError("ENSEMBLE_FROZEN_MEMBER_ANSWERS_CHANGED")
                indexed = {(r["code"], r["u"]): r for r in raw}
                if len(indexed) != len(raw):
                    raise ValueError("ENSEMBLE_DUPLICATE_MEMBER_QUESTION")
                member_rows[name] = indexed
                output[name].extend(raw)
            expected = sorted(next(iter(member_rows.values())))
            if any(sorted(v) != expected for v in member_rows.values()):
                raise ValueError("ENSEMBLE_MEMBER_QUESTION_SET_CHANGED")
            combined = []
            for key in expected:
                rows = [member_rows[name][key] for name, _, _ in MEMBERS]
                if (
                    len(
                        {
                            (r["y"], r["baseline_prediction"], r["family"], r["group"], r["actual_direction"])
                            for r in rows
                        }
                    )
                    != 1
                ):
                    raise ValueError("ENSEMBLE_MEMBER_LABEL_OR_BASELINE_CHANGED")
                combined.append(
                    {k: rows[0][k] for k in ("code", "family", "group", "u", "y", "actual_direction")}
                    | blend([r["research_score"] for r in rows], rows[0]["baseline_prediction"])
                )
            path = root() / f"folds/{q}-{group}-{CANDIDATES[0]}.json"
            if path.exists():
                if base.read(path) != combined:
                    raise ValueError("ENSEMBLE_SAVED_ANSWER_CHANGED")
            else:
                base.save(path, combined)
            output[CANDIDATES[0]].extend(combined)
            output["SPX_SIGN"].extend(base.read(sparse.root() / f"folds/{q}-{group}-SPX_SIGN.json"))
    output["ALWAYS_UP"] = [r | {"prediction": 1} for r in output["SPX_SIGN"]]
    expected = sorted((r["code"], r["u"], r["y"]) for r in output["SPX_SIGN"])
    if len(expected) != 5670 or any(
        sorted((r["code"], r["u"], r["y"]) for r in v) != expected for v in output.values()
    ):
        raise ValueError("ROUND_23_COMMON_EXAM_CHANGED")
    manifest = {
        "kind": "FROZEN_MEMBER_REFERENCE_MANIFEST",
        "members": p["members"],
        "rule": p["rule"],
        "weights": p["weights"],
        "plan_hash": base.digest(p),
    }
    path = root() / "ensemble-manifest.json"
    if path.exists():
        if base.read(path) != manifest:
            raise ValueError("ENSEMBLE_MANIFEST_CHANGED")
    else:
        base.save(path, manifest)
    result = {
        "at": base.now().isoformat(),
        "winner": CANDIDATES[0],
        "fingerprint": fingerprint(),
        "metrics": {n: base.metrics(v) for n, v in output.items()},
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
        "model_hash_kind": "ENSEMBLE_MANIFEST_FILE_SHA256_WITH_FROZEN_MEMBER_MODEL_HASHES",
        "plan_hash": base.digest(p),
        "calendar_hash": p["calendar_hash"],
        "development_fits": 0,
        "current_fits": 0,
        "reused_development_member_models": 36,
        "reused_current_member_models": 9,
        "this_round_2026_scores_read": False,
        "kind": "DEVELOPMENT_ONLY_AFTER_RESERVED_AUDIT_CONSUMPTION",
        "new_cost_cny": 0,
        "first_forward_target": FIRST_TARGET,
    }
    base.save(root() / "result.json", result)
    return result


def models():
    result = base.read(root() / "result.json")
    path = root() / "ensemble-manifest.json"
    if (
        result["fingerprint"] != fingerprint()
        or result["calendar_hash"] != base.calendar()[1]
        or hashlib.sha256(path.read_bytes()).hexdigest() != result["model_sha256"]
    ):
        raise ValueError("ROUND_23_MANIFEST_OR_CODE_CHANGED")
    manifest = base.read(path)
    if manifest["plan_hash"] != result["plan_hash"]:
        raise ValueError("ENSEMBLE_MANIFEST_PLAN_CHANGED")
    source_bundles = {}
    for name, _, module in MEMBERS:
        spec = manifest["members"][name]
        source_result, source_bundle = module.models()
        if base.digest(source_result) != spec["result_hash"] or source_result["model_sha256"] != spec["model_sha256"]:
            raise ValueError("ENSEMBLE_SOURCE_MODEL_CHANGED")
        source_bundles[name] = source_bundle[name]
    groups = set(next(iter(source_bundles.values())))
    if any(set(v) != groups for v in source_bundles.values()):
        raise ValueError("ENSEMBLE_MEMBER_GROUPS_CHANGED")
    bundle = {CANDIDATES[0]: {g: {n: models[g] for n, models in source_bundles.items()} for g in groups}}
    return result, bundle


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
    vix_input = vix_data.capture(base.now())
    deadline = datetime.combine(date.fromisoformat(target), time(8, 30), base.ZONE)
    with threadpool_limits(limits=2):
        for path in paths:
            if base.now() >= deadline:
                break
            p5, _, source, original = dual.read_parent(path)
            z = live_vector(source, original, hk_input["rows"], vix_input["rows"])
            choices = {n: answer(z, n, bundle[n][original["group"]]) for n in CANDIDATES}
            value = {
                "at": base.now().isoformat(),
                "u": target,
                "code": original["code"],
                "parent_hash": base.digest(p5),
                "source_hash": base.digest(source),
                "hk_input_hash": base.digest(hk_input),
                "vix_input_hash": base.digest(vix_input),
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
            raise ValueError("ROUND_23_PARENT_OR_MODEL_CHANGED")
        hk_input = hk_live.load(value["u"])
        vix_input = vix_data.load(value["u"])
        if value["vix_input_hash"] != base.digest(vix_input):
            raise ValueError("ENSEMBLE_MODEL_LIVE_INPUT_CHANGED")
        if value["hk_input_hash"] != base.digest(hk_input):
            raise ValueError("ENSEMBLE_MODEL_LIVE_INPUT_CHANGED")
        if not np.allclose(
            value["z"], live_vector(source, original, hk_input["rows"], vix_input["rows"]), rtol=0, atol=1e-12
        ):
            raise ValueError("ROUND_23_VECTOR_CHANGED")
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
