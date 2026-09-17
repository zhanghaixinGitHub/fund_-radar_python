"""第六十六轮：三类基金共用一套市场信号参数，减少分别估计的波动。

每类原504个成熟日期的训练题保持完整，合并后采用自然日期家族权重；仅5次拟合。
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
from app.services import direction_1d_sprint_cnya_data as cnya_data
from app.services import direction_1d_sprint_dual_us as dual
from app.services import direction_1d_sprint_etf_decay as previous_round
from app.services import direction_1d_sprint_etf_joint as reference
from app.services import direction_1d_sprint_etf_runtime_v2 as runtime
from app.services import direction_1d_sprint_fund_response as previous
from app.services import direction_1d_sprint_hk as hk
from app.services import direction_1d_sprint_hk_live as hk_live
from app.services import direction_1d_sprint_overnight as overnight
from app.services import direction_1d_sprint_return_target as regression
from app.services import direction_1d_sprint_sequence as sequence
from app.services import direction_1d_sprint_sparse as sparse
from app.services import direction_1d_sprint_us_etf_data as us_etf_data

CANDIDATES = ("POOLED_SPX_FXI_CNYA_LR3_504",)
LEARNED = CANDIDATES
FIRST_TARGET = "2026-09-16"
UP_THRESHOLD = 0.5
CONTROL = "HK_EXTRA12_ERR504"


def root():
    return base.ROOT / "round-66"


def active():
    regression.active()


def fingerprint():
    value = previous_round.fingerprint()
    for name in (
        "app/services/direction_1d_sprint_etf_pooled.py",
        "scripts/direction_1d_sprint_etf_pooled.py",
        "tests/test_direction_1d_sprint_etf_pooled.py",
    ):
        value["code"][name] = hashlib.sha256((base.PROJECT / name).read_bytes()).hexdigest()
    return value


vector = reference.vector
live_vector = reference.live_vector
available = reference.available
dataset = reference.dataset


def selected_features(z, name):
    if name not in CANDIDATES:
        raise ValueError("ETF_POOLED_RECIPE_INVALID")
    return reference.selected_features(z, reference.LEARNED[0])


def pooled_rows(rows, cutoff):
    """分别取每类原504成熟日期，再合并；不能先合并截取而悄悄删掉某类原训练题。"""
    chosen = []
    for group in sorted({r["group"] for r in rows}):
        chosen.extend(
            adaptive.training_rows(
                [
                    r
                    for r in rows
                    if r["group"] == group and r["u"] < cutoff and r["mature"] < cutoff and available(r["z"])
                ],
                cutoff,
                "MONTHLY_BAL504",
            )
        )
    return chosen


def plan():
    path = root() / "plan.json"
    if path.exists():
        value = base.read(path)
        if value["fingerprint"] != fingerprint() or value["calendar_hash"] != base.calendar()[1]:
            raise ValueError("ROUND_66_CODE_OR_CALENDAR_CHANGED")
        return value
    active()
    previous_round.models()
    proposal = base.read(root() / "proposal-before-implementation.json")
    if (
        proposal["candidates"] != list(CANDIDATES)
        or proposal["max_development_fits"] != 4
        or proposal["max_current_fits"] != 1
    ):
        raise ValueError("ETF_POOLED_PROPOSAL_CHANGED")
    parent = reference.plan()
    value = {
        "at": base.now().isoformat(),
        "fingerprint": fingerprint(),
        "candidates": CANDIDATES,
        "hypothesis": proposal["hypothesis"],
        "recipe": proposal["recipe"],
        "target": "Raw unit NAV T to adjacent CN U UP/NON_UP, flatseparate",
        "features": "ExactR50raw20, selected0/12/17; one shared LR3, no group flags or fund flags",
        "training": "Unionexactpergroup504maturerows, sharedquarterlyhead, natural date-family weights and scales",
        "max_development_fits": 4,
        "max_current_fits": 1,
        "max_reproduction_count": 1,
        "expected_questions": 5670,
        "expected_dates": 199,
        "calendar_hash": parent["calendar_hash"],
        "input_hashes": parent["input_hashes"]
        | {"round-66/proposal-before-implementation.json": base.digest(proposal)},
        "current_fit_cutoff": parent["current_fit_cutoff"],
        "first_forward_target": FIRST_TARGET,
        "new_cost_cny": 0,
        "new_history_requests": 0,
        "this_round_2026_scores_read": False,
        "source_timing": parent["source_timing"],
        "live_budget": "ReuseR50existinginputs;zero additional GET",
        "source_limit": "Historical source first-publication unverified; repeated2025developmentnotfutureaccuracy",
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
    if name not in LEARNED:
        return None
    chosen = pooled_rows(rows, cutoff)
    if len({r["u"] for r in chosen}) < 120 or any(not available(r["z"]) for r in chosen):
        raise ValueError("US_ETF_TRAINING_SCOPE_OR_DATES_INVALID")
    x = np.asarray([selected_features(r["z"], name) for r in chosen])
    y = np.asarray([r["y"] for r in chosen])
    if len(Counter(y)) != 2 or min(Counter(y).values()) < 20:
        raise ValueError("US_ETF_DIRECT_UP_CLASSES_INSUFFICIENT")
    weights = regression.weights(chosen)
    # 仅用已成熟训练样本估计尺度；不居中且无截距，使零市场变化对应中性0.5分数。
    _, training_mean, scale = sequence.normalize_training(x, regression.weights(chosen))
    mean = [0.0] * x.shape[1]
    x = x / np.asarray(scale)
    active()
    with warnings.catch_warnings(), threadpool_limits(limits=2):
        warnings.simplefilter("error", ConvergenceWarning)
        model = LogisticRegression(C=0.1, max_iter=1500, random_state=17, fit_intercept=False)
        model.fit(x, y, sample_weight=weights)
    active()
    return {
        "model": model,
        "mean": mean,
        "scale": scale,
        "fit_hash": base.digest(chosen),
        "weight_hash": base.digest(weights.tolist()),
        "pooled_group_fit_hashes": {
            g: base.digest([r for r in chosen if r["group"] == g]) for g in sorted({r["group"] for r in chosen})
        },
        "scale_weighting": "POOLED_NATURAL_DATE_FAMILY",
        "fit_rows": len(chosen),
        "fit_dates": len({r["u"] for r in chosen}),
        "fit_end": max(r["u"] for r in chosen),
        "max_mature_date": max(r["mature"] for r in chosen),
        "cutoff": cutoff,
        "weighted_up_rate": float(np.average(y, weights=weights)),
        "training_feature_mean_not_subtracted": training_mean,
        "training_target": "raw_NAV_UP",
        "source_route": "BOTH_ETF_AVAILABLE_AFTER_CHINA_CLOSE",
    }


def batch_answers(values, name, trained):
    """推断及缺源回退完全复用原LR3路径，三个类别共用同一组已训练系数。"""
    if name not in CANDIDATES:
        raise ValueError("ETF_POOLED_RECIPE_INVALID")
    return reference.batch_answers(values, reference.LEARNED[0], trained)


def answer(z, name, trained=None):
    return batch_answers([z], name, trained)[0]


def shared_path(name, cutoff):
    return root() / "shared" / f"{cutoff}-{name}.joblib"


def load_shared(name, cutoff):
    path = shared_path(name, cutoff)
    receipt = base.read(path.with_suffix(".json"))
    if (
        receipt["sha256"] != hashlib.sha256(path.read_bytes()).hexdigest()
        or receipt["cutoff"] != cutoff
        or receipt["name"] != name
    ):
        raise ValueError("ETF_POOLED_SHARED_CHANGED")
    head = joblib.load(path)
    if head["fit_hash"] != receipt["fit_hash"] or head["cutoff"] != cutoff:
        raise ValueError("ETF_POOLED_SHARED_ROWS_CHANGED")
    return head


def shared_checkpoint(rows, name, cutoff):
    """相同季度的三类基金只拟合一次；已存在检查点只验证，失败不重复调参重试。"""
    path = shared_path(name, cutoff)
    if path.with_suffix(".json").exists():
        head = load_shared(name, cutoff)
        if head["fit_hash"] != base.digest(pooled_rows(rows, cutoff)):
            raise ValueError("ETF_POOLED_REUSED_ROWS_CHANGED")
        return head
    attempt = path.with_suffix(".attempt.json")
    if attempt.exists():
        raise ValueError("ETF_POOLED_PREVIOUS_FIT_INTERRUPTED")
    base.save(attempt, {"at": base.now().isoformat(), "name": name, "cutoff": cutoff, "new_fit_count": 1})
    head = fit(rows, name, cutoff)
    joblib.dump(head, path)
    base.save(
        path.with_suffix(".json"),
        {
            "at": base.now().isoformat(),
            "name": name,
            "cutoff": cutoff,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "fit_hash": head["fit_hash"],
            "new_fit_count": 1,
        },
    )
    return head


def fit_checkpoint(rows, name, cutoff, label):
    """仅绑定该类旧HK回退与本季度共用模型；此函数不执行新拟合。"""
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
    base.save(attempt, {"at": base.now().isoformat(), "name": name, "cutoff": cutoff, "new_fit": False})
    us_etf = load_shared(name, cutoff)
    subgroup = adaptive.training_rows([r for r in rows if available(r["z"])], cutoff, "MONTHLY_BAL504")
    if us_etf["pooled_group_fit_hashes"][group] != base.digest(subgroup):
        raise ValueError("ETF_POOLED_GROUP_ROWS_CHANGED")
    trained = {
        "control": control,
        "us_etf": us_etf,
        "group": group,
        "control_fit_hash": control["fit_hash"],
        "new_fit_count": 0,
        "shared_sha256": base.read(shared_path(name, cutoff).with_suffix(".json"))["sha256"],
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
            raise ValueError("ROUND_66_INPUT_CHANGED")
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
            for name in CANDIDATES:
                shared_checkpoint(rows, name, start)
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
            base.save(root() / "progress.json", {"at": base.now().isoformat(), "quarter": q}, replace=True)
        output["ALWAYS_UP"] = [r | {"prediction": 1} for r in output["SPX_SIGN"]]
        expected = sorted((r["code"], r["u"], r["y"]) for r in output["SPX_SIGN"])
        if len(expected) != 5670 or any(
            sorted((r["code"], r["u"], r["y"]) for r in v) != expected for v in output.values()
        ):
            raise ValueError("ROUND_66_COMMON_EXAM_CHANGED")
        metrics = {n: base.metrics(v) for n, v in output.items()}
        winner = max(CANDIDATES, key=lambda n: (metrics[n]["accuracy"], -CANDIDATES.index(n)))
        active()
        for name in CANDIDATES:
            shared_checkpoint(rows, name, p["current_fit_cutoff"])
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
        raise ValueError("ROUND_66_MODEL_OR_CODE_CHANGED")
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
    cn_input_path = cnya_data.root() / target / "input.json"
    cn_input = cnya_data.load(target) if cn_input_path.exists() else None
    if hk_input is None or us_etf_input is None or cn_input is None:
        return report()
    deadline = datetime.combine(date.fromisoformat(target), time(8, 30), base.ZONE)
    with threadpool_limits(limits=2):
        for path in paths:
            if base.now() >= deadline:
                break
            p5, _, source, original = dual.read_parent(path)
            z = live_vector(source, original, hk_input["rows"], us_etf_input["rows"], cn_input["rows"])
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
            raise ValueError("ROUND_66_PARENT_OR_MODEL_CHANGED")
        hk_input = hk_live.load(value["u"])
        us_etf_input = us_etf_data.load(value["u"])
        cn_input = cnya_data.load(value["u"])
        if value["cnya_input_hash"] != base.digest(cn_input):
            raise ValueError("ETF_JOINT_CNYA_INPUT_CHANGED")
        if value["us_etf_input_hash"] != base.digest(us_etf_input):
            raise ValueError("US_ETF_MODEL_LIVE_INPUT_CHANGED")
        if value["hk_input_hash"] != base.digest(hk_input):
            raise ValueError("US_ETF_MODEL_LIVE_INPUT_CHANGED")
        if not np.allclose(
            value["z"],
            live_vector(source, original, hk_input["rows"], us_etf_input["rows"], cn_input["rows"]),
            rtol=0,
            atol=1e-12,
        ):
            raise ValueError("ROUND_66_VECTOR_CHANGED")
        expected_answers = {n: answer(value["z"], n, bundle[n][original["group"]]) for n in CANDIDATES}
        if not runtime.answers_match(value["answers"], expected_answers):
            raise ValueError("ROUND_66_SAVED_ANSWER_CHANGED")
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
