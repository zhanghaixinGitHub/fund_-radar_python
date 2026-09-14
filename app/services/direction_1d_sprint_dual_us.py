"""第六轮：纳指与标普同一隔夜窗口，新增信息须实际到达后才能保存未来答案。"""

import hashlib
import json
from collections import defaultdict
from datetime import date, datetime, time

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from app.integrations.tushare_sprint_ixic import fetch_ixic
from app.services import direction_1d_sprint as base
from app.services import direction_1d_sprint_fund_response as previous
from app.services import direction_1d_sprint_ixic_data as data
from app.services import direction_1d_sprint_overnight as overnight
from app.services import direction_1d_sprint_sparse as sparse

CANDIDATES = ("LR3_US_BAL504", "TREE3_US_BAL504", "IXIC_SIGN", "US_EQUAL_SIGN")


def root():
    return base.ROOT / "round-06"


def fingerprint():
    value = previous.fingerprint() | data.fingerprint()
    for name in ("app/services/direction_1d_sprint_dual_us.py", "scripts/direction_1d_sprint_dual_us.py"):
        value[name] = hashlib.sha256((base.PROJECT / name).read_bytes()).hexdigest()
    return value


def plan():
    path = root() / "plan.json"
    if path.exists():
        value = base.read(path)
        if value["code"] != fingerprint():
            raise ValueError("ROUND_06_CODE_CHANGED")
        return value
    value = {
        "at": base.now().isoformat(),
        "code": fingerprint(),
        "candidates": list(CANDIDATES),
        "features": ["SPX_T15_U0830_RETURN", "IXIC_T15_U0830_RETURN", "NEW_US_SESSION_COUNT"],
        "training_days": 504,
        "development_year": 2025,
        "threshold": 0.5,
        "logistic_C": 0.1,
        "tree": {"iterations": 60, "leaves": 3, "min_leaf": 60, "learning_rate": 0.05, "l2": 10},
        "monotonicity": "SPX and IXIC scores nondecreasing for equity/mixed, bond unconstrained",
        "rules": "IXIC return >=0; equally weighted SPX+IXIC return >=0; neither outputs a probability",
        "selection": "highest development accuracy among four frozen branches before future outcomes",
        "maximum_development_fits": 24,
        "maximum_forward_fits": 6,
        "maximum_new_ixic_requests_per_target": 3,
        "live_slots": ["0700", "0730", "0800"],
        "ixic_data_hash": base.digest(base.read(data.root() / "history.json")),
        "ixic_source_proof_hash": base.digest(base.read(data.root() / "source-proof.json")),
        "parent_result_hash": base.digest(base.read(previous.root() / "result.json")),
        "held_out_2026_scores_read": False,
        "historical_availability": "ASSUMED_NOT_TRUE_FORWARD",
        "new_cost_cny": 0,
    }
    base.save(path, value)
    return value


def vector(x32, alignment, points):
    """两个指数必须使用相同中国收盘至次日08:30区间，缺日不能按0填补。"""
    sparse.vector(x32, "LR2_BAL252")
    ixic, sessions = overnight.extend([0.0] * 30, alignment, points)[-2:]
    if sessions != x32[31]:
        raise ValueError("US_SESSION_COUNT_MISMATCH")
    return [float(x32[30]), ixic, sessions]


def dataset():
    history = base.read(data.root() / "history.json")
    if history["code"] != data.fingerprint():
        raise ValueError("IXIC_DATA_CODE_CHANGED")
    return [
        r | {"z": vector(r["x"], overnight.alignment(r["t"], r["u"]), history["rows"])} for r in overnight.dataset()
    ]


def fit(rows, name, cutoff):
    if name not in CANDIDATES[:2]:
        raise ValueError("CANDIDATE_NOT_TRAINABLE")
    chosen = previous.selected(rows, cutoff)
    groups = {r["group"] for r in chosen}
    x = np.asarray([r["z"] for r in chosen])
    if len(groups) != 1 or x.shape[1] != 3 or not np.isfinite(x).all():
        raise ValueError("DUAL_US_TRAIN_INPUT_INVALID")
    y = np.asarray([r["y"] for r in chosen])
    weights = sparse.training_weights(chosen)
    if name == "LR3_US_BAL504":
        model = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=1500, random_state=0))
        model.fit(x, y, standardscaler__sample_weight=weights, logisticregression__sample_weight=weights)
    else:
        direction = int(next(iter(groups)) in ("CN_EQUITY", "CN_MIXED"))
        model = HistGradientBoostingClassifier(
            max_iter=60,
            max_leaf_nodes=3,
            min_samples_leaf=60,
            learning_rate=0.05,
            l2_regularization=10,
            monotonic_cst=[direction, direction, 0],
            early_stopping=False,
            random_state=0,
        )
        model.fit(x, y, sample_weight=weights)
    return {
        "model": model,
        "fit_hash": base.digest(chosen),
        "fit_dates": len({r["u"] for r in chosen}),
        "fit_end": max(r["u"] for r in chosen),
    }


def answer(z, name, trained=None):
    if len(z) != 3 or not np.isfinite(z).all() or name not in CANDIDATES:
        raise ValueError("DUAL_US_INPUT_INVALID")
    if name in CANDIDATES[2:]:
        score = z[1] if name == "IXIC_SIGN" else (z[0] + z[1]) / 2
        return {"prediction": int(score >= 0), "research_score": None, "kind": "FIXED_RULE"}
    score = float(trained["model"].predict_proba([z])[0, 1])
    if not np.isfinite(score) or not 0 <= score <= 1:
        raise ValueError("DUAL_US_SCORE_INVALID")
    return {"prediction": int(score > 0.5), "research_score": score, "kind": "UNCALIBRATED_MODEL_SCORE"}


def train():
    p = plan()
    overnight.source()
    if (root() / "result.json").exists():
        models()
        return base.read(root() / "result.json")
    if p["ixic_data_hash"] != base.digest(base.read(data.root() / "history.json")) or p[
        "parent_result_hash"
    ] != base.digest(base.read(previous.root() / "result.json")):
        raise ValueError("ROUND_06_PARENT_CHANGED")
    rows = dataset()
    groups = sorted({r["group"] for r in rows})
    output = defaultdict(list)
    with threadpool_limits(limits=2):
        for q in range(1, 5):
            start = f"2025-{q * 3 - 2:02d}-01"
            end = "2026-01-01" if q == 4 else f"2025-{q * 3 + 1:02d}-01"
            for group in groups:
                grouped = [r for r in rows if r["group"] == group]
                exam = [r for r in grouped if start <= r["u"] < end]
                control = base.read(sparse.root() / f"folds/{q}-{group}-SPX_SIGN.json")
                if {(r["code"], r["u"]) for r in exam} != {(r["code"], r["u"]) for r in control}:
                    raise ValueError("COMMON_EXAM_MISMATCH")
                output["SPX_SIGN"].extend(control)
                output["ALWAYS_UP"].extend(r | {"prediction": 1} for r in control)
                for name in CANDIDATES:
                    path = root() / f"folds/{q}-{group}-{name}.json"
                    if path.exists():
                        values = base.read(path)
                    else:
                        trained = fit(grouped, name, start) if name in CANDIDATES[:2] else None
                        values = [
                            {k: r[k] for k in ("code", "family", "group", "u", "y", "actual_direction")}
                            | answer(r["z"], name, trained)
                            for r in exam
                        ]
                        base.save(path, values)
                    output[name].extend(values)
            base.save(root() / "progress.json", {"at": base.now().isoformat(), "quarter": q}, replace=True)
        metrics = {n: base.metrics(v) for n, v in output.items()}
        winner = max(CANDIDATES, key=lambda n: (metrics[n]["accuracy"], -CANDIDATES.index(n)))
        bundle = {
            n: {g: fit([r for r in rows if r["group"] == g], n, str(base.now().date())) for g in groups}
            for n in CANDIDATES[:2]
        }
    path = root() / "models.joblib"
    joblib.dump(bundle, path)
    value = {
        "at": base.now().isoformat(),
        "winner": winner,
        "metrics": metrics,
        "code": fingerprint(),
        "plan_hash": base.digest(p),
        "model_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "held_out_2026_scores_read": False,
        "kind": "DEVELOPMENT_ASSUMED_AVAILABILITY_NOT_FORWARD",
        "new_cost_cny": 0,
    }
    base.save(root() / "result.json", value)
    return value


def models():
    result = base.read(root() / "result.json")
    path = root() / "models.joblib"
    if result["code"] != fingerprint() or hashlib.sha256(path.read_bytes()).hexdigest() != result["model_sha256"]:
        raise ValueError("ROUND_06_MODEL_OR_CODE_CHANGED")
    return result, joblib.load(path)


def preflight():
    _, bundle = models()
    chosen = {}
    for row in reversed(dataset()):
        chosen.setdefault(row["code"], row)
    with threadpool_limits(limits=2):
        for r in chosen.values():
            for n in CANDIDATES:
                answer(r["z"], n, bundle[n][r["group"]] if n in bundle else None)
    value = {
        "at": base.now().isoformat(),
        "kind": "DRY_RUN_NOT_FORWARD",
        "branch_checks": len(chosen) * len(CANDIDATES),
    }
    base.save(root() / "preflight.json", value)
    return value


def read_parent(path):
    value = base.read(path)
    receipt = base.read(previous.root() / "receipts" / value["u"] / path.name)
    deadline = datetime.combine(date.fromisoformat(value["u"]), time(8, 30), base.ZONE)
    if (
        receipt["status"] != "VERIFIED"
        or receipt["forecast_hash"] != base.digest(value)
        or datetime.fromisoformat(receipt["readback_at"]) >= deadline
        or datetime.fromisoformat(value["at"]) >= deadline
    ):
        raise ValueError("ROUND_05_PARENT_NOT_VERIFIED")
    p4, source, original = previous.read_parent(sparse.root() / "forward" / value["u"] / path.name)
    if any(
        value[key] != base.digest(v)
        for key, v in (("parent_hash", p4), ("source_hash", source), ("original_hash", original))
    ):
        raise ValueError("ROUND_05_PARENT_CHANGED")
    return value, p4, source, original


def tick():
    at = base.now()
    end = datetime.fromisoformat(base.read(base.ROOT / "protocol.json")["deadline_at"])
    w = base.window(at)
    if at >= end or w["status"] != "OPEN" or not time(7) <= at.time() < time(8, 30):
        return report()
    target = w["target_nav_date"]
    paths = [
        p
        for p in (previous.root() / "forward" / target).glob("*.json")
        if not (root() / "forward" / target / p.name).exists()
    ]
    if not paths:
        return report()
    overnight.source()
    manifest, bundle = models()
    aligned = overnight.alignment(w["base_nav_date"], target)
    if any(datetime.fromisoformat(t) >= at for t in aligned["close_events"].values()):
        raise ValueError("IXIC_UNFINISHED_CASH_SESSION")
    slot = f"{at.hour:02d}{'00' if at.minute < 30 else '30'}"
    reservation = root() / f"live/{target}/{slot}-reserved.json"
    if reservation.exists():
        return report()
    base.save(reservation, {"at": at.isoformat(), "maximum_requests": 1, "alignment": aligned})
    required = aligned["required_us_dates"]
    body = fetch_ixic(date.fromisoformat(required[0]), date.fromisoformat(required[-1]))
    observation = {"received_at": base.now().isoformat(), "response": json.loads(body), "alignment": aligned}
    observation_path = root() / f"live/{target}/{slot}-response.json"
    base.save(observation_path, observation)
    deadline = datetime.fromisoformat(aligned["deadline"])
    if base.now() >= deadline:
        return report()
    points = data.validate(body, date.fromisoformat(required[0]), date.fromisoformat(required[-1]))
    with threadpool_limits(limits=2):
        for path in paths:
            if base.now() >= deadline:
                break
            p5, _, source, original = read_parent(path)
            z = vector(source["x"], aligned, points)
            choices = {n: answer(z, n, bundle[n][original["group"]] if n in bundle else None) for n in CANDIDATES}
            value = {
                "at": base.now().isoformat(),
                "u": target,
                "code": original["code"],
                "z": z,
                "parent_hash": base.digest(p5),
                "source_hash": base.digest(source),
                "original_hash": base.digest(original),
                "ixic_observation_hash": base.digest(observation),
                "ixic_observation_file": f"live/{target}/{slot}-response.json",
                "model_hash": manifest["model_sha256"],
                "answers": choices,
                "status": "MODEL_NOT_RELEASED",
            }
            saved = root() / "forward" / target / path.name
            base.save(saved, value)
            readback = base.now()
            valid = base.read(saved) == value and readback < deadline
            base.save(
                root() / "receipts" / target / path.name,
                {
                    "readback_at": readback.isoformat(),
                    "forecast_hash": base.digest(value),
                    "status": "VERIFIED" if valid else "LATE_OR_INVALID",
                },
            )
    return report()


def report():
    if not (root() / "result.json").exists():
        return {"phase": "NOT_TRAINED"}
    original_report = base.report()
    paired, good, late, pending, closed = defaultdict(list), 0, 0, 0, 0
    for path in (root() / "forward").glob("*/*.json"):
        value = base.read(path)
        receipt_path = root() / "receipts" / value["u"] / path.name
        receipt = base.read(receipt_path) if receipt_path.exists() else {}
        deadline = datetime.combine(date.fromisoformat(value["u"]), time(8, 30), base.ZONE)
        if (
            receipt.get("status") != "VERIFIED"
            or receipt.get("forecast_hash") != base.digest(value)
            or datetime.fromisoformat(receipt["readback_at"]) >= deadline
            or datetime.fromisoformat(value["at"]) >= deadline
        ):
            late += 1
            continue
        p5, p4, source, original = read_parent(previous.root() / "forward" / value["u"] / path.name)
        observation = base.read(root() / value["ixic_observation_file"])
        if (
            any(
                value[key] != base.digest(v)
                for key, v in (
                    ("parent_hash", p5),
                    ("source_hash", source),
                    ("original_hash", original),
                    ("ixic_observation_hash", observation),
                )
            )
            or datetime.fromisoformat(observation["received_at"]) >= deadline
        ):
            raise ValueError("ROUND_06_FORECAST_EVIDENCE_CHANGED")
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
    due = original_report["due_eligible_predictions"]
    whole = len(original_report["closed_targets"]) * original_report["watchlist_funds"]
    value = {
        "at": base.now().isoformat(),
        "primary_candidate": base.read(root() / "result.json")["winner"],
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
