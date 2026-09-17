"""第十轮：比较小盘、工业、科技相对大盘的隔夜表现，新增四个固定方案。

RUT早年异常数据隔离后，严格证明所有252日期训练样本和2025考试题仍完整。
未来输入依赖已验证的第六轮父答案，并等待RUT、DJI实际到达，不用历史回填冒充。
"""

import hashlib
import json
import shutil
import time as walltime
import warnings
from collections import defaultdict
from datetime import date, datetime, time

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from app.integrations.tushare_sprint_style import fetch_style
from app.services import direction_1d_sprint as base
from app.services import direction_1d_sprint_adaptive as adaptive
from app.services import direction_1d_sprint_dual_us as dual
from app.services import direction_1d_sprint_exposure as exposure
from app.services import direction_1d_sprint_fund_response as previous
from app.services import direction_1d_sprint_overnight as overnight
from app.services import direction_1d_sprint_return_target as regression
from app.services import direction_1d_sprint_sparse as sparse
from app.services import direction_1d_sprint_style_data as data

CANDIDATES = ("STYLE_LR5_BAL252", "STYLE_TREE5_BAL252", "STYLE_RIDGE5_RET252", "US4_EQUAL_SIGN")
LEARNED = CANDIDATES[:3]
FIRST_TARGET = "2026-09-16"


def root():
    return base.ROOT / "round-10"


def fingerprint():
    value = exposure.fingerprint()
    value["code"].update(data.fingerprint())
    for name in (
        "app/services/direction_1d_sprint_style.py",
        "scripts/direction_1d_sprint_style.py",
        "tests/test_direction_1d_sprint_style.py",
        ".local-runs/direction-1d-sprint-20260914/market-style-data-v1/prepare.py",
    ):
        value["code"][name] = hashlib.sha256((base.PROJECT / name).read_bytes()).hexdigest()
    return value


def active():
    regression.active()


def vector(spx, ixic, rut, dji, sessions):
    """五项输入依次为大盘、三类风格相对大盘的差值、同一窗口内美股新增交易日数。"""
    values = [spx, ixic, rut, dji, sessions]
    if not np.isfinite(values).all() or sessions < 0 or int(sessions) != sessions:
        raise ValueError("STYLE_VECTOR_INVALID")
    return [float(spx), float(ixic - spx), float(rut - spx), float(dji - spx), float(sessions)]


def from_points(x, aligned, points):
    spx, count = x[30:32]
    moves = {}
    for code, prices in points.items():
        move, sessions = overnight.extend([0.0] * 30, aligned, prices)[-2:]
        if sessions != count:
            raise ValueError("STYLE_SESSION_COUNT_MISMATCH")
        moves[code] = move
    return vector(spx, moves["IXIC"], moves["RUT"], moves["DJI"], count)


def dataset():
    styles = base.read(data.root() / "usable-history.json")
    ixic = base.read(dual.data.root() / "history.json")
    if styles["code"] != data.fingerprint() or ixic["code"] != dual.data.fingerprint():
        raise ValueError("STYLE_SOURCE_CODE_CHANGED")
    points = styles["rows"] | {"IXIC": ixic["rows"]}
    rows, raw, cache = [], regression.dataset(), {}
    for row in raw:
        aligned = overnight.alignment(row["t"], row["u"])
        if not all(day in points["RUT"] for day in aligned["required_us_dates"]):
            # 仅隔离已明确停用的早年训练历史；后面逐周期证明并未影响实际训练/考试题。
            if row["u"] >= "2023-01-10":
                raise ValueError("STYLE_REQUIRED_RECENT_HISTORY_MISSING")
            continue
        if row["u"] not in cache:
            cache[row["u"]] = from_points(row["x"], aligned, points)
        rows.append(row | {"z": cache[row["u"]]})
    proofs = []
    for cutoff in ("2025-01-01", "2025-04-01", "2025-07-01", "2025-10-01", str(base.now().date())):
        for group in sorted({r["group"] for r in raw}):
            before = adaptive.training_rows([r for r in raw if r["group"] == group], cutoff, "MONTHLY_BAL252")
            after = adaptive.training_rows([r for r in rows if r["group"] == group], cutoff, "MONTHLY_BAL252")

            def keys(items):
                return sorted((r["code"], r["u"], r["mature"]) for r in items)

            if keys(before) != keys(after):
                raise ValueError("STYLE_QUARANTINE_CHANGED_TRAINING_QUESTIONS")
            proofs.append(
                {
                    "cutoff": cutoff,
                    "group": group,
                    "rows": len(after),
                    "first_target": min(r["u"] for r in after),
                    "question_hash": base.digest(keys(after)),
                }
            )
    return rows, proofs


def fit(rows, name, cutoff):
    if name not in LEARNED:
        raise ValueError("STYLE_NOT_LEARNED_RECIPE")
    chosen = adaptive.training_rows(rows, cutoff, "MONTHLY_BAL252")
    x = np.asarray([r["z"] for r in chosen])
    is_return = name == "STYLE_RIDGE5_RET252"
    y = np.asarray([r["return_target"] if is_return else r["y"] for r in chosen])
    weights = regression.weights(chosen) if is_return else sparse.training_weights(chosen)
    if x.shape[1] != 5 or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("STYLE_TRAIN_INPUT_INVALID")
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        if name == "STYLE_TREE5_BAL252":
            direction = int(chosen[0]["group"] in ("CN_EQUITY", "CN_MIXED"))
            model = HistGradientBoostingClassifier(
                max_iter=60,
                max_leaf_nodes=3,
                min_samples_leaf=60,
                learning_rate=0.05,
                l2_regularization=10,
                early_stopping=False,
                monotonic_cst=[direction, 0, 0, 0, 0],
                random_state=0,
            )
            model.fit(x, y, sample_weight=weights)
        else:
            estimator = Ridge(alpha=10) if is_return else LogisticRegression(C=0.1, max_iter=1500, random_state=0)
            model = make_pipeline(StandardScaler(), estimator)
            model.fit(x, y, standardscaler__sample_weight=weights, **{f"{model.steps[-1][0]}__sample_weight": weights})
    return {
        "model": model,
        "fit_hash": base.digest(chosen),
        "fit_rows": len(chosen),
        "fit_dates": len({r["u"] for r in chosen}),
        "fit_end": max(r["u"] for r in chosen),
        "max_mature_date": max(r["mature"] for r in chosen),
        "cutoff": cutoff,
    }


def answer(z, name, trained=None):
    if name not in CANDIDATES or len(z) != 5 or not np.isfinite(z).all():
        raise ValueError("STYLE_ANSWER_INPUT_INVALID")
    if name == "US4_EQUAL_SIGN":
        score = z[0] + sum(z[1:4]) / 4
        return {"prediction": int(score >= 0), "research_score": None, "kind": "FIXED_RULE"}
    if name == "STYLE_RIDGE5_RET252":
        score = float(trained["model"].predict([z])[0])
        prediction, kind = int(score > 0), "NORMALIZED_RETURN_RESEARCH_SCORE"
    else:
        score = float(trained["model"].predict_proba([z])[0, 1])
        prediction, kind = int(score > 0.5), "UNCALIBRATED_RESEARCH_SCORE"
    if not np.isfinite(score):
        raise ValueError("STYLE_SCORE_INVALID")
    return {"prediction": prediction, "research_score": score, "kind": kind}


def plan():
    path = root() / "plan.json"
    if path.exists():
        value = base.read(path)
        if value["fingerprint"] != fingerprint():
            raise ValueError("ROUND_10_CODE_CHANGED")
        return value
    active()
    value = {
        "at": base.now().isoformat(),
        "fingerprint": fingerprint(),
        "candidates": CANDIDATES,
        "hypothesis": "Small-cap/industrial relative overnight returns may add information missing from SPX/IXIC.",
        "features": ["SPX", "IXIC_minus_SPX", "RUT_minus_SPX", "DJI_minus_SPX", "US_sessions"],
        "training_dates": 252,
        "development_year": 2025,
        "quarterly_refit": True,
        "max_development_fits": 36,
        "max_current_fits": 9,
        "max_reproduction_count": 1,
        "logistic_C": 0.1,
        "ridge_alpha": 10,
        "tree": {"iterations": 60, "leaves": 3, "minimum_leaf": 60, "learning_rate": 0.05, "l2": 10},
        "thresholds": "classifiers >0.5; regression >0; fixed equal-average >=0",
        "controls": ["SPX_SIGN", "LR2_BAL252", "ALWAYS_UP"],
        "selection": "highest 2025 common5670 date-family accuracy; keep all four and report monthly stability",
        "input_hashes": {
            n: base.digest(base.read(base.ROOT / n))
            for n in (
                "history.json",
                "round-02/market.json",
                "round-03/spx.json",
                "ixic-data-v1/history.json",
                "market-style-data-v1/usable-history.json",
            )
        },
        "first_forward_target": FIRST_TARGET,
        "current_fit_cutoff": str(base.now().date()),
        "live_policy": "Verified round06 parent; RUT/DJI each max3 queries in0700/0730/0800; save before0830",
        "maximum_new_requests_per_target": 6,
        "new_cost_cny": 0,
        "historical_availability": "RECONSTRUCTED_NOT_TRUE_FORWARD",
        "this_round_2026_scores_read": False,
    }
    base.save(path, value)
    for name in value["fingerprint"]["code"]:
        dest = root() / "code" / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(base.PROJECT / name, dest)
    return value


def train():
    p = plan()
    if (root() / "result.json").exists():
        return models()[0]
    active()
    for name, expected in p["input_hashes"].items():
        if base.digest(base.read(base.ROOT / name)) != expected:
            raise ValueError("ROUND_10_INPUT_CHANGED")
    rows, proofs = dataset()
    base.save(root() / "training-question-proof.json", proofs)
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
                            fit([r for r in rows if r["group"] == group], name, start) if name in LEARNED else None
                        )
                        scored = [
                            {k: r[k] for k in ("code", "family", "group", "u", "y", "actual_direction")}
                            | answer(r["z"], name, trained)
                            for r in exam
                        ]
                        base.save(path, scored)
                        if trained:
                            base.save(
                                root() / f"training/{q}-{group}-{name}.json",
                                {k: v for k, v in trained.items() if k != "model"},
                            )
                    output[name].extend(scored)
                for control in ("SPX_SIGN", "LR2_BAL252"):
                    output[control].extend(base.read(sparse.root() / f"folds/{q}-{group}-{control}.json"))
            base.save(root() / "progress.json", {"at": base.now().isoformat(), "quarter": q}, replace=True)
        output["ALWAYS_UP"] = [r | {"prediction": 1} for r in output["SPX_SIGN"]]
        expected = sorted((r["code"], r["u"], r["y"]) for r in output["SPX_SIGN"])
        if len(expected) != 5670 or any(
            sorted((r["code"], r["u"], r["y"]) for r in v) != expected for v in output.values()
        ):
            raise ValueError("ROUND_10_COMMON_EXAM_CHANGED")
        metrics = {n: base.metrics(v) for n, v in output.items()}
        winner = max(CANDIDATES, key=lambda n: (metrics[n]["accuracy"], -CANDIDATES.index(n)))
        active()
        bundle = {
            n: {g: fit([r for r in rows if r["group"] == g], n, p["current_fit_cutoff"]) for g in groups}
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
        "development_fits": 36,
        "current_fits": 9,
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
        or hashlib.sha256(path.read_bytes()).hexdigest() != result["model_sha256"]
    ):
        raise ValueError("ROUND_10_MODEL_OR_CODE_CHANGED")
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


def read_parent(path):
    """验证第六轮、上游净值和IXIC原始响应，拒绝尚未验证或已经被替换的父答案。"""
    p6 = base.read(path)
    receipt = base.read(dual.root() / "receipts" / p6["u"] / path.name)
    deadline = datetime.combine(date.fromisoformat(p6["u"]), time(8, 30), base.ZONE)
    if (
        receipt["status"] != "VERIFIED"
        or receipt["forecast_hash"] != base.digest(p6)
        or datetime.fromisoformat(receipt["readback_at"]) >= deadline
        or datetime.fromisoformat(p6["at"]) >= deadline
    ):
        raise ValueError("STYLE_PARENT_NOT_VERIFIED")
    p5, p4, source, original = dual.read_parent(previous.root() / "forward" / p6["u"] / path.name)
    observation = base.read(dual.root() / p6["ixic_observation_file"])
    if (
        any(
            p6[k] != base.digest(v)
            for k, v in (
                ("parent_hash", p5),
                ("source_hash", source),
                ("original_hash", original),
                ("ixic_observation_hash", observation),
            )
        )
        or p6["model_hash"] != base.read(dual.root() / "result.json")["model_sha256"]
    ):
        raise ValueError("STYLE_PARENT_EVIDENCE_CHANGED")
    aligned = overnight.alignment(original["base"], original["u"])
    if observation["alignment"] != aligned or datetime.fromisoformat(
        observation["received_at"]
    ) > datetime.fromisoformat(p6["at"]):
        raise ValueError("STYLE_PARENT_OBSERVATION_TIME_INVALID")
    required = aligned["required_us_dates"]
    ixic = dual.data.validate(json.dumps(observation["response"]).encode(), required[0], required[-1])
    recomputed = dual.vector(source["x"], aligned, ixic)
    if not np.allclose(p6["z"], recomputed, rtol=0, atol=1e-14):
        raise ValueError("STYLE_PARENT_VECTOR_CHANGED")
    return p6, p5, p4, source, original


def collect_points(code, aligned, slot):
    """每个指数每个半小时时槽至多一次；成功接收的同日输入复用，失败不在同槽重试。"""
    target = aligned["target"]
    folder = root() / "live" / target
    deadline = datetime.fromisoformat(aligned["deadline"])
    for path in sorted(folder.glob(f"{code}-*-verified.json")):
        value = base.read(path)
        raw = base.read(root() / value["raw_file"])
        if value["alignment_hash"] != base.digest(aligned) or value["raw_hash"] != base.digest(raw):
            raise ValueError("STYLE_LIVE_INPUT_CHANGED")
        return value
    reservation = folder / f"{code}-{slot}-reserved.json"
    if reservation.exists() or base.now() >= deadline:
        return None
    if slot not in ("0700", "0730", "0800") or len(list(folder.glob(f"{code}-*-reserved.json"))) >= 3:
        return None
    required = aligned["required_us_dates"]
    base.save(reservation, {"at": base.now().isoformat(), "code": code, "maximum_requests": 1, "alignment": aligned})
    try:
        body = fetch_style(code, date.fromisoformat(required[0]), date.fromisoformat(required[-1]))
        observation = {"received_at": base.now().isoformat(), "response": json.loads(body), "alignment": aligned}
        relative = f"live/{target}/{code}-{slot}-response.json"
        base.save(root() / relative, observation)
        if base.now() >= deadline:
            return None
        points = data.validate(code, body, required[0], required[-1])
        value = {
            "raw_file": relative,
            "raw_hash": base.digest(observation),
            "alignment_hash": base.digest(aligned),
            "points": points,
        }
        base.save(folder / f"{code}-{slot}-verified.json", value)
        return value
    except Exception as exc:
        base.save(folder / f"{code}-{slot}-error.json", {"at": base.now().isoformat(), "error": base.error_code(exc)})
        return None
    finally:
        walltime.sleep(max(7, 60 / overnight.source()["rate_limit_per_minute"]))


def tick():
    at = base.now()
    end = datetime.fromisoformat(base.read(base.ROOT / "protocol.json")["deadline_at"])
    window = base.window(at)
    if at >= end or window["status"] != "OPEN" or not time(7) <= at.time() < time(8, 30):
        return report()
    target = window["target_nav_date"]
    if target < FIRST_TARGET:
        return report()
    paths = [
        p
        for p in (dual.root() / "forward" / target).glob("*.json")
        if not (root() / "forward" / target / p.name).exists()
    ]
    if not paths:
        return report()
    read_parent(paths[0])
    manifest, bundle = models()
    aligned = overnight.alignment(window["base_nav_date"], target)
    if any(datetime.fromisoformat(v) >= at for v in aligned["close_events"].values()):
        raise ValueError("STYLE_UNFINISHED_SESSION")
    slot = f"{at.hour:02d}{'00' if at.minute < 30 else '30'}"
    observed = {code: collect_points(code, aligned, slot) for code in data.INDICES}
    deadline = datetime.fromisoformat(aligned["deadline"])
    if any(v is None for v in observed.values()) or base.now() >= deadline:
        return report() | {"input_state": "STYLE_INPUTS_NOT_READY", "slot": slot}
    moves = {code: overnight.extend([0.0] * 30, aligned, v["points"])[-2:] for code, v in observed.items()}
    with threadpool_limits(limits=2):
        for path in paths:
            if base.now() >= deadline:
                break
            p6, _, _, _, original = read_parent(path)
            if any(v[1] != p6["z"][2] for v in moves.values()):
                raise ValueError("STYLE_SESSION_COUNT_MISMATCH")
            z = vector(p6["z"][0], p6["z"][1], moves["RUT"][0], moves["DJI"][0], p6["z"][2])
            value = {
                "at": base.now().isoformat(),
                "u": target,
                "code": original["code"],
                "z": z,
                "parent_hash": base.digest(p6),
                "original_hash": base.digest(original),
                "observations": {c: {k: v[k] for k in ("raw_file", "raw_hash")} for c, v in observed.items()},
                "model_hash": manifest["model_sha256"],
                "status": "MODEL_NOT_RELEASED",
                "answers": {
                    n: answer(z, n, bundle[n][original["group"]] if n in LEARNED else None) for n in CANDIDATES
                },
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
        p6, p5, p4, _, original = read_parent(dual.root() / "forward" / value["u"] / path.name)
        if (
            value["parent_hash"] != base.digest(p6)
            or value["original_hash"] != base.digest(original)
            or value["model_hash"] != result["model_sha256"]
        ):
            raise ValueError("ROUND_10_PARENT_OR_MODEL_CHANGED")
        aligned = overnight.alignment(original["base"], original["u"])
        moves = {}
        for code in data.INDICES:
            reference = value["observations"][code]
            raw = base.read(root() / reference["raw_file"])
            if (
                base.digest(raw) != reference["raw_hash"]
                or raw["alignment"] != aligned
                or datetime.fromisoformat(raw["received_at"]) > datetime.fromisoformat(value["at"])
            ):
                raise ValueError("ROUND_10_INPUT_EVIDENCE_CHANGED")
            required = aligned["required_us_dates"]
            points = data.validate(code, json.dumps(raw["response"]).encode(), required[0], required[-1])
            moves[code] = overnight.extend([0.0] * 30, aligned, points)[-2]
        z = vector(p6["z"][0], p6["z"][1], moves["RUT"], moves["DJI"], p6["z"][2])
        if not np.allclose(value["z"], z, rtol=0, atol=1e-14):
            raise ValueError("ROUND_10_VECTOR_CHANGED")
        good += 1
        closed += value["u"] in original_report["closed_targets"]
        outcome_path = base.ROOT / "outcomes" / value["u"] / path.name
        if not outcome_path.exists():
            pending += 1
            continue
        outcome = base.read(outcome_path)
        if outcome["forecast_hash"] != base.digest(original):
            raise ValueError("OUTCOME_INPUT_CHANGED")
        for name, choice in (
            value["answers"] | p6["answers"] | p5["answers"] | p4["answers"] | original["answers"]
        ).items():
            paired[name].append(original | outcome | {"prediction": choice["prediction"]})
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
