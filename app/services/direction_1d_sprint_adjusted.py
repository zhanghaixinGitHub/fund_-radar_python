"""第十四轮：只替换为复权净值输入，保持原始单位净值方向标签及原学习器。

同题比较原八项逻辑回归/浅层树，不能把分红后的复权收益方向偷换成预测目标。
未来使用独立的限量实际采集；数据库旧值不得冒充基准日最新输入。
"""

import hashlib
import json
import shutil
import time as clock
import traceback
import warnings
from collections import defaultdict
from datetime import date, datetime, time, timedelta
from decimal import Decimal

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

from app.integrations.tushare_sprint_adjusted_nav import FIELDS, fetch_adjusted_nav
from app.services import direction_1d_sprint as base
from app.services import direction_1d_sprint_adaptive as adaptive
from app.services import direction_1d_sprint_correction as previous_round
from app.services import direction_1d_sprint_dual_us as dual
from app.services import direction_1d_sprint_fund_response as previous
from app.services import direction_1d_sprint_overnight as overnight
from app.services import direction_1d_sprint_return_target as regression
from app.services import direction_1d_sprint_sequence as sequence
from app.services import direction_1d_sprint_sparse as sparse

CANDIDATES = ("ADJ_LR8_BAL252", "ADJ_TREE8_BAL252")
LEARNED = CANDIDATES
FIRST_TARGET = "2026-09-16"


def root():
    return base.ROOT / "round-14"


def active():
    regression.active()


def fingerprint():
    value = previous_round.fingerprint()
    for name in (
        "app/integrations/tushare_sprint_adjusted_nav.py",
        "app/services/direction_1d_sprint_adjusted.py",
        "scripts/direction_1d_sprint_adjusted.py",
        "tests/test_direction_1d_sprint_adjusted.py",
    ):
        value["code"][name] = hashlib.sha256((base.PROJECT / name).read_bytes()).hexdigest()
    return value


def parse(code, raw, start, end):
    """只接受对应基金、请求范围内的唯一日期及正数净值；保留源公告日期，不补缺失。"""
    payload = json.loads(raw)
    data = payload.get("data") or {}
    if (
        type(payload.get("code")) is not int
        or payload.get("code") != 0
        or data.get("fields") != FIELDS
        or not isinstance(data.get("items"), list)
        or len(data["items"]) > 160
    ):
        raise ValueError("ADJUSTED_RESPONSE_INVALID")
    points = {}
    for row in data["items"]:
        if len(row) != len(FIELDS):
            raise ValueError("ADJUSTED_ROW_INVALID")
        item = dict(zip(FIELDS, row, strict=True))
        day = date.fromisoformat(item["nav_date"]).isoformat()
        ann = date.fromisoformat(item["ann_date"]).isoformat() if item["ann_date"] else None
        if item["ts_code"] != code or not start <= day <= end or day in points:
            raise ValueError("ADJUSTED_CODE_OR_DATE_INVALID")
        values = [Decimal(str(item[n])) for n in ("unit_nav", "adj_nav")]
        if any(not v.is_finite() or v <= 0 for v in values):
            raise ValueError("ADJUSTED_VALUE_INVALID")
        points[day] = {"date": day, "ann_date": ann, "nav": str(values[0]), "adjusted_nav": str(values[1])}
    return points


def vector(x, original, points):
    """复权序列必须覆盖与原答案相同的61天，并逐日核对单位净值，防止混入修订数据。"""
    wanted = list(map(str, base.input_days(date.fromisoformat(original["base"]))))
    inputs = original["inputs"]
    if [r["date"] for r in inputs] != wanted or any(d not in points for d in wanted):
        raise ValueError("ADJUSTED_INPUT_INCOMPLETE")
    values = []
    for old in inputs:
        r = points[old["date"]]
        if Decimal(str(old["nav"])) != Decimal(str(r["nav"])) or (r.get("ann_date") or r["date"]) > original["u"]:
            raise ValueError("ADJUSTED_ORIGINAL_INPUT_CHANGED")
        values.append(float(r["adjusted_nav"]))
    if (
        len(x) != 32
        or not np.isfinite(x).all()
        or not np.allclose(x[:18], base.vector([r["nav"] for r in inputs], True), rtol=0, atol=1e-12)
    ):
        raise ValueError("ADJUSTED_PARENT_VECTOR_CHANGED")
    new_x = base.vector(values, True) + list(x[18:])
    return [new_x[i] for i in sparse.FEATURE_INDICES]


def dataset():
    snapshot = base.read(base.ROOT / "adjusted-nav-data-v1/history.json")
    points = {f["fund_code"]: {r["date"]: r for r in f["rows"]} for f in snapshot["funds"]}
    original = {
        f["fund_code"]: {r["date"]: r for r in f["rows"]} for f in base.read(base.ROOT / "history.json")["funds"]
    }
    output, cache = [], {}
    for r in overnight.dataset():
        if r["t"] not in cache:
            cache[r["t"]] = list(map(str, base.input_days(date.fromisoformat(r["t"]))))
        fake = {"base": r["t"], "u": r["u"], "inputs": [original[r["code"]][d] for d in cache[r["t"]]]}
        output.append(r | {"z": vector(r["x"], fake, points[r["code"]])})
    return output, {"at": base.now().isoformat(), "rows": len(output), "target": "unchanged raw next unit NAV UP"}


def plan():
    path = root() / "plan.json"
    if path.exists():
        p = base.read(path)
        if p["fingerprint"] != fingerprint():
            raise ValueError("ROUND_14_CODE_CHANGED")
        return p
    active()
    p = {
        "at": base.now().isoformat(),
        "fingerprint": fingerprint(),
        "candidates": CANDIDATES,
        "hypothesis": "Adjusted NAV lag features versus matched original LR8/TREE8 inputs.",
        "features": sparse.FEATURE_NAMES,
        "target": "original raw next unit NAV UP, flat NON_UP; unchanged",
        "training_dates": 252,
        "development_year": 2025,
        "quarterly_refit": True,
        "max_development_fits": 24,
        "max_current_fits": 6,
        "max_reproduction_count": 1,
        "logistic_C": 0.1,
        "tree": {"iterations": 60, "leaves": 3, "min_leaf": 60, "learning_rate": 0.05, "l2": 10, "seed": 0},
        "weights": "date/family equal then class-balanced",
        "threshold": 0.5,
        "controls": ["SPX_SIGN", "LR2_BAL252", "LR8_BAL252", "TREE8_BAL252", "ALWAYS_UP"],
        "selection": "highest same5670 date-family accuracy among two; no retuning",
        "input_hashes": {
            n: base.digest(base.read(base.ROOT / n))
            for n in ("history.json", "round-02/market.json", "round-03/spx.json", "adjusted-nav-data-v1/history.json")
        },
        "first_forward_target": FIRST_TARGET,
        "current_fit_cutoff": str(base.now().date()),
        "live_capture": {
            "max_attempts_per_fund_target": 3,
            "slots": ["0700", "0730", "0800"],
            "days": 130,
            "max_rows": 160,
            "min_spacing_seconds": 7,
            "max_retries": 0,
            "scope_funds": 30,
        },
        "new_cost_cny": 0,
        "this_round_2026_scores_read": False,
        "historical_availability": "CURRENT_RECONSTRUCTED_ADJUSTED_HISTORY_NOT_ORIGINAL_PUBLICATION_VINTAGE",
    }
    base.save(path, p)
    for name in p["fingerprint"]["code"]:
        dest = root() / "code" / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(base.PROJECT / name, dest)
    return p


def fit(rows, name, cutoff):
    chosen = adaptive.training_rows(rows, cutoff, "MONTHLY_BAL252")
    x = np.asarray([r["z"] for r in chosen])
    y = np.asarray([r["y"] for r in chosen])
    w = sparse.training_weights(chosen)
    active()
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        if name == "ADJ_LR8_BAL252":
            x, mean, scale = sequence.normalize_training(x, w)
            model = LogisticRegression(C=0.1, max_iter=1500, random_state=0)
        elif name == "ADJ_TREE8_BAL252":
            # 与原始树保持相同输入尺度和SPX单调约束，隔离净值口径这个变化。
            mean, scale = [0.0] * 8, [1.0] * 8
            direction = int(chosen[0]["group"] in ("CN_EQUITY", "CN_MIXED"))
            model = HistGradientBoostingClassifier(
                max_iter=60,
                max_leaf_nodes=3,
                min_samples_leaf=60,
                learning_rate=0.05,
                l2_regularization=10,
                early_stopping=False,
                monotonic_cst=[direction] + [0] * 7,
                random_state=0,
            )
        else:
            raise ValueError("ADJUSTED_UNKNOWN_RECIPE")
        model.fit(x, y, sample_weight=w)
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
    }


def answer(z, name, trained=None):
    if name not in CANDIDATES or len(z) != 8 or not np.isfinite(z).all():
        raise ValueError("ADJUSTED_ANSWER_INPUT_INVALID")
    x = (np.asarray(z) - trained["mean"]) / trained["scale"]
    if not np.isfinite(x).all():
        raise ValueError("ADJUSTED_NORMALIZATION_INVALID")
    score = float(trained["model"].predict_proba([x])[0, 1])
    if not np.isfinite(score) or not 0 <= score <= 1:
        raise ValueError("ADJUSTED_SCORE_INVALID")
    return {"prediction": int(score > 0.5), "research_score": score, "kind": "UNCALIBRATED_RESEARCH_SCORE"}


def live_points(code, source_code, anchor, target, slot):
    """同一基金每个预定时间段最多请求一次；成功观测可复用，失败不在同一段重试。"""
    prefix = root() / "live" / target
    ready = prefix / f"{code}-ready.json"
    if ready.exists():
        return base.read(ready)
    attempt = prefix / f"{code}-{slot}-attempt.json"
    if attempt.exists():
        return None
    active()
    start = date.fromisoformat(anchor) - timedelta(days=130)
    began = clock.monotonic()
    base.save(
        attempt, {"at": base.now().isoformat(), "code": code, "source_code": source_code, "base": anchor, "u": target}
    )
    try:
        raw = fetch_adjusted_nav(source_code, start, date.fromisoformat(anchor))
        received = base.now().isoformat()
        observation = {
            "received_at": received,
            "code": code,
            "source_code": source_code,
            "base": anchor,
            "u": target,
            "raw": raw.decode("utf-8"),
        }
        relative = f"live/{target}/{code}-{slot}-response.json"
        base.save(root() / relative, observation)
        points = parse(source_code, raw, start.isoformat(), anchor)
        wanted = list(map(str, base.input_days(date.fromisoformat(anchor))))
        if not all(d in points for d in wanted):
            return None
        value = {
            "points": points,
            "received_at": received,
            "source_file": relative,
            "source_hash": base.digest(observation),
            "u": target,
            "base": anchor,
            "code": code,
        }
        base.save(ready, value)
        return value
    finally:
        clock.sleep(max(0, 7 - (clock.monotonic() - began)))


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
            raise ValueError("ADJUSTED_CHECKPOINT_CHANGED")
        return joblib.load(path)
    if attempt.exists():
        raise ValueError("ADJUSTED_PREVIOUS_FIT_INTERRUPTED")
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
            raise ValueError("ROUND_14_INPUT_CHANGED")
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
                            fit_checkpoint([r for r in rows if r["group"] == group], name, start, f"{q}-{group}-{name}")
                            if name in LEARNED
                            else None
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
                for control in ("SPX_SIGN", "LR2_BAL252", "LR8_BAL252", "TREE8_BAL252"):
                    output[control].extend(base.read(sparse.root() / f"folds/{q}-{group}-{control}.json"))
            base.save(root() / "progress.json", {"at": base.now().isoformat(), "quarter": q}, replace=True)
        output["ALWAYS_UP"] = [r | {"prediction": 1} for r in output["SPX_SIGN"]]
        expected = sorted((r["code"], r["u"], r["y"]) for r in output["SPX_SIGN"])
        if len(expected) != 5670 or any(
            sorted((r["code"], r["u"], r["y"]) for r in v) != expected for v in output.values()
        ):
            raise ValueError("ROUND_14_COMMON_EXAM_CHANGED")
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
        or hashlib.sha256(path.read_bytes()).hexdigest() != result["model_sha256"]
    ):
        raise ValueError("ROUND_14_MODEL_OR_CODE_CHANGED")
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
    profiles = {f["fund_code"]: f for f in base.read(base.ROOT / "history.json")["funds"]}
    slot = "0800" if at.time() >= time(8) else "0730" if at.time() >= time(7, 30) else "0700"
    deadline = datetime.combine(date.fromisoformat(target), time(8, 30), base.ZONE)
    failures = {}
    with threadpool_limits(limits=2):
        for path in paths:
            if base.now() >= deadline:
                break
            p5, _, source, original = dual.read_parent(path)
            try:
                context = live_points(
                    original["code"], profiles[original["code"]]["source_fund_code"], original["base"], target, slot
                )
                if context is None or base.now() >= deadline:
                    continue
                z = vector(source["x"], original, context["points"])
            except Exception as exc:
                failures[original["code"]] = {
                    "error": base.error_code(exc),
                    "type": type(exc).__name__,
                    "stack": [
                        {"file": f.filename, "line": f.lineno, "function": f.name}
                        for f in traceback.extract_tb(exc.__traceback__)
                    ],
                }
                continue
            choices = {n: answer(z, n, bundle[n][original["group"]]) for n in CANDIDATES}
            value = {
                "at": base.now().isoformat(),
                "u": target,
                "code": original["code"],
                "parent_hash": base.digest(p5),
                "source_hash": base.digest(source),
                "original_hash": base.digest(original),
                "model_hash": manifest["model_sha256"],
                "answers": choices,
                "z": z,
                "context": context,
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
    if failures:
        base.save(root() / f"live/{target}/{slot}-errors.json", failures, replace=True)
        raise ValueError("ADJUSTED_SOME_FUND_INPUTS_FAILED")
    return report()


def report():
    if not (root() / "result.json").exists():
        return {"phase": "NOT_TRAINED"}
    original_report = base.report()
    result = base.read(root() / "result.json")
    paired, good, late, pending, closed = defaultdict(list), 0, 0, 0, 0
    profiles = {f["fund_code"]: f for f in base.read(base.ROOT / "history.json")["funds"]}
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
            raise ValueError("ROUND_14_PARENT_OR_MODEL_CHANGED")
        context = value["context"]
        observation = base.read(root() / context["source_file"])
        if (
            base.digest(observation) != context["source_hash"]
            or datetime.fromisoformat(context["received_at"]) >= deadline
            or datetime.fromisoformat(context["received_at"]) > datetime.fromisoformat(value["at"])
        ):
            raise ValueError("ROUND_14_OBSERVATION_INVALID")
        if (
            context["received_at"] != observation["received_at"]
            or context["u"] != value["u"]
            or observation["u"] != value["u"]
            or context["code"] != value["code"]
            or observation["code"] != value["code"]
            or context["base"] != original["base"]
            or observation["base"] != original["base"]
            or observation["source_code"] != profiles[value["code"]]["source_fund_code"]
        ):
            raise ValueError("ROUND_14_OBSERVATION_LINEAGE_CHANGED")
        parsed = parse(
            observation["source_code"],
            observation["raw"],
            str(date.fromisoformat(original["base"]) - timedelta(days=130)),
            original["base"],
        )
        if context["points"] != parsed:
            raise ValueError("ROUND_14_PARSED_INPUT_CHANGED")
        if not np.allclose(value["z"], vector(source["x"], original, context["points"]), rtol=0, atol=1e-12):
            raise ValueError("ROUND_14_VECTOR_CHANGED")
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
