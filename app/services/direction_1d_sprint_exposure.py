"""第九轮：用已成熟历史估计每只基金的近期市场反应，并向同类基金收缩。

新增特征只依赖预测日之前已成熟的标签。它们是历史反应的估计，不能当作真实
持仓或经济因果关系；所有模型继续回答原来单位净值的一日涨跌问题。
"""

import hashlib
import shutil
import warnings
from collections import Counter, defaultdict
from datetime import date, datetime, time

import joblib
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from app.services import direction_1d_sprint as base
from app.services import direction_1d_sprint_adaptive as adaptive
from app.services import direction_1d_sprint_dual_us as dual
from app.services import direction_1d_sprint_fund_response as previous
from app.services import direction_1d_sprint_overnight as overnight
from app.services import direction_1d_sprint_return_target as regression
from app.services import direction_1d_sprint_sparse as sparse

CANDIDATES = ("EXPOSURE_LR504", "EXPOSURE_RIDGE504", "EXPOSURE_SIGN126", "EXPOSURE_SLOPE126")
LEARNED = CANDIDATES[:2]
FIRST_TARGET = "2026-09-16"


def root():
    return base.ROOT / "round-09"


def fingerprint():
    value = adaptive.fingerprint()
    for name in (
        "app/services/direction_1d_sprint_exposure.py",
        "scripts/direction_1d_sprint_exposure.py",
        "tests/test_direction_1d_sprint_exposure.py",
    ):
        value["code"][name] = hashlib.sha256((base.PROJECT / name).read_bytes()).hexdigest()
    return value


def active():
    regression.active()


def moments(s, returns, up, weights):
    """美股变动用百分数，基金变动用事前波动标准化值；0.25稳定项抑制低方差斜率爆炸。"""
    mass = weights.sum()
    if mass <= 0:
        return {"slope": 0.0, "mean_s": 0.0, "mean_return": 0.0, "up_rate": 0.5}
    mean_s = float(np.average(s, weights=weights))
    mean_return = float(np.average(returns, weights=weights))
    variance = float(np.average((s - mean_s) ** 2, weights=weights))
    covariance = float(np.average((s - mean_s) * (returns - mean_return), weights=weights))
    return {
        "slope": float(np.clip(covariance / (variance + 0.25), -3, 3)),
        "mean_s": mean_s,
        "mean_return": mean_return,
        "up_rate": float(np.average(up, weights=weights)),
    }


class ExposureHistory:
    """每次仅查询截止前成熟、最近126个目标日期；冷启动使用中性先验而不删除基金。"""

    def __init__(self, rows):
        if len({(r["code"], r["u"]) for r in rows}) != len(rows):
            raise ValueError("EXPOSURE_DUPLICATE_FUND_DATE")
        self.rows = rows
        self.target = np.asarray([r["u"] for r in rows])
        self.mature = np.asarray([r["mature"] for r in rows])
        self.codes = sorted({r["code"] for r in rows})
        self.groups = {r["code"]: r["group"] for r in rows}

    def snapshot(self, cutoff):
        eligible = np.flatnonzero((self.target < cutoff) & (self.mature < cutoff))
        dates = sorted(set(self.target[eligible]))[-126:]
        selected_dates = set(dates)
        chosen = [self.rows[i] for i in eligible if self.target[i] in selected_dates]
        by_group, by_code = defaultdict(list), defaultdict(list)
        for row in chosen:
            by_group[row["group"]].append(row)
            by_code[row["code"]].append(row)
        group_stats = {}
        for group, rows in by_group.items():
            group_stats[group] = self.summarize(rows, family_weighted=True)
        neutral = {"slope": 0.0, "mean_s": 0.0, "mean_return": 0.0, "up_rate": 0.5}
        funds = {}
        for code in self.codes:
            rows = by_code[code]
            own = self.summarize(rows, family_weighted=False) if rows else neutral
            group = group_stats.get(self.groups[code], neutral)
            # 126个先验日期的收缩强度事先固定，避免少量个体噪声压过同类基金证据。
            n = len(rows)
            share = n / (n + 126)
            funds[code] = {key: share * own[key] + (1 - share) * group[key] for key in neutral}
            funds[code].update({"dates": n, "group": self.groups[code]})
        return {"cutoff": cutoff, "max_mature": max((r["mature"] for r in chosen), default=None), "funds": funds}

    @staticmethod
    def summarize(rows, *, family_weighted):
        s = np.asarray([r["x"][30] * 100 for r in rows])
        returns = np.asarray([r["return_target"] for r in rows])
        up = np.asarray([r["y"] for r in rows])
        if not np.isfinite(s).all() or not np.isfinite(returns).all() or np.max(np.abs(returns)) > 5:
            raise ValueError("EXPOSURE_HISTORY_INVALID")
        if family_weighted:
            counts = Counter((r["u"], r["family"]) for r in rows)
            families = Counter(u for u, _ in counts)
            weights = np.asarray([1 / (families[r["u"]] * counts[r["u"], r["family"]]) for r in rows])
        else:
            weights = np.ones(len(rows))
        return moments(s, returns, up, weights)


def vector(x, code, snapshot):
    if len(x) != 32 or not np.isfinite(x).all() or code not in snapshot["funds"]:
        raise ValueError("EXPOSURE_INPUT_INVALID")
    fund = snapshot["funds"][code]
    s, slope = float(x[30] * 100), fund["slope"]
    intercept = fund["mean_return"] - slope * fund["mean_s"]
    z = [s, x[31], slope * s, intercept, fund["up_rate"] - 0.5, slope]
    if not np.isfinite(z).all():
        raise ValueError("EXPOSURE_FEATURE_NOT_FINITE")
    return z


def dataset(rows=None):
    rows = regression.dataset() if rows is None else rows
    history, cache, result = ExposureHistory(rows), {}, []
    for row in rows:
        if row["u"] not in cache:
            cache[row["u"]] = history.snapshot(row["u"])
        snapshot = cache[row["u"]]
        if snapshot["max_mature"] is not None and snapshot["max_mature"] >= row["u"]:
            raise ValueError("EXPOSURE_FUTURE_CONTEXT")
        result.append(row | {"z": vector(row["x"], row["code"], snapshot)})
    return result


def fit(rows, name, cutoff):
    chosen = previous.selected(rows, cutoff)
    if name not in LEARNED:
        raise ValueError("EXPOSURE_RECIPE_NOT_LEARNED")
    x = np.asarray([r["z"] for r in chosen])
    is_lr = name == "EXPOSURE_LR504"
    y = np.asarray([r["y"] if is_lr else r["return_target"] for r in chosen])
    w = sparse.training_weights(chosen) if is_lr else regression.weights(chosen)
    estimator = LogisticRegression(C=0.1, max_iter=1500, random_state=0) if is_lr else Ridge(alpha=10)
    model = make_pipeline(StandardScaler(), estimator)
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        model.fit(x, y, standardscaler__sample_weight=w, **{f"{model.steps[-1][0]}__sample_weight": w})
    return {
        "model": model,
        "fit_hash": base.digest(chosen),
        "fit_rows": len(chosen),
        "fit_end": max(r["u"] for r in chosen),
        "max_mature_date": max(r["mature"] for r in chosen),
        "fit_dates": len({r["u"] for r in chosen}),
        "cutoff": cutoff,
    }


def score(z, name, trained=None):
    if len(z) != 6 or not np.isfinite(z).all() or name not in CANDIDATES:
        raise ValueError("EXPOSURE_SCORE_INPUT_INVALID")
    if name == "EXPOSURE_LR504":
        value = float(trained["model"].predict_proba([z])[0, 1])
        prediction = int(value > 0.5)
        kind = "UNCALIBRATED_RESEARCH_SCORE"
    else:
        value = float(trained["model"].predict([z])[0]) if name == "EXPOSURE_RIDGE504" else z[2]
        if name == "EXPOSURE_SIGN126":
            value += z[3]
        prediction, kind = int(value > 0), "NORMALIZED_RETURN_RESEARCH_SCORE"
    if not np.isfinite(value):
        raise ValueError("EXPOSURE_SCORE_INVALID")
    return {"prediction": prediction, "research_score": value, "kind": kind}


def plan():
    path = root() / "plan.json"
    if path.exists():
        value = base.read(path)
        if value["fingerprint"] != fingerprint():
            raise ValueError("ROUND_09_CODE_CHANGED")
        return value
    active()
    value = {
        "at": base.now().isoformat(),
        "fingerprint": fingerprint(),
        "candidates": CANDIDATES,
        "hypothesis": "Rolling fund response and prior can distinguish funds missed by pooled SPX models.",
        "context_dates": 126,
        "prior_strength_dates": 126,
        "variance_penalty_percent_squared": 0.25,
        "slope_clip": [-3, 3],
        "features": ["SPX_percent", "sessions", "slope_times_SPX", "intercept", "up_rate_minus_half", "slope"],
        "development_year": 2025,
        "quarters": 4,
        "train_dates": 504,
        "max_development_fits": 24,
        "max_current_fits": 6,
        "max_reproduction_count": 1,
        "C": 0.1,
        "ridge_alpha": 10,
        "lr_threshold": 0.5,
        "other_threshold": 0,
        "context_policy": "Historical context uses mature<target; current exposure snapshot freezes at fit date.",
        "current_fit_cutoff": str(base.now().date()),
        "first_forward_target": FIRST_TARGET,
        "historical_availability": "RECONSTRUCTED_NOT_TRUE_FORWARD",
        "this_round_2026_scores_read": False,
        "reserved_2026_already_consumed": True,
        "new_cost_cny": 0,
        "new_provider_calls": 0,
        "selection": "2025 common5670 date-family accuracy among four; report all, no exclusion or retuning",
        "input_hashes": {
            n: base.digest(base.read(base.ROOT / n))
            for n in (
                "history.json",
                "round-02/market.json",
                "round-03/spx.json",
                "round-08/result.json",
            )
        },
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
            raise ValueError("ROUND_09_INPUT_CHANGED")
    raw = regression.dataset()
    rows = dataset(raw)
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
                            | score(r["z"], name, trained)
                            for r in exam
                        ]
                        base.save(path, scored)
                        if trained:
                            base.save(
                                root() / f"training/{q}-{group}-{name}.json",
                                {k: v for k, v in trained.items() if k != "model"},
                            )
                    output[name].extend(scored)
            control = base.read(previous.root() / f"folds/{q}.json")["GROUP_LR2_BAL504"]
            output["GROUP_LR2_BAL504"].extend(control)
            output["ALWAYS_UP"].extend(r | {"prediction": 1} for r in control)
            for group in groups:
                output["SPX_SIGN"].extend(base.read(base.ROOT / f"round-04/folds/{q}-{group}-SPX_SIGN.json"))
            base.save(root() / "progress.json", {"at": base.now().isoformat(), "quarter": q}, replace=True)
        expected = sorted((r["code"], r["u"], r["y"]) for r in output["GROUP_LR2_BAL504"])
        if len(expected) != 5670 or any(
            sorted((r["code"], r["u"], r["y"]) for r in v) != expected for v in output.values()
        ):
            raise ValueError("ROUND_09_COMMON_EXAM_CHANGED")
        metrics = {n: base.metrics(v) for n, v in output.items()}
        winner = max(CANDIDATES, key=lambda n: (metrics[n]["accuracy"], -CANDIDATES.index(n)))
        active()
        bundle = {
            n: {g: fit([r for r in rows if r["group"] == g], n, p["current_fit_cutoff"]) for g in groups}
            for n in LEARNED
        }
        bundle["context"] = ExposureHistory(raw).snapshot(p["current_fit_cutoff"])
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
        "plan_hash": base.digest(p),
        "model_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "feature_dataset_hash": base.digest(rows),
        "development_fits": 24,
        "current_fits": 6,
        "kind": "DEVELOPMENT_ONLY_AFTER_RESERVED_AUDIT_CONSUMPTION",
        "this_round_2026_scores_read": False,
        "new_cost_cny": 0,
        "new_provider_calls": 0,
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
        raise ValueError("ROUND_09_MODEL_OR_CODE_CHANGED")
    return result, joblib.load(path)


def answers(bundle, original, source):
    z = vector(source["x"], original["code"], bundle["context"])
    return {n: score(z, n, bundle[n][original["group"]] if n in LEARNED else None) for n in CANDIDATES}


def preflight():
    _, bundle = models()
    selected = {}
    for row in reversed(overnight.dataset()):
        selected.setdefault(row["code"], row)
    with threadpool_limits(limits=2):
        for row in selected.values():
            answers(bundle, row, row)
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
    deadline = datetime.combine(date.fromisoformat(target), time(8, 30), base.ZONE)
    with threadpool_limits(limits=2):
        for path in paths:
            if base.now() >= deadline:
                break
            p5, _, source, original = dual.read_parent(path)
            if bundle["context"]["cutoff"] >= target:
                raise ValueError("EXPOSURE_FORWARD_CONTEXT_NOT_PRIOR")
            choices = answers(bundle, original, source)
            value = {
                "at": base.now().isoformat(),
                "u": target,
                "code": original["code"],
                "parent_hash": base.digest(p5),
                "source_hash": base.digest(source),
                "original_hash": base.digest(original),
                "model_hash": manifest["model_sha256"],
                "answers": choices,
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
            receipt.get("status") != "VERIFIED"
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
            raise ValueError("ROUND_09_PARENT_OR_MODEL_CHANGED")
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
