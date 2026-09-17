"""第十一轮：同类基金的已知涨跌广度、离散度及自身相对强弱。

同伴由基准日输入是否已可用决定，不能用下一日标签是否存在来选择同伴。
产品份额先按家族聚合，缺失比例作为输入，不通过删除难题来提高准确率。
"""

import hashlib
import shutil
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

from app.services import direction_1d_sprint as base
from app.services import direction_1d_sprint_adaptive as adaptive
from app.services import direction_1d_sprint_dual_us as dual
from app.services import direction_1d_sprint_fund_response as previous
from app.services import direction_1d_sprint_return_target as regression
from app.services import direction_1d_sprint_sparse as sparse
from app.services import direction_1d_sprint_style as previous_round

CANDIDATES = ("BREADTH_LR8_BAL252", "BREADTH_TREE8_BAL252", "BREADTH_RIDGE8_RET252")
LEARNED = CANDIDATES
FIRST_TARGET = "2026-09-16"


def root():
    return base.ROOT / "round-11"


def fingerprint():
    value = previous_round.fingerprint()
    for name in (
        "app/services/direction_1d_sprint_breadth.py",
        "scripts/direction_1d_sprint_breadth.py",
        "tests/test_direction_1d_sprint_breadth.py",
    ):
        value["code"][name] = hashlib.sha256((base.PROJECT / name).read_bytes()).hexdigest()
    return value


def active():
    regression.active()


def summarize(peers, profiles):
    """同一家族的份额先平均，再计算同类分布；完整范围作为覆盖率分母。"""
    expected, grouped = defaultdict(set), defaultdict(lambda: defaultdict(list))
    for fund in profiles:
        expected[fund["group"]].add(fund["family"])
    for peer in peers.values():
        grouped[peer["group"]][peer["family"]].append(peer["x"])
    summaries = {}
    for group, families in grouped.items():
        x = np.asarray([np.mean(values, axis=0) for values in families.values()])
        summaries[group] = {
            "median_ret1": float(np.median(x[:, 7])),
            "median_ret5": float(np.median(x[:, 0])),
            "breadth": float(np.mean(x[:, 7] > 0)),
            "dispersion": float(np.std(x[:, 7])),
            "median_vol20": float(np.median(x[:, 3])),
            "coverage": len(families) / len(expected[group]),
        }
    return {"peers": peers, "groups": summaries}


class PeerHistory:
    """只读取基准日及以前61个净值与公告日，不读取下一交易日净值或标签。"""

    def __init__(self, history):
        self.profiles = [{k: f[k] for k in ("fund_code", "family", "group")} for f in history["funds"]]
        self.points = {f["fund_code"]: {r["date"]: r for r in f["rows"]} for f in history["funds"]}

    def snapshot(self, base_day, target):
        wanted = list(map(str, base.input_days(date.fromisoformat(base_day))))
        peers = {}
        for fund in self.profiles:
            points = self.points[fund["fund_code"]]
            if not all(d in points and (points[d].get("ann_date") or d) <= target for d in wanted):
                continue
            try:
                x = base.vector([points[d]["nav"] for d in wanted], True)
            except ValueError:
                continue
            peers[fund["fund_code"]] = {"group": fund["group"], "family": fund["family"], "x": x}
        return summarize(peers, self.profiles)


def vector(x, code, snapshot):
    if len(x) != 32 or not np.isfinite(x).all() or code not in snapshot["peers"]:
        raise ValueError("BREADTH_OWN_INPUT_MISSING")
    peer = snapshot["peers"][code]
    if not np.allclose(x[:18], peer["x"], rtol=0, atol=1e-12):
        raise ValueError("BREADTH_OWN_INPUT_CHANGED")
    group = snapshot["groups"][peer["group"]]
    own_vol = max(float(x[3]), 0.0001)
    group_vol = max(group["median_vol20"], 0.0001)
    z = [
        x[30],
        x[31],
        float(np.clip((x[7] - group["median_ret1"]) / own_vol, -5, 5)),
        float(np.clip((x[0] - group["median_ret5"]) / (own_vol * np.sqrt(5)), -5, 5)),
        group["breadth"] - 0.5,
        float(np.clip(group["dispersion"] / group_vol, 0, 5)),
        float(np.clip(group["median_ret1"] / group_vol, -5, 5)),
        group["coverage"],
    ]
    if not np.isfinite(z).all():
        raise ValueError("BREADTH_FEATURE_NOT_FINITE")
    return z


def dataset():
    history = PeerHistory(base.read(base.ROOT / "history.json"))
    rows, cache, output = regression.dataset(), {}, []
    for row in rows:
        if row["u"] not in cache:
            cache[row["u"]] = history.snapshot(row["t"], row["u"])
        output.append(row | {"z": vector(row["x"], row["code"], cache[row["u"]])})
    return output, {
        "at": base.now().isoformat(),
        "peer_snapshot_dates": len(cache),
        "label_used_for_peer_selection": False,
        "context_policy": "61NAV inputs through base day with announcement<=target only",
    }


def plan():
    path = root() / "plan.json"
    if path.exists():
        value = base.read(path)
        if value["fingerprint"] != fingerprint():
            raise ValueError("ROUND_11_CODE_CHANGED")
        return value
    active()
    value = {
        "at": base.now().isoformat(),
        "fingerprint": fingerprint(),
        "candidates": CANDIDATES,
        "hypothesis": "Domestic peer breadth and relative strength may add information beyond US direction.",
        "features": [
            "SPX",
            "US_sessions",
            "own_relative_ret1",
            "own_relative_ret5",
            "peer_breadth_centered",
            "peer_dispersion",
            "peer_median_ret1",
            "peer_coverage",
        ],
        "feature_policy": "Only base-day available NAVs; family aggregation; no future-label-based peer membership",
        "training_dates": 252,
        "development_year": 2025,
        "quarterly_refit": True,
        "max_development_fits": 36,
        "max_current_fits": 9,
        "max_reproduction_count": 1,
        "logistic_C": 0.1,
        "ridge_alpha": 10,
        "tree": {"iterations": 60, "leaves": 3, "minimum_leaf": 60, "learning_rate": 0.05, "l2": 10},
        "thresholds": "classifiers >0.5; regression >0",
        "controls": ["SPX_SIGN", "LR2_BAL252", "ALWAYS_UP"],
        "selection": "highest 2025 common5670 date-family accuracy among three; no retuning",
        "input_hashes": {
            n: base.digest(base.read(base.ROOT / n))
            for n in (
                "history.json",
                "round-02/market.json",
                "round-03/spx.json",
            )
        },
        "first_forward_target": FIRST_TARGET,
        "current_fit_cutoff": str(base.now().date()),
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
    if name not in LEARNED:
        raise ValueError("BREADTH_NOT_LEARNED_RECIPE")
    chosen = adaptive.training_rows(rows, cutoff, "MONTHLY_BAL252")
    x = np.asarray([r["z"] for r in chosen])
    is_return = name == "BREADTH_RIDGE8_RET252"
    y = np.asarray([r["return_target"] if is_return else r["y"] for r in chosen])
    weights = regression.weights(chosen) if is_return else sparse.training_weights(chosen)
    if x.shape[1] != 8 or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("BREADTH_TRAIN_INPUT_INVALID")
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        if name == "BREADTH_TREE8_BAL252":
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
    if name not in CANDIDATES or len(z) != 8 or not np.isfinite(z).all():
        raise ValueError("BREADTH_ANSWER_INPUT_INVALID")
    if name == "BREADTH_RIDGE8_RET252":
        score = float(trained["model"].predict([z])[0])
        prediction, kind = int(score > 0), "NORMALIZED_RETURN_RESEARCH_SCORE"
    else:
        score = float(trained["model"].predict_proba([z])[0, 1])
        prediction, kind = int(score > 0.5), "UNCALIBRATED_RESEARCH_SCORE"
    if not np.isfinite(score):
        raise ValueError("BREADTH_SCORE_INVALID")
    return {"prediction": prediction, "research_score": score, "kind": kind}


def train():
    p = plan()
    if (root() / "result.json").exists():
        return models()[0]
    active()
    for name, expected in p["input_hashes"].items():
        if base.digest(base.read(base.ROOT / name)) != expected:
            raise ValueError("ROUND_11_INPUT_CHANGED")
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
            raise ValueError("ROUND_11_COMMON_EXAM_CHANGED")
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
        raise ValueError("ROUND_11_MODEL_OR_CODE_CHANGED")
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


def validate_peer(value, profile, anchor, target, model_hash):
    """实际同伴只接受同一目标日已落盘的原始预测输入，不能接受晚到或错期记录。"""
    deadline = datetime.combine(date.fromisoformat(target), time(8, 30), base.ZONE)
    if (
        value["code"] != profile["fund_code"]
        or value["group"] != profile["group"]
        or value["family"] != profile["family"]
        or value["base"] != anchor
        or value["u"] != target
        or value["models_hash"] != model_hash
        or datetime.fromisoformat(value["at"]) >= deadline
        or datetime.fromisoformat(value["at"]) > base.now()
    ):
        raise ValueError("BREADTH_PEER_NOT_VALID")
    wanted = list(map(str, base.input_days(date.fromisoformat(anchor))))
    if [row["date"] for row in value["inputs"]] != wanted or any(
        (row.get("ann_date") or row["date"]) > target for row in value["inputs"]
    ):
        raise ValueError("BREADTH_PEER_INPUT_DATES_INVALID")
    return {
        "group": profile["group"],
        "family": profile["family"],
        "x": base.vector([row["nav"] for row in value["inputs"]], True),
    }


def peer_snapshot(anchor, target):
    profiles = base.read(base.ROOT / "history.json")["funds"]
    model_hash = base.read(base.ROOT / "round-01/result.json")["model_sha256"]
    peers, references = {}, {}
    for profile in profiles:
        code = profile["fund_code"]
        path = base.ROOT / "forward" / target / f"{code}.json"
        if not path.exists():
            continue
        value = base.read(path)
        peers[code] = validate_peer(value, profile, anchor, target, model_hash)
        references[code] = base.digest(value)
    return summarize(peers, profiles) | {
        "at": base.now().isoformat(),
        "base": anchor,
        "u": target,
        "original_references": references,
    }


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
        for p in (previous.root() / "forward" / target).glob("*.json")
        if not (root() / "forward" / target / p.name).exists()
    ]
    if not paths:
        return report()
    manifest, bundle = models()
    context = peer_snapshot(window["base_nav_date"], target)
    relative = f"live/{target}/peers-{base.now().strftime('%Y%m%dT%H%M%S%f')}.json"
    if not (root() / relative).exists():
        base.save(root() / relative, context)
    elif base.read(root() / relative) != context:
        raise ValueError("BREADTH_CONTEXT_ALREADY_EXISTS")
    deadline = datetime.combine(date.fromisoformat(target), time(8, 30), base.ZONE)
    with threadpool_limits(limits=2):
        for path in paths:
            if base.now() >= deadline:
                break
            p5, _, source, original = dual.read_parent(path)
            z = vector(source["x"], original["code"], context)
            value = {
                "at": base.now().isoformat(),
                "u": target,
                "code": original["code"],
                "z": z,
                "parent_hash": base.digest(p5),
                "source_hash": base.digest(source),
                "original_hash": base.digest(original),
                "context_file": relative,
                "context_hash": base.digest(context),
                "model_hash": manifest["model_sha256"],
                "status": "MODEL_NOT_RELEASED",
                "answers": {n: answer(z, n, bundle[n][original["group"]]) for n in CANDIDATES},
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
    paired, cache = defaultdict(list), {}
    good, late, pending, closed = 0, 0, 0, 0
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
            raise ValueError("ROUND_11_PARENT_OR_MODEL_CHANGED")
        key = value["context_file"]
        if key not in cache:
            context = base.read(root() / key)
            for code, expected in context["original_references"].items():
                if base.digest(base.read(base.ROOT / "forward" / value["u"] / f"{code}.json")) != expected:
                    raise ValueError("ROUND_11_PEER_INPUT_CHANGED")
            cache[key] = context
        context = cache[key]
        if (
            base.digest(context) != value["context_hash"]
            or context["u"] != value["u"]
            or datetime.fromisoformat(context["at"]) > datetime.fromisoformat(value["at"])
        ):
            raise ValueError("ROUND_11_CONTEXT_CHANGED")
        if not np.allclose(value["z"], vector(source["x"], original["code"], context), rtol=0, atol=1e-12):
            raise ValueError("ROUND_11_VECTOR_CHANGED")
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
    targets = [d for d in original_report["closed_targets"] if d >= FIRST_TARGET]
    due, whole = len(targets) * original_report["eligible_funds"], len(targets) * original_report["watchlist_funds"]
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
