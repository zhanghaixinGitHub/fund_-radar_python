"""第五轮：比较基金自身的隔夜反应与同类基金共同反应，不新增外部读取。

使用固定504日窗口，分别学习资产组和单基金的2项隔夜输入；混合权重按样本数
预先确定。另以组内反应作为先验平滑每只基金的方向频率，所有统计只读成熟标签。
"""

import hashlib
from collections import Counter, defaultdict
from datetime import date, datetime, time

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from app.services import direction_1d_sprint as base
from app.services import direction_1d_sprint_overnight as overnight
from app.services import direction_1d_sprint_sparse as sparse

CANDIDATES = ("GROUP_LR2_BAL504", "FUND_LR2_BAL504", "BLEND_LR2_BAL504", "FUND_CONDITIONAL504")


def root():
    return base.ROOT / "round-05"


def fingerprint():
    result = sparse.fingerprint()
    for name in ("app/services/direction_1d_sprint_fund_response.py", "scripts/direction_1d_sprint_fund_response.py"):
        result[name] = hashlib.sha256((base.PROJECT / name).read_bytes()).hexdigest()
    return result


def plan():
    path = root() / "plan.json"
    if path.exists():
        value = base.read(path)
        if value["code"] != fingerprint():
            raise ValueError("ROUND_05_CODE_CHANGED")
        return value
    value = {
        "at": base.now().isoformat(),
        "code": fingerprint(),
        "candidates": list(CANDIDATES),
        "features": ["SPX_T15_U0830_RETURN", "NEW_US_SESSION_COUNT"],
        "training_days": 504,
        "logistic_C": 0.1,
        "logistic_weights": "equal date and family then balance train classes",
        "blend_fund_weight": "fund training dates / (fund training dates + 252)",
        "conditional_bins": ["SPX_NEGATIVE", "SPX_ZERO_OR_NO_SESSION", "SPX_POSITIVE"],
        "conditional_group_smoothing": "one pseudo-UP plus one pseudo-NON_UP day",
        "conditional_fund_prior_days": 60,
        "threshold": 0.5,
        "development_year": 2025,
        "max_logistic_fits": {"development": 132, "forward": 33},
        "max_fund_conditional_estimates": {"development": 120, "forward": 30},
        "selection": "highest same-question development accuracy among four fixed branches before future outcomes",
        "parent_result_hash": base.digest(base.read(sparse.root() / "result.json")),
        "spx_data_hash": base.digest(base.read(overnight.root() / "spx.json")),
        "nav_data_hash": base.digest(base.read(base.ROOT / "history.json")),
        "held_out_2026_scores_read": False,
        "historical_availability": "ASSUMED_NOT_TRUE_FORWARD",
        "new_provider_requests": 0,
        "new_cost_cny": 0,
    }
    base.save(path, value)
    return value


def selected(rows, cutoff):
    """按标签成熟日严格截断，再取最近504个目标日期；不把未来标签用于个体先验。"""
    eligible = [r for r in rows if r["mature"] < cutoff]
    dates = set(sorted({r["u"] for r in eligible})[-504:])
    chosen = [r for r in eligible if r["u"] in dates]
    counts = Counter(r["y"] for r in chosen)
    if len(dates) < 120 or len(counts) != 2 or min(counts.values()) < 20:
        raise ValueError("FUND_RESPONSE_TRAIN_INSUFFICIENT")
    return chosen


def fit_lr(rows):
    x = np.asarray([sparse.vector(r["x"], "LR2_BAL252") for r in rows])
    y = np.asarray([r["y"] for r in rows])
    weights = sparse.training_weights(rows)
    model = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=1500, random_state=0))
    model.fit(x, y, standardscaler__sample_weight=weights, logisticregression__sample_weight=weights)
    return {
        "model": model,
        "fit_rows": len(rows),
        "fit_dates": len({r["u"] for r in rows}),
        "fit_end": max(r["u"] for r in rows),
        "fit_hash": base.digest(rows),
    }


def bucket(x):
    move, sessions = sparse.vector(x, "LR2_BAL252")
    if sessions == 0 or move == 0:
        return 1
    return 0 if move < 0 else 2


def conditional_rates(rows):
    """按日期/家族等权计算组先验，保留真实两类比例；不能用平衡类权重冒充频率。"""
    family_counts = Counter((r["u"], r["family"]) for r in rows)
    families = Counter(day for day, _ in family_counts)
    totals, ups = np.zeros(3), np.zeros(3)
    for r in rows:
        w = 1 / (family_counts[r["u"], r["family"]] * families[r["u"]])
        index = bucket(r["x"])
        totals[index] += w
        ups[index] += w * r["y"]
    return ((ups + 1) / (totals + 2)).tolist()


def fit_bundle(rows, cutoff):
    """同一次拟合给四条分支共享参数，不为混合分支重复训练相同学习器。"""
    groups, funds, priors = {}, {}, {}
    for group in sorted({r["group"] for r in rows}):
        chosen = selected([r for r in rows if r["group"] == group], cutoff)
        groups[group] = fit_lr(chosen)
        priors[group] = conditional_rates(chosen)
    for code in sorted({r["code"] for r in rows}):
        chosen = selected([r for r in rows if r["code"] == code], cutoff)
        group_set = {r["group"] for r in chosen}
        if len(group_set) != 1:
            raise ValueError("FUND_GROUP_CHANGED_WITHIN_SNAPSHOT")
        group = next(iter(group_set))
        counts = np.zeros(3)
        ups = np.zeros(3)
        for r in chosen:
            index = bucket(r["x"])
            counts[index] += 1
            ups[index] += r["y"]
        funds[code] = fit_lr(chosen) | {
            "group": group,
            "conditional_counts": counts.tolist(),
            "conditional_ups": ups.tolist(),
            "conditional_scores": ((ups + 60 * np.asarray(priors[group])) / (counts + 60)).tolist(),
        }
    return {"cutoff": cutoff, "groups": groups, "funds": funds, "group_conditional_prior": priors}


def answers(bundle, row):
    code, group = row["code"], row["group"]
    if code not in bundle["funds"] or bundle["funds"][code]["group"] != group:
        raise ValueError("FUND_RESPONSE_SCOPE_CHANGED")
    fund = bundle["funds"][code]
    x = [sparse.vector(row["x"], "LR2_BAL252")]
    gp = float(bundle["groups"][group]["model"].predict_proba(x)[0, 1])
    fp = float(fund["model"].predict_proba(x)[0, 1])
    weight = fund["fit_dates"] / (fund["fit_dates"] + 252)
    scores = (gp, fp, weight * fp + (1 - weight) * gp, fund["conditional_scores"][bucket(row["x"])])
    if not np.isfinite(scores).all() or min(scores) < 0 or max(scores) > 1:
        raise ValueError("FUND_RESPONSE_SCORE_INVALID")
    return {
        name: {"prediction": int(score > 0.5), "research_score": float(score), "kind": "UNCALIBRATED_RESEARCH_SCORE"}
        for name, score in zip(CANDIDATES, scores, strict=True)
    }


def train():
    p = plan()
    overnight.source()
    if (root() / "result.json").exists():
        models()
        return base.read(root() / "result.json")
    if (
        p["parent_result_hash"] != base.digest(base.read(sparse.root() / "result.json"))
        or p["spx_data_hash"] != base.digest(base.read(overnight.root() / "spx.json"))
        or p["nav_data_hash"] != base.digest(base.read(base.ROOT / "history.json"))
    ):
        raise ValueError("ROUND_05_INPUT_CHANGED")
    rows = overnight.dataset()
    output = defaultdict(list)
    with threadpool_limits(limits=2):
        for q in range(1, 5):
            start = f"2025-{q * 3 - 2:02d}-01"
            end = "2026-01-01" if q == 4 else f"2025-{q * 3 + 1:02d}-01"
            exam = [r for r in rows if start <= r["u"] < end]
            for name in ("LR2_BAL252", "SPX_SIGN"):
                control = []
                for path in sorted((sparse.root() / "folds").glob(f"{q}-*-{name}.json")):
                    control.extend(base.read(path))
                if {(r["code"], r["u"]) for r in exam} != {(r["code"], r["u"]) for r in control}:
                    raise ValueError("COMMON_EXAM_MISMATCH")
                output[name].extend(control)
                if name == "SPX_SIGN":
                    output["ALWAYS_UP"].extend(r | {"prediction": 1} for r in control)
            path = root() / f"folds/{q}.json"
            if path.exists():
                predictions = base.read(path)
            else:
                fitted = fit_bundle(rows, start)
                predictions = {name: [] for name in CANDIDATES}
                for r in exam:
                    choices = answers(fitted, r)
                    for name in CANDIDATES:
                        predictions[name].append(
                            {k: r[k] for k in ("code", "family", "group", "u", "y", "actual_direction")} | choices[name]
                        )
                base.save(path, predictions)
                base.save(
                    root() / f"training/{q}.json",
                    {
                        "cutoff": start,
                        "groups": {
                            g: {k: v for k, v in t.items() if k != "model"} for g, t in fitted["groups"].items()
                        },
                        "funds": {c: {k: v for k, v in t.items() if k != "model"} for c, t in fitted["funds"].items()},
                    },
                )
            for name, values in predictions.items():
                output[name].extend(values)
            base.save(root() / "progress.json", {"at": base.now().isoformat(), "quarter": q}, replace=True)
        metrics = {n: base.metrics(v) for n, v in output.items()}
        winner = max(CANDIDATES, key=lambda n: (metrics[n]["accuracy"], -CANDIDATES.index(n)))
        bundle = fit_bundle(rows, str(base.now().date()))
    model_path = root() / "models.joblib"
    joblib.dump(bundle, model_path)
    result = {
        "at": base.now().isoformat(),
        "winner": winner,
        "metrics": metrics,
        "code": fingerprint(),
        "plan_hash": base.digest(p),
        "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
        "held_out_2026_scores_read": False,
        "kind": "DEVELOPMENT_ASSUMED_AVAILABILITY_NOT_FORWARD",
        "new_provider_requests": 0,
        "new_cost_cny": 0,
    }
    base.save(root() / "result.json", result)
    return result


def models():
    result = base.read(root() / "result.json")
    path = root() / "models.joblib"
    if result["code"] != fingerprint() or hashlib.sha256(path.read_bytes()).hexdigest() != result["model_sha256"]:
        raise ValueError("ROUND_05_MODEL_OR_CODE_CHANGED")
    return result, joblib.load(path)


def preflight():
    _, bundle = models()
    chosen = {}
    for row in reversed(overnight.dataset()):
        chosen.setdefault(row["code"], row)
    with threadpool_limits(limits=2):
        count = sum(len(answers(bundle, r)) for r in chosen.values())
    value = {"at": base.now().isoformat(), "kind": "DRY_RUN_NOT_FORWARD", "branch_checks": count}
    base.save(root() / "preflight.json", value)
    return value


def read_parent(path):
    previous = base.read(path)
    receipt = base.read(sparse.root() / "receipts" / previous["u"] / path.name)
    deadline = datetime.combine(date.fromisoformat(previous["u"]), time(8, 30), base.ZONE)
    if (
        receipt["status"] != "VERIFIED"
        or receipt["forecast_hash"] != base.digest(previous)
        or datetime.fromisoformat(receipt["readback_at"]) >= deadline
        or datetime.fromisoformat(previous["at"]) >= deadline
    ):
        raise ValueError("ROUND_04_PARENT_NOT_VERIFIED")
    source, original = sparse.read_parent(overnight.root() / "forward" / previous["u"] / path.name)
    if previous["parent_hash"] != base.digest(source) or previous["original_hash"] != base.digest(original):
        raise ValueError("ROUND_04_PARENT_INPUT_CHANGED")
    return previous, source, original


def tick():
    at = base.now()
    end = datetime.fromisoformat(base.read(base.ROOT / "protocol.json")["deadline_at"])
    w = base.window(at)
    if at >= end or w["status"] != "OPEN" or not time(7) <= at.time() < time(8, 30):
        return report()
    target = w["target_nav_date"]
    paths = [
        p
        for p in (sparse.root() / "forward" / target).glob("*.json")
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
            previous, source, original = read_parent(path)
            choices = answers(bundle, original | {"x": source["x"]})
            value = {
                "at": base.now().isoformat(),
                "u": target,
                "code": original["code"],
                "parent_hash": base.digest(previous),
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
        previous, source, original = read_parent(sparse.root() / "forward" / value["u"] / path.name)
        if any(
            value[key] != base.digest(payload)
            for key, payload in (
                ("parent_hash", previous),
                ("source_hash", source),
                ("original_hash", original),
            )
        ):
            raise ValueError("ROUND_05_PARENT_CHANGED")
        good += 1
        closed += value["u"] in original_report["closed_targets"]
        path = base.ROOT / "outcomes" / value["u"] / path.name
        if not path.exists():
            pending += 1
            continue
        outcome = base.read(path)
        if outcome["forecast_hash"] != base.digest(original):
            raise ValueError("OUTCOME_INPUT_CHANGED")
        for name, choice in (value["answers"] | previous["answers"] | original["answers"]).items():
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
        "matched_forward_metrics": {k: base.metrics(v) for k, v in paired.items()},
        "new_provider_requests": 0,
        "new_cost_cny": 0,
        "model_released": False,
    }
    base.save(root() / "report.json", value, replace=True)
    return value
