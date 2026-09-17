"""三天研究的2026保留期一次性复核：固定配方滚动重训，不更换未来预测模型。

这份成绩检验开发期选择能否延续到另一段历史，不代表当年实际提前预测。
净值公告和隔夜行情历史可用时间仍有假设；真正未来证据继续由六轮任务采集。
"""

import hashlib
import importlib.metadata
import math
import shutil
from collections import defaultdict
from datetime import datetime

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

from app.services import direction_1d_sprint as base
from app.services import direction_1d_sprint_fund_response as response
from app.services import direction_1d_sprint_overnight as overnight
from app.services import direction_1d_sprint_sparse as sparse

NAMES = ("GROUP_LR2_BAL504", "SPX_SIGN", "LR7_504", "ALWAYS_UP")
FOLDS = (("2026-01-01", "2026-04-01"), ("2026-04-01", "2026-07-01"), ("2026-07-01", "2026-09-12"))


def root():
    return base.ROOT / "reserved-audit-v1"


def active():
    """保留期复核同样受三天期限约束；已生成报告可在期限后直接读取。"""
    if base.now() >= datetime.fromisoformat(base.read(base.ROOT / "protocol.json")["deadline_at"]):
        raise ValueError("SPRINT_DEADLINE_REACHED")


def fingerprint():
    code = response.fingerprint()
    for name in (
        "app/services/direction_1d_sprint_reserved_audit.py",
        "scripts/direction_1d_sprint_reserved_audit.py",
        "tests/test_direction_1d_sprint_reserved_audit.py",
    ):
        code[name] = hashlib.sha256((base.PROJECT / name).read_bytes()).hexdigest()
    return {
        "code": code,
        "libraries": {n: importlib.metadata.version(n) for n in ("numpy", "scikit-learn", "joblib", "threadpoolctl")},
        "calendar_hash": base.calendar()[1],
    }


def plan():
    """在读取任何本轮2026评分前固定候选、切分、输入、预算和完整代码。"""
    path = root() / "plan.json"
    if path.exists():
        p = base.read(path)
        if p["fingerprint"] != fingerprint():
            raise ValueError("RESERVED_AUDIT_CODE_CHANGED")
        return p
    active()
    p = {
        "at": base.now().isoformat(),
        "fingerprint": fingerprint(),
        "models": list(NAMES),
        "folds": [list(fold) for fold in FOLDS],
        "max_fits": 18,
        "max_reproduction_count": 1,
        "threshold": "learned score strictly greater than 0.5; SPX nonnegative implies UP",
        "fit_policy": "same frozen recipes, each group refit using mature dates strictly before each quarter",
        "selection_policy": "report all fixed candidates; no ranking-based replacement or threshold tuning",
        "historical_availability": "RECONSTRUCTED_ANN_DATE_AND_ASSUMED_INDEX_TIMING_NOT_TRUE_FORWARD",
        "reserved_consumption": "2026-01-01 through 2026-09-11 is consumed once this audit starts",
        "bootstrap": {"block_dates": [5, 20], "draws": 5000, "seed": 20260914},
        "new_provider_calls": 0,
        "new_cost_cny": 0,
        "future_model_binding_changed": False,
        "input_hashes": {
            name: base.digest(base.read(base.ROOT / name))
            for name in ("history.json", "round-02/market.json", "round-03/spx.json", "round-05/result.json")
        },
    }
    base.save(path, p)
    for name in p["fingerprint"]["code"]:
        destination = root() / "code" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(base.PROJECT / name, destination)
    base.save(root() / "code/manifest.json", p["fingerprint"])
    return p


def choose_training(rows, cutoff):
    """标签成熟日与目标日都要严格早于切分日，防止错误元数据令未来样本进入训练。"""
    chosen = response.selected(rows, cutoff)
    if any(r["u"] >= cutoff or r["mature"] >= cutoff for r in chosen):
        raise ValueError("RESERVED_AUDIT_FUTURE_LABEL")
    return chosen


def validate_exam(rows, start, end):
    """同一题只保留一次，且输入日早于目标日；不通过筛掉困难基金来改善分数。"""
    if not rows or any(not start <= r["u"] < end or r["t"] >= r["u"] for r in rows):
        raise ValueError("RESERVED_AUDIT_EXAM_RANGE")
    if len({(r["code"], r["u"]) for r in rows}) != len(rows):
        raise ValueError("RESERVED_AUDIT_DUPLICATE_QUESTION")


def score_models(group_model, original_model, rows):
    """对同一批问题同时给出四个答案；七项对照使用当季重新训练的配方，不载入今天的模型。"""
    x2 = np.asarray([sparse.vector(r["x"], "LR2_BAL252") for r in rows])
    x7 = np.asarray([r["x"][:7] for r in rows])
    learned = group_model["model"].predict_proba(x2)[:, 1]
    original = original_model["model"].predict_proba(x7)[:, 1]
    if not np.isfinite(learned).all() or not np.isfinite(original).all():
        raise ValueError("RESERVED_AUDIT_INVALID_SCORE")
    result = defaultdict(list)
    for i, row in enumerate(rows):
        common = {k: row[k] for k in ("code", "family", "group", "u", "y", "actual_direction")}
        decisions = (int(learned[i] > 0.5), int(x2[i, 0] >= 0), int(original[i] > 0.5), 1)
        for name, prediction in zip(NAMES, decisions, strict=True):
            result[name].append(common | {"prediction": prediction})
    return dict(result)


def daily(rows):
    family = defaultdict(list)
    for r in rows:
        family[r["u"], r["family"]].append(int(r["prediction"] == r["y"]))
    dates = defaultdict(list)
    for (u, _), values in family.items():
        dates[u].append(float(np.mean(values)))
    return {u: float(np.mean(v)) for u, v in sorted(dates.items())}


def paired_intervals(predictions, spec):
    """整段交易日重抽样，四个方法使用同一套日期；区间仅衡量该历史样本的不确定性。"""
    values = np.asarray([list(daily(predictions[n]).values()) for n in NAMES]).T
    result = {}
    for block in spec["block_dates"]:
        if len(values) < block:
            raise ValueError("RESERVED_AUDIT_TOO_FEW_DATES")
        rng = np.random.default_rng(spec["seed"] + block)
        starts = rng.integers(0, len(values) - block + 1, size=(spec["draws"], math.ceil(len(values) / block)))
        indices = (starts[:, :, None] + np.arange(block)).reshape(spec["draws"], -1)[:, : len(values)]
        draws = values[indices].mean(axis=1)
        result[str(block)] = {
            name: {
                "accuracy_95_interval": np.quantile(draws[:, i], [0.025, 0.975]).tolist(),
                "excess_vs_always_up_95_interval": np.quantile(draws[:, i] - draws[:, 3], [0.025, 0.975]).tolist(),
            }
            for i, name in enumerate(NAMES)
        }
    return result


def run():
    p = plan()
    if (root() / "result.json").exists():
        return base.read(root() / "result.json")
    active()
    for name, expected in p["input_hashes"].items():
        if base.digest(base.read(base.ROOT / name)) != expected:
            raise ValueError("RESERVED_AUDIT_INPUT_CHANGED")
    marker = base.ROOT / "reserved-audit-consumption.json"
    if not marker.exists():
        base.save(marker, {"at": base.now().isoformat(), "plan_hash": base.digest(p), "period": p["folds"]})
    elif base.read(marker)["plan_hash"] != base.digest(p):
        raise ValueError("RESERVED_PERIOD_ALREADY_CONSUMED")
    rows = overnight.dataset()
    groups = sorted({r["group"] for r in rows})
    if len(groups) != 3:
        raise ValueError("RESERVED_AUDIT_GROUP_SCOPE_CHANGED")
    output, quarters = defaultdict(list), {}
    with threadpool_limits(limits=2):
        for q, (start, end) in enumerate(FOLDS, 1):
            active()
            exam = [r for r in rows if start <= r["u"] < end]
            validate_exam(exam, start, end)
            path = root() / f"folds/{q}.json"
            if path.exists():
                folded = base.read(path)
                if folded["plan_hash"] != base.digest(p) or folded["exam_hash"] != base.digest(exam):
                    raise ValueError("RESERVED_AUDIT_FOLD_CHANGED")
            else:
                predictions, training = defaultdict(list), {}
                for group in groups:
                    chosen = choose_training([r for r in rows if r["group"] == group], start)
                    model = response.fit_lr(chosen)
                    # 两种配方共享同一成熟日和504日窗口；原配方内部仍按原权重与7列拟合。
                    control = base.fit(chosen, "LR7_504", start)
                    batch = [r for r in exam if r["group"] == group]
                    if not batch:
                        raise ValueError("RESERVED_AUDIT_GROUP_EXAM_EMPTY")
                    for name, answers in score_models(model, control, batch).items():
                        predictions[name].extend(answers)
                    artifact = root() / f"models/{q}-{group}.joblib"
                    artifact.parent.mkdir(parents=True, exist_ok=True)
                    if artifact.exists():
                        raise ValueError("RESERVED_AUDIT_ORPHAN_MODEL_REQUIRES_REVIEW")
                    joblib.dump({"group_model": model, "seven_feature_control": control}, artifact)
                    training[group] = {
                        "fit_hash": base.digest(chosen),
                        "fit_dates": len({r["u"] for r in chosen}),
                        "fit_rows": len(chosen),
                        "max_target_date": max(r["u"] for r in chosen),
                        "max_mature_date": max(r["mature"] for r in chosen),
                        "cutoff": start,
                        "model_sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
                    }
                folded = {
                    "plan_hash": base.digest(p),
                    "exam_hash": base.digest(exam),
                    "training": training,
                    "predictions": dict(predictions),
                }
                base.save(path, folded)
            for name in NAMES:
                predictions = folded["predictions"][name]
                if {(r["code"], r["u"], r["y"]) for r in predictions} != {
                    (r["code"], r["u"], r["y"]) for r in exam
                } or len(predictions) != len(exam):
                    raise ValueError("RESERVED_AUDIT_COMMON_EXAM_MISMATCH")
                output[name].extend(predictions)
            quarters[str(q)] = {n: base.metrics(folded["predictions"][n]) for n in NAMES}
            base.save(root() / "progress.json", {"at": base.now().isoformat(), "completed_quarter": q}, replace=True)
    sessions = [str(d) for d in base.calendar()[0] if "2026-01-01" <= str(d) < "2026-09-12"]
    eligible = len(base.read(base.ROOT / "history.json")["funds"])
    count = len(output[NAMES[0]])
    result = {
        "at": base.now().isoformat(),
        "kind": "ONE_TIME_RESERVED_HISTORICAL_AUDIT_NOT_TRUE_FORWARD",
        "plan_hash": base.digest(p),
        "held_out_2026_consumed": True,
        "metrics": {n: base.metrics(output[n]) for n in NAMES},
        "quarters": quarters,
        "groups": {g: {n: base.metrics([r for r in output[n] if r["group"] == g]) for n in NAMES} for g in groups},
        "paired_intervals": paired_intervals(output, p["bootstrap"]),
        "coverage": {
            "eligible_funds": eligible,
            "whole_watchlist_funds": 43,
            "calendar_target_dates": len(sessions),
            "answered_fund_dates": count,
            "eligible_coverage": count / (eligible * len(sessions)),
            "whole_watchlist_coverage": count / (43 * len(sessions)),
            "missing_eligible_fund_dates": eligible * len(sessions) - count,
        },
        "model_selection_changed": False,
        "future_model_binding_changed": False,
        "new_cost_cny": 0,
        "new_provider_calls": 0,
        "model_released": False,
    }
    base.save(root() / "result.json", result)
    return result
