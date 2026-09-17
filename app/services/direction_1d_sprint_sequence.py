"""第十二轮：完整收益时序与小型神经网络的有界对照。

输入始终截止基准日；按训练集拟合标准化，固定迭代次数，不用考试结果提前停止。
沿用同一5670题、日期/家族权重与次日原始单位净值方向，分数不是正式上涨概率。
"""

import hashlib
import shutil
import warnings
from collections import defaultdict
from datetime import date, datetime, time

import joblib
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from threadpoolctl import threadpool_limits

from app.services import direction_1d_sprint as base
from app.services import direction_1d_sprint_adaptive as adaptive
from app.services import direction_1d_sprint_breadth as previous_round
from app.services import direction_1d_sprint_dual_us as dual
from app.services import direction_1d_sprint_fund_response as previous
from app.services import direction_1d_sprint_overnight as overnight
from app.services import direction_1d_sprint_return_target as regression
from app.services import direction_1d_sprint_sparse as sparse

CANDIDATES = ("LR22_SEQ252", "MLP22_16_E40", "MLP62_32x16_E60")
LEARNED = CANDIDATES
FIRST_TARGET = "2026-09-16"
RECIPES = {"LR22_SEQ252": (20, (), 0), "MLP22_16_E40": (20, (16,), 40), "MLP62_32x16_E60": (60, (32, 16), 60)}


def root():
    return base.ROOT / "round-12"


def active():
    regression.active()


def fingerprint():
    value = previous_round.fingerprint()
    for name in (
        "app/services/direction_1d_sprint_sequence.py",
        "scripts/direction_1d_sprint_sequence.py",
        "tests/test_direction_1d_sprint_sequence.py",
    ):
        value["code"][name] = hashlib.sha256((base.PROJECT / name).read_bytes()).hexdigest()
    return value


def vector(x, values):
    """61个净值生成按旧到新排列的60个日收益；只用T日已知20日波动归一化。

    每项限于正负5个波动单位，最低波动为0.0001；末两项保留SPX小数收益和会话数。
    与原18项净值特征核对，拒绝历史与实际推断中误用不同的净值窗口。
    """
    v = np.asarray(values, dtype=float)
    if len(x) != 32 or v.shape != (61,) or not np.isfinite(x).all() or not np.isfinite(v).all() or min(v) <= 0:
        raise ValueError("SEQUENCE_INPUT_INVALID")
    if not np.allclose(x[:18], base.vector(v, True), rtol=0, atol=1e-12):
        raise ValueError("SEQUENCE_NAV_WINDOW_CHANGED")
    returns = v[1:] / v[:-1] - 1
    z = np.clip(returns / max(float(x[3]), 0.0001), -5, 5).tolist() + list(x[30:32])
    if not np.isfinite(z).all():
        raise ValueError("SEQUENCE_FEATURE_NOT_FINITE")
    return z


def live_vector(source, original):
    """使用已验证父预测的61条原始输入，拒绝错序、跨期或公告日晚于目标日的净值。"""
    wanted = list(map(str, base.input_days(date.fromisoformat(original["base"]))))
    inputs = original["inputs"]
    if [r["date"] for r in inputs] != wanted or any((r.get("ann_date") or r["date"]) > original["u"] for r in inputs):
        raise ValueError("SEQUENCE_LIVE_DATES_INVALID")
    return vector(source["x"], [r["nav"] for r in inputs])


def dataset():
    points = {f["fund_code"]: {r["date"]: r for r in f["rows"]} for f in base.read(base.ROOT / "history.json")["funds"]}
    output, wanted_cache = [], {}
    for row in overnight.dataset():
        if row["t"] not in wanted_cache:
            wanted_cache[row["t"]] = list(map(str, base.input_days(date.fromisoformat(row["t"]))))
        inputs = [points[row["code"]][d] for d in wanted_cache[row["t"]]]
        if any((r.get("ann_date") or r["date"]) > row["u"] for r in inputs):
            raise ValueError("SEQUENCE_HISTORICAL_INPUT_NOT_AVAILABLE")
        output.append(row | {"z": vector(row["x"], [r["nav"] for r in inputs])})
    return output, {
        "at": base.now().isoformat(),
        "rows": len(output),
        "input_order": "oldest-to-newest through T, SPX, sessions",
    }


def plan():
    path = root() / "plan.json"
    if path.exists():
        value = base.read(path)
        if value["fingerprint"] != fingerprint():
            raise ValueError("ROUND_12_CODE_CHANGED")
        return value
    active()
    value = {
        "at": base.now().isoformat(),
        "fingerprint": fingerprint(),
        "candidates": CANDIDATES,
        "hypothesis": "Return order may add information beyond summaries; fixed small MLP versus linear control.",
        "recipes": RECIPES,
        "training_dates": 252,
        "development_year": 2025,
        "quarterly_refit": True,
        "max_development_fits": 36,
        "max_current_fits": 9,
        "max_reproduction_count": 1,
        "feature_policy": "returns old-to-new / max(T vol20,0.0001), clip +-5; then SPX, sessions",
        "normalization": "train-only weighted mean and centered-square variance, constant variance<=1e-24 scale1",
        "training_weights": "date/family equal then class-balanced, original weight total preserved",
        "logistic_C": 0.1,
        "neural": {
            "solver": "adam",
            "learning_rate": 0.001,
            "alpha": 0.1,
            "batch": 256,
            "seed": 17,
            "activation": "relu",
            "epochs": "fixed per recipe",
            "early_stopping": False,
        },
        "thresholds": ">0.5; uncalibrated research scores only",
        "controls": ["SPX_SIGN", "LR2_BAL252", "ALWAYS_UP"],
        "selection": "highest 2025 common5670 date-family accuracy among three; no retuning",
        "input_hashes": {
            n: base.digest(base.read(base.ROOT / n))
            for n in ("history.json", "round-02/market.json", "round-03/spx.json")
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


def normalize_training(x, weights):
    """均值和方差只看训练集；直接计算中心平方，避免常量列因浮点抵消得到负方差。"""
    if x.ndim != 2 or not np.isfinite(x).all() or not np.isfinite(weights).all() or min(weights) <= 0:
        raise ValueError("SEQUENCE_SCALE_INPUT_INVALID")
    mean = np.average(x, weights=weights, axis=0)
    variance = np.average((x - mean) ** 2, weights=weights, axis=0)
    scale = np.where(variance <= 1e-24, 1.0, np.sqrt(variance))
    return (x - mean) / scale, mean.tolist(), scale.tolist()


def selected_features(z, name):
    if name not in RECIPES or len(z) != 62 or not np.isfinite(z).all():
        raise ValueError("SEQUENCE_ANSWER_INPUT_INVALID")
    return z[-(RECIPES[name][0] + 2) :]


def fit(rows, name, cutoff):
    """每次拟合前排除未成熟标签；神经网络固定循环次数并逐轮检查三天研究截止时间。"""
    chosen = adaptive.training_rows(rows, cutoff, "MONTHLY_BAL252")
    x = np.asarray([selected_features(r["z"], name) for r in chosen])
    y = np.asarray([r["y"] for r in chosen])
    weights = sparse.training_weights(chosen)
    x, mean, scale = normalize_training(x, weights)
    _, hidden, epochs = RECIPES[name]
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        if hidden:
            model = MLPClassifier(
                hidden_layer_sizes=hidden,
                solver="adam",
                activation="relu",
                alpha=0.1,
                learning_rate_init=0.001,
                batch_size=min(256, len(chosen)),
                random_state=17,
                early_stopping=False,
                tol=0,
                n_iter_no_change=epochs + 1,
            )
            for _ in range(epochs):
                active()
                model.partial_fit(x, y, classes=np.array([0, 1]), sample_weight=weights)
            losses = [float(v) for v in model.loss_curve_]
            if len(losses) != epochs or not np.isfinite(losses).all():
                raise ValueError("SEQUENCE_EPOCH_OR_LOSS_INVALID")
        else:
            active()
            model = LogisticRegression(C=0.1, max_iter=1500, random_state=17)
            model.fit(x, y, sample_weight=weights)
            losses = []
    return {
        "model": model,
        "mean": mean,
        "scale": scale,
        "loss_curve": losses,
        "epochs": epochs,
        "fit_hash": base.digest(chosen),
        "fit_rows": len(chosen),
        "fit_dates": len({r["u"] for r in chosen}),
        "fit_end": max(r["u"] for r in chosen),
        "max_mature_date": max(r["mature"] for r in chosen),
        "cutoff": cutoff,
    }


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
            raise ValueError("SEQUENCE_CHECKPOINT_CHANGED")
        return joblib.load(path)
    if attempt.exists():
        raise ValueError("SEQUENCE_PREVIOUS_FIT_INTERRUPTED")
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


def answer(z, name, trained=None):
    x = (np.asarray(selected_features(z, name)) - trained["mean"]) / trained["scale"]
    if not np.isfinite(x).all():
        raise ValueError("SEQUENCE_NORMALIZED_INPUT_INVALID")
    score = float(trained["model"].predict_proba([x])[0, 1])
    if not np.isfinite(score) or not 0 <= score <= 1:
        raise ValueError("SEQUENCE_SCORE_INVALID")
    return {"prediction": int(score > 0.5), "research_score": score, "kind": "UNCALIBRATED_RESEARCH_SCORE"}


def train():
    p = plan()
    if (root() / "result.json").exists():
        return models()[0]
    active()
    for name, expected in p["input_hashes"].items():
        if base.digest(base.read(base.ROOT / name)) != expected:
            raise ValueError("ROUND_12_INPUT_CHANGED")
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
                for control in ("SPX_SIGN", "LR2_BAL252"):
                    output[control].extend(base.read(sparse.root() / f"folds/{q}-{group}-{control}.json"))
            base.save(root() / "progress.json", {"at": base.now().isoformat(), "quarter": q}, replace=True)
        output["ALWAYS_UP"] = [r | {"prediction": 1} for r in output["SPX_SIGN"]]
        expected = sorted((r["code"], r["u"], r["y"]) for r in output["SPX_SIGN"])
        if len(expected) != 5670 or any(
            sorted((r["code"], r["u"], r["y"]) for r in v) != expected for v in output.values()
        ):
            raise ValueError("ROUND_12_COMMON_EXAM_CHANGED")
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
        raise ValueError("ROUND_12_MODEL_OR_CODE_CHANGED")
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
    deadline = datetime.combine(date.fromisoformat(target), time(8, 30), base.ZONE)
    with threadpool_limits(limits=2):
        for path in paths:
            if base.now() >= deadline:
                break
            p5, _, source, original = dual.read_parent(path)
            z = live_vector(source, original)
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
            raise ValueError("ROUND_12_PARENT_OR_MODEL_CHANGED")
        if not np.allclose(value["z"], live_vector(source, original), rtol=0, atol=1e-12):
            raise ValueError("ROUND_12_VECTOR_CHANGED")
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
