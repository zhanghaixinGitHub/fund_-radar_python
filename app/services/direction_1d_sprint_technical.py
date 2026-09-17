"""第十九轮：在一日港股纠错模型上追加六项有限历史技术变换。

仅比较一个新树方案；保持原模型的结构、权重和阈值，隔离新增特征的影响。
所有指标以基准日及以前61条原始单位净值计算，不是预测未来6/14/20/60日。
"""

import hashlib
import shutil
from collections import Counter, defaultdict
from datetime import date, datetime, time

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from threadpoolctl import threadpool_limits

from app.services import direction_1d_sprint as base
from app.services import direction_1d_sprint_adaptive as adaptive
from app.services import direction_1d_sprint_calibrated as previous_round
from app.services import direction_1d_sprint_dual_us as dual
from app.services import direction_1d_sprint_fund_response as previous
from app.services import direction_1d_sprint_hk as hk
from app.services import direction_1d_sprint_hk_live as hk_live
from app.services import direction_1d_sprint_overnight as overnight
from app.services import direction_1d_sprint_return_target as regression
from app.services import direction_1d_sprint_sequence as sequence
from app.services import direction_1d_sprint_sparse as sparse

CANDIDATES = ("MOM_EXTRA18_ERR504",)
LEARNED = CANDIDATES
FIRST_TARGET = "2026-09-16"
FLIP_THRESHOLD = 0.55


def root():
    return base.ROOT / "round-19"


def active():
    regression.active()


def fingerprint():
    value = previous_round.fingerprint()
    for name in (
        "app/services/direction_1d_sprint_technical.py",
        "scripts/direction_1d_sprint_technical.py",
        "tests/test_direction_1d_sprint_technical.py",
    ):
        value["code"][name] = hashlib.sha256((base.PROJECT / name).read_bytes()).hexdigest()
    return value


def ema(values, period):
    """有限61点EMA从首个观测值起步，不冒充无限历史或第三方完整指标。"""
    values = np.asarray(values, dtype=float)
    output = [float(values[0])]
    alpha = 2 / (period + 1)
    for value in values[1:]:
        output.append(alpha * float(value) + (1 - alpha) * output[-1])
    return np.asarray(output)


def technical_features(x, values):
    """固定六项变换的顺序、单位与常量处理；与原18项输入核对原始单位净值。

    RSI以净值差的前N项算术均值初始化，再按Wilder递推；输出(gain-loss)/(gain+loss)。
    EMA12-26及其EMA9信号差除以末日净值*已知20日收益波动，限于正负5；
    布林位置用20点净值总体标准差的两倍，限于正负3；14点区间位置映射到[-1,1]。
    原始60日窗口全平仍按旧规则拒绝，不扩大样本范围；通过原校验后，
    单个指标分母不超过最大净值的1e-12时取中性0，避免局部平盘产生除零。
    """
    sequence.vector(x, values)
    nav = np.asarray(values, dtype=float)
    delta = np.diff(nav)
    epsilon = float(max(nav)) * 1e-12
    result = []
    for period in (6, 14):
        gain = float(np.maximum(delta[:period], 0).mean())
        loss = float(np.maximum(-delta[:period], 0).mean())
        for change in delta[period:]:
            gain = ((period - 1) * gain + max(float(change), 0)) / period
            loss = ((period - 1) * loss + max(-float(change), 0)) / period
        result.append(0.0 if gain + loss <= epsilon else (gain - loss) / (gain + loss))
    macd = ema(nav, 12) - ema(nav, 26)
    signal = ema(macd, 9)
    unit = float(nav[-1]) * max(float(x[3]), 0.0001)
    result.extend([float(np.clip(macd[-1] / unit, -5, 5)), float(np.clip((macd[-1] - signal[-1]) / unit, -5, 5))])
    deviation = float(np.std(nav[-20:], ddof=0))
    result.append(
        0.0 if deviation <= epsilon else float(np.clip((nav[-1] - nav[-20:].mean()) / (2 * deviation), -3, 3))
    )
    low, high = float(min(nav[-14:])), float(max(nav[-14:]))
    result.append(0.0 if high - low <= epsilon else 2 * (float(nav[-1]) - low) / (high - low) - 1)
    if not np.isfinite(result).all():
        raise ValueError("TECHNICAL_FEATURE_NOT_FINITE")
    return result


def vector(x, t, u, markets, values):
    return hk.vector(x, t, u, markets) + technical_features(x, values)


def live_vector(source, original, markets):
    """只复用父预测中已验证的61条净值；错序和迟公告均拒绝。"""
    sequence.live_vector(source, original)
    return vector(source["x"], original["base"], original["u"], markets, [r["nav"] for r in original["inputs"]])


def selected_features(z, name):
    if name not in CANDIDATES or len(z) != 18 or not np.isfinite(z).all():
        raise ValueError("TECHNICAL_INPUT_INVALID")
    return z


def dataset():
    points = {f["fund_code"]: {r["date"]: r for r in f["rows"]} for f in base.read(base.ROOT / "history.json")["funds"]}
    rows, proof = hk.dataset()
    output, date_cache = [], {}
    for r in rows:
        if r["t"] not in date_cache:
            date_cache[r["t"]] = list(map(str, base.input_days(date.fromisoformat(r["t"]))))
        inputs = [points[r["code"]][d] for d in date_cache[r["t"]]]
        if any((v.get("ann_date") or v["date"]) > r["u"] for v in inputs):
            raise ValueError("TECHNICAL_INPUT_NOT_YET_ANNOUNCED")
        output.append(r | {"z": r["z"] + technical_features(r["x"], [v["nav"] for v in inputs])})
    return output, proof | {
        "at": base.now().isoformat(),
        "technical_input_length": 61,
        "target": "next CN day raw unit NAV; past indicator periods do not change forecast horizon",
    }


def plan():
    path = root() / "plan.json"
    if path.exists():
        value = base.read(path)
        if value["fingerprint"] != fingerprint() or value["calendar_hash"] != base.calendar()[1]:
            raise ValueError("ROUND_19_CODE_OR_CALENDAR_CHANGED")
        return value
    active()
    parent, _ = hk.models()
    value = {
        "at": base.now().isoformat(),
        "fingerprint": fingerprint(),
        "candidates": CANDIDATES,
        "hypothesis": (
            "Six fixed nonlinear NAV transforms may add useful inductive bias beyond existing summary/sequence inputs."
        ),
        "base_candidate": "HK_EXTRA12_ERR504",
        "parent_result_hash": base.digest(parent),
        "target": "raw unit NAV T to adjacent CN trading day U, strictly up versus NON_UP; flat separate",
        "technical_inputs": "original61unitNAV through T; same announcements and original vector validation",
        "features": [
            "WilderRSI6 centered[-1,1]",
            "WilderRSI14 centered[-1,1]",
            "finiteEMA12minus26 / NAV_T*vol20",
            "finiteMACD minusEMA9signal / NAV_T*vol20",
            "Bollinger20 /2populationstd",
            "stochastic14[-1,1]",
        ],
        "initialization": (
            "EMA starts first of61NAV values. RSI starts mean firstN NAV differences, then Wilder recursion."
        ),
        "original_eligibility": "Original flat60NAV window remains rejected; no changed sample coverage",
        "constant_epsilon": "Eligible input only: indicator denominator<=maxNAV*1e-12 gives neutral0",
        "normalization": "MACD scale is NAV_T times max(vol20,0.0001); clipMACD/hist +-5, Bollinger +-3",
        "extra_trees": {
            "trees": 256,
            "depth": 6,
            "min_leaf": 80,
            "max_features": 1.0,
            "bootstrap": False,
            "seed": 17,
            "threads": 2,
        },
        "weights": "date/family natural error prior, no class balance",
        "flip_threshold": 0.55,
        "training_dates": 504,
        "quarterly_refit": True,
        "development_year": 2025,
        "max_development_fits": 12,
        "max_current_fits": 3,
        "max_reproduction_count": 1,
        "expected_questions": 5670,
        "expected_dates": 199,
        "controls": ["HK_EXTRA12_ERR504", "SPX_SIGN", "ALWAYS_UP"],
        "selection": "one predeclared candidate; no parameter or threshold search",
        "calendar_hash": base.calendar()[1],
        "input_hashes": hk.plan()["input_hashes"],
        "current_fit_cutoff": str(base.now().date()),
        "first_forward_target": FIRST_TARGET,
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
    """严格沿用第17轮随机树配方；本轮唯一变化为追加六项历史净值变换。"""
    if name not in CANDIDATES:
        raise ValueError("TECHNICAL_RECIPE_INVALID")
    chosen = adaptive.training_rows(rows, cutoff, "MONTHLY_BAL504")
    x = np.asarray([selected_features(r["z"], name) for r in chosen])
    y = np.asarray([int(int(r["z"][0] >= 0) != r["y"]) for r in chosen])
    if len(Counter(y)) != 2 or min(Counter(y).values()) < 20:
        raise ValueError("TECHNICAL_ERROR_LABELS_INSUFFICIENT")
    weights = regression.weights(chosen)
    x, mean, scale = sequence.normalize_training(x, weights)
    active()
    model = ExtraTreesClassifier(
        n_estimators=256, max_depth=6, min_samples_leaf=80, max_features=1.0, bootstrap=False, random_state=17, n_jobs=2
    )
    model.fit(x, y, sample_weight=weights)
    active()
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
        "weighted_error_rate": float(np.average(y, weights=weights)),
    }


def batch_answers(values, name, trained):
    """同一模型批量推理，避免逐题调度线程；与实时单题使用同一阈值实现。"""
    if not values:
        return []
    x = (np.asarray([selected_features(z, name) for z in values]) - trained["mean"]) / trained["scale"]
    if not np.isfinite(x).all():
        raise ValueError("HK_MODEL_NORMALIZED_INPUT_INVALID")
    scores = trained["model"].predict_proba(x)[:, 1]
    if not np.isfinite(scores).all() or min(scores) < 0 or max(scores) > 1:
        raise ValueError("HK_MODEL_SCORE_INVALID")
    result = []
    for z, score in zip(values, scores, strict=True):
        value = {"research_score": float(score), "kind": "UNCALIBRATED_RESEARCH_SCORE"}
        baseline, flipped = int(z[0] >= 0), bool(score > FLIP_THRESHOLD)
        value |= {
            "prediction": 1 - baseline if flipped else baseline,
            "baseline_prediction": baseline,
            "flipped": flipped,
            "kind": "UNCALIBRATED_BASELINE_ERROR_SCORE",
        }
        result.append(value)
    return result


def answer(z, name, trained=None):
    return batch_answers([z], name, trained)[0]


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
            raise ValueError("HK_MODEL_CHECKPOINT_CHANGED")
        return joblib.load(path)
    if attempt.exists():
        raise ValueError("HK_MODEL_PREVIOUS_FIT_INTERRUPTED")
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
            raise ValueError("ROUND_19_INPUT_CHANGED")
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
                            {k: r[k] for k in ("code", "family", "group", "u", "y", "actual_direction")} | choice
                            for r, choice in zip(
                                exam, batch_answers([r["z"] for r in exam], name, trained), strict=True
                            )
                        ]
                        base.save(path, scored)
                        if trained:
                            base.save(
                                root() / f"training/{q}-{group}-{name}.json",
                                {k: v for k, v in trained.items() if k != "model"},
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
            raise ValueError("ROUND_19_COMMON_EXAM_CHANGED")
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
        "calendar_hash": p["calendar_hash"],
        "development_fits": 12,
        "current_fits": 3,
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
        raise ValueError("ROUND_19_MODEL_OR_CODE_CHANGED")
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
    hk_input = hk_live.capture(at)
    deadline = datetime.combine(date.fromisoformat(target), time(8, 30), base.ZONE)
    with threadpool_limits(limits=2):
        for path in paths:
            if base.now() >= deadline:
                break
            p5, _, source, original = dual.read_parent(path)
            z = live_vector(source, original, hk_input["rows"])
            choices = {n: answer(z, n, bundle[n][original["group"]]) for n in CANDIDATES}
            value = {
                "at": base.now().isoformat(),
                "u": target,
                "code": original["code"],
                "parent_hash": base.digest(p5),
                "source_hash": base.digest(source),
                "hk_input_hash": base.digest(hk_input),
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
            raise ValueError("ROUND_19_PARENT_OR_MODEL_CHANGED")
        hk_input = hk_live.load(value["u"])
        if value["hk_input_hash"] != base.digest(hk_input):
            raise ValueError("HK_MODEL_LIVE_INPUT_CHANGED")
        if not np.allclose(value["z"], live_vector(source, original, hk_input["rows"]), rtol=0, atol=1e-12):
            raise ValueError("ROUND_19_VECTOR_CHANGED")
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
