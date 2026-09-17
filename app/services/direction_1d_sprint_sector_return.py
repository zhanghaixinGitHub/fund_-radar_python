"""第五十七轮：相同市场与行业输入，学习下一日涨跌幅度后判断方向。

固定零截距岭回归5/155列，配对比较二元标签与连续目标；研究分数不是概率或承诺收益。
保留全部30只基金、5670道开发题，不用历史多轮择优冒充独立未来验证。
"""

import hashlib
import shutil
from collections import Counter, defaultdict
from datetime import date, datetime, time

import joblib
import numpy as np
from sklearn.linear_model import Ridge
from threadpoolctl import threadpool_limits

from app.services import direction_1d_sprint as base
from app.services import direction_1d_sprint_adaptive as adaptive
from app.services import direction_1d_sprint_cnya_data as cnya_data
from app.services import direction_1d_sprint_dual_us as dual
from app.services import direction_1d_sprint_etf_joint as majority
from app.services import direction_1d_sprint_etf_runtime_v2 as runtime
from app.services import direction_1d_sprint_fund_response as previous
from app.services import direction_1d_sprint_hk as hk
from app.services import direction_1d_sprint_hk_live as hk_live
from app.services import direction_1d_sprint_overnight as overnight
from app.services import direction_1d_sprint_return_target as regression
from app.services import direction_1d_sprint_sector_pooling as previous_round
from app.services import direction_1d_sprint_sequence as sequence
from app.services import direction_1d_sprint_sparse as sparse
from app.services import direction_1d_sprint_us_etf_data as us_etf_data
from app.services import direction_1d_sprint_us_sector_etf_data as sectors

CANDIDATES = ("SECTOR_RIDGE5_RET504", "SECTOR_RIDGE155_RET504")
FEATURE_INDICES = (0, 12, 17, 20, 21)
SHRINKAGE = 0.25
CODE_ORDER = (
    "001021",
    "001632",
    "002112",
    "002170",
    "004237",
    "004605",
    "005187",
    "005284",
    "005312",
    "006038",
    "006730",
    "007045",
    "007509",
    "007832",
    "007950",
    "008164",
    "008888",
    "008960",
    "010737",
    "011036",
    "011103",
    "013180",
    "013275",
    "013330",
    "013472",
    "014156",
    "015596",
    "016008",
    "017493",
    "160323",
)
LEARNED = CANDIDATES
FIRST_TARGET = "2026-09-16"
UP_THRESHOLD = 0.0
CONTROL = "HK_EXTRA12_ERR504"


def root():
    return base.ROOT / "round-57"


def active():
    regression.active()


def fingerprint():
    value = previous_round.fingerprint()
    for name in (
        "app/services/direction_1d_sprint_sector_return.py",
        "scripts/direction_1d_sprint_sector_return.py",
        "tests/test_direction_1d_sprint_sector_return.py",
    ):
        value["code"][name] = hashlib.sha256((base.PROJECT / name).read_bytes()).hexdigest()
    return value


def fund_index(code):
    """基金顺序来自原始30只研究范围并写入冻结源码；不在预测时接受新的基金编号。"""
    if code not in CODE_ORDER:
        raise ValueError("SECTOR_RETURN_UNKNOWN_CODE")
    return CODE_ORDER.index(code)


def vector(x, code, t, u, markets, etf_points, cn_points, sector_points):
    """原三信号之后追加两项完整美国交易时段行业涨跌及可用标记，最后绑定基金身份。"""
    return (
        majority.vector(x, code, t, u, markets, etf_points, cn_points)
        + sectors.features(t, u, sector_points)
        + [fund_index(code)]
    )


def live_vector(source, original, markets, etf_points, cn_points, sector_points):
    sequence.live_vector(source, original)
    return vector(
        source["x"], original["code"], original["base"], original["u"], markets, etf_points, cn_points, sector_points
    )


def available(z):
    return majority.available(z[:20]) and z[22] == 1


def selected_features(z, name):
    if (
        name not in CANDIDATES
        or len(z) != 24
        or not np.isfinite(z).all()
        or isinstance(z[23], bool)
        or not float(z[23]).is_integer()
        or not 0 <= z[23] < len(CODE_ORDER)
        or z[22] not in (0, 1)
        or (z[22] == 0 and any(z[20:22]))
    ):
        raise ValueError("SECTOR_RETURN_INPUT_INVALID")
    majority.selected_features(z[:20], majority.CANDIDATES[0])
    return [float(z[i]) for i in FEATURE_INDICES]


def design(values, scale, name):
    """五项按训练尺度缩放，基金交互在该尺度后乘0.25；全局对照直接使用同五项。"""
    scales = np.asarray(scale, dtype=float)
    if scales.shape != (5,) or not np.isfinite(scales).all() or min(scales) <= 0:
        raise ValueError("SECTOR_RETURN_SCALE_INVALID")
    core = np.asarray([selected_features(z, name) for z in values]) / scales
    if name == CANDIDATES[0]:
        return core
    out = np.zeros((len(values), 5 + 5 * len(CODE_ORDER)))
    out[:, :5] = core
    for i, z in enumerate(values):
        offset = 5 + 5 * int(z[23])
        out[i, offset : offset + 5] = SHRINKAGE * core[i]
    return out


def dataset():
    """沿用R56原始24列与题目，不往行记录附加标签破坏配对训练题哈希。"""
    return previous_round.dataset()


def target_values(rows):
    """仅由已成熟训练题的原始T/U净值构建连续标签；波动使用T日已知值，方向必须与原y一致。"""
    nav = base.read(base.ROOT / "history.json")
    lookup = {f["fund_code"]: {r["date"]: r["nav"] for r in f["rows"]} for f in nav["funds"]}
    values = []
    for row in rows:
        value = regression.normalized_target(lookup[row["code"]][row["t"]], lookup[row["code"]][row["u"]], row["x"])
        if int(value > 0) != row["y"]:
            raise ValueError("SECTOR_RETURN_RAW_DIRECTION_CHANGED")
        values.append(value)
    return np.asarray(values, dtype=float)


def plan():
    path = root() / "plan.json"
    if path.exists():
        value = base.read(path)
        if value["fingerprint"] != fingerprint() or value["calendar_hash"] != base.calendar()[1]:
            raise ValueError("ROUND_57_CODE_OR_CALENDAR_CHANGED")
        return value
    active()
    previous_round.models()
    prior = previous_round.plan()
    proposal = base.read(root() / "proposal-before-implementation.json")
    if (
        proposal["candidates"] != list(CANDIDATES)
        or proposal["budget"]
        != {
            "development_fits": 24,
            "current_fits": 6,
            "reproductions": 1,
            "new_provider_requests": 0,
            "new_cost_cny": 0,
        }
        or proposal["current_fit_cutoff"] != prior["current_fit_cutoff"]
    ):
        raise ValueError("SECTOR_RETURN_PROPOSAL_CHANGED")
    value = prior | {
        "at": base.now().isoformat(),
        "fingerprint": fingerprint(),
        "candidates": CANDIDATES,
        "hypothesis": "ContinuousnextNAV magnitude may preserve information discarded bybinaryclassification",
        "target": "Raw unit NAV T->adjacentCN U UP/NON_UP,flat separate;score>0 UP,zero NON_UP",
        "training_target": "clip(raw nextNAVreturn/max(T20dvol,.0001),-5,5),existing R7formula",
        "recipe": "Ridgealpha10,nointercept,cholesky,naturaldatefamilyweights,same5training scales,.25fundinteractions",
        "input_hashes": prior["input_hashes"] | {"round-57/proposal-before-implementation.json": base.digest(proposal)},
        "control": "Reuse pairedR56LR5/LR155 andR50majority;exact15maturetrainingwindows;nooldfit repeats",
        "live_budget": "Reuse R56sector andexistingETF/CNYAcaptures,0extraGET",
        "new_cost_cny": 0,
        "this_round_2026_scores_read": False,
        "limits": "Normalizedreturnresearchscore isnotprobability/promisedreturn;historicaldevelopmentonly",
    }
    base.save(path, value)
    for name in value["fingerprint"]["code"]:
        dest = root() / "code" / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(base.PROJECT / name, dest)
    return value


def fit(rows, name, cutoff):
    """五项市场输入与基金部分共享使用相同成熟训练题；不根据历史答案选择行业基金映射。"""
    if name not in CANDIDATES:
        raise ValueError("SECTOR_RETURN_RECIPE_INVALID")
    chosen = adaptive.training_rows(
        [r for r in rows if r["u"] < cutoff and r["mature"] < cutoff and available(r["z"])], cutoff, "MONTHLY_BAL504"
    )
    if len({r["u"] for r in chosen}) < 120:
        raise ValueError("SECTOR_RETURN_TRAINING_SHORT")
    core = np.asarray([selected_features(r["z"], name) for r in chosen])
    labels = np.asarray([r["y"] for r in chosen])
    y = target_values(chosen)
    if len(Counter(labels)) != 2 or min(Counter(labels).values()) < 20:
        raise ValueError("SECTOR_RETURN_CLASSES_INSUFFICIENT")
    weights = regression.weights(chosen)
    _, training_mean, scale = sequence.normalize_training(core, weights)
    x = design([r["z"] for r in chosen], scale, name)
    active()
    with threadpool_limits(limits=2):
        model = Ridge(alpha=10, fit_intercept=False, solver="cholesky")
        model.fit(x, y, sample_weight=weights)
    active()
    return {
        "model": model,
        "mean": [0.0] * 5,
        "scale": scale,
        "fit_hash": base.digest(chosen),
        "code_order": CODE_ORDER,
        "shrinkage": SHRINKAGE,
        "fit_rows": len(chosen),
        "fit_dates": len({r["u"] for r in chosen}),
        "fit_end": max(r["u"] for r in chosen),
        "max_mature_date": max(r["mature"] for r in chosen),
        "cutoff": cutoff,
        "weighted_up_rate": float(np.average(labels, weights=weights)),
        "weighted_return_mean": float(np.average(y, weights=weights)),
        "continuous_target_hash": base.digest(y.tolist()),
        "training_feature_mean_not_subtracted": training_mean,
        "training_target": "normalized_next_raw_NAV_return",
    }


def batch_answers(values, name, trained):
    if not values:
        return []
    for z in values:
        selected_features(z, name)
    choices = majority.batch_answers([z[:20] for z in values], majority.CANDIDATES[0], trained)
    indices = [i for i, z in enumerate(values) if available(z)]
    if indices:
        head = trained["us_etf"]
        if (
            head is None
            or tuple(head["code_order"]) != CODE_ORDER
            or head["shrinkage"] != SHRINKAGE
            or head["model"].n_features_in_ != (5 if name == CANDIDATES[0] else 155)
            or head["mean"] != [0.0] * 5
        ):
            raise ValueError("SECTOR_RETURN_MODEL_SCHEMA_CHANGED")
        x = design([values[i] for i in indices], head["scale"], name)
        scores = head["model"].predict(x)
        if not np.isfinite(scores).all():
            raise ValueError("SECTOR_RETURN_SCORE_INVALID")
        for i, score in zip(indices, scores, strict=True):
            baseline, prediction = int(values[i][0] >= 0), int(score > UP_THRESHOLD)
            choices[i] = {
                "research_score": float(score),
                "kind": "NORMALIZED_RETURN_RESEARCH_SCORE",
                "prediction": prediction,
                "baseline_prediction": baseline,
                "flipped": prediction != baseline,
            }
    return choices


def answer(z, name, trained=None):
    return batch_answers([z], name, trained)[0]


def fit_checkpoint(rows, name, cutoff, label):
    """沿用旧HK，另校验LR3对照训练题一致；本轮仅拟合新增部分共享模型一次。"""
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
    # LR3对照已在第50轮训练；逐模型核对相同成熟行，避免重训或换题提高成绩。
    prior_path = majority.root() / "checkpoints" / f"{label.replace(name, majority.LEARNED[0])}.joblib"
    prior_receipt = base.read(prior_path.with_suffix(".json"))
    if (
        prior_receipt["cutoff"] != cutoff
        or prior_receipt["sha256"] != hashlib.sha256(prior_path.read_bytes()).hexdigest()
    ):
        raise ValueError("SECTOR_RETURN_PRIOR_CHECKPOINT_CHANGED")
    prior_head = joblib.load(prior_path)["us_etf"]
    shared = adaptive.training_rows(
        [r for r in rows if r["u"] < cutoff and r["mature"] < cutoff and available(r["z"])], cutoff, "MONTHLY_BAL504"
    )
    if base.digest([r | {"z": r["z"][:20]} for r in shared]) != prior_head["fit_hash"]:
        raise ValueError("SECTOR_RETURN_LR3_CONTROL_ROWS_CHANGED")
    # 分类器与回归器逐基金/季度使用相同题目，区别仅在既定训练目标和损失函数。
    paired_name = previous_round.CANDIDATES[CANDIDATES.index(name)]
    paired_path = previous_round.root() / "checkpoints" / f"{label.replace(name, paired_name)}.joblib"
    paired_meta = base.read(paired_path.with_suffix(".json"))
    if (
        paired_meta["cutoff"] != cutoff
        or paired_meta["sha256"] != hashlib.sha256(paired_path.read_bytes()).hexdigest()
        or joblib.load(paired_path)["us_etf"]["fit_hash"] != base.digest(shared)
    ):
        raise ValueError("SECTOR_RETURN_CLASSIFIER_TRAINING_ROWS_CHANGED")
    base.save(attempt, {"at": base.now().isoformat(), "name": name, "cutoff": cutoff, "new_fit": name in LEARNED})
    us_etf = fit(rows, name, cutoff)
    trained = {
        "control": control,
        "us_etf": us_etf,
        "group": group,
        "control_fit_hash": control["fit_hash"],
        "new_fit_count": int(us_etf is not None),
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
            raise ValueError("ROUND_57_INPUT_CHANGED")
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
                output[majority.LEARNED[0]].extend(
                    base.read(majority.root() / f"folds/{q}-{group}-{majority.LEARNED[0]}.json")
                )
                parent_candidate = previous_round.CANDIDATES[0]
                output[parent_candidate].extend(
                    base.read(previous_round.root() / f"folds/{q}-{group}-{parent_candidate}.json")
                )
            base.save(root() / "progress.json", {"at": base.now().isoformat(), "quarter": q}, replace=True)
        output["ALWAYS_UP"] = [r | {"prediction": 1} for r in output["SPX_SIGN"]]
        expected = sorted((r["code"], r["u"], r["y"]) for r in output["SPX_SIGN"])
        if len(expected) != 5670 or any(
            sorted((r["code"], r["u"], r["y"]) for r in v) != expected for v in output.values()
        ):
            raise ValueError("ROUND_57_COMMON_EXAM_CHANGED")
        metrics = {n: base.metrics(v) for n, v in output.items()}
        winner = max(CANDIDATES, key=lambda n: (metrics[n]["accuracy"], -CANDIDATES.index(n)))
        active()
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
        or result["calendar_hash"] != base.calendar()[1]
        or hashlib.sha256(path.read_bytes()).hexdigest() != result["model_sha256"]
    ):
        raise ValueError("ROUND_57_MODEL_OR_CODE_CHANGED")
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
    sector_input = sectors.capture(base.now())
    cn_input_path = cnya_data.root() / target / "input.json"
    cn_input = cnya_data.load(target) if cn_input_path.exists() else None
    if hk_input is None or us_etf_input is None or cn_input is None or sector_input is None:
        return report()
    deadline = datetime.combine(date.fromisoformat(target), time(8, 30), base.ZONE)
    with threadpool_limits(limits=2):
        for path in paths:
            if base.now() >= deadline:
                break
            p5, _, source, original = dual.read_parent(path)
            z = live_vector(
                source, original, hk_input["rows"], us_etf_input["rows"], cn_input["rows"], sector_input["rows"]
            )
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
                "sector_input_hash": base.digest(sector_input),
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
            raise ValueError("ROUND_57_PARENT_OR_MODEL_CHANGED")
        hk_input = hk_live.load(value["u"])
        us_etf_input = us_etf_data.load(value["u"])
        cn_input = cnya_data.load(value["u"])
        sector_input = sectors.load(value["u"])
        if value["sector_input_hash"] != base.digest(sector_input):
            raise ValueError("SECTOR_RETURN_INPUT_CHANGED")
        if value["cnya_input_hash"] != base.digest(cn_input):
            raise ValueError("ETF_JOINT_CNYA_INPUT_CHANGED")
        if value["us_etf_input_hash"] != base.digest(us_etf_input):
            raise ValueError("US_ETF_MODEL_LIVE_INPUT_CHANGED")
        if value["hk_input_hash"] != base.digest(hk_input):
            raise ValueError("US_ETF_MODEL_LIVE_INPUT_CHANGED")
        if not np.allclose(
            value["z"],
            live_vector(
                source, original, hk_input["rows"], us_etf_input["rows"], cn_input["rows"], sector_input["rows"]
            ),
            rtol=0,
            atol=1e-12,
        ):
            raise ValueError("ROUND_57_VECTOR_CHANGED")
        expected_answers = {n: answer(value["z"], n, bundle[n][original["group"]]) for n in CANDIDATES}
        if not runtime.answers_match(value["answers"], expected_answers):
            raise ValueError("ROUND_57_SAVED_ANSWER_CHANGED")
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
