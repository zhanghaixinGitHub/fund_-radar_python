"""固定浅树的近期权重对照；只读旧研究，不登记模型，不改真实预测。"""

from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits

from app.services import direction_1d_tree_study as tree
from app.services.direction_1d_protocol import ZONE, calendar, digest
from app.services.direction_1d_training import read, weights, write_new
from app.services.direction_training_artifacts import file_hash

sector = tree.sector
CANDIDATE, REFERENCE = "TREE_RECENCY126", "TREE11"
QUARTERS = tree.QUARTERS
# 唯一预先固定的半衰期；全局保持权重总和，避免同时改变L2惩罚的相对强度。
WEIGHT_RECIPE = {
    "half_life_trading_days": 126,
    "anchor": "LATEST_T_IN_ORIGINAL_FIT",
    "base": "ORIGINAL_FAMILY_DATE_WEIGHTS",
    "normalization": "PRESERVE_ORIGINAL_TOTAL_WEIGHT",
    "scaler": "REUSE_ORIGINAL_FAMILY_DATE_CONTROL_SCALER",
    "evaluation_weights": "ORIGINAL_FAMILY_DATE_WEIGHTS",
}


def recency_weights(rows: list[dict]) -> tuple[np.ndarray, dict]:
    """返回逐行新权重及可复核摘要；按冻结交易日历计龄，不以当前时间给FINAL衰减。"""
    positions = {str(d): i for i, d in enumerate(calendar()[0]) if d.year <= 2024}
    if not rows or any(r["t"] not in positions for r in rows):
        raise ValueError("RECENCY_FIT_DATE_INVALID_OR_PROTECTED")
    anchor = max(r["t"] for r in rows)
    ages = np.asarray([positions[anchor] - positions[r["t"]] for r in rows], dtype=int)
    old = weights(rows)
    if not np.isfinite(old).all() or np.min(old) <= 0:
        raise ValueError("RECENCY_BASE_WEIGHT_INVALID")
    decayed = old * np.exp2(-ages / WEIGHT_RECIPE["half_life_trading_days"])
    multiplier = float(old.sum() / decayed.sum())
    new = decayed * multiplier
    if not np.isfinite(new).all() or np.min(new) <= 0 or not np.isclose(new.sum(), old.sum(), rtol=1e-13):
        raise ValueError("RECENCY_WEIGHT_TOTAL_INVALID")
    return new, {
        "anchor_t": anchor,
        "max_age_trading_days": int(ages.max()),
        "fit_hash": digest(rows),
        "age_hash": digest(ages.tolist()),
        "original_weight_hash": digest(old.tolist()),
        "weight_hash": digest(new.tolist()),
        "original_weight_sum": float(old.sum()),
        "weight_sum": float(new.sum()),
        "normalization_multiplier": multiplier,
        "min_weight": float(new.min()),
        "max_weight": float(new.max()),
        "original_effective_row_count": float(old.sum() ** 2 / (old @ old)),
        "effective_row_count": float(new.sum() ** 2 / (new @ new)),
    }


def fingerprint() -> dict:
    """扩展旧树指纹；旧实现及已有冻结包无需重新解释或降低校验。"""
    result = tree.fingerprint()
    project = Path(__file__).resolve().parents[2]
    for name in ("app/services/direction_1d_recency_study.py", "scripts/direction_1d_recency_study.py"):
        result[name] = file_hash(project / name)
    return result


def require_replay(base: Path) -> None:
    if (
        read(base / "main/models.json") != read(base / "replay/models.json")
        or read(base / "main/predictions.json") != read(base / "replay/predictions.json")
        or read(base / "replay-proof.json")["max_score_diff"] != 0
    ):
        raise ValueError("RECENCY_CONTROL_REPLAY_INVALID")


def freeze(base: Path, plan: Path, root: Path) -> dict:
    """冻结已完成树对照、全部输入、方案与五份权重摘要；不进行模型拟合。"""
    tree.verify(base)
    require_replay(base)
    old = read(base / "study.json")
    controls = read(base / "main/models.json")
    audits = {}
    for quarter in (*QUARTERS, "FINAL"):
        rows = sector.selected_fit(base / "base", quarter)
        _, info = recency_weights(rows)
        if (
            digest(rows) != old["fit_hashes"][quarter]
            or info["original_weight_hash"] != controls[quarter]["weight_hash"]
        ):
            raise ValueError("RECENCY_CONTROL_FIT_CHANGED")
        audits[quarter] = info
    root.mkdir(parents=True, exist_ok=False)
    names = {
        *old["input_files"],
        "study.json",
        "study-receipt.json",
        "replay-proof.json",
        "comparison.json",
        "comparison-receipt.json",
    }
    names.update(
        p.relative_to(base).as_posix() for mode in ("main", "replay") for p in (base / mode).iterdir() if p.is_file()
    )
    for name in sorted(names):
        if not (base / name).resolve().is_relative_to(base.resolve()):
            raise ValueError("RECENCY_CONTROL_PATH_INVALID")
        tree.copy_file(base / name, root / "base" / name)
    tree.copy_file(plan, root / "plan.md")
    files = {"base/" + n: file_hash(root / "base" / n) for n in sorted(names)}
    files["plan.md"] = file_hash(root / "plan.md")
    spec = {
        **{
            k: old[k]
            for k in (
                "features",
                "fund_codes",
                "exam_count",
                "fit_hashes",
                "fit_counts",
                "source",
                "protected_years_excluded",
                "primary_metric",
                "bootstrap_seed",
                "bootstrap_repetitions",
                "bootstrap_block_days",
            )
        },
        "kind": "HISTORICAL_RECENCY_COMPARISON_ONLY",
        "created_at": datetime.now(ZONE).isoformat(),
        "fingerprint": fingerprint(),
        "candidate": CANDIDATE,
        "reference": REFERENCE,
        "control_refitted": False,
        "tree_recipe": tree.TREE_RECIPE,
        "weight_recipe": WEIGHT_RECIPE,
        "weight_audits": audits,
        "threshold": 0.5,
        "max_main_fits": 5,
        "max_replay_fits": 5,
        "input_files": files,
        "model_released": False,
    }
    spec["cohort_id"] = "D1-RECENCY-" + digest({"inputs": files, "weights": WEIGHT_RECIPE})[:24]
    write_new(root / "study.json", spec)
    write_new(root / "study-receipt.json", {"hash": digest(spec)})
    return {k: spec[k] for k in ("cohort_id", "created_at", "exam_count", "fit_counts", "weight_audits")}


def verify_inputs(root: Path) -> dict:
    spec = read(root / "study.json")
    if (
        digest(spec) != read(root / "study-receipt.json")["hash"]
        or spec["fingerprint"] != fingerprint()
        or spec["tree_recipe"] != tree.TREE_RECIPE
        or spec["weight_recipe"] != WEIGHT_RECIPE
        or spec["candidate"] != CANDIDATE
        or spec["reference"] != REFERENCE
        or spec["model_released"]
    ):
        raise ValueError("RECENCY_CODE_OR_RECIPE_CHANGED")
    for name, expected in spec["input_files"].items():
        path = root / name
        if not path.resolve().is_relative_to(root.resolve()) or path.is_symlink() or file_hash(path) != expected:
            raise ValueError("RECENCY_FROZEN_INPUT_CHANGED")
    if datetime.now(ZONE) >= datetime.fromisoformat(spec["source"]["source_expires_at"]):
        raise ValueError("RECENCY_SOURCE_RETENTION_EXPIRED")
    return spec


def predict(model: dict, rows: list[dict]) -> np.ndarray:
    """权重只用于训练；显式校验候选身份后，复用旧树的严格数值结构与推理协议。"""
    if (
        model.get("candidate") != CANDIDATE
        or model.get("weight_recipe") != WEIGHT_RECIPE
        or model.get("model_released") is not False
    ):
        raise ValueError("RECENCY_MODEL_PROTOCOL_INVALID")
    return tree.predict({**model, "candidate": tree.CANDIDATE}, rows)


def fit(rows: list[dict], control: dict, spec: dict, quarter: str, exam: list[dict]) -> dict:
    """保持原FIT与标准化，仅拟合新权重；对全部训练行及考题核对JSON恢复误差。"""
    w, info = recency_weights(rows)
    if (
        info != spec["weight_audits"][quarter]
        or digest(rows) != control["fit_hash"]
        or info["original_weight_hash"] != control["weight_hash"]
    ):
        raise ValueError("RECENCY_FIT_OR_WEIGHT_CHANGED")
    tree.validate_model(control)
    cutoff = datetime.fromisoformat(control["train_as_of"])
    if any(datetime.fromisoformat(r["mature_at"]) > cutoff for r in rows):
        raise ValueError("RECENCY_FIT_LABEL_NOT_MATURE")
    if len({r["t"] for r in rows}) < 252 or min(sum(r["y"] == y for r in rows) for y in (0, 1)) < 30:
        raise ValueError("RECENCY_INSUFFICIENT_SAMPLES")
    mean, scale = np.asarray(control["mean"]), np.asarray(control["scale"])
    x = (np.asarray([sector.vector(r, "SPECIFIC11") for r in rows]) - mean) / scale
    y = np.asarray([r["y"] for r in rows])
    with threadpool_limits(limits=1):
        estimator = HistGradientBoostingClassifier(**tree.TREE_RECIPE).fit(x, y, sample_weight=w)
        model = {
            **{
                k: control[k]
                for k in (
                    "kind",
                    "target_definition",
                    "train_as_of",
                    "features",
                    "recipe",
                    "mean",
                    "scale",
                    "fit_hash",
                    "fit_count",
                    "distinct_dates",
                    "classes",
                )
            },
            "candidate": CANDIDATE,
            "cohort_id": spec["cohort_id"],
            "model_released": False,
            "weight_recipe": WEIGHT_RECIPE,
            "weight_audit": info,
            "weight_hash": info["weight_hash"],
            "control_model_hash": digest(control),
            "baseline": float(estimator._baseline_prediction[0, 0]),
            "trees": tree.export_trees(estimator),
        }
        differences = {}
        for name, checked in (("fit", rows), ("exam", exam)):
            if checked:
                values = (np.asarray([sector.vector(r, "SPECIFIC11") for r in checked]) - mean) / scale
                differences[name] = float(
                    np.max(np.abs(predict(model, checked) - estimator.predict_proba(values)[:, 1]))
                )
        if max(differences.values()) > 1e-12:
            raise ValueError("RECENCY_RESTORE_MISMATCH")
        model["restore_max_score_diff"] = differences
    return model


def expected_predictions(original: list[dict], scores: dict) -> list[dict]:
    """完整保留原标签、元数据和所有基线，只追加一个候选分数与方向。"""
    return [
        {
            **old,
            "scores": {**old["scores"], CANDIDATE: scores[old["fund_code"], old["t"]]},
            "directions": {**old["directions"], CANDIDATE: int(scores[old["fund_code"], old["t"]] > 0.5)},
        }
        for old in original
    ]


def run(root: Path, *, replay: bool = False) -> dict:
    spec = verify_inputs(root)
    if replay:
        verify(root)
    out = root / ("replay" if replay else "main")
    out.mkdir(exist_ok=False)
    write_new(out / "budget-reserved.json", {"at": datetime.now(ZONE).isoformat(), "max_fits": 5})
    controls = read(root / "base/main/models.json")
    exam = read(root / "base/base/common-exam.json")
    models, scores = {}, {}
    for quarter in (*QUARTERS, "FINAL"):
        rows = sector.selected_fit(root / "base/base", quarter)
        if digest(rows) != spec["fit_hashes"][quarter]:
            raise ValueError("RECENCY_FIT_CHANGED")
        selected = [r for r in exam if r["quarter"] == quarter]
        write_new(out / (quarter + "-reserved.json"), {"at": datetime.now(ZONE).isoformat(), "fit_hash": digest(rows)})
        model = fit(rows, controls[quarter], spec, quarter, selected)
        write_new(out / (quarter + ".json"), model)
        models[quarter] = model
        scores.update(
            {(r["fund_code"], r["t"]): float(s) for r, s in zip(selected, predict(model, selected), strict=True)}
        )
    predictions = expected_predictions(read(root / "base/main/predictions.json"), scores)
    write_new(out / "models.json", models)
    write_new(out / "predictions.json", predictions)
    result = {
        "finished_at": datetime.now(ZONE).isoformat(),
        "successful_fits": len(models),
        "models_hash": digest(models),
        "predictions_hash": digest(predictions),
        "model_released": False,
    }
    write_new(out / "completion.json", result)
    if replay:
        require_replay_outputs(root)
        write_new(
            root / "replay-proof.json",
            {
                "models_equal": True,
                "predictions_equal": True,
                "max_score_diff": 0.0,
                "successful_fits": 5,
                "prediction_count": len(predictions),
            },
        )
    return result


def require_replay_outputs(root: Path) -> None:
    if any(read(root / "main" / name) != read(root / "replay" / name) for name in ("models.json", "predictions.json")):
        raise ValueError("RECENCY_REPLAY_MISMATCH")


def verify(root: Path) -> dict:
    """不拟合地恢复所有分数，并重算FIT/权重与基线；同时检查每次预算预约和时间顺序。"""
    spec = verify_inputs(root)
    tree.verify(root / "base")
    require_replay(root / "base")
    quarters = (*QUARTERS, "FINAL")
    controls = read(root / "base/main/models.json")
    exam = read(root / "base/base/common-exam.json")
    original = read(root / "base/main/predictions.json")
    infos = {q: recency_weights(sector.selected_fit(root / "base/base", q))[1] for q in quarters}
    if infos != spec["weight_audits"]:
        raise ValueError("RECENCY_FROZEN_WEIGHTS_CHANGED")
    for mode in ("main", "replay"):
        folder = root / mode
        if not (folder / "completion.json").exists():
            if mode == "main" or folder.exists():
                raise ValueError("RECENCY_TRAINING_INCOMPLETE")
            continue
        complete, models, predictions = (
            read(folder / name) for name in ("completion.json", "models.json", "predictions.json")
        )
        if complete["models_hash"] != digest(models) or complete["predictions_hash"] != digest(predictions):
            raise ValueError("RECENCY_OUTPUT_CHANGED")
        budget = read(folder / "budget-reserved.json")
        start, end = datetime.fromisoformat(budget["at"]), datetime.fromisoformat(complete["finished_at"])
        if (
            set(models) != set(quarters)
            or budget["max_fits"] != 5
            or complete["successful_fits"] != 5
            or not datetime.fromisoformat(spec["created_at"]) <= start <= end
            or {p.name for p in folder.glob("*-reserved.json")}
            != {"budget-reserved.json", *(q + "-reserved.json" for q in quarters)}
        ):
            raise ValueError("RECENCY_FIT_BUDGET_INVALID")
        scores = {}
        for q, model in models.items():
            reserved = read(folder / (q + "-reserved.json"))
            old = controls[q]
            if (
                model != read(folder / (q + ".json"))
                or model["cohort_id"] != spec["cohort_id"]
                or model["fit_hash"] != spec["fit_hashes"][q]
                or model["weight_audit"] != infos[q]
                or model["weight_hash"] != infos[q]["weight_hash"]
                or model["control_model_hash"] != digest(old)
                or reserved["fit_hash"] != model["fit_hash"]
                or not start <= datetime.fromisoformat(reserved["at"]) <= end
                or any(
                    model[k] != old[k]
                    for k in ("mean", "scale", "fit_count", "train_as_of", "classes", "distinct_dates")
                )
                or not all(np.isfinite(v) and 0 <= v <= 1e-12 for v in model["restore_max_score_diff"].values())
                or set(model["restore_max_score_diff"]) != ({"fit"} if q == "FINAL" else {"fit", "exam"})
            ):
                raise ValueError("RECENCY_MODEL_CHANGED")
            selected = [r for r in exam if r["quarter"] == q]
            scores.update(
                {(r["fund_code"], r["t"]): float(s) for r, s in zip(selected, predict(model, selected), strict=True)}
            )
        if len(predictions) != spec["exam_count"] or predictions != expected_predictions(original, scores):
            raise ValueError("RECENCY_RESTORED_OUTPUT_CHANGED")
    if (root / "replay/completion.json").exists():
        require_replay_outputs(root)
    return read(root / "main/completion.json")


def evaluate(root: Path) -> dict:
    verify(root)
    require_replay_outputs(root)
    if read(root / "replay-proof.json") != {
        "models_equal": True,
        "predictions_equal": True,
        "max_score_diff": 0.0,
        "successful_fits": 5,
        "prediction_count": read(root / "study.json")["exam_count"],
    }:
        raise ValueError("RECENCY_REPLAY_PROOF_INVALID")
    spec, rows = read(root / "study.json"), read(root / "main/predictions.json")
    branches = (
        REFERENCE,
        CANDIDATE,
        "LINEAR11",
        "GROUPED_7",
        "ALWAYS_UP",
        "ALWAYS_NON_UP",
        "INITIAL_MAJORITY",
        "MOMENTUM",
    )
    result = {
        "kind": "DEVELOPMENT_COMPARISON_NOT_INDEPENDENT_TEST",
        "overall": {b: sector.comparison.metrics(rows, b) for b in branches},
        "strata": {
            field: {
                v: {b: sector.comparison.metrics([r for r in rows if r[field] == v], b) for b in branches}
                for v in sorted({r[field] for r in rows})
            }
            for field in ("quarter", "fund_code", "group")
        },
        "paired_primary": sector.paired(rows, CANDIDATE, REFERENCE, spec),
        "paired_linear": sector.paired(rows, CANDIDATE, "LINEAR11", spec),
        "runtime_model_changed": False,
        "model_released": False,
        "limitations": [
            "2021—2024为重复使用的开发数据，非独立验证",
            "只检验126交易日这一套权重，不搜索其他半衰期",
            "历史可用时刻仍有假设",
            "有效样本量是权重诊断，不是独立市场日数量",
            "真实未来效果待到期",
        ],
    }
    write_new(root / "comparison.json", result)
    write_new(root / "comparison-receipt.json", {"hash": digest(result), "at": datetime.now(ZONE).isoformat()})
    return result
