"""固定11项输入的线性/浅层树对照；数值JSON保存树结构，不注册或替换在用模型。"""

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from app.services import direction_1d_sector_study as sector
from app.services import direction_1d_tree_audit as audit
from app.services.direction_1d_protocol import ZONE, digest
from app.services.direction_1d_training import read, weights, write_new
from app.services.direction_training_artifacts import file_hash

CANDIDATE = "TREE11"
REFERENCE = "LINEAR11"
QUARTERS = sector.QUARTERS
# 仅这一套候选：二层树最多四个叶子；禁用随机验证集与早停，100轮全部固定。
TREE_RECIPE = {
    "loss": "log_loss",
    "learning_rate": 0.05,
    "max_iter": 100,
    "max_leaf_nodes": 4,
    "max_depth": 2,
    "min_samples_leaf": 100,
    "l2_regularization": 10.0,
    "max_features": 1.0,
    "max_bins": 63,
    "categorical_features": None,
    "monotonic_cst": None,
    "interaction_cst": None,
    "warm_start": False,
    "early_stopping": False,
    "validation_fraction": None,
    "random_state": 0,
    "class_weight": None,
}


def fingerprint() -> dict:
    """旧实验指纹保持原样，另绑定本轮审计、树实现和命令入口。"""
    result = sector.fingerprint()
    project = Path(__file__).resolve().parents[2]
    for name in (
        "app/services/direction_1d_tree_study.py",
        "app/services/direction_1d_tree_audit.py",
        "scripts/direction_1d_tree_study.py",
    ):
        result[name] = file_hash(project / name)
    return result


def copy_file(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if source.is_symlink():
        raise ValueError("TREE_SOURCE_SYMLINK_REJECTED")
    with target.open("xb") as stream:
        stream.write(source.read_bytes())


def freeze(base: Path, audit_dir: Path, output: Path) -> dict:
    """原25只、5704题及504日FIT不变；新树仅5次主拟合和5次唯一复现。"""
    sector.verify(base)
    before, diagnostic = read(base / "study.json"), read(audit_dir / "audit.json")
    receipt = read(audit_dir / "audit-receipt.json")
    if receipt["code_hash"] != file_hash(Path(audit.__file__)) or receipt["audit_hash"] != digest(diagnostic):
        raise ValueError("TREE_DIAGNOSTIC_CHANGED")
    for name, expected in receipt["files"].items():
        if Path(name).name != name or file_hash(audit_dir / name) != expected:
            raise ValueError("TREE_DIAGNOSTIC_SOURCE_CHANGED")
    if (
        diagnostic["requires_data_repair"]
        or diagnostic["nav_mismatch_count"]
        or diagnostic["scope"] != before["fund_codes"]
        or diagnostic["exam_count"] != before["exam_count"]
        or diagnostic["base_study_hash"] != file_hash(base / "study.json")
    ):
        raise ValueError("TREE_DIAGNOSTIC_NOT_READY")
    now = datetime.now(ZONE)
    if not timedelta(0) <= now - datetime.fromisoformat(diagnostic["audited_at"]) <= timedelta(days=1):
        raise ValueError("TREE_DIAGNOSTIC_STALE")
    if read(base / "main/models.json") != read(base / "replay/models.json") or read(
        base / "main/predictions.json"
    ) != read(base / "replay/predictions.json"):
        raise ValueError("TREE_LINEAR_REPLAY_CHANGED")
    output.mkdir(parents=True, exist_ok=False)
    # 完整复制原研究的冻结输入和两次完成记录，使本包恢复不依赖旧目录仍存在。
    names = {
        *before["input_files"],
        "study.json",
        "study-receipt.json",
        "replay-proof.json",
        "comparison.json",
        "comparison-receipt.json",
    }
    names.update(
        p.relative_to(base).as_posix() for mode in ("main", "replay") for p in (base / mode).iterdir() if p.is_file()
    )
    frozen_files = []
    for name in sorted(names):
        if not (base / name).resolve().is_relative_to(base.resolve()):
            raise ValueError("TREE_BASE_PATH_INVALID")
        copy_file(base / name, output / "base" / name)
        frozen_files.append("base/" + name)
    for name in (*receipt["files"], "audit-receipt.json"):
        copy_file(audit_dir / name, output / "audit" / name)
        frozen_files.append("audit/" + name)
    spec = {
        "kind": "HISTORICAL_TREE_COMPARISON_ONLY",
        "created_at": now.isoformat(),
        "fingerprint": fingerprint(),
        "candidate": CANDIDATE,
        "reference": REFERENCE,
        "tree_recipe": TREE_RECIPE,
        "linear_recipe": before["recipe"],
        "linear_refitted": False,
        "features": sector.feature_names("SPECIFIC11"),
        "fund_codes": before["fund_codes"],
        "exam_count": before["exam_count"],
        "fit_hashes": before["fit_hashes"],
        "fit_counts": before["fit_counts"],
        "threshold": 0.5,
        "max_main_fits": 5,
        "max_replay_fits": 5,
        "primary_metric": "FAMILY_DATE_WEIGHTED_ACCURACY",
        "bootstrap_seed": 20260912,
        "bootstrap_repetitions": 2000,
        "bootstrap_block_days": 5,
        "input_files": {name: file_hash(output / name) for name in frozen_files},
        "source": before["source"],
        "protected_years_excluded": [2025, 2026],
        "model_released": False,
    }
    spec["cohort_id"] = "D1-TREE-" + digest({"inputs": spec["input_files"], "recipe": TREE_RECIPE})[:24]
    write_new(output / "study.json", spec)
    write_new(output / "study-receipt.json", {"hash": digest(spec)})
    return {k: spec[k] for k in ("cohort_id", "created_at", "fund_codes", "fit_counts", "exam_count", "max_main_fits")}


def verify_inputs(root: Path) -> dict:
    spec = read(root / "study.json")
    if (
        digest(spec) != read(root / "study-receipt.json")["hash"]
        or spec["fingerprint"] != fingerprint()
        or spec["tree_recipe"] != TREE_RECIPE
    ):
        raise ValueError("TREE_CODE_OR_RECIPE_CHANGED")
    for name, expected in spec["input_files"].items():
        path = root / name
        if not path.resolve().is_relative_to(root.resolve()) or path.is_symlink() or file_hash(path) != expected:
            raise ValueError("TREE_FROZEN_INPUT_CHANGED")
    if datetime.now(ZONE) >= datetime.fromisoformat(spec["source"]["source_expires_at"]):
        raise ValueError("TREE_SOURCE_RETENTION_EXPIRED")
    return spec


def export_trees(estimator: HistGradientBoostingClassifier) -> list[list[dict]]:
    """保存已含学习率的数值叶子和真实阈值；拒绝分类分裂，避免依赖可执行序列化文件。"""
    result = []
    for trees in estimator._predictors:
        if len(trees) != 1:
            raise ValueError("TREE_BINARY_TARGET_REQUIRED")
        rows = []
        for n in trees[0].nodes:
            if n["is_categorical"]:
                raise ValueError("TREE_CATEGORICAL_SPLIT_REJECTED")
            rows.append(
                {
                    "leaf": bool(n["is_leaf"]),
                    "value": float(n["value"]),
                    "feature": int(n["feature_idx"]),
                    "threshold": float(n["num_threshold"]),
                    "left": int(n["left"]),
                    "right": int(n["right"]),
                    "depth": int(n["depth"]),
                    "count": int(n["count"]),
                }
            )
        result.append(rows)
    return result


def validate_model(model: dict) -> None:
    """验证完整树拓扑、数值和配方；遇到缺树、循环、越界或非法值立即停止评分。"""
    if (
        model.get("candidate") != CANDIDATE
        or model.get("features") != sector.feature_names("SPECIFIC11")
        or model.get("recipe") != TREE_RECIPE
        or model.get("target_definition") != "UNIT_NAV_DIRECTION_V1"
    ):
        raise ValueError("TREE_MODEL_PROTOCOL_INVALID")
    values = [model["baseline"], *model["mean"], *model["scale"]]
    if (
        len(model["mean"]) != 11
        or len(model["scale"]) != 11
        or not np.isfinite(values).all()
        or min(model["scale"]) <= 0
    ):
        raise ValueError("TREE_MODEL_SCALE_INVALID")
    if len(model["trees"]) != TREE_RECIPE["max_iter"]:
        raise ValueError("TREE_MODEL_ROUNDS_INVALID")
    for nodes in model["trees"]:
        if not 1 <= len(nodes) <= 2 * TREE_RECIPE["max_leaf_nodes"] - 1:
            raise ValueError("TREE_MODEL_NODES_INVALID")
        seen, stack, leaves = set(), [(0, 0)], 0
        while stack:
            i, depth = stack.pop()
            if i in seen or not 0 <= i < len(nodes) or depth > TREE_RECIPE["max_depth"]:
                raise ValueError("TREE_MODEL_TOPOLOGY_INVALID")
            seen.add(i)
            n = nodes[i]
            if n["depth"] != depth or not np.isfinite([n["value"], n["threshold"]]).all() or n["count"] <= 0:
                raise ValueError("TREE_MODEL_NODE_INVALID")
            if n["leaf"]:
                leaves += 1
            else:
                if not 0 <= n["feature"] < 11 or n["left"] <= i or n["right"] <= i:
                    raise ValueError("TREE_MODEL_BRANCH_INVALID")
                stack.extend(((n["left"], depth + 1), (n["right"], depth + 1)))
        if len(seen) != len(nodes) or leaves > TREE_RECIPE["max_leaf_nodes"]:
            raise ValueError("TREE_MODEL_UNREACHABLE_NODE")


def predict(model: dict, rows: list[dict]) -> np.ndarray:
    """按原标准化及数值阈值逐树累加；每行每棵树只到一个叶子，输入缺失不补值。"""
    validate_model(model)
    if not rows:
        return np.asarray([], dtype=float)
    x = (np.asarray([sector.vector(r, "SPECIFIC11") for r in rows]) - np.asarray(model["mean"])) / np.asarray(
        model["scale"]
    )
    raw = np.full(len(rows), model["baseline"], dtype=float)
    for nodes in model["trees"]:
        stack = [(0, np.arange(len(rows)))]
        while stack:
            i, selected = stack.pop()
            if not len(selected):
                continue
            node = nodes[i]
            if node["leaf"]:
                raw[selected] += node["value"]
            else:
                left = x[selected, node["feature"]] <= node["threshold"]
                stack.extend(((node["left"], selected[left]), (node["right"], selected[~left])))
    return 1 / (1 + np.exp(-np.clip(raw, -700, 700)))


def fit(rows: list[dict], linear: dict, spec: dict, exam: list[dict]) -> dict:
    """只拟合新树；FIT与权重、Scaler必须同原线性版本，完整FIT和考题均校验JSON恢复。"""
    if digest(rows) != linear["fit_hash"] or digest(weights(rows).tolist()) != linear["weight_hash"]:
        raise ValueError("TREE_FIT_OR_WEIGHTS_CHANGED")
    if len({r["t"] for r in rows}) < 252 or min(sum(r["y"] == y for r in rows) for y in (0, 1)) < 30:
        raise ValueError("TREE_INSUFFICIENT_SAMPLES")
    x, y, w = (
        np.asarray([sector.vector(r, "SPECIFIC11") for r in rows]),
        np.asarray([r["y"] for r in rows]),
        weights(rows),
    )
    with threadpool_limits(limits=1):
        scaler = StandardScaler().fit(x, sample_weight=w)
        if scaler.mean_.tolist() != linear["mean"] or scaler.scale_.tolist() != linear["scale"]:
            raise ValueError("TREE_SCALER_CHANGED")
        estimator = HistGradientBoostingClassifier(**TREE_RECIPE).fit(scaler.transform(x), y, sample_weight=w)
        model = {
            "candidate": CANDIDATE,
            "kind": "RESEARCH_ONLY",
            "target_definition": "UNIT_NAV_DIRECTION_V1",
            "cohort_id": spec["cohort_id"],
            "model_released": False,
            "train_as_of": linear["train_as_of"],
            "features": sector.feature_names("SPECIFIC11"),
            "recipe": TREE_RECIPE,
            "mean": scaler.mean_.tolist(),
            "scale": scaler.scale_.tolist(),
            "baseline": float(estimator._baseline_prediction[0, 0]),
            "trees": export_trees(estimator),
            "fit_hash": digest(rows),
            "fit_count": len(rows),
            "weight_hash": digest(w.tolist()),
            "distinct_dates": len({r["t"] for r in rows}),
            "classes": {str(v): int(sum(y == v)) for v in (0, 1)},
        }
        differences = {}
        for name, checked in (("fit", rows), ("exam", exam)):
            if checked:
                values = np.asarray([sector.vector(r, "SPECIFIC11") for r in checked])
                differences[name] = float(
                    np.max(np.abs(predict(model, checked) - estimator.predict_proba(scaler.transform(values))[:, 1]))
                )
        if max(differences.values()) > 1e-12:
            raise ValueError("TREE_RESTORED_SCORES_MISMATCH")
        model["restore_max_score_diff"] = differences
    return model


def run(root: Path, *, replay: bool = False) -> dict:
    spec = verify_inputs(root)
    if replay:
        verify(root)
    out = root / ("replay" if replay else "main")
    out.mkdir(exist_ok=False)
    write_new(out / "budget-reserved.json", {"at": datetime.now(ZONE).isoformat(), "max_fits": 5})
    linear = read(root / "base/main/models.json")
    exam = read(root / "base/common-exam.json")
    models, scores = {}, {}
    for quarter in (*QUARTERS, "FINAL"):
        rows = sector.selected_fit(root / "base", quarter)
        if digest(rows) != spec["fit_hashes"][quarter]:
            raise ValueError("TREE_FIT_CHANGED")
        selected_exam = [r for r in exam if r["quarter"] == quarter]
        write_new(out / (quarter + "-reserved.json"), {"at": datetime.now(ZONE).isoformat(), "fit_hash": digest(rows)})
        model = fit(rows, linear["SPECIFIC11-" + quarter], spec, selected_exam)
        write_new(out / (quarter + ".json"), model)
        models[quarter] = model
        scores.update(
            {
                (r["fund_code"], r["t"]): float(p)
                for r, p in zip(selected_exam, predict(model, selected_exam), strict=True)
            }
        )
    predictions = []
    for old in read(root / "base/main/predictions.json"):
        s = scores[old["fund_code"], old["t"]]
        predictions.append(
            {
                **old,
                "scores": {**old["scores"], REFERENCE: old["scores"]["SPECIFIC11"], CANDIDATE: s},
                "directions": {
                    **old["directions"],
                    REFERENCE: old["directions"]["SPECIFIC11"],
                    CANDIDATE: int(s > 0.5),
                },
            }
        )
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
        if models != read(root / "main/models.json") or predictions != read(root / "main/predictions.json"):
            raise ValueError("TREE_REPLAY_MISMATCH")
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


def verify(root: Path) -> dict:
    spec = verify_inputs(root)
    expected_keys = {*QUARTERS, "FINAL"}
    exam = read(root / "base/common-exam.json")
    original = read(root / "base/main/predictions.json")
    for mode in ("main", "replay"):
        folder = root / mode
        if not (folder / "completion.json").exists():
            if mode == "main":
                raise ValueError("TREE_TRAINING_INCOMPLETE")
            continue
        complete, models, predictions = (
            read(folder / n) for n in ("completion.json", "models.json", "predictions.json")
        )
        if complete["models_hash"] != digest(models) or complete["predictions_hash"] != digest(predictions):
            raise ValueError("TREE_OUTPUT_CHANGED")
        if (
            set(models) != expected_keys
            or complete["successful_fits"] != 5
            or {p.name for p in folder.glob("*-reserved.json")}
            != {"budget-reserved.json", *(k + "-reserved.json" for k in expected_keys)}
        ):
            raise ValueError("TREE_FIT_BUDGET_INVALID")
        expected_scores = {}
        for quarter, model in models.items():
            if model != read(folder / (quarter + ".json")) or model["fit_hash"] != spec["fit_hashes"][quarter]:
                raise ValueError("TREE_MODEL_CHANGED")
            rows = [r for r in exam if r["quarter"] == quarter]
            expected_scores.update(
                {(r["fund_code"], r["t"]): float(s) for r, s in zip(rows, predict(model, rows), strict=True)}
            )
        if len(predictions) != spec["exam_count"]:
            raise ValueError("TREE_EXAM_CHANGED")
        for row, old in zip(predictions, original, strict=True):
            key = (row["fund_code"], row["t"])
            if (
                any(row[k] != old[k] for k in ("fund_code", "t", "u", "y", "quarter", "kind"))
                or row["scores"][REFERENCE] != old["scores"]["SPECIFIC11"]
                or row["scores"][CANDIDATE] != expected_scores[key]
                or row["directions"][CANDIDATE] != int(expected_scores[key] > 0.5)
            ):
                raise ValueError("TREE_RESTORED_OUTPUT_CHANGED")
    return read(root / "main/completion.json")


def evaluate(root: Path) -> dict:
    verify(root)
    proof = read(root / "replay-proof.json")
    if (
        not proof["models_equal"]
        or proof["max_score_diff"] != 0
        or read(root / "main/models.json") != read(root / "replay/models.json")
        or read(root / "main/predictions.json") != read(root / "replay/predictions.json")
    ):
        raise ValueError("TREE_REPLAY_REQUIRED")
    spec, rows = read(root / "study.json"), read(root / "main/predictions.json")
    branches = (REFERENCE, CANDIDATE, "GROUPED_7", "ALWAYS_UP", "ALWAYS_NON_UP", "INITIAL_MAJORITY", "MOMENTUM")
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
        "paired_old_group": sector.paired(rows, CANDIDATE, "GROUPED_7", spec),
        "runtime_model_changed": False,
        "model_released": False,
        "limitations": [
            "2021—2024已使用开发数据，并非独立验证",
            "只验证这一套浅树配方，不代表所有非线性模型",
            "历史公告/收盘可用时刻仍是假设",
            "未用诊断删题或调参",
            "真实未来效果待到期",
        ],
    }
    write_new(root / "comparison.json", result)
    write_new(root / "comparison-receipt.json", {"hash": digest(result), "at": datetime.now(ZONE).isoformat()})
    return result
