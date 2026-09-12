"""只增加沪深300成交活跃度的固定树对照；旧模型与真实预测只读。"""

from datetime import date, datetime, time
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from app.services import direction_1d_activity_data as acquisition
from app.services import direction_1d_tree_study as tree
from app.services.direction_1d_protocol import ZONE, calendar, digest
from app.services.direction_1d_training import read, weights, write_new
from app.services.direction_training_artifacts import file_hash

sector = tree.sector
CANDIDATE, REFERENCE = "TREE_ACTIVITY12", "TREE11"
QUARTERS = tree.QUARTERS
FEATURES = [*sector.feature_names("SPECIFIC11"), "hs300_amount_relative_previous_20d"]
FEATURE_RECIPE = {
    "formula": "HS300_AMOUNT_T_DIV_MEAN_PREVIOUS_20_SESSIONS_MINUS_ONE",
    "unit": "FRACTION",
    "input_dates": "PREVIOUS_20_SESSIONS_AND_T_ONLY",
    "availability": "T_18_ASSUMED",
    "scaler": "KEEP_CONTROL_11_APPEND_FIT_WEIGHTED_1",
    "weights": "ORIGINAL_FAMILY_DATE_NO_RECENCY",
}


def fingerprint() -> dict:
    result = tree.fingerprint()
    project = Path(__file__).resolve().parents[2]
    for name in (
        "app/services/direction_1d_activity_study.py",
        "app/services/direction_1d_activity_data.py",
        "scripts/direction_1d_activity_study.py",
    ):
        result[name] = file_hash(project / name)
    return result


def attach(rows: list[dict], series: dict) -> list[dict]:
    """仅使用截至T的21个连续交易日；分母为T之前20日，不含T或目标U日。"""
    days = [str(d) for d in calendar()[0] if d.year <= 2024]
    positions = {d: i for i, d in enumerate(days)}
    result, cached = [], {}
    for row in rows:
        t = row["t"]
        if t not in positions or positions[t] < 20 or "activity_input" in row:
            raise ValueError("ACTIVITY_WINDOW_DATE_INVALID")
        if t not in cached:
            wanted = days[positions[t] - 20 : positions[t] + 1]
            if any(d not in series for d in wanted):
                raise ValueError("ACTIVITY_WINDOW_INCOMPLETE")
            selected = [series[d] for d in wanted]
            if any(
                r["index_code"] != acquisition.INDEX or r["date"] != d for d, r in zip(wanted, selected, strict=True)
            ):
                raise ValueError("ACTIVITY_WINDOW_IDENTITY_INVALID")
            values = np.asarray([float(r["amount"]) for r in selected])
            if not np.isfinite(values).all() or np.min(values) <= 0:
                raise ValueError("ACTIVITY_WINDOW_VALUE_INVALID")
            denominator = float(values[:-1].mean())
            value = float(values[-1] / denominator - 1)
            if not np.isfinite([denominator, value]).all() or denominator <= 0:
                raise ValueError("ACTIVITY_WINDOW_RETURN_INVALID")
            cached[t] = {
                "x": [value],
                "index_code": acquisition.INDEX,
                "window_dates": wanted,
                "source_hash": digest(selected),
                "available_at_assumed": datetime.combine(date.fromisoformat(t), time(18), ZONE).isoformat(),
            }
        result.append({**row, "activity_input": cached[t]})
    return result


def vector(row: dict) -> list[float]:
    old = sector.vector(row, "SPECIFIC11")
    extra = row["activity_input"]["x"]
    if len(extra) != 1 or not np.isfinite(extra).all():
        raise ValueError("ACTIVITY_FEATURE_INVALID")
    return [*old, *extra]


def control_rows(rows: list[dict]) -> list[dict]:
    return [{k: v for k, v in row.items() if k != "activity_input"} for row in rows]


def inputs(base: Path, series: dict) -> tuple[dict, list[dict]]:
    """原FIT和考题重新派生成交额特征；晚到、缺日数据使整轮停止，不按成绩删题。"""
    controls = read(base / "main/models.json")
    fits = {}
    for q in (*QUARTERS, "FINAL"):
        original = sector.selected_fit(base / "base", q)
        if digest(original) != controls[q]["fit_hash"]:
            raise ValueError("ACTIVITY_CONTROL_FIT_CHANGED")
        fits[q] = attach(original, series)
        cutoff = datetime.fromisoformat(controls[q]["train_as_of"])
        if any(
            datetime.fromisoformat(r["activity_input"]["available_at_assumed"]) > cutoff
            or datetime.fromisoformat(r["mature_at"]) > cutoff
            for r in fits[q]
        ):
            raise ValueError("ACTIVITY_FIT_NOT_AVAILABLE")
    exam = attach(read(base / "base/common-exam.json"), series)
    if any(
        datetime.fromisoformat(r["activity_input"]["available_at_assumed"])
        > datetime.combine(date.fromisoformat(r["u"]), time(8), ZONE)
        for r in exam
    ):
        raise ValueError("ACTIVITY_EXAM_NOT_AVAILABLE")
    return fits, exam


def require_control(base: Path) -> None:
    tree.verify(base)
    if any(read(base / "main" / n) != read(base / "replay" / n) for n in ("models.json", "predictions.json")):
        raise ValueError("ACTIVITY_CONTROL_REPLAY_INVALID")


def freeze(base: Path, data_root: Path, plan: Path, root: Path) -> dict:
    """先核验四次数据回执与旧收盘，再冻结完整12维输入；此步骤不拟合。"""
    require_control(base)
    payload = acquisition.verify(data_root)
    budget, seal = read(data_root / "budget.json"), read(data_root / "data-seal.json")
    if budget["plan_hash"] != file_hash(plan) or budget["base_study_hash"] != file_hash(base / "study.json"):
        raise ValueError("ACTIVITY_PLAN_OR_BASE_CHANGED")
    if read(data_root / "old-prices.json") != read(base / "base/prices.json")["prices"][acquisition.INDEX]:
        raise ValueError("ACTIVITY_REFERENCE_PRICES_CHANGED")
    before = read(base / "study.json")
    fits, exam = inputs(base, payload["rows"])
    root.mkdir(parents=True, exist_ok=False)
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
    for name in sorted(names):
        if not (base / name).resolve().is_relative_to(base.resolve()):
            raise ValueError("ACTIVITY_CONTROL_PATH_INVALID")
        tree.copy_file(base / name, root / "base" / name)
    for name in (*seal["files"], "data-seal.json", "data-seal-receipt.json"):
        if Path(name).name != name:
            raise ValueError("ACTIVITY_DATA_PATH_INVALID")
        tree.copy_file(data_root / name, root / "data" / name)
    tree.copy_file(plan, root / "plan.md")
    write_new(root / "fits.json", fits)
    write_new(root / "exam.json", exam)
    files = {"base/" + name: file_hash(root / "base" / name) for name in sorted(names)}
    files.update(
        {
            "data/" + name: file_hash(root / "data" / name)
            for name in (*seal["files"], "data-seal.json", "data-seal-receipt.json")
        }
    )
    files.update({name: file_hash(root / name) for name in ("plan.md", "fits.json", "exam.json")})
    source = {
        **before["source"],
        "activity_source": budget["source"],
        "activity_data_seal_hash": digest(seal),
        "activity_source_expires_at": payload["source_expires_at"],
        "activity_api_calls": 4,
        "source_expires_at": min(
            datetime.fromisoformat(before["source"]["source_expires_at"]),
            datetime.fromisoformat(payload["source_expires_at"]),
        ).isoformat(),
    }
    spec = {
        **{
            k: before[k]
            for k in (
                "fund_codes",
                "exam_count",
                "fit_counts",
                "protected_years_excluded",
                "primary_metric",
                "bootstrap_seed",
                "bootstrap_repetitions",
                "bootstrap_block_days",
            )
        },
        "source": source,
        "kind": "HISTORICAL_ACTIVITY_COMPARISON_ONLY",
        "created_at": datetime.now(ZONE).isoformat(),
        "fingerprint": fingerprint(),
        "candidate": CANDIDATE,
        "reference": REFERENCE,
        "features": FEATURES,
        "feature_recipe": FEATURE_RECIPE,
        "tree_recipe": tree.TREE_RECIPE,
        "original_fit_hashes": before["fit_hashes"],
        "fit_hashes": {q: digest(r) for q, r in fits.items()},
        "weight_hashes": {q: digest(weights(r).tolist()) for q, r in fits.items()},
        "threshold": 0.5,
        "max_main_fits": 5,
        "max_replay_fits": 5,
        "control_refitted": False,
        "model_released": False,
        "input_files": files,
    }
    spec["cohort_id"] = "D1-ACTIVITY-" + digest({"inputs": files, "feature": FEATURE_RECIPE})[:24]
    write_new(root / "study.json", spec)
    write_new(root / "study-receipt.json", {"hash": digest(spec)})
    return {k: spec[k] for k in ("cohort_id", "created_at", "exam_count", "fit_counts", "features")}


def verify_inputs(root: Path) -> dict:
    spec = read(root / "study.json")
    if (
        digest(spec) != read(root / "study-receipt.json")["hash"]
        or spec["fingerprint"] != fingerprint()
        or spec["tree_recipe"] != tree.TREE_RECIPE
        or spec["feature_recipe"] != FEATURE_RECIPE
        or spec["features"] != FEATURES
        or spec["candidate"] != CANDIDATE
        or spec["reference"] != REFERENCE
        or spec["model_released"]
    ):
        raise ValueError("ACTIVITY_CODE_OR_RECIPE_CHANGED")
    for name, expected in spec["input_files"].items():
        path = root / name
        if not path.resolve().is_relative_to(root.resolve()) or path.is_symlink() or file_hash(path) != expected:
            raise ValueError("ACTIVITY_FROZEN_INPUT_CHANGED")
    if datetime.now(ZONE) >= datetime.fromisoformat(spec["source"]["source_expires_at"]):
        raise ValueError("ACTIVITY_SOURCE_RETENTION_EXPIRED")
    return spec


def validate_model(model: dict) -> None:
    """12维模型独立验证；不修改旧11维验证器，缺树、非法节点或输入配方变化一律拒绝。"""
    if (
        model.get("candidate") != CANDIDATE
        or model.get("features") != FEATURES
        or model.get("recipe") != tree.TREE_RECIPE
        or model.get("feature_recipe") != FEATURE_RECIPE
        or model.get("target_definition") != "UNIT_NAV_DIRECTION_V1"
        or model.get("model_released") is not False
    ):
        raise ValueError("ACTIVITY_MODEL_PROTOCOL_INVALID")
    if (
        len(model["mean"]) != 12
        or len(model["scale"]) != 12
        or not np.isfinite([model["baseline"], *model["mean"], *model["scale"]]).all()
        or min(model["scale"]) <= 0
    ):
        raise ValueError("ACTIVITY_MODEL_SCALE_INVALID")
    if len(model["trees"]) != tree.TREE_RECIPE["max_iter"]:
        raise ValueError("ACTIVITY_MODEL_ROUNDS_INVALID")
    for nodes in model["trees"]:
        if not 1 <= len(nodes) <= 2 * tree.TREE_RECIPE["max_leaf_nodes"] - 1:
            raise ValueError("ACTIVITY_MODEL_NODES_INVALID")
        seen, stack, leaves = set(), [(0, 0)], 0
        while stack:
            i, depth = stack.pop()
            if i in seen or not 0 <= i < len(nodes) or depth > tree.TREE_RECIPE["max_depth"]:
                raise ValueError("ACTIVITY_MODEL_TOPOLOGY_INVALID")
            seen.add(i)
            n = nodes[i]
            if n["depth"] != depth or not np.isfinite([n["value"], n["threshold"]]).all() or n["count"] <= 0:
                raise ValueError("ACTIVITY_MODEL_NODE_INVALID")
            if n["leaf"]:
                leaves += 1
            else:
                if not 0 <= n["feature"] < 12 or n["left"] <= i or n["right"] <= i:
                    raise ValueError("ACTIVITY_MODEL_BRANCH_INVALID")
                stack.extend(((n["left"], depth + 1), (n["right"], depth + 1)))
        if len(seen) != len(nodes) or leaves > tree.TREE_RECIPE["max_leaf_nodes"]:
            raise ValueError("ACTIVITY_MODEL_UNREACHABLE_NODE")


def predict(model: dict, rows: list[dict]) -> np.ndarray:
    """与旧树相同的数值阈值/叶子累加，但明确使用12维输入，不用pickle恢复。"""
    validate_model(model)
    if not rows:
        return np.asarray([], dtype=float)
    x = (np.asarray([vector(r) for r in rows]) - np.asarray(model["mean"])) / np.asarray(model["scale"])
    raw = np.full(len(rows), model["baseline"], dtype=float)
    for nodes in model["trees"]:
        stack = [(0, np.arange(len(rows)))]
        while stack:
            i, selected = stack.pop()
            if not len(selected):
                continue
            n = nodes[i]
            if n["leaf"]:
                raw[selected] += n["value"]
            else:
                left = x[selected, n["feature"]] <= n["threshold"]
                stack.extend(((n["left"], selected[left]), (n["right"], selected[~left])))
    return 1 / (1 + np.exp(-np.clip(raw, -700, 700)))


def fit(rows: list[dict], control: dict, spec: dict, quarter: str, exam: list[dict]) -> dict:
    """仅新增特征维度；原11维尺度和全体样本权重逐值保持，完整FIT/考题核对恢复。"""
    w = weights(rows)
    if (
        digest(rows) != spec["fit_hashes"][quarter]
        or digest(control_rows(rows)) != control["fit_hash"]
        or digest(w.tolist()) != control["weight_hash"]
        or digest(w.tolist()) != spec["weight_hashes"][quarter]
    ):
        raise ValueError("ACTIVITY_FIT_OR_WEIGHT_CHANGED")
    tree.validate_model(control)
    cutoff = datetime.fromisoformat(control["train_as_of"])
    if any(
        datetime.fromisoformat(r["mature_at"]) > cutoff
        or datetime.fromisoformat(r["activity_input"]["available_at_assumed"]) > cutoff
        for r in rows
    ):
        raise ValueError("ACTIVITY_FIT_NOT_AVAILABLE")
    if len({r["t"] for r in rows}) < 252 or min(sum(r["y"] == y for r in rows) for y in (0, 1)) < 30:
        raise ValueError("ACTIVITY_INSUFFICIENT_SAMPLES")
    values, y = np.asarray([vector(r) for r in rows]), np.asarray([r["y"] for r in rows])
    with threadpool_limits(limits=1):
        extra_scaler = StandardScaler().fit(values[:, -1:], sample_weight=w)
        mean = [*control["mean"], float(extra_scaler.mean_[0])]
        scale = [*control["scale"], float(extra_scaler.scale_[0])]
        x = (values - np.asarray(mean)) / np.asarray(scale)
        estimator = HistGradientBoostingClassifier(**tree.TREE_RECIPE).fit(x, y, sample_weight=w)
        model = {
            **{
                k: control[k]
                for k in (
                    "kind",
                    "target_definition",
                    "train_as_of",
                    "recipe",
                    "fit_count",
                    "distinct_dates",
                    "classes",
                    "weight_hash",
                )
            },
            "candidate": CANDIDATE,
            "cohort_id": spec["cohort_id"],
            "model_released": False,
            "features": FEATURES,
            "feature_recipe": FEATURE_RECIPE,
            "mean": mean,
            "scale": scale,
            "baseline": float(estimator._baseline_prediction[0, 0]),
            "trees": tree.export_trees(estimator),
            "fit_hash": digest(rows),
            "original_fit_hash": control["fit_hash"],
            "control_model_hash": digest(control),
        }
        differences = {}
        for name, checked in (("fit", rows), ("exam", exam)):
            if checked:
                check_x = (np.asarray([vector(r) for r in checked]) - np.asarray(mean)) / np.asarray(scale)
                differences[name] = float(
                    np.max(np.abs(predict(model, checked) - estimator.predict_proba(check_x)[:, 1]))
                )
        if not all(np.isfinite(v) and v <= 1e-12 for v in differences.values()):
            raise ValueError("ACTIVITY_RESTORE_MISMATCH")
        model["restore_max_score_diff"] = differences
    return model


def expected_predictions(original: list[dict], scores: dict) -> list[dict]:
    return [
        {
            **r,
            "scores": {**r["scores"], CANDIDATE: scores[r["fund_code"], r["t"]]},
            "directions": {**r["directions"], CANDIDATE: int(scores[r["fund_code"], r["t"]] > 0.5)},
        }
        for r in original
    ]


def require_replay(root: Path) -> None:
    if any(read(root / "main" / n) != read(root / "replay" / n) for n in ("models.json", "predictions.json")):
        raise ValueError("ACTIVITY_REPLAY_MISMATCH")


def run(root: Path, *, replay: bool = False) -> dict:
    spec = verify_inputs(root)
    if replay:
        verify(root)
    out = root / ("replay" if replay else "main")
    out.mkdir(exist_ok=False)
    write_new(out / "budget-reserved.json", {"at": datetime.now(ZONE).isoformat(), "max_fits": 5})
    fits, exam = read(root / "fits.json"), read(root / "exam.json")
    controls, models, scores = read(root / "base/main/models.json"), {}, {}
    for q in (*QUARTERS, "FINAL"):
        selected = [r for r in exam if r["quarter"] == q]
        write_new(out / (q + "-reserved.json"), {"at": datetime.now(ZONE).isoformat(), "fit_hash": digest(fits[q])})
        model = fit(fits[q], controls[q], spec, q, selected)
        write_new(out / (q + ".json"), model)
        models[q] = model
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
        require_replay(root)
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
    """从封存成交额重新派生全部新增输入，核对尺度、权重、预算、模型及每条预测，不重新拟合。"""
    spec = verify_inputs(root)
    require_control(root / "base")
    payload = acquisition.verify(root / "data")
    fits, exam = inputs(root / "base", payload["rows"])
    if fits != read(root / "fits.json") or exam != read(root / "exam.json"):
        raise ValueError("ACTIVITY_DERIVED_INPUT_CHANGED")
    controls, original = read(root / "base/main/models.json"), read(root / "base/main/predictions.json")
    quarters = {*QUARTERS, "FINAL"}
    for mode in ("main", "replay"):
        folder = root / mode
        if not (folder / "completion.json").exists():
            if mode == "main" or folder.exists():
                raise ValueError("ACTIVITY_TRAINING_INCOMPLETE")
            continue
        complete, models, predictions = (
            read(folder / n) for n in ("completion.json", "models.json", "predictions.json")
        )
        if complete["models_hash"] != digest(models) or complete["predictions_hash"] != digest(predictions):
            raise ValueError("ACTIVITY_OUTPUT_CHANGED")
        budget = read(folder / "budget-reserved.json")
        start, end = datetime.fromisoformat(budget["at"]), datetime.fromisoformat(complete["finished_at"])
        if (
            set(models) != quarters
            or budget["max_fits"] != 5
            or complete["successful_fits"] != 5
            or not datetime.fromisoformat(spec["created_at"]) <= start <= end
            or {p.name for p in folder.glob("*-reserved.json")}
            != {"budget-reserved.json", *(q + "-reserved.json" for q in quarters)}
        ):
            raise ValueError("ACTIVITY_FIT_BUDGET_INVALID")
        scores = {}
        for q, model in models.items():
            old, rows = controls[q], fits[q]
            reserved = read(folder / (q + "-reserved.json"))
            if (
                model != read(folder / (q + ".json"))
                or model["cohort_id"] != spec["cohort_id"]
                or model["fit_hash"] != digest(rows)
                or model["fit_hash"] != spec["fit_hashes"][q]
                or model["original_fit_hash"] != digest(control_rows(rows))
                or model["original_fit_hash"] != spec["original_fit_hashes"][q]
                or model["control_model_hash"] != digest(old)
                or model["weight_hash"] != digest(weights(rows).tolist())
                or model["weight_hash"] != spec["weight_hashes"][q]
                or model["mean"][:11] != old["mean"]
                or model["scale"][:11] != old["scale"]
                or any(
                    model[k] != old[k] for k in ("fit_count", "train_as_of", "classes", "distinct_dates", "weight_hash")
                )
                or reserved["fit_hash"] != model["fit_hash"]
                or not start <= datetime.fromisoformat(reserved["at"]) <= end
                or set(model["restore_max_score_diff"]) != ({"fit"} if q == "FINAL" else {"fit", "exam"})
                or not all(np.isfinite(v) and 0 <= v <= 1e-12 for v in model["restore_max_score_diff"].values())
            ):
                raise ValueError("ACTIVITY_MODEL_CHANGED")
            # 仅核算一维统计量，不调用分类器fit；校验新增尺度未遭修改。
            values = np.asarray([r["activity_input"]["x"][0] for r in rows])
            w = weights(rows)
            mean = float(np.average(values, weights=w))
            variance = float(np.average((values - mean) ** 2, weights=w))
            expected_scale = float(np.sqrt(variance)) if variance else 1.0
            if not np.isclose(model["mean"][-1], mean, rtol=1e-12, atol=1e-15) or not np.isclose(
                model["scale"][-1], expected_scale, rtol=1e-12, atol=1e-15
            ):
                raise ValueError("ACTIVITY_EXTRA_SCALE_CHANGED")
            selected = [r for r in exam if r["quarter"] == q]
            scores.update(
                {(r["fund_code"], r["t"]): float(s) for r, s in zip(selected, predict(model, selected), strict=True)}
            )
        if len(predictions) != spec["exam_count"] or predictions != expected_predictions(original, scores):
            raise ValueError("ACTIVITY_RESTORED_OUTPUT_CHANGED")
    if (root / "replay/completion.json").exists():
        require_replay(root)
    return read(root / "main/completion.json")


def evaluate(root: Path) -> dict:
    verify(root)
    require_replay(root)
    spec, rows = read(root / "study.json"), read(root / "main/predictions.json")
    if read(root / "replay-proof.json") != {
        "models_equal": True,
        "predictions_equal": True,
        "max_score_diff": 0.0,
        "successful_fits": 5,
        "prediction_count": spec["exam_count"],
    }:
        raise ValueError("ACTIVITY_REPLAY_PROOF_INVALID")
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
            "2021—2024是重复使用开发数据，非独立验证",
            "仅增加沪深300成交活跃度，不搜索配方",
            "历史数据可用时刻仍有假设",
            "真实未来效果待到期",
        ],
    }
    write_new(root / "comparison.json", result)
    write_new(root / "comparison-receipt.json", {"hash": digest(result), "at": datetime.now(ZONE).isoformat()})
    return result
