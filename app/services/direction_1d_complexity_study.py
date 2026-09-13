"""一日树复杂度研究：先诊断旧模型，再按较早日期选择深度，保持全体考题与截止不变。"""

from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits

from app.services import direction_1d_full_threshold as full
from app.services.direction_1d_protocol import ZONE, digest
from app.services.direction_1d_training import read, weights, write_new
from app.services.direction_training_artifacts import file_hash

activity, tree, sector = full.activity, full.tree, full.sector
REFERENCE, SELECTED = "ACTIVITY12", "SELECTED_DEPTH"
QUARTERS = full.QUARTERS
# 深度和与其对应的最大叶数一起限定表达能力；其余输入、学习率、轮数、权重均不变。
RECIPES = {f"DEPTH{d}": {**tree.TREE_RECIPE, "max_depth": d, "max_leaf_nodes": 2**d} for d in (1, 3)}
BRANCHES = (REFERENCE, *RECIPES)
SELECTION = {
    "metric": "FAMILY_DATE_WEIGHTED_ACCURACY",
    "validation": "EXISTING_3_PAST_BLOCKS_21_TARGET_DAYS_EACH",
    "tie_priority": list(BRANCHES),
    "tie_tolerance": 1e-12,
    "threshold": 0.5,
    "exam_used_for_selection": False,
}
OBSERVATION = {"positive_quarters": 3, "positive_funds": 17, "maximum_fund_drop": 0.05}


def score_summary(rows: list[dict], scores: list | np.ndarray) -> dict:
    """同时显示整体和两类识别能力；分数仅是研究输出，不声明为已校准上涨概率。"""
    p = np.asarray(scores, dtype=float)
    if not rows or p.shape != (len(rows),) or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("COMPLEXITY_SCORES_INVALID")
    y, w = np.asarray([r["y"] for r in rows]), weights(rows)
    if not np.isin(y, (0, 1)).all():
        raise ValueError("COMPLEXITY_LABEL_INVALID")
    predicted = p > 0.5
    safe = np.clip(p, 1e-15, 1 - 1e-15)
    recalls = [float(np.mean(predicted[y == c] == c)) if np.any(y == c) else None for c in (0, 1)]
    return {
        "count": len(rows),
        "target_days": len({r["u"] for r in rows}),
        "correct": int(np.sum(predicted == y)),
        "accuracy": float(np.mean(predicted == y)),
        "weighted_accuracy": float(np.average(predicted == y, weights=w)),
        "actual_up_fraction": float(np.mean(y)),
        "predicted_up_fraction": float(np.mean(predicted)),
        "up_recall": recalls[1],
        "non_up_recall": recalls[0],
        "weighted_log_loss": float(np.average(-y * np.log(safe) - (1 - y) * np.log(1 - safe), weights=w)),
        "weighted_brier": float(np.average((p - y) ** 2, weights=w)),
    }


def diagnose(base: Path) -> dict:
    """只用已封存模型重算练习题与后续题；不训练、不改变样本、不读取保护年份。"""
    full.verify(base)
    activity.require_replay(base)
    universe, schedule, exam = (read(base / n) for n in ("universe.json", "schedule.json", "exam.json"))
    models = read(base / "main/models.json")
    jobs = {}
    for key, info in schedule.items():
        model = models[key][REFERENCE]
        fit_rows, checked = ([universe[i] for i in info[n]] for n in ("fit_indexes", "exam_indexes"))
        train, validation = (score_summary(rows, activity.predict(model, rows)) for rows in (fit_rows, checked))
        jobs[key] = {
            "train": train,
            "following": validation,
            "accuracy_gap": train["weighted_accuracy"] - validation["weighted_accuracy"],
        }
    scores = {full.identity(r): r["scores"][REFERENCE] for r in read(base / "main/predictions.json")}
    strata = {}
    for field in ("month", "fund_code", "group", "restored_question", "market_t_direction"):

        def value(r, field=field):
            if field == "month":
                return r["u"][:7]
            if field == "market_t_direction":
                return "UP" if r["market_input"]["x"][0] > 0 else "NON_UP"
            return str(r[field])

        strata[field] = {
            v: score_summary(
                [r for r in exam if value(r) == v], [scores[full.identity(r)] for r in exam if value(r) == v]
            )
            for v in sorted({value(r) for r in exam})
        }
    return {
        "kind": "EXISTING_MODEL_DIAGNOSTIC_NOT_CAUSAL_PROOF",
        "at": datetime.now(ZONE).isoformat(),
        "base_study_hash": file_hash(base / "study.json"),
        "jobs": jobs,
        "strata": strata,
        "overall": score_summary(exam, [scores[full.identity(r)] for r in exam]),
        "new_fits": 0,
        "model_released": False,
    }


def fingerprint() -> dict:
    result = full.fingerprint()
    project = Path(__file__).resolve().parents[2]
    for name in ("app/services/direction_1d_complexity_study.py", "scripts/direction_1d_complexity_study.py"):
        result[name] = file_hash(project / name)
    return result


def validate_schedule(universe: list[dict], schedule: dict, exam: list[dict]) -> None:
    """复用原16个时间窗口；按整日隔离训练/验证，所有候选必须回答完整的相同考题。"""
    keys = {q + suffix for q in QUARTERS for suffix in ("", "-V1", "-V2", "-V3")}
    if set(schedule) != keys or len({full.identity(r) for r in universe}) != len(universe):
        raise ValueError("COMPLEXITY_SCHEDULE_OR_DUPLICATE")
    if any(not "2021-01-01" <= r["t"] < r["u"] <= "2024-12-31" for r in universe):
        raise ValueError("COMPLEXITY_PROTECTED_YEAR")
    for info in schedule.values():
        for name in ("fit_indexes", "exam_indexes"):
            indexes = info[name]
            if len(set(indexes)) != len(indexes) or any(
                type(i) is not int or not 0 <= i < len(universe) for i in indexes
            ):
                raise ValueError("COMPLEXITY_MEMBER_INDEX_INVALID")
        expected = full.job(
            universe, info["fit_indexes"], info["exam_indexes"], datetime.fromisoformat(info["train_as_of"])
        )
        if expected != info:
            raise ValueError("COMPLEXITY_MEMBERS_OR_TIME_CHANGED")
    for q in QUARTERS:
        rows = [universe[i] for k in (1, 2, 3) for i in schedule[q + f"-V{k}"]["exam_indexes"]]
        if (
            len({full.identity(r) for r in rows}) != len(rows)
            or len({r["u"] for r in rows}) != 63
            or any(r["mature_at"] > schedule[q]["train_as_of"] for r in rows)
        ):
            raise ValueError("COMPLEXITY_SELECTION_FUTURE_OR_DUPLICATE")
        checked = [r for r in exam if r["quarter"] == q]
        if [
            {
                k: v
                for k, v in r.items()
                if k not in ("quarter", "restored_question", "universe_index", "legacy_group_score")
            }
            for r in checked
        ] != [universe[i] for i in schedule[q]["exam_indexes"]]:
            raise ValueError("COMPLEXITY_OUTER_EXAM_CHANGED")


def freeze(base: Path, plan: Path, diagnosis: Path, scope: Path, root: Path) -> dict:
    """固定两种新复杂度，复用旧深度2；每种16个窗口，主训练及唯一复现各32次。"""
    old = full.verify_inputs(base)
    activity.require_replay(base)
    diagnostic, owner = read(diagnosis), read(scope)
    full.audit.validate_scope(owner)
    if (
        diagnostic["base_study_hash"] != file_hash(base / "study.json")
        or diagnostic["new_fits"] != 0
        or owner["scope_hash"] != old["owner_scope_hash"]
    ):
        raise ValueError("COMPLEXITY_DIAGNOSTIC_OR_SCOPE_CHANGED")
    universe, schedule, exam = (read(base / n) for n in ("universe.json", "schedule.json", "exam.json"))
    validate_schedule(universe, schedule, exam)
    root.mkdir(parents=True, exist_ok=False)
    names = {*old["input_files"], "study.json", "study-receipt.json", "replay-proof.json"}
    names.update(
        p.relative_to(base).as_posix() for mode in ("main", "replay") for p in (base / mode).iterdir() if p.is_file()
    )
    for n in sorted(names):
        if not (base / n).resolve().is_relative_to(base.resolve()):
            raise ValueError("COMPLEXITY_SOURCE_PATH_INVALID")
        tree.copy_file(base / n, root / "control" / n)
    for name, path in (("plan.md", plan), ("diagnosis.json", diagnosis), ("owner-scope.json", scope)):
        tree.copy_file(path, root / name)
    files = {"control/" + n: file_hash(root / "control" / n) for n in names}
    files.update({n: file_hash(root / n) for n in ("plan.md", "diagnosis.json", "owner-scope.json")})
    spec = {
        "kind": "HISTORICAL_DEPTH_SELECTION_DEVELOPMENT_ONLY",
        "created_at": datetime.now(ZONE).isoformat(),
        "fingerprint": fingerprint(),
        "recipes": RECIPES,
        "selection": SELECTION,
        "observation": OBSERVATION,
        "reference": REFERENCE,
        "primary_candidate": SELECTED,
        "features": activity.FEATURES,
        "feature_recipe": activity.FEATURE_RECIPE,
        "threshold": 0.5,
        "fund_codes": old["fund_codes"],
        "owner_scope_hash": old["owner_scope_hash"],
        "exam_count": len(exam),
        "restored_count": old["restored_count"],
        "job_keys": sorted(schedule),
        "source_expires_at": old["source"]["source_expires_at"],
        "input_files": files,
        "max_main_fits": 32,
        "max_replay_fits": 32,
        "protected_years_excluded": [2025, 2026],
        "new_api_calls": 0,
        "database_writes": 0,
        "model_released": False,
        "final_fit": False,
        "bootstrap_seed": 20260912,
        "bootstrap_repetitions": 2000,
        "bootstrap_block_days": 5,
    }
    spec["cohort_id"] = "D1-COMPLEXITY-" + digest({"files": files, "recipes": RECIPES, "selection": SELECTION})[:24]
    write_new(root / "study.json", spec)
    write_new(root / "study-receipt.json", {"hash": digest(spec), "at": spec["created_at"]})
    return {k: spec[k] for k in ("cohort_id", "created_at", "exam_count", "max_main_fits", "max_replay_fits")}


def verify_inputs(root: Path) -> dict:
    spec = read(root / "study.json")
    expected = {
        "recipes": RECIPES,
        "selection": SELECTION,
        "observation": OBSERVATION,
        "reference": REFERENCE,
        "primary_candidate": SELECTED,
        "features": activity.FEATURES,
        "feature_recipe": activity.FEATURE_RECIPE,
        "threshold": 0.5,
        "max_main_fits": 32,
        "max_replay_fits": 32,
        "protected_years_excluded": [2025, 2026],
        "new_api_calls": 0,
        "database_writes": 0,
        "model_released": False,
        "final_fit": False,
        "bootstrap_seed": 20260912,
        "bootstrap_repetitions": 2000,
        "bootstrap_block_days": 5,
    }
    if (
        digest(spec) != read(root / "study-receipt.json")["hash"]
        or spec["fingerprint"] != fingerprint()
        or any(spec.get(k) != v for k, v in expected.items())
    ):
        raise ValueError("COMPLEXITY_SPEC_CODE_OR_BUDGET_CHANGED")
    if datetime.now(ZONE) >= datetime.fromisoformat(spec["source_expires_at"]):
        raise ValueError("COMPLEXITY_SOURCE_EXPIRED")
    for n, expected_hash in spec["input_files"].items():
        p = root / n
        if not p.resolve().is_relative_to(root.resolve()) or p.is_symlink() or file_hash(p) != expected_hash:
            raise ValueError("COMPLEXITY_INPUT_CHANGED")
    return spec


def validate_model(model: dict) -> None:
    """独立检查可变深度树的完整拓扑；旧深度2校验器维持原有协议，不做兼容性放宽。"""
    recipe = RECIPES.get(model.get("candidate"))
    if (
        recipe is None
        or model.get("recipe") != recipe
        or model.get("features") != activity.FEATURES
        or model.get("feature_recipe") != activity.FEATURE_RECIPE
        or model.get("model_released") is not False
        or model.get("target_definition") != "UNIT_NAV_DIRECTION_V1"
    ):
        raise ValueError("COMPLEXITY_MODEL_PROTOCOL")
    if (
        len(model["mean"]) != 12
        or len(model["scale"]) != 12
        or min(model["scale"]) <= 0
        or not np.isfinite([model["baseline"], *model["mean"], *model["scale"]]).all()
        or len(model["trees"]) != recipe["max_iter"]
    ):
        raise ValueError("COMPLEXITY_MODEL_SCALE_OR_ROUNDS")
    for nodes in model["trees"]:
        if not 1 <= len(nodes) <= 2 * recipe["max_leaf_nodes"] - 1:
            raise ValueError("COMPLEXITY_MODEL_NODE_COUNT")
        seen, stack, leaves = set(), [(0, 0)], 0
        while stack:
            i, depth = stack.pop()
            if i in seen or not 0 <= i < len(nodes) or depth > recipe["max_depth"]:
                raise ValueError("COMPLEXITY_MODEL_TOPOLOGY")
            seen.add(i)
            n = nodes[i]
            if (
                n["depth"] != depth
                or type(n["leaf"]) is not bool
                or n["count"] <= 0
                or not np.isfinite([n["value"], n["threshold"]]).all()
            ):
                raise ValueError("COMPLEXITY_MODEL_NODE")
            if n["leaf"]:
                leaves += 1
            else:
                if not 0 <= n["feature"] < 12 or n["left"] <= i or n["right"] <= i:
                    raise ValueError("COMPLEXITY_MODEL_BRANCH")
                stack.extend(((n["left"], depth + 1), (n["right"], depth + 1)))
        if len(seen) != len(nodes) or leaves > recipe["max_leaf_nodes"]:
            raise ValueError("COMPLEXITY_MODEL_UNREACHABLE")


def predict(model: dict, rows: list[dict]) -> np.ndarray:
    """仅以12项输入和JSON数值树恢复分数，推理路径不读取答案。"""
    if model.get("candidate") == activity.CANDIDATE:
        return activity.predict(model, rows)
    validate_model(model)
    if not rows:
        return np.asarray([], dtype=float)
    x = (np.asarray([activity.vector(r) for r in rows]) - model["mean"]) / model["scale"]
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


def fit(rows: list[dict], checked: list[dict], control: dict, spec: dict, branch: str) -> dict:
    """复用对照的全部尺度与权重；拟合后核对练习题和随后题的原生/JSON恢复分数。"""
    activity.validate_model(control)
    if digest(rows) != control["fit_hash"] or digest(weights(rows).tolist()) != control["weight_hash"]:
        raise ValueError("COMPLEXITY_FIT_MEMBERS_OR_WEIGHT_CHANGED")
    full.check_available(rows, datetime.fromisoformat(control["train_as_of"]), fit=True)
    x = (np.asarray([activity.vector(r) for r in rows]) - control["mean"]) / control["scale"]
    with threadpool_limits(limits=1):
        estimator = HistGradientBoostingClassifier(**RECIPES[branch]).fit(
            x, [r["y"] for r in rows], sample_weight=weights(rows)
        )
        model = {
            k: control[k]
            for k in (
                "kind",
                "target_definition",
                "train_as_of",
                "fit_count",
                "distinct_dates",
                "classes",
                "weight_hash",
                "fit_hash",
                "mean",
                "scale",
                "features",
                "feature_recipe",
            )
        }
        model.update(
            candidate=branch,
            recipe=RECIPES[branch],
            cohort_id=spec["cohort_id"],
            model_released=False,
            control_model_hash=digest(control),
            baseline=float(estimator._baseline_prediction[0, 0]),
            trees=tree.export_trees(estimator),
        )
        differences = {}
        for name, selected in (("fit", rows), ("following", checked)):
            values = (np.asarray([activity.vector(r) for r in selected]) - model["mean"]) / model["scale"]
            difference = float(np.max(np.abs(predict(model, selected) - estimator.predict_proba(values)[:, 1])))
            if difference > 1e-12:
                raise ValueError("COMPLEXITY_RESTORE_MISMATCH")
            differences[name] = difference
        model["restore_max_score_diff"] = differences
    return model


def select_branch(rows: list[dict], scores: dict) -> dict:
    """选择只接收较早折外分数；并列保留旧模型，禁止按2024外层成绩事后选赢家。"""
    if set(scores) != set(BRANCHES):
        raise ValueError("COMPLEXITY_SELECTION_BRANCHES")
    reports = {b: score_summary(rows, scores[b]) for b in BRANCHES}
    best = max(v["weighted_accuracy"] for v in reports.values())
    selected = next(b for b in BRANCHES if best - reports[b]["weighted_accuracy"] <= SELECTION["tie_tolerance"])
    return {"branch": selected, "reports": reports, "rows_hash": digest(rows), "scores_hash": digest(scores)}


def selections(universe: list[dict], schedule: dict, models: dict) -> dict:
    chosen = {}
    for q in QUARTERS:
        rows, scores = [], {b: [] for b in BRANCHES}
        for k in (1, 2, 3):
            key = q + f"-V{k}"
            checked = [universe[i] for i in schedule[key]["exam_indexes"]]
            rows.extend(checked)
            for b in BRANCHES:
                scores[b].extend(predict(models[key][b], checked).tolist())
        if any(r["mature_at"] > schedule[q]["train_as_of"] for r in rows):
            raise ValueError("COMPLEXITY_SELECTION_ANSWER_TOO_LATE")
        chosen[q] = {
            **select_branch(rows, scores),
            "selection_as_of": schedule[q]["train_as_of"],
            "latest_answer_mature_at": max(r["mature_at"] for r in rows),
        }
    return chosen


def outputs(base: Path, models: dict, selected: dict) -> tuple[list[dict], dict]:
    universe, schedule, exam = (read(base / n) for n in ("universe.json", "schedule.json", "exam.json"))
    original = {full.identity(r): r for r in read(base / "main/predictions.json")}
    result, diagnostic = [], {}
    for key, info in schedule.items():
        diagnostic[key] = {}
        for branch in BRANCHES:
            diagnostic[key][branch] = {
                name: score_summary(rows, predict(models[key][branch], rows))
                for name, rows in (
                    ("train", [universe[i] for i in info["fit_indexes"]]),
                    ("following", [universe[i] for i in info["exam_indexes"]]),
                )
            }
    for q in QUARTERS:
        checked = [r for r in exam if r["quarter"] == q]
        values = {b: predict(models[q][b], checked).tolist() for b in BRANCHES}
        for i, r in enumerate(checked):
            prior = original[full.identity(r)]
            if values[REFERENCE][i] != prior["scores"][REFERENCE]:
                raise ValueError("COMPLEXITY_CONTROL_SCORE_CHANGED")
            scores = {b: values[b][i] for b in BRANCHES}
            scores[SELECTED] = scores[selected[q]["branch"]]
            result.append(
                {
                    **r,
                    "scores": scores,
                    "selected_branch": selected[q]["branch"],
                    "directions": {
                        **{b: int(v > 0.5) for b, v in scores.items()},
                        **{
                            b: prior["directions"][b]
                            for b in ("ALWAYS_UP", "ALWAYS_NON_UP", "INITIAL_MAJORITY", "MOMENTUM")
                        },
                    },
                }
            )
    return sorted(result, key=lambda r: (r["t"], r["family"], r["fund_code"])), diagnostic


def run(root: Path, *, replay: bool = False) -> dict:
    spec = verify_inputs(root)
    if replay:
        verify(root)
    base = root / "control"
    universe, schedule, exam = (read(base / n) for n in ("universe.json", "schedule.json", "exam.json"))
    validate_schedule(universe, schedule, exam)
    control = read(base / "main/models.json")
    models = {k: {REFERENCE: v[REFERENCE]} for k, v in control.items()}
    out = root / ("replay" if replay else "main")
    out.mkdir(exist_ok=False)
    write_new(out / "budget-reserved.json", {"at": datetime.now(ZONE).isoformat(), "max_fits": 32})
    # 所有较早折外训练结束后先固化选择；之后才生成外层候选，无法利用外层答案选深度。
    order = [q + f"-V{k}" for q in QUARTERS for k in (1, 2, 3)] + list(QUARTERS)
    for key in order:
        info = schedule[key]
        rows, checked = ([universe[i] for i in info[n]] for n in ("fit_indexes", "exam_indexes"))
        for branch in RECIPES:
            write_new(
                out / (key + "-" + branch + "-reserved.json"),
                {"at": datetime.now(ZONE).isoformat(), "fit_hash": info["fit_hash"]},
            )
            model = fit(rows, checked, control[key][REFERENCE], spec, branch)
            models[key][branch] = model
            write_new(out / (key + "-" + branch + ".json"), model)
        if key == QUARTERS[-1] + "-V3":
            selected = selections(universe, schedule, models)
            write_new(out / "selections.json", selected)
            write_new(out / "selection-receipt.json", {"at": datetime.now(ZONE).isoformat(), "hash": digest(selected)})
    predicted, diagnostic = outputs(base, models, selected)
    artifacts = {"models": models, "predictions": predicted, "diagnostics": diagnostic}
    for name, value in artifacts.items():
        write_new(out / (name + ".json"), value)
    completion = {
        "finished_at": datetime.now(ZONE).isoformat(),
        "successful_fits": 32,
        "model_released": False,
        **{name + "_hash": digest(value) for name, value in artifacts.items()},
        "selections_hash": digest(selected),
    }
    write_new(out / "completion.json", completion)
    if replay:
        for name in (*artifacts, "selections"):
            if read(out / (name + ".json")) != read(root / "main" / (name + ".json")):
                raise ValueError("COMPLEXITY_REPLAY_MISMATCH")
        write_new(
            root / "replay-proof.json",
            {
                "models_equal": True,
                "predictions_equal": True,
                "selections_equal": True,
                "diagnostics_equal": True,
                "max_score_diff": 0.0,
                "successful_fits": 32,
                "prediction_count": len(predicted),
            },
        )
    return completion


def verify(root: Path) -> dict:
    """只读重建每个模型的成员、时间、选择及全部分数；验证不进行任何拟合。"""
    spec = verify_inputs(root)
    base = root / "control"
    universe, schedule, exam = (read(base / n) for n in ("universe.json", "schedule.json", "exam.json"))
    validate_schedule(universe, schedule, exam)
    if (
        len(exam) != spec["exam_count"]
        or sorted({r["fund_code"] for r in exam}) != spec["fund_codes"]
        or sorted(schedule) != spec["job_keys"]
    ):
        raise ValueError("COMPLEXITY_COHORT_CHANGED")
    control = read(base / "main/models.json")
    for mode in ("main", "replay"):
        out = root / mode
        if not (out / "completion.json").exists():
            if mode == "main" or out.exists():
                raise ValueError("COMPLEXITY_INCOMPLETE_RUN")
            continue
        complete, models = read(out / "completion.json"), read(out / "models.json")
        budget, receipt = read(out / "budget-reserved.json"), read(out / "selection-receipt.json")
        reservations = {"budget-reserved.json", *(k + "-" + b + "-reserved.json" for k in schedule for b in RECIPES)}
        if (
            complete["successful_fits"] != 32
            or budget["max_fits"] != 32
            or set(models) != set(schedule)
            or {p.name for p in out.glob("*-reserved.json")} != reservations
            or digest(models) != complete["models_hash"]
            or complete["model_released"] is not False
        ):
            raise ValueError("COMPLEXITY_RUN_BUDGET_OR_MODELS")
        chosen = selections(universe, schedule, models)
        if (
            chosen != read(out / "selections.json")
            or digest(chosen) != complete["selections_hash"]
            or receipt["hash"] != digest(chosen)
        ):
            raise ValueError("COMPLEXITY_SELECTION_CHANGED")
        for key, info in schedule.items():
            if set(models[key]) != set(BRANCHES) or models[key][REFERENCE] != control[key][REFERENCE]:
                raise ValueError("COMPLEXITY_CONTROL_OR_BRANCH_CHANGED")
            for b in RECIPES:
                model = models[key][b]
                validate_model(model)
                reserved = read(out / (key + "-" + b + "-reserved.json"))
                if (
                    model != read(out / (key + "-" + b + ".json"))
                    or model["cohort_id"] != spec["cohort_id"]
                    or reserved["fit_hash"] != info["fit_hash"]
                    or model["fit_hash"] != info["fit_hash"]
                    or model["weight_hash"] != info["weight_hash"]
                    or model["control_model_hash"] != digest(control[key][REFERENCE])
                    or any(
                        model[k] != control[key][REFERENCE][k]
                        for k in ("mean", "scale", "train_as_of", "fit_count", "distinct_dates", "classes")
                    )
                    or not spec["created_at"] <= budget["at"] <= reserved["at"] <= complete["finished_at"]
                    or not budget["at"] <= receipt["at"] <= complete["finished_at"]
                    or (key in QUARTERS and reserved["at"] < receipt["at"])
                    or (key not in QUARTERS and reserved["at"] > receipt["at"])
                ):
                    raise ValueError("COMPLEXITY_MODEL_METADATA_OR_RESERVATION")
        predicted, diagnostic = outputs(base, models, chosen)
        prior_diagnostic = read(root / "diagnosis.json")["jobs"]
        for key in schedule:
            if any(diagnostic[key][REFERENCE][name] != prior_diagnostic[key][name] for name in ("train", "following")):
                raise ValueError("COMPLEXITY_ORIGINAL_DIAGNOSTIC_CHANGED")
        for name, value in (("predictions", predicted), ("diagnostics", diagnostic)):
            if value != read(out / (name + ".json")) or digest(value) != complete[name + "_hash"]:
                raise ValueError("COMPLEXITY_RESTORED_OUTPUT_CHANGED")
    if (root / "replay").exists():
        expected = {
            "models_equal": True,
            "predictions_equal": True,
            "selections_equal": True,
            "diagnostics_equal": True,
            "max_score_diff": 0.0,
            "successful_fits": 32,
            "prediction_count": len(exam),
        }
        if read(root / "replay-proof.json") != expected:
            raise ValueError("COMPLEXITY_REPLAY_PROOF_CHANGED")
        for name in ("models", "predictions", "selections", "diagnostics"):
            if read(root / "main" / (name + ".json")) != read(root / "replay" / (name + ".json")):
                raise ValueError("COMPLEXITY_REPLAY_CHANGED")
    return {"verified": True, "exam_count": len(exam), "new_fits": 0, "api_calls": 0}


def comparison(rows: list[dict], spec: dict) -> dict:
    metrics = sector.comparison.metrics
    branches = (*BRANCHES, SELECTED, "ALWAYS_UP", "ALWAYS_NON_UP", "INITIAL_MAJORITY", "MOMENTUM")
    overall = {b: metrics(rows, b) for b in branches}
    strata = {
        field: {
            str(v): {b: metrics([r for r in rows if r[field] == v], b) for b in branches}
            for v in sorted({r[field] for r in rows})
        }
        for field in ("quarter", "fund_code", "group", "restored_question")
    }
    paired = {b: sector.paired(rows, b, REFERENCE, spec) for b in (*RECIPES, SELECTED)}
    deltas = {
        field: [v[SELECTED]["weighted_accuracy"] - v[REFERENCE]["weighted_accuracy"] for v in strata[field].values()]
        for field in ("quarter", "fund_code")
    }
    conditions = {
        "overall_improved": paired[SELECTED]["weighted_accuracy_difference"] > 0,
        "interval_positive": paired[SELECTED]["block_bootstrap_interval_95"][0] > 0,
        "quarters_improved": sum(d > 0 for d in deltas["quarter"]) >= OBSERVATION["positive_quarters"],
        "funds_improved": sum(d > 0 for d in deltas["fund_code"]) >= OBSERVATION["positive_funds"],
        "fund_drop_controlled": min(deltas["fund_code"]) >= -OBSERVATION["maximum_fund_drop"],
        "both_recalls_preserved": all(
            overall[SELECTED][k] >= overall[REFERENCE][k] for k in ("up_recall", "non_up_recall")
        ),
    }
    return {
        "kind": "DEVELOPMENT_COMPLEXITY_COMPARISON_NOT_INDEPENDENT_TEST",
        "overall": overall,
        "strata": strata,
        "paired": paired,
        "conditions": conditions,
        "primary_candidate": SELECTED,
        "status": "DEVELOPMENT_OBSERVATION_CONDITIONS_MET" if all(conditions.values()) else "NO_STABLE_GAIN",
        "model_released": False,
        "runtime_model_changed": False,
        "limitations": [
            "2021—2024重复使用，外层历史成绩也是开发证据",
            "历史首次发布时间与首版本未验证",
            "本轮只比较固定两种复杂度，不能证明其他模型无效",
            "2025/2026保护不变，真实未来效果未验证",
        ],
    }


def evaluate(root: Path) -> dict:
    verify(root)
    if not (root / "replay-proof.json").exists():
        raise ValueError("COMPLEXITY_REPLAY_REQUIRED")
    report = comparison(read(root / "main/predictions.json"), read(root / "study.json"))
    write_new(root / "comparison.json", report)
    write_new(root / "comparison-receipt.json", {"hash": digest(report), "at": datetime.now(ZONE).isoformat()})
    return report
