"""完整历史净值可用性与阈值对照；仅离线研究，原日期、真实预测和在用模型只读。"""

from datetime import date, datetime, time
from pathlib import Path

import numpy as np

from app.services import direction_1d_ann_audit as audit
from app.services import direction_1d_weekly_study as weekly
from app.services.direction_1d_protocol import ZONE, calendar, digest
from app.services.direction_1d_training import read, weights, write_new
from app.services.direction_training_artifacts import file_hash

activity, tree, sector, original = weekly.activity, weekly.tree, weekly.sector, weekly.original
QUARTERS = tree.QUARTERS
BRANCHES = ("TREE11", "ACTIVITY12")
THRESHOLDS = (0.45, 0.50, 0.55)
RECIPE = {
    "availability": "NAV_NEXT_SESSION_08_ASSUMED_V2_NOT_FIRST_PUBLICATION_PROOF",
    "raw_ann_date": "PRESERVE_AS_SOURCE_LINEAGE_ONLY",
    "label_mature": "SESSION_AFTER_TARGET_U_08_ASSUMED",
    "quarter": "INHERIT_ORIGINAL_T_QUARTER_AND_CUTOFF",
    "fit_window": "LAST_504_TRADING_T_DAYS_PER_ORIGINAL_GROUP",
    "validation": "LAST_63_MATURE_TARGET_U_DAYS_BEFORE_OUTER_CUTOFF_3_BLOCKS_OF_21",
    "inner_cutoff": "FIRST_VALIDATION_T_00_ASIA_SHANGHAI",
    "selection": "MAX_FAMILY_DATE_WEIGHTED_ACCURACY_ON_ALL_63_OOF_DAYS",
    "tie_break": "WITHIN_1E_MINUS_12_CLOSEST_TO_0_5_THEN_LOWER",
    "thresholds": list(THRESHOLDS),
    "direction": "SCORE_STRICTLY_GREATER_THAN_THRESHOLD",
    "final_fit": False,
}


def fingerprint() -> dict:
    result = audit.fingerprint()
    result["weekly"] = weekly.fingerprint()
    root = Path(__file__).resolve().parents[2]
    for name in ("app/services/direction_1d_full_threshold.py", "scripts/direction_1d_full_threshold.py"):
        result[name] = file_hash(root / name)
    return result


def identity(row: dict) -> tuple[str, str]:
    return row["fund_code"], row["t"]


def redate(rows: list[dict]) -> list[dict]:
    """只更改独立研究行的可用性元数据；输入/答案hash和原mature_at全部保留。"""
    days = calendar()[0]
    following = dict(zip(days[:-1], days[1:], strict=True))
    result, seen = [], set()
    for row in rows:
        t, u = date.fromisoformat(row["t"]), date.fromisoformat(row["u"])
        if (
            identity(row) in seen
            or row["kind"] != "HISTORICAL_RECONSTRUCTION"
            or not 2021 <= t.year <= u.year <= 2024
            or following.get(t) != u
            or u not in following
        ):
            raise ValueError("FULL_THRESHOLD_SOURCE_OR_NEXT_SESSION_INVALID")
        seen.add(identity(row))
        result.append(
            {
                **row,
                "legacy_mature_at": row["mature_at"],
                "mature_at": datetime.combine(following[u], time(8), ZONE).isoformat(),
                "nav_available_at_assumed": datetime.combine(u, time(8), ZONE).isoformat(),
                "availability_version": RECIPE["availability"],
            }
        )
    return result


def check_available(rows: list[dict], cutoff: datetime, *, fit: bool) -> None:
    """训练需答案成熟；考题只需预测输入可用，绝不提前要求或使用U日答案。"""
    for r in rows:
        stamps = [datetime.fromisoformat(r["nav_available_at_assumed"])]
        stamps += [
            datetime.fromisoformat(r[k]["available_at_assumed"])
            for k in ("market_input", "specific_input", "activity_input")
        ]
        if fit:
            stamps.append(datetime.fromisoformat(r["mature_at"]))
        if max(stamps) > cutoff or r["u"] > "2024-12-31":
            raise ValueError("FULL_THRESHOLD_DATA_NOT_AVAILABLE")


def choose_validation(universe: list[dict], cutoff: datetime) -> list[list[int]]:
    """按目标日整组取63个已成熟日期，连续三段各21日；同日基金不会跨训练/验证。"""
    eligible = [i for i, r in enumerate(universe) if datetime.fromisoformat(r["mature_at"]) <= cutoff]
    dates = sorted({universe[i]["u"] for i in eligible})[-63:]
    if len(dates) != 63:
        raise ValueError("FULL_THRESHOLD_VALIDATION_TOO_SHORT")
    return [[i for i in eligible if universe[i]["u"] in set(dates[k : k + 21])] for k in (0, 21, 42)]


def fit_indexes(raw: list[dict], universe: list[dict], cutoff: datetime) -> list[int]:
    available = {identity(r): datetime.fromisoformat(r["nav_available_at_assumed"]) for r in raw}
    chosen = {identity(r) for r in weekly.select_week(raw, available, cutoff)}
    indexes = [i for i, r in enumerate(universe) if identity(r) in chosen]
    rows = [universe[i] for i in indexes]
    check_available(rows, cutoff, fit=True)
    if len({r["t"] for r in rows}) < 252 or min(sum(r["y"] == y for r in rows) for y in (0, 1)) < 30:
        raise ValueError("FULL_THRESHOLD_INSUFFICIENT_FIT")
    return indexes


def job(universe: list[dict], selected: list[int], tested: list[int], cutoff: datetime) -> dict:
    fit, exam = [universe[i] for i in selected], [universe[i] for i in tested]
    if set(selected) & set(tested) or {r["u"] for r in fit} & {r["u"] for r in exam}:
        raise ValueError("FULL_THRESHOLD_FIT_EXAM_OVERLAP")
    check_available(fit, cutoff, fit=True)
    for r in exam:
        deadline = datetime.combine(date.fromisoformat(r["u"]), time(8), ZONE)
        check_available([r], deadline, fit=False)
        if cutoff >= deadline:
            raise ValueError("FULL_THRESHOLD_MODEL_AFTER_EXAM")
    return {
        "train_as_of": cutoff.isoformat(),
        "fit_indexes": selected,
        "exam_indexes": tested,
        "fit_hash": digest(fit),
        "control_fit_hash": digest(activity.control_rows(fit)),
        "weight_hash": digest(weights(fit).tolist()),
        "fit_count": len(fit),
        "distinct_dates": len({r["t"] for r in fit}),
        "fit_start": fit[0]["t"],
        "fit_end": fit[-1]["t"],
        "latest_mature_at": max(r["mature_at"] for r in fit),
        "exam_count": len(exam),
        "exam_target_dates": sorted({r["u"] for r in exam}),
    }


def derive(base: Path) -> tuple[list[dict], dict, list[dict]]:
    """从原始冻结净值重建全部行；映射、数值特征、标签和旧季度划分均不改变。"""
    history, previous = read(base / "history.json"), read(base / "dataset.json")
    if history.get("forward_samples") or original.build_samples(history) != previous:
        raise ValueError("FULL_THRESHOLD_ORIGINAL_DATASET_CHANGED")
    raw = redate(previous)
    mapped, _ = sector.attach_specific(raw, read(base / "mapping.json")["funds"], read(base / "prices.json")["prices"])
    universe = activity.attach(mapped, read(base / "activity-data.json")["rows"])
    by_key = {identity(r): i for i, r in enumerate(universe)}
    old_predictions, old_models = read(base / "baseline-predictions.json"), read(base / "baseline-models.json")
    exam = []
    old_exam = {identity(r): r for r in read(base / "old-exam.json")}
    for p in old_predictions:
        key = identity(p)
        if key not in by_key:
            continue
        row = universe[by_key[key]]
        if (
            row["y"] != p["y"]
            or row["u"] != p["u"]
            or abs(original.score(old_models[p["job"]], row["x"]) - p["score"]) > 1e-12
        ):
            raise ValueError("FULL_THRESHOLD_ORIGINAL_ANSWER_CHANGED")
        q = p["job"].rsplit("-", 1)[1]
        exam.append(
            {
                **row,
                "quarter": q,
                "universe_index": by_key[key],
                "restored_question": key not in old_exam,
                "legacy_group_score": p["score"],
            }
        )
    exam.sort(key=lambda r: (r["t"], r["family"], r["fund_code"]))
    if len({identity(r) for r in exam}) != len(exam) or not set(old_exam) <= {identity(r) for r in exam}:
        raise ValueError("FULL_THRESHOLD_QUESTION_DUPLICATE_OR_LOST")
    proof = read(base / "audit-result.json")["exam_impact"]
    if (
        len(exam) != proof["mapped_25_before_ann_gate"]
        or sum(r["restored_question"] for r in exam) != proof["mapped_25_ann_rejected"]
    ):
        raise ValueError("FULL_THRESHOLD_COVERAGE_DIFFERS_FROM_AUDIT")
    schedule = {}
    for q in QUARTERS:
        cutoff = datetime.fromisoformat(old_models["CN_EQUITY-" + q]["train_as_of"])
        tested = [r["universe_index"] for r in exam if r["quarter"] == q]
        schedule[q] = job(universe, fit_indexes(raw, universe, cutoff), tested, cutoff)
        folds = choose_validation(universe, cutoff)
        for k, indexes in enumerate(folds):
            first_t = min(universe[i]["t"] for i in indexes)
            inner = datetime.combine(date.fromisoformat(first_t), time(), ZONE)
            schedule[q + f"-V{k + 1}"] = job(universe, fit_indexes(raw, universe, inner), indexes, inner)
    return universe, schedule, exam


def freeze(base: Path, audit_root: Path, plan: Path, scope_path: Path, root: Path) -> dict:
    """事前固定32次主拟合和32次唯一复现；不在冻结阶段训练、评分新增题或联网。"""
    activity.verify(base)
    activity.require_replay(base)
    audit.verify(audit_root)
    scope = read(scope_path)
    audit.validate_scope(scope)
    old_spec = read(base / "study.json")
    if read(audit_root / "audit-result.json")["scope_hash"] != scope["scope_hash"] or not set(
        old_spec["fund_codes"]
    ) <= set(scope["codes"]):
        raise ValueError("FULL_THRESHOLD_OWNER_SCOPE_CHANGED")
    root.mkdir(parents=True, exist_ok=False)
    source = base / "base/base"
    files = {
        "history.json": source / "baseline/history.json",
        "dataset.json": source / "baseline/dataset.json",
        "baseline-models.json": source / "baseline/models.json",
        "baseline-predictions.json": source / "baseline/predictions.json",
        "mapping.json": source / "mapping.json",
        "prices.json": source / "prices.json",
        "old-exam.json": base / "exam.json",
        "old-predictions.json": base / "main/predictions.json",
        "old-tree-models.json": base / "base/main/models.json",
        "old-activity-models.json": base / "main/models.json",
        "activity-data.json": base / "data/activity-data.json",
        "activity-study.json": base / "study.json",
        "audit-result.json": audit_root / "audit-result.json",
        "audit-result-receipt.json": audit_root / "audit-result-receipt.json",
    }
    for name, source_path in files.items():
        tree.copy_file(source_path, root / "base" / name)
    tree.copy_file(plan, root / "plan.md")
    tree.copy_file(scope_path, root / "owner-scope.json")
    universe, schedule, exam = derive(root / "base")
    for name, value in (("universe.json", universe), ("schedule.json", schedule), ("exam.json", exam)):
        write_new(root / name, value)
    names = [
        *("base/" + n for n in files),
        "plan.md",
        "owner-scope.json",
        "universe.json",
        "schedule.json",
        "exam.json",
    ]
    spec = {
        "kind": "HISTORICAL_FULL_COHORT_AND_THRESHOLD_ONLY",
        "created_at": datetime.now(ZONE).isoformat(),
        "fingerprint": fingerprint(),
        "recipe": RECIPE,
        "tree_recipe": tree.TREE_RECIPE,
        "activity_recipe": activity.FEATURE_RECIPE,
        "fund_codes": old_spec["fund_codes"],
        "source": old_spec["source"],
        "owner_scope_hash": scope["scope_hash"],
        "input_files": {n: file_hash(root / n) for n in names},
        "exam_count": len(exam),
        "restored_count": sum(r["restored_question"] for r in exam),
        "job_keys": sorted(schedule),
        "max_main_fits": 2 * len(schedule),
        "max_replay_fits": 2 * len(schedule),
        "protected_years_excluded": [2025, 2026],
        "new_api_calls": 0,
        "database_writes": 0,
        "model_released": False,
        "primary_metric": "FAMILY_DATE_WEIGHTED_ACCURACY",
        "bootstrap_seed": 20260912,
        "bootstrap_repetitions": 2000,
        "bootstrap_block_days": 5,
    }
    spec["cohort_id"] = "D1-FULL-THRESHOLD-" + digest({"inputs": spec["input_files"], "recipe": RECIPE})[:24]
    write_new(root / "study.json", spec)
    write_new(root / "study-receipt.json", {"at": spec["created_at"], "hash": digest(spec)})
    return {
        k: spec[k]
        for k in ("cohort_id", "created_at", "exam_count", "restored_count", "max_main_fits", "max_replay_fits")
    }


def verify_inputs(root: Path) -> dict:
    spec = read(root / "study.json")
    if digest(spec) != read(root / "study-receipt.json")["hash"] or spec["fingerprint"] != fingerprint():
        raise ValueError("FULL_THRESHOLD_CODE_OR_SPEC_CHANGED")
    if (
        spec["recipe"] != RECIPE
        or spec["tree_recipe"] != tree.TREE_RECIPE
        or spec["activity_recipe"] != activity.FEATURE_RECIPE
        or spec["model_released"] is not False
        or spec["protected_years_excluded"] != [2025, 2026]
        or spec["max_main_fits"] != 32
        or spec["max_replay_fits"] != 32
    ):
        raise ValueError("FULL_THRESHOLD_RECIPE_OR_BUDGET_CHANGED")
    if datetime.now(ZONE) >= datetime.fromisoformat(spec["source"]["source_expires_at"]):
        raise ValueError("FULL_THRESHOLD_SOURCE_EXPIRED")
    for name, expected in spec["input_files"].items():
        p = root / name
        if not p.resolve().is_relative_to(root.resolve()) or p.is_symlink() or file_hash(p) != expected:
            raise ValueError("FULL_THRESHOLD_INPUT_CHANGED")
    return spec


def select_threshold(rows: list[dict], scores: list[float]) -> dict:
    """只接收较早的折外验证分数；先比加权准确率，并列优先0.5，再优先较低阈值。"""
    scores = np.asarray(scores, dtype=float)
    if not rows or len(rows) != len(scores) or not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
        raise ValueError("FULL_THRESHOLD_VALIDATION_SCORES_INVALID")
    w, y = weights(rows), np.asarray([r["y"] for r in rows])
    reports = {str(t): float(np.average((scores > t) == y, weights=w)) for t in THRESHOLDS}
    best = max(reports.values())
    tied = [t for t in THRESHOLDS if best - reports[str(t)] <= 1e-12]
    threshold = min(tied, key=lambda t: (round(abs(t - 0.5), 12), t))
    return {
        "threshold": threshold,
        "weighted_accuracies": reports,
        "validation_count": len(rows),
        "target_dates": len({r["u"] for r in rows}),
        "rows_hash": digest(rows),
        "scores_hash": digest(scores.tolist()),
    }


def predict(branch: str, model: dict, rows: list[dict]) -> np.ndarray:
    return (tree if branch == "TREE11" else activity).predict(model, rows)


def selections(universe: list[dict], schedule: dict, models: dict) -> dict:
    result = {}
    for q in QUARTERS:
        indexes = [i for k in range(1, 4) for i in schedule[q + f"-V{k}"]["exam_indexes"]]
        rows = [universe[i] for i in indexes]
        if len(indexes) != len(set(indexes)) or len({r["u"] for r in rows}) != 63:
            raise ValueError("FULL_THRESHOLD_VALIDATION_MEMBERS_CHANGED")
        cutoff = datetime.fromisoformat(schedule[q]["train_as_of"])
        if any(datetime.fromisoformat(r["mature_at"]) > cutoff for r in rows):
            raise ValueError("FULL_THRESHOLD_VALIDATION_ANSWER_TOO_LATE")
        result[q] = {}
        for b in BRANCHES:
            scores = []
            for k in range(1, 4):
                key = q + f"-V{k}"
                checked = [universe[i] for i in schedule[key]["exam_indexes"]]
                scores.extend(predict(b, models[key][b], checked).tolist())
            result[q][b] = {
                **select_threshold(rows, scores),
                "selection_as_of": schedule[q]["train_as_of"],
                "latest_answer_mature_at": max(r["mature_at"] for r in rows),
            }
    return result


def predictions(
    root: Path, universe: list[dict], schedule: dict, exam: list[dict], models: dict, selected: dict
) -> list[dict]:
    """旧树在新增题上的分数明确标为本次历史重算；旧5704题仍逐值核对原封存。"""
    old_models = {
        "TREE11": read(root / "base/old-tree-models.json"),
        "ACTIVITY12": read(root / "base/old-activity-models.json"),
    }
    old = {identity(r): r for r in read(root / "base/old-predictions.json")}
    result = []
    for q in QUARTERS:
        checked = [r for r in exam if r["quarter"] == q]
        scores = {b: predict(b, models[q][b], checked) for b in BRANCHES}
        old_scores = {b: predict(b, old_models[b][q], checked) for b in BRANCHES}
        fit_rows = [universe[i] for i in schedule[q]["fit_indexes"]]
        majority = int(np.average([r["y"] for r in fit_rows], weights=weights(fit_rows)) > 0.5)
        for i, r in enumerate(checked):
            values = {b: float(scores[b][i]) for b in BRANCHES}
            values.update({"LEGACY_" + b: float(old_scores[b][i]) for b in BRANCHES})
            directions = {"ALWAYS_UP": 1, "ALWAYS_NON_UP": 0, "INITIAL_MAJORITY": majority, "MOMENTUM": r["momentum"]}
            for b in BRANCHES:
                directions[b] = int(values[b] > 0.5)
                directions[b + "_TUNED"] = int(values[b] > selected[q][b]["threshold"])
                directions["LEGACY_" + b] = int(values["LEGACY_" + b] > 0.5)
                for threshold in (0.45, 0.55):
                    directions[b + f"_FIXED_{int(threshold * 100)}"] = int(values[b] > threshold)
            if identity(r) in old:
                previous = old[identity(r)]
                if (
                    values["LEGACY_TREE11"] != previous["scores"]["TREE11"]
                    or values["LEGACY_ACTIVITY12"] != previous["scores"]["TREE_ACTIVITY12"]
                ):
                    raise ValueError("FULL_THRESHOLD_LEGACY_SCORES_CHANGED")
            result.append(
                {
                    **r,
                    "kind": "HISTORICAL_RECONSTRUCTION_ASSUMED_V2",
                    "scores": values,
                    "directions": directions,
                    "selected_thresholds": {b: selected[q][b]["threshold"] for b in BRANCHES},
                }
            )
    return sorted(result, key=lambda r: (r["t"], r["family"], r["fund_code"]))


def run(root: Path, *, replay: bool = False) -> dict:
    spec = verify_inputs(root)
    if replay:
        verify(root)
    universe, schedule, exam = (read(root / n) for n in ("universe.json", "schedule.json", "exam.json"))
    out = root / ("replay" if replay else "main")
    out.mkdir(exist_ok=False)
    write_new(out / "budget-reserved.json", {"at": datetime.now(ZONE).isoformat(), "max_fits": 32})
    models = {}
    # 先完成较早折外模型及选择，再训练季度模型；每项拟合前独占预算回执。
    order = [q + f"-V{k}" for q in QUARTERS for k in (1, 2, 3)] + list(QUARTERS)
    for key in order:
        info = schedule[key]
        rows, checked = [universe[i] for i in info["fit_indexes"]], [universe[i] for i in info["exam_indexes"]]
        models[key] = {}
        for b in BRANCHES:
            write_new(
                out / (key + "-" + b + "-reserved.json"),
                {"at": datetime.now(ZONE).isoformat(), "fit_hash": info["fit_hash"]},
            )
            if b == "TREE11":
                model = weekly.fit_control(activity.control_rows(rows), info, spec, activity.control_rows(checked))
            else:
                model = activity.fit(
                    rows,
                    models[key]["TREE11"],
                    {**spec, "fit_hashes": {key: info["fit_hash"]}, "weight_hashes": {key: info["weight_hash"]}},
                    key,
                    checked,
                )
            models[key][b] = model
            write_new(out / (key + "-" + b + ".json"), model)
        if key == QUARTERS[-1] + "-V3":
            selected = selections(universe, schedule, models)
            write_new(out / "thresholds.json", selected)
            write_new(
                out / "threshold-selection-receipt.json",
                {"at": datetime.now(ZONE).isoformat(), "hash": digest(selected)},
            )
    predicted = predictions(root, universe, schedule, exam, models, selected)
    write_new(out / "models.json", models)
    write_new(out / "predictions.json", predicted)
    completion = {
        "finished_at": datetime.now(ZONE).isoformat(),
        "successful_fits": 32,
        "models_hash": digest(models),
        "predictions_hash": digest(predicted),
        "thresholds_hash": digest(selected),
        "model_released": False,
    }
    write_new(out / "completion.json", completion)
    if replay:
        for n in ("models.json", "predictions.json", "thresholds.json"):
            if read(out / n) != read(root / "main" / n):
                raise ValueError("FULL_THRESHOLD_REPLAY_MISMATCH")
        write_new(
            root / "replay-proof.json",
            {
                "models_equal": True,
                "predictions_equal": True,
                "thresholds_equal": True,
                "max_score_diff": 0.0,
                "successful_fits": 32,
                "prediction_count": len(predicted),
            },
        )
    return completion


def verify(root: Path) -> dict:
    """只读复算全部行、时间划分、尺度、阈值选择及分数；不增加任何拟合或下载。"""
    spec = verify_inputs(root)
    universe, schedule, exam = derive(root / "base")
    for name, value in (("universe.json", universe), ("schedule.json", schedule), ("exam.json", exam)):
        if value != read(root / name):
            raise ValueError("FULL_THRESHOLD_RECONSTRUCTION_CHANGED")
    if (
        sorted(schedule) != spec["job_keys"]
        or len(exam) != spec["exam_count"]
        or sorted({r["fund_code"] for r in exam}) != spec["fund_codes"]
    ):
        raise ValueError("FULL_THRESHOLD_SCOPE_OR_SCHEDULE_CHANGED")
    for mode in ("main", "replay"):
        out = root / mode
        if not (out / "completion.json").exists():
            if mode == "main":
                raise ValueError("FULL_THRESHOLD_RUN_INCOMPLETE")
            continue
        complete, models = read(out / "completion.json"), read(out / "models.json")
        budget = read(out / "budget-reserved.json")
        expected_reservations = {
            "budget-reserved.json",
            *(k + "-" + b + "-reserved.json" for k in schedule for b in BRANCHES),
        }
        if (
            complete["successful_fits"] != 32
            or set(models) != set(schedule)
            or budget["max_fits"] != 32
            or {p.name for p in out.glob("*-reserved.json")} != expected_reservations
            or complete["models_hash"] != digest(models)
        ):
            raise ValueError("FULL_THRESHOLD_RUN_BUDGET_OR_MODELS_CHANGED")
        chosen = selections(universe, schedule, models)
        choice_receipt = read(out / "threshold-selection-receipt.json")
        if (
            chosen != read(out / "thresholds.json")
            or digest(chosen) != complete["thresholds_hash"]
            or choice_receipt["hash"] != digest(chosen)
        ):
            raise ValueError("FULL_THRESHOLD_CHOICE_CHANGED")
        for key, info in schedule.items():
            rows = [universe[i] for i in info["fit_indexes"]]
            if set(models[key]) != set(BRANCHES):
                raise ValueError("FULL_THRESHOLD_BRANCH_CHANGED")
            for b in BRANCHES:
                model = models[key][b]
                if model != read(out / (key + "-" + b + ".json")):
                    raise ValueError("FULL_THRESHOLD_MODEL_FILE_CHANGED")
                reservation = read(out / (key + "-" + b + "-reserved.json"))
                if (
                    reservation["fit_hash"] != info["fit_hash"]
                    or not budget["at"] <= reservation["at"] <= complete["finished_at"]
                ):
                    raise ValueError("FULL_THRESHOLD_RESERVATION_CHANGED")
                if key in QUARTERS and reservation["at"] < choice_receipt["at"]:
                    raise ValueError("FULL_THRESHOLD_CHOICE_AFTER_OUTER_FIT")
                weekly.verify_model(
                    model,
                    activity.control_rows(rows) if b == "TREE11" else rows,
                    info,
                    spec,
                    control=None if b == "TREE11" else models[key]["TREE11"],
                )
        expected = predictions(root, universe, schedule, exam, models, chosen)
        if expected != read(out / "predictions.json") or digest(expected) != complete["predictions_hash"]:
            raise ValueError("FULL_THRESHOLD_RESTORED_PREDICTIONS_CHANGED")
    return {
        "verified": True,
        "exam_count": len(exam),
        "restored_count": spec["restored_count"],
        "new_fits": 0,
        "api_calls": 0,
    }


def evaluate(root: Path) -> dict:
    verify(root)
    spec, proof = read(root / "study.json"), read(root / "replay-proof.json")
    if (
        not all(proof[k] for k in ("models_equal", "predictions_equal", "thresholds_equal"))
        or proof["max_score_diff"] != 0
    ):
        raise ValueError("FULL_THRESHOLD_REPLAY_REQUIRED")
    for n in ("models.json", "predictions.json", "thresholds.json"):
        if read(root / "main" / n) != read(root / "replay" / n):
            raise ValueError("FULL_THRESHOLD_REPLAY_CHANGED")
    rows = read(root / "main/predictions.json")
    branches = list(rows[0]["directions"])
    metrics = sector.comparison.metrics
    pairs = [(b, "LEGACY_" + b) for b in BRANCHES] + [(b + "_TUNED", b) for b in BRANCHES] + [("ACTIVITY12", "TREE11")]
    result = {
        "kind": "DEVELOPMENT_FULL_COHORT_AND_THRESHOLD_NOT_INDEPENDENT_TEST",
        "overall": {b: metrics(rows, b) for b in branches},
        "strata": {
            field: {
                str(value): {b: metrics([r for r in rows if r[field] == value], b) for b in branches}
                for value in sorted({r[field] for r in rows})
            }
            for field in ("quarter", "fund_code", "restored_question")
        },
        "paired": {a + "_VS_" + b: sector.paired(rows, a, b, spec) for a, b in pairs},
        "thresholds": read(root / "main/thresholds.json"),
        "runtime_model_changed": False,
        "model_released": False,
        "limitations": [
            "2021—2024反复使用开发数据",
            "V2早间净值及答案可用时间是假设，未证明首版本",
            "阈值仅在较早验证段选择，不按外层成绩择优",
            "日期和基金相关，按目标日联合重采样",
            "没有新增真实预测、没有解封2025/2026、没有切换模型",
        ],
    }
    write_new(root / "comparison.json", result)
    write_new(root / "comparison-receipt.json", {"at": datetime.now(ZONE).isoformat(), "hash": digest(result)})
    return result
