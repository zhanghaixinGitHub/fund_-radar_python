"""固定两套树配方的周更对照；只做历史研究，不注册模型或改写真实预测。"""

from datetime import date, datetime, time, timedelta
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from app.services import direction_1d_activity_study as activity
from app.services import direction_1d_training as original
from app.services.direction_1d_protocol import ZONE, calendar, digest
from app.services.direction_training_artifacts import file_hash

tree, sector = activity.tree, activity.sector
read, write_new = original.read, original.write_new
CANDIDATES = ("WEEKLY_TREE11", "WEEKLY_ACTIVITY12")
PAIRS = ((CANDIDATES[0], "TREE11"), (CANDIDATES[1], "TREE_ACTIVITY12"), (CANDIDATES[1], CANDIDATES[0]))
SCHEDULE_RECIPE = {
    "cutoff": "SUNDAY_12_ASIA_SHANGHAI_BEFORE_TARGET_U_WEEK",
    "fit_window": "LAST_504_TRADING_T_DAYS_PER_ORIGINAL_GROUP",
    "availability": "LABEL_MATURE_AND_ALL_INPUTS_AVAILABLE_AT_CUTOFF",
    "weights": "ORIGINAL_FAMILY_DATE_NO_RECENCY",
    "scaler": "REFIT_ON_WEEKLY_FIT_SHARED_FIRST_11",
    "exam": "UNCHANGED_2024_ACTIVITY_EXAM",
    "empty_exam_weeks": "NO_FIT_REQUIRED",
    "final_fit": False,
    "threshold": 0.5,
}


def fingerprint() -> dict:
    result = activity.fingerprint()
    project = Path(__file__).resolve().parents[2]
    for name in ("app/services/direction_1d_weekly_study.py", "scripts/direction_1d_weekly_study.py"):
        result[name] = file_hash(project / name)
    return result


def week_cutoff(target: str) -> datetime:
    """按目标U日所属自然周取前一周日12点；跨年周仍按U的研究年份限制。"""
    u = date.fromisoformat(target)
    if u.year != 2024 or u not in calendar()[0]:
        raise ValueError("WEEKLY_EXAM_DATE_INVALID")
    return datetime.combine(u - timedelta(days=u.weekday() + 1), time(12), ZONE)


def select_week(raw: list[dict], available: dict, cutoff: datetime) -> list[dict]:
    """先排除截止时尚未成熟/未公布的输入，再按原分组截504日；不按答案删样本。"""
    known = [
        r
        for r in raw
        if datetime.fromisoformat(r["mature_at"]) <= cutoff and available[r["fund_code"], r["t"]] <= cutoff
    ]
    keys = {
        (r["fund_code"], r["t"])
        for group in sorted({r["group"] for r in raw})
        for r in original.select_fit([r for r in known if r["group"] == group], cutoff)
    }
    return [r for r in raw if (r["fund_code"], r["t"]) in keys]


def ensure_known(rows: list[dict], available: dict, cutoff: datetime) -> None:
    """净值公告、目标成熟及两种指数输入均不能晚于训练时刻。"""
    for r in rows:
        stamps = [datetime.fromisoformat(r["mature_at"]), available[r["fund_code"], r["t"]]]
        stamps.extend(
            datetime.fromisoformat(r[k]["available_at_assumed"])
            for k in ("market_input", "specific_input", "activity_input")
        )
        if max(stamps) > cutoff or r["u"] > "2024-12-31" or r["t"] >= r["u"]:
            raise ValueError("WEEKLY_FIT_NOT_AVAILABLE")


def build_inputs(base: Path) -> tuple[list[dict], dict, list[dict]]:
    """只从封存原始数据重建全量候选行；各周存行号以免重复保存50份历史数据。"""
    source = base / "base/base"
    raw = read(source / "baseline/dataset.json")
    available = sector.comparison.availability(raw, read(source / "baseline/history.json"))
    if len({(r["fund_code"], r["t"]) for r in raw}) != len(raw):
        raise ValueError("WEEKLY_DUPLICATE_SOURCE")
    mapped, _ = sector.attach_specific(
        raw, read(source / "mapping.json")["funds"], read(source / "prices.json")["prices"]
    )
    universe = activity.attach(mapped, read(base / "data/activity-data.json")["rows"])
    exam = read(base / "exam.json")
    codes = read(base / "study.json")["fund_codes"]
    if sorted({r["fund_code"] for r in universe}) != codes or sorted({r["fund_code"] for r in exam}) != codes:
        raise ValueError("WEEKLY_SCOPE_CHANGED")
    next_days = dict(zip(calendar()[0][:-1], calendar()[0][1:], strict=True))
    for r in universe:
        if next_days[date.fromisoformat(r["t"])] != date.fromisoformat(r["u"]):
            raise ValueError("WEEKLY_TARGET_NOT_NEXT_SESSION")
    schedule = {}
    cutoffs = {u: week_cutoff(u) for u in {r["u"] for r in exam}}
    for cutoff in sorted(set(cutoffs.values())):
        key = str(cutoff.date())
        selected = select_week(raw, available, cutoff)
        chosen = {(r["fund_code"], r["t"]) for r in selected}
        indexes = [i for i, r in enumerate(universe) if (r["fund_code"], r["t"]) in chosen]
        rows = [universe[i] for i in indexes]
        ensure_known(rows, available, cutoff)
        if len({r["t"] for r in rows}) < 252 or min(sum(r["y"] == y for r in rows) for y in (0, 1)) < 30:
            raise ValueError("WEEKLY_INSUFFICIENT_FIT")
        questions = [i for i, r in enumerate(exam) if cutoffs[r["u"]] == cutoff]
        for i in questions:
            r = exam[i]
            deadline = datetime.combine(date.fromisoformat(r["u"]), time(8), ZONE)
            if (
                cutoff >= deadline
                or available[r["fund_code"], r["t"]] > deadline
                or any(
                    datetime.fromisoformat(r[k]["available_at_assumed"]) > deadline
                    for k in ("market_input", "specific_input", "activity_input")
                )
            ):
                raise ValueError("WEEKLY_EXAM_NOT_AVAILABLE")
        schedule[key] = {
            "train_as_of": cutoff.isoformat(),
            "fit_indexes": indexes,
            "exam_indexes": questions,
            "fit_hash": digest(rows),
            "control_fit_hash": digest(activity.control_rows(rows)),
            "weight_hash": digest(original.weights(rows).tolist()),
            "fit_count": len(rows),
            "distinct_dates": len({r["t"] for r in rows}),
            "fit_start": min(r["t"] for r in rows),
            "fit_end": max(r["t"] for r in rows),
            "latest_mature_at": max(r["mature_at"] for r in rows),
            "exam_count": len(questions),
        }
    return universe, schedule, exam


def require_base(base: Path) -> None:
    activity.verify(base)
    activity.require_replay(base)


def freeze(base: Path, plan: Path, root: Path) -> dict:
    """先封存周界、每周成员、两套配方及100次预算；不进行分类器训练。"""
    require_base(base)
    universe, schedule, exam = build_inputs(base)
    before = read(base / "study.json")
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
            raise ValueError("WEEKLY_BASE_PATH_INVALID")
        tree.copy_file(base / name, root / "base" / name)
    tree.copy_file(plan, root / "plan.md")
    for name, value in (("universe.json", universe), ("schedule.json", schedule), ("exam.json", exam)):
        write_new(root / name, value)
    files = {"base/" + name: file_hash(root / "base" / name) for name in sorted(names)}
    files.update({n: file_hash(root / n) for n in ("plan.md", "universe.json", "schedule.json", "exam.json")})
    spec = {
        **{
            k: before[k]
            for k in (
                "fund_codes",
                "exam_count",
                "source",
                "protected_years_excluded",
                "primary_metric",
                "bootstrap_seed",
                "bootstrap_repetitions",
                "bootstrap_block_days",
            )
        },
        "kind": "HISTORICAL_WEEKLY_FREQUENCY_COMPARISON_ONLY",
        "created_at": datetime.now(ZONE).isoformat(),
        "fingerprint": fingerprint(),
        "candidates": CANDIDATES,
        "pairs": PAIRS,
        "schedule_recipe": SCHEDULE_RECIPE,
        "tree_recipe": tree.TREE_RECIPE,
        "activity_recipe": activity.FEATURE_RECIPE,
        "input_files": files,
        "weeks": sorted(schedule),
        "max_main_fits": 2 * len(schedule),
        "max_replay_fits": 2 * len(schedule),
        "quarterly_controls_refitted": False,
        "new_api_calls": 0,
        "database_writes": 0,
        "model_released": False,
        "cohort_id": "D1-WEEKLY-"
        + digest({"schedule": digest(schedule), "base": digest(before), "recipe": SCHEDULE_RECIPE})[:24],
    }
    write_new(root / "study.json", spec)
    write_new(root / "study-receipt.json", {"hash": digest(spec), "at": datetime.now(ZONE).isoformat()})
    return {k: spec[k] for k in ("cohort_id", "created_at", "exam_count", "weeks", "max_main_fits", "max_replay_fits")}


def verify_inputs(root: Path) -> dict:
    spec = read(root / "study.json")
    if digest(spec) != read(root / "study-receipt.json")["hash"] or spec["fingerprint"] != fingerprint():
        raise ValueError("WEEKLY_CODE_OR_SPEC_CHANGED")
    if (
        spec["candidates"] != list(CANDIDATES)
        or spec["pairs"] != [list(p) for p in PAIRS]
        or spec["schedule_recipe"] != SCHEDULE_RECIPE
        or spec["tree_recipe"] != tree.TREE_RECIPE
        or spec["activity_recipe"] != activity.FEATURE_RECIPE
        or spec["model_released"] is not False
    ):
        raise ValueError("WEEKLY_RECIPE_CHANGED")
    if datetime.now(ZONE) >= datetime.fromisoformat(spec["source"]["source_expires_at"]):
        raise ValueError("WEEKLY_SOURCE_EXPIRED")
    for name, expected in spec["input_files"].items():
        p = root / name
        if not p.resolve().is_relative_to(root.resolve()) or p.is_symlink() or file_hash(p) != expected:
            raise ValueError("WEEKLY_INPUT_CHANGED")
    schedule = read(root / "schedule.json")
    if (
        spec["weeks"] != sorted(schedule)
        or spec["max_main_fits"] != 2 * len(schedule)
        or spec["max_replay_fits"] != 2 * len(schedule)
    ):
        raise ValueError("WEEKLY_BUDGET_CHANGED")
    return spec


def fit_control(rows: list[dict], week: dict, spec: dict, exam: list[dict]) -> dict:
    """每周原11项树只拟合一次；Scaler由该周FIT计算，不伪造旧线性模型。"""
    if digest(rows) != week["control_fit_hash"] or digest(original.weights(rows).tolist()) != week["weight_hash"]:
        raise ValueError("WEEKLY_CONTROL_FIT_CHANGED")
    cutoff = datetime.fromisoformat(week["train_as_of"])
    if any(
        datetime.fromisoformat(r["mature_at"]) > cutoff
        or any(
            datetime.fromisoformat(r[k]["available_at_assumed"]) > cutoff for k in ("market_input", "specific_input")
        )
        for r in rows
    ):
        raise ValueError("WEEKLY_FIT_NOT_AVAILABLE")
    if len({r["t"] for r in rows}) < 252 or min(sum(r["y"] == y for r in rows) for y in (0, 1)) < 30:
        raise ValueError("WEEKLY_INSUFFICIENT_FIT")
    x, y, w = (
        np.asarray([sector.vector(r, "SPECIFIC11") for r in rows]),
        np.asarray([r["y"] for r in rows]),
        original.weights(rows),
    )
    with threadpool_limits(limits=1):
        scaler = StandardScaler().fit(x, sample_weight=w)
        estimator = HistGradientBoostingClassifier(**tree.TREE_RECIPE).fit(scaler.transform(x), y, sample_weight=w)
        model = {
            "candidate": "TREE11",
            "kind": "RESEARCH_ONLY",
            "target_definition": "UNIT_NAV_DIRECTION_V1",
            "cohort_id": spec["cohort_id"],
            "model_released": False,
            "train_as_of": week["train_as_of"],
            "features": sector.feature_names("SPECIFIC11"),
            "recipe": tree.TREE_RECIPE,
            "mean": scaler.mean_.tolist(),
            "scale": scaler.scale_.tolist(),
            "baseline": float(estimator._baseline_prediction[0, 0]),
            "trees": tree.export_trees(estimator),
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
                    np.max(
                        np.abs(tree.predict(model, checked) - estimator.predict_proba(scaler.transform(values))[:, 1])
                    )
                )
        if not all(np.isfinite(v) and 0 <= v <= 1e-12 for v in differences.values()):
            raise ValueError("WEEKLY_RESTORE_MISMATCH")
        model["restore_max_score_diff"] = differences
    return model


def expected_predictions(old: list[dict], scores: dict) -> list[dict]:
    """保留全部旧分支，增加每周分支与截止日期；历史输出明确沿用研究标识。"""
    return [
        {
            **r,
            "weekly_cutoff": week_cutoff(r["u"]).isoformat(),
            "scores": {**r["scores"], **{c: scores[c][r["fund_code"], r["t"]] for c in CANDIDATES}},
            "directions": {**r["directions"], **{c: int(scores[c][r["fund_code"], r["t"]] > 0.5) for c in CANDIDATES}},
        }
        for r in old
    ]


def run(root: Path, *, replay: bool = False) -> dict:
    spec = verify_inputs(root)
    if replay:
        verify(root)
    out = root / ("replay" if replay else "main")
    out.mkdir(exist_ok=False)
    write_new(out / "budget-reserved.json", {"at": datetime.now(ZONE).isoformat(), "max_fits": spec["max_main_fits"]})
    universe, schedule, exam = (read(root / n) for n in ("universe.json", "schedule.json", "exam.json"))
    models, scores = {}, {c: {} for c in CANDIDATES}
    for key in spec["weeks"]:
        week = schedule[key]
        rows, selected = [universe[i] for i in week["fit_indexes"]], [exam[i] for i in week["exam_indexes"]]
        for c in CANDIDATES:
            name = c + "-" + key
            write_new(
                out / (name + "-reserved.json"),
                {
                    "at": datetime.now(ZONE).isoformat(),
                    "fit_hash": week["control_fit_hash"] if c == CANDIDATES[0] else week["fit_hash"],
                },
            )
            if c == CANDIDATES[0]:
                model = fit_control(activity.control_rows(rows), week, spec, selected)
            else:
                model = activity.fit(
                    rows,
                    models[CANDIDATES[0] + "-" + key],
                    {
                        "cohort_id": spec["cohort_id"],
                        "fit_hashes": {key: week["fit_hash"]},
                        "weight_hashes": {key: week["weight_hash"]},
                    },
                    key,
                    selected,
                )
            write_new(out / (name + ".json"), model)
            models[name] = model
            predictor = tree.predict if c == CANDIDATES[0] else activity.predict
            scores[c].update(
                {(r["fund_code"], r["t"]): float(s) for r, s in zip(selected, predictor(model, selected), strict=True)}
            )
        if len(models) % 20 == 0:
            print(
                f"周更训练 {'复现' if replay else '主跑'}：{len(models)}/{spec['max_main_fits']} 次，截止 {key}",
                flush=True,
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
        activity.require_replay(root)
        write_new(
            root / "replay-proof.json",
            {
                "models_equal": True,
                "predictions_equal": True,
                "max_score_diff": 0.0,
                "successful_fits": len(models),
                "prediction_count": len(predictions),
            },
        )
    return result


def verify_model(model: dict, rows: list[dict], week: dict, spec: dict, *, control: dict | None = None) -> None:
    """核查每周训练成员与统计量、12项对照血缘；此校验不重新拟合分类器。"""
    extended = control is not None
    predictor = activity if extended else tree
    predictor.validate_model(model)
    if (
        model["cohort_id"] != spec["cohort_id"]
        or model["fit_hash"] != digest(rows)
        or model["weight_hash"] != week["weight_hash"]
        or model["weight_hash"] != digest(original.weights(rows).tolist())
        or model["train_as_of"] != week["train_as_of"]
        or model["fit_count"] != len(rows)
        or model["distinct_dates"] != len({r["t"] for r in rows})
        or model["classes"] != {str(y): sum(r["y"] == y for r in rows) for y in (0, 1)}
        or set(model["restore_max_score_diff"]) != {"fit", "exam"}
        or not all(np.isfinite(v) and 0 <= v <= 1e-12 for v in model["restore_max_score_diff"].values())
    ):
        raise ValueError("WEEKLY_MODEL_METADATA_CHANGED")
    values = np.asarray([activity.vector(r) if extended else sector.vector(r, "SPECIFIC11") for r in rows])
    w = original.weights(rows)
    mean = np.average(values, weights=w, axis=0)
    variance = np.average((values - mean) ** 2, weights=w, axis=0)
    scale = np.where(variance > 0, np.sqrt(variance), 1.0)
    if not np.allclose(model["mean"], mean, rtol=1e-12, atol=1e-15) or not np.allclose(
        model["scale"], scale, rtol=1e-12, atol=1e-15
    ):
        raise ValueError("WEEKLY_MODEL_SCALE_CHANGED")
    if extended and (
        model["original_fit_hash"] != control["fit_hash"]
        or model["control_model_hash"] != digest(control)
        or model["mean"][:11] != control["mean"]
        or model["scale"][:11] != control["scale"]
    ):
        raise ValueError("WEEKLY_MODEL_CONTROL_CHANGED")


def verify(root: Path) -> dict:
    spec = verify_inputs(root)
    require_base(root / "base")
    universe, schedule, exam = build_inputs(root / "base")
    if any(
        read(root / name) != value
        for name, value in (("universe.json", universe), ("schedule.json", schedule), ("exam.json", exam))
    ):
        raise ValueError("WEEKLY_DERIVED_INPUT_CHANGED")
    expected_keys = {c + "-" + k for c in CANDIDATES for k in spec["weeks"]}
    for mode in ("main", "replay"):
        folder = root / mode
        if not (folder / "completion.json").exists():
            if mode == "main" or folder.exists():
                raise ValueError("WEEKLY_TRAINING_INCOMPLETE")
            continue
        complete, models, predictions = (
            read(folder / n) for n in ("completion.json", "models.json", "predictions.json")
        )
        budget = read(folder / "budget-reserved.json")
        start, end = datetime.fromisoformat(budget["at"]), datetime.fromisoformat(complete["finished_at"])
        if (
            complete["models_hash"] != digest(models)
            or complete["predictions_hash"] != digest(predictions)
            or set(models) != expected_keys
            or budget["max_fits"] != len(expected_keys)
            or complete["successful_fits"] != len(expected_keys)
            or not datetime.fromisoformat(spec["created_at"]) <= start <= end
            or {p.name for p in folder.glob("*-reserved.json")}
            != {"budget-reserved.json", *(k + "-reserved.json" for k in expected_keys)}
        ):
            raise ValueError("WEEKLY_OUTPUT_OR_BUDGET_CHANGED")
        scores = {c: {} for c in CANDIDATES}
        for key, week in schedule.items():
            rows, selected = [universe[i] for i in week["fit_indexes"]], [exam[i] for i in week["exam_indexes"]]
            control = models[CANDIDATES[0] + "-" + key]
            for c in CANDIDATES:
                name = c + "-" + key
                model, reserved = models[name], read(folder / (name + "-reserved.json"))
                if (
                    model != read(folder / (name + ".json"))
                    or reserved["fit_hash"] != model["fit_hash"]
                    or not start <= datetime.fromisoformat(reserved["at"]) <= end
                ):
                    raise ValueError("WEEKLY_MODEL_RECEIPT_CHANGED")
                verify_model(
                    model,
                    activity.control_rows(rows) if c == CANDIDATES[0] else rows,
                    week,
                    spec,
                    control=control if c == CANDIDATES[1] else None,
                )
                predictor = tree.predict if c == CANDIDATES[0] else activity.predict
                scores[c].update(
                    {
                        (r["fund_code"], r["t"]): float(s)
                        for r, s in zip(selected, predictor(model, selected), strict=True)
                    }
                )
        if len(predictions) != spec["exam_count"] or predictions != expected_predictions(
            read(root / "base/main/predictions.json"), scores
        ):
            raise ValueError("WEEKLY_RESTORED_OUTPUT_CHANGED")
    if (root / "replay/completion.json").exists():
        activity.require_replay(root)
    return read(root / "main/completion.json")


def evaluate(root: Path) -> dict:
    verify(root)
    activity.require_replay(root)
    spec, rows = read(root / "study.json"), read(root / "main/predictions.json")
    proof = {
        "models_equal": True,
        "predictions_equal": True,
        "max_score_diff": 0.0,
        "successful_fits": spec["max_main_fits"],
        "prediction_count": spec["exam_count"],
    }
    if read(root / "replay-proof.json") != proof:
        raise ValueError("WEEKLY_REPLAY_PROOF_INVALID")
    branches = (
        *CANDIDATES,
        "TREE11",
        "TREE_ACTIVITY12",
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
        "paired": {c + "_VS_" + reference: sector.paired(rows, c, reference, spec) for c, reference in PAIRS},
        "runtime_model_changed": False,
        "model_released": False,
        "limitations": [
            "2021—2024重复使用开发数据，非独立验证",
            "周更截至前一周日12点，旧季度模型保留原截止时刻",
            "历史可用时刻是假设，未验证历史首版",
            "无考题的周不拟合，不删除已有考题",
            "真实未来效果待到期",
        ],
    }
    write_new(root / "comparison.json", result)
    write_new(root / "comparison-receipt.json", {"hash": digest(result), "at": datetime.now(ZONE).isoformat()})
    return result
