"""一日单日市场广度对照：复用冻结12维树，仅增加T日上涨股票比例，离线有界执行。"""

from datetime import date, datetime, time, timedelta
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from app.services import direction_1d_full_threshold as full
from app.services import direction_breadth_data as breadth
from app.services.direction_1d_protocol import ZONE, calendar, digest
from app.services.direction_1d_training import read, weights, write_new
from app.services.direction_training_artifacts import file_hash, read_seal

activity, tree, sector = full.activity, full.tree, full.sector
REFERENCE, CANDIDATE = "ACTIVITY12", "BREADTH13"
QUARTERS = full.QUARTERS
FEATURES = [*activity.FEATURES, "sh_sz_a_share_up_fraction_t"]
FEATURE_RECIPE = {
    "numerator": "OFFICIAL_PCT_CHG_STRICTLY_POSITIVE",
    "denominator": "POSITIVE_VOLUME_VALID_SH_SZ_A_SHARE_ROWS_INCLUDING_FLAT",
    "input_date": "T_ONLY_NO_ROLLING_AVERAGE_NO_TARGET_U",
    "unit": "FRACTION_0_TO_1",
    "availability": "T_18_ASSUMED_NOT_HISTORICAL_FIRST_PUBLICATION_PROOF",
    "scaler": "KEEP_CONTROL_12_APPEND_FIT_WEIGHTED_1",
    "missing": "FAIL_WHOLE_EXPERIMENT_NO_IMPUTATION_OR_QUESTION_DROPPING",
}
OBSERVATION_RULE = {
    "positive_quarters": 3,
    "positive_funds": 17,
    "maximum_fund_accuracy_drop": 0.05,
    "require_positive_paired_lower_bound": True,
    "require_both_recalls_non_decreasing": True,
}


def fingerprint() -> dict:
    """绑定本轮与所有复用的实现；新增文件不改写旧实验指纹。"""
    result = full.fingerprint()
    project = Path(__file__).resolve().parents[2]
    for name in (
        "app/services/direction_1d_breadth_study.py",
        "scripts/direction_1d_breadth_study.py",
        "app/services/direction_breadth_data.py",
        "app/integrations/tushare_breadth.py",
        "app/services/trading_calendar.py",
    ):
        result[name] = file_hash(project / name)
    return result


def validate_source(source: dict) -> None:
    """仅接纳当前已登记daily能力的原来源，凭据不进入研究包。"""
    if (
        source.get("source_code") != "TUSHARE_PRO_FUND"
        or source.get("enabled") is not True
        or not source.get("authorization_verified_at")
        or "daily" not in source.get("authorized_api_names", [])
        or source.get("retention_days", 0) <= 0
    ):
        raise ValueError("BREADTH1D_SOURCE_UNAVAILABLE")


def source_files(folder: Path) -> dict[str, str]:
    """先核验原月度封存及总数据封存，只选择数据依赖，不复制或修改20日模型。"""
    acquired = read_seal(folder, "breadth-acquired.json")
    names = {"breadth-acquired.json", *acquired["files"]}
    for month in sorted({d[:7] for d in breadth.days()}):
        name = f"breadth-source-month-{month}.json"
        names.update(read_seal(folder, name)["files"])
        names.add(name)
    return {n: file_hash(folder / n) for n in sorted(names)}


def attach(rows: list[dict], daily: dict) -> list[dict]:
    """仅附加T日单日比例；整数计数重新求比值，缺日或未来/重复样本拒绝整轮。"""
    days = calendar()[0]
    next_day = dict(zip(days[:-1], days[1:], strict=True))
    result, seen = [], set()
    for row in rows:
        t, u = date.fromisoformat(row["t"]), date.fromisoformat(row["u"])
        if (
            not 2021 <= t.year <= u.year <= 2024
            or next_day.get(t) != u
            or full.identity(row) in seen
            or "breadth_input" in row
        ):
            raise ValueError("BREADTH1D_DATE_IDENTITY_OR_DUPLICATE")
        seen.add(full.identity(row))
        item = daily.get(row["t"])
        if item is None or item.get("date") != row["t"] or item.get("status") != "READY":
            raise ValueError("BREADTH1D_DAY_MISSING_OR_NOT_READY")
        c = item["counts"]
        if (
            any(type(c[k]) is not int or c[k] < 0 for k in ("up", "down", "flat", "valid", "SH", "SZ"))
            or c["valid"] < breadth.RULES["daily_minimum"]
            or c["up"] + c["down"] + c["flat"] != c["valid"]
            or c["SH"] + c["SZ"] != c["valid"]
            or min(c["SH"], c["SZ"]) < breadth.RULES["exchange_minimum"]
        ):
            raise ValueError("BREADTH1D_COUNTS_INVALID")
        fraction = c["up"] / c["valid"]
        if not np.isfinite(item["up_fraction"]) or fraction != item["up_fraction"]:
            raise ValueError("BREADTH1D_FRACTION_INVALID")
        result.append(
            {
                **row,
                "breadth_input": {
                    "x": [fraction],
                    "date": row["t"],
                    "counts": c,
                    "source_hash": digest(item),
                    "available_at_assumed": datetime.combine(t, time(18), ZONE).isoformat(),
                },
            }
        )
    return result


def controls(rows: list[dict]) -> list[dict]:
    return [{k: v for k, v in r.items() if k != "breadth_input"} for r in rows]


def vector(row: dict) -> list[float]:
    extra = row["breadth_input"]["x"]
    if len(extra) != 1 or not np.isfinite(extra).all() or not 0 <= extra[0] <= 1:
        raise ValueError("BREADTH1D_FEATURE_INVALID")
    result = [*activity.vector(row), *extra]
    if len(result) != 13 or not np.isfinite(result).all():
        raise ValueError("BREADTH1D_VECTOR_INVALID")
    return result


def available(rows: list[dict], cutoff: datetime, *, fit: bool) -> None:
    full.check_available(rows, cutoff, fit=fit)
    for r in rows:
        if datetime.fromisoformat(r["breadth_input"]["available_at_assumed"]) > cutoff:
            raise ValueError("BREADTH1D_INPUT_TOO_LATE")


def freeze(base: Path, data: Path, plan: Path, scope_path: Path, source_path: Path, root: Path) -> dict:
    """验明旧模型及原始数据后固定4次主拟合和4次复现；此阶段不训练、不联网。"""
    full.verify(base)
    activity.require_replay(base)
    scope, source = read(scope_path), read(source_path)
    full.audit.validate_scope(scope)
    validate_source(source)
    old = read(base / "study.json")
    if old["owner_scope_hash"] != scope["scope_hash"] or not set(old["fund_codes"]) <= set(scope["codes"]):
        raise ValueError("BREADTH1D_OWNER_SCOPE_CHANGED")
    data_names = source_files(data)
    payload = read(data / "breadth-data.json")
    if payload["source_metadata"]["source_id"] != source["source_id"] or payload["rules"] != breadth.RULES:
        raise ValueError("BREADTH1D_SOURCE_IDENTITY_CHANGED")
    # 从全部969份原始股票响应重算分子/分母，不把旧五日特征改名为单日特征。
    daily = breadth.rebuild_daily(data)
    acquired = min(datetime.fromisoformat(r["retrieved_at"]) for r in payload["requests"])
    expires = min(
        acquired + timedelta(days=source["retention_days"]), datetime.fromisoformat(old["source"]["source_expires_at"])
    )
    if datetime.now(ZONE) >= expires:
        raise ValueError("BREADTH1D_SOURCE_EXPIRED")
    universe, exam = attach(read(base / "universe.json"), daily), attach(read(base / "exam.json"), daily)
    schedule = {q: read(base / "schedule.json")[q] for q in QUARTERS}
    fits = {}
    for q, info in schedule.items():
        rows = [universe[i] for i in info["fit_indexes"]]
        available(rows, datetime.fromisoformat(info["train_as_of"]), fit=True)
        fits[q] = digest(rows)
    for row in exam:
        available([row], datetime.combine(date.fromisoformat(row["u"]), time(8), ZONE), fit=False)
    root.mkdir(parents=True, exist_ok=False)
    names = set(old["input_files"]) | {"study.json", "study-receipt.json", "replay-proof.json"}
    for mode in ("main", "replay"):
        names.update(p.relative_to(base).as_posix() for p in (base / mode).iterdir() if p.is_file())
    for name in sorted(names):
        tree.copy_file(base / name, root / "control" / name)
    for name in data_names:
        tree.copy_file(data / name, root / "data" / name)
    for name, path in (("plan.md", plan), ("owner-scope.json", scope_path), ("source-status.json", source_path)):
        tree.copy_file(path, root / name)
    for name, value in (("universe.json", universe), ("exam.json", exam), ("schedule.json", schedule)):
        write_new(root / name, value)
    files = {"control/" + n: file_hash(root / "control" / n) for n in names}
    files.update({"data/" + n: file_hash(root / "data" / n) for n in data_names})
    files.update(
        {
            n: file_hash(root / n)
            for n in (
                "plan.md",
                "owner-scope.json",
                "source-status.json",
                "universe.json",
                "exam.json",
                "schedule.json",
            )
        }
    )
    spec = {
        "kind": "HISTORICAL_ONE_DAY_BREADTH_ONLY",
        "created_at": datetime.now(ZONE).isoformat(),
        "fingerprint": fingerprint(),
        "candidate": CANDIDATE,
        "reference": REFERENCE,
        "features": FEATURES,
        "feature_recipe": FEATURE_RECIPE,
        "tree_recipe": tree.TREE_RECIPE,
        "observation_rule": OBSERVATION_RULE,
        "source_expires_at": expires.isoformat(),
        "first_acquired_at": acquired.isoformat(),
        "source_id": source["source_id"],
        "source_daily_count": len(daily),
        "historical_first_versions_verified": False,
        "fund_codes": old["fund_codes"],
        "owner_scope_hash": scope["scope_hash"],
        "exam_count": len(exam),
        "restored_count": sum(r["restored_question"] for r in exam),
        "fit_hashes": fits,
        "fit_counts": {q: schedule[q]["fit_count"] for q in QUARTERS},
        "threshold": 0.5,
        "max_main_fits": 4,
        "max_replay_fits": 4,
        "input_files": files,
        "protected_years_excluded": [2025, 2026],
        "new_api_calls": 0,
        "database_writes": 0,
        "model_released": False,
        "primary_metric": old["primary_metric"],
        "bootstrap_seed": 20260912,
        "bootstrap_repetitions": 2000,
        "bootstrap_block_days": 5,
    }
    spec["cohort_id"] = "D1-BREADTH-" + digest({"files": files, "recipe": FEATURE_RECIPE})[:24]
    write_new(root / "study.json", spec)
    write_new(root / "study-receipt.json", {"at": spec["created_at"], "hash": digest(spec)})
    return {
        k: spec[k]
        for k in ("cohort_id", "created_at", "exam_count", "restored_count", "fit_counts", "source_daily_count")
    }


def verify_inputs(root: Path) -> dict:
    spec = read(root / "study.json")
    if digest(spec) != read(root / "study-receipt.json")["hash"] or spec["fingerprint"] != fingerprint():
        raise ValueError("BREADTH1D_CODE_OR_SPEC_CHANGED")
    expected = {
        "candidate": CANDIDATE,
        "reference": REFERENCE,
        "features": FEATURES,
        "feature_recipe": FEATURE_RECIPE,
        "tree_recipe": tree.TREE_RECIPE,
        "observation_rule": OBSERVATION_RULE,
        "threshold": 0.5,
        "max_main_fits": 4,
        "max_replay_fits": 4,
        "protected_years_excluded": [2025, 2026],
        "new_api_calls": 0,
        "database_writes": 0,
        "model_released": False,
    }
    if any(spec.get(k) != v for k, v in expected.items()):
        raise ValueError("BREADTH1D_RECIPE_OR_BUDGET_CHANGED")
    if datetime.now(ZONE) >= datetime.fromisoformat(spec["source_expires_at"]):
        raise ValueError("BREADTH1D_SOURCE_EXPIRED")
    for name, expected_hash in spec["input_files"].items():
        path = root / name
        if not path.resolve().is_relative_to(root.resolve()) or path.is_symlink() or file_hash(path) != expected_hash:
            raise ValueError("BREADTH1D_INPUT_CHANGED")
    return spec


def validate_model(model: dict) -> None:
    """13维模型单独校验，保留原树的完整拓扑/有限数检查，不放宽旧12维协议。"""
    if (
        model.get("candidate") != CANDIDATE
        or model.get("features") != FEATURES
        or model.get("recipe") != tree.TREE_RECIPE
        or model.get("feature_recipe") != FEATURE_RECIPE
        or model.get("model_released") is not False
        or model.get("target_definition") != "UNIT_NAV_DIRECTION_V1"
        or len(model["mean"]) != 13
        or len(model["scale"]) != 13
        or not np.isfinite([model["baseline"], *model["mean"], *model["scale"]]).all()
        or min(model["scale"]) <= 0
        or len(model["trees"]) != tree.TREE_RECIPE["max_iter"]
    ):
        raise ValueError("BREADTH1D_MODEL_PROTOCOL_OR_SCALE")
    for nodes in model["trees"]:
        if not 1 <= len(nodes) <= 2 * tree.TREE_RECIPE["max_leaf_nodes"] - 1:
            raise ValueError("BREADTH1D_MODEL_NODES")
        seen, stack, leaves = set(), [(0, 0)], 0
        while stack:
            i, depth = stack.pop()
            if i in seen or not 0 <= i < len(nodes) or depth > tree.TREE_RECIPE["max_depth"]:
                raise ValueError("BREADTH1D_MODEL_TOPOLOGY")
            seen.add(i)
            n = nodes[i]
            if n["depth"] != depth or not np.isfinite([n["value"], n["threshold"]]).all() or n["count"] <= 0:
                raise ValueError("BREADTH1D_MODEL_NODE")
            if n["leaf"]:
                leaves += 1
            else:
                if not 0 <= n["feature"] < 13 or n["left"] <= i or n["right"] <= i:
                    raise ValueError("BREADTH1D_MODEL_BRANCH")
                stack.extend(((n["left"], depth + 1), (n["right"], depth + 1)))
        if len(seen) != len(nodes) or leaves > tree.TREE_RECIPE["max_leaf_nodes"]:
            raise ValueError("BREADTH1D_MODEL_UNREACHABLE")


def predict(model: dict, rows: list[dict]) -> np.ndarray:
    """仅用输入与数值树评分，不读取答案；不使用可执行序列化文件。"""
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
    """训练仅增加第13维；原12维尺度、成员、权重及截止必须与冻结对照逐值一致。"""
    activity.validate_model(control)
    w = weights(rows)
    if (
        digest(rows) != spec["fit_hashes"][quarter]
        or digest(controls(rows)) != control["fit_hash"]
        or digest(w.tolist()) != control["weight_hash"]
        or len({r["t"] for r in rows}) < 252
        or min(sum(r["y"] == y for r in rows) for y in (0, 1)) < 30
    ):
        raise ValueError("BREADTH1D_FIT_MEMBERS_OR_WEIGHTS")
    cutoff = datetime.fromisoformat(control["train_as_of"])
    available(rows, cutoff, fit=True)
    if {r["u"] for r in rows} & {r["u"] for r in exam}:
        raise ValueError("BREADTH1D_FIT_EXAM_OVERLAP")
    for r in exam:
        deadline = datetime.combine(date.fromisoformat(r["u"]), time(8), ZONE)
        available([r], deadline, fit=False)
        if cutoff >= deadline:
            raise ValueError("BREADTH1D_MODEL_TOO_LATE")
    values, y = np.asarray([vector(r) for r in rows]), np.asarray([r["y"] for r in rows])
    with threadpool_limits(limits=1):
        scaler = StandardScaler().fit(values[:, -1:], sample_weight=w)
        mean, scale = [*control["mean"], float(scaler.mean_[0])], [*control["scale"], float(scaler.scale_[0])]
        estimator = HistGradientBoostingClassifier(**tree.TREE_RECIPE).fit(
            (values - np.asarray(mean)) / np.asarray(scale), y, sample_weight=w
        )
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
            "features": FEATURES,
            "feature_recipe": FEATURE_RECIPE,
            "cohort_id": spec["cohort_id"],
            "model_released": False,
            "mean": mean,
            "scale": scale,
            "baseline": float(estimator._baseline_prediction[0, 0]),
            "trees": tree.export_trees(estimator),
            "fit_hash": digest(rows),
            "control_fit_hash": control["fit_hash"],
            "control_model_hash": digest(control),
        }
        differences = {}
        for name, checked in (("fit", rows), ("exam", exam)):
            if checked:
                x = (np.asarray([vector(r) for r in checked]) - np.asarray(mean)) / np.asarray(scale)
                differences[name] = float(np.max(np.abs(predict(model, checked) - estimator.predict_proba(x)[:, 1])))
        if not all(np.isfinite(v) and v <= 1e-12 for v in differences.values()):
            raise ValueError("BREADTH1D_RESTORE_MISMATCH")
        model["restore_max_score_diff"] = differences
    return model


def predictions(root: Path, exam: list[dict], models: dict) -> list[dict]:
    previous = {full.identity(r): r for r in read(root / "control/main/predictions.json")}
    result = []
    for q in QUARTERS:
        checked = [r for r in exam if r["quarter"] == q]
        scores = predict(models[q], checked)
        for row, score in zip(checked, scores, strict=True):
            old = previous[full.identity(row)]
            if any(old[k] != row[k] for k in ("y", "u", "input_hash", "label_hash", "restored_question")):
                raise ValueError("BREADTH1D_ANSWER_OR_QUESTION_CHANGED")
            branches = (REFERENCE, "TREE11", "ALWAYS_UP", "ALWAYS_NON_UP", "INITIAL_MAJORITY", "MOMENTUM")
            result.append(
                {
                    **row,
                    "scores": {
                        REFERENCE: old["scores"][REFERENCE],
                        "TREE11": old["scores"]["TREE11"],
                        CANDIDATE: float(score),
                    },
                    "directions": {**{b: old["directions"][b] for b in branches}, CANDIDATE: int(score > 0.5)},
                }
            )
    return sorted(result, key=lambda r: (r["t"], r["family"], r["fund_code"]))


def run(root: Path, *, replay: bool = False) -> dict:
    spec = verify_inputs(root)
    if replay:
        verify(root, rebuild_source=False)
    universe, schedule, exam = (read(root / n) for n in ("universe.json", "schedule.json", "exam.json"))
    old = read(root / "control/main/models.json")
    out = root / ("replay" if replay else "main")
    out.mkdir(exist_ok=False)
    write_new(out / "budget-reserved.json", {"at": datetime.now(ZONE).isoformat(), "max_fits": 4})
    models = {}
    for q in QUARTERS:
        rows = [universe[i] for i in schedule[q]["fit_indexes"]]
        checked = [r for r in exam if r["quarter"] == q]
        write_new(out / (q + "-reserved.json"), {"at": datetime.now(ZONE).isoformat(), "fit_hash": digest(rows)})
        models[q] = fit(rows, old[q][REFERENCE], spec, q, checked)
        write_new(out / (q + ".json"), models[q])
    predicted = predictions(root, exam, models)
    for name, value in (("models.json", models), ("predictions.json", predicted)):
        write_new(out / name, value)
    complete = {
        "finished_at": datetime.now(ZONE).isoformat(),
        "successful_fits": 4,
        "models_hash": digest(models),
        "predictions_hash": digest(predicted),
        "model_released": False,
    }
    write_new(out / "completion.json", complete)
    if replay:
        for name in ("models.json", "predictions.json"):
            if read(out / name) != read(root / "main" / name):
                raise ValueError("BREADTH1D_REPLAY_MISMATCH")
        write_new(
            root / "replay-proof.json",
            {
                "models_equal": True,
                "predictions_equal": True,
                "max_score_diff": 0.0,
                "successful_fits": 4,
                "prediction_count": len(predicted),
            },
        )
    return complete


def verify(root: Path, *, rebuild_source: bool = True) -> dict:
    """只读恢复全部输入、时间、尺度和分数；默认从原始股票行复算，无任何模型拟合。"""
    spec = verify_inputs(root)
    if rebuild_source:
        full.verify(root / "control")
    daily = breadth.rebuild_daily(root / "data") if rebuild_source else read(root / "data/breadth-data.json")["daily"]
    universe, exam = (
        attach(read(root / "control/universe.json"), daily),
        attach(read(root / "control/exam.json"), daily),
    )
    schedule = {q: read(root / "control/schedule.json")[q] for q in QUARTERS}
    if any(
        value != read(root / name)
        for name, value in (("universe.json", universe), ("exam.json", exam), ("schedule.json", schedule))
    ):
        raise ValueError("BREADTH1D_RECONSTRUCTION_CHANGED")
    if len(exam) != spec["exam_count"] or sorted({r["fund_code"] for r in exam}) != spec["fund_codes"]:
        raise ValueError("BREADTH1D_SCOPE_CHANGED")
    old = read(root / "control/main/models.json")
    for mode in ("main", "replay"):
        out = root / mode
        if not (out / "completion.json").exists():
            if mode == "main":
                raise ValueError("BREADTH1D_RUN_INCOMPLETE")
            continue
        complete, models, budget = (
            read(out / "completion.json"),
            read(out / "models.json"),
            read(out / "budget-reserved.json"),
        )
        if (
            set(models) != set(QUARTERS)
            or complete["successful_fits"] != 4
            or budget["max_fits"] != 4
            or digest(models) != complete["models_hash"]
            or {p.name for p in out.glob("*-reserved.json")}
            != {"budget-reserved.json", *(q + "-reserved.json" for q in QUARTERS)}
        ):
            raise ValueError("BREADTH1D_BUDGET_OR_MODEL_CHANGED")
        for q in QUARTERS:
            model, control = models[q], old[q][REFERENCE]
            validate_model(model)
            rows = [universe[i] for i in schedule[q]["fit_indexes"]]
            available(rows, datetime.fromisoformat(control["train_as_of"]), fit=True)
            w = weights(rows)
            scaler = StandardScaler().fit(np.asarray([vector(r)[-1:] for r in rows]), sample_weight=w)
            reservation = read(out / (q + "-reserved.json"))
            if (
                model != read(out / (q + ".json"))
                or model["control_model_hash"] != digest(control)
                or model["control_fit_hash"] != digest(controls(rows))
                or model["fit_hash"] != digest(rows)
                or digest(rows) != spec["fit_hashes"][q]
                or reservation["fit_hash"] != digest(rows)
                or model["weight_hash"] != digest(w.tolist())
                or model["mean"] != [*control["mean"], float(scaler.mean_[0])]
                or model["scale"] != [*control["scale"], float(scaler.scale_[0])]
                or not budget["at"] <= reservation["at"] <= complete["finished_at"]
                or any(model[k] != control[k] for k in ("train_as_of", "fit_count", "distinct_dates", "classes"))
            ):
                raise ValueError("BREADTH1D_MODEL_MEMBERS_SCALE_OR_RESERVATION_CHANGED")
        expected = predictions(root, exam, models)
        if expected != read(out / "predictions.json") or digest(expected) != complete["predictions_hash"]:
            raise ValueError("BREADTH1D_SCORES_CHANGED")
    if (root / "comparison.json").exists():
        result = comparison(read(root / "main/predictions.json"), spec)
        if result != read(root / "comparison.json") or digest(result) != read(root / "comparison-receipt.json")["hash"]:
            raise ValueError("BREADTH1D_EVALUATION_CHANGED")
    return {"verified": True, "exam_count": len(exam), "source_rebuilt": rebuild_source, "new_fits": 0, "api_calls": 0}


def comparison(rows: list[dict], spec: dict) -> dict:
    """完整报告事前主配对与方向失衡；既有开发题只提供探索性证据。"""
    branches = list(rows[0]["directions"])
    metrics = sector.comparison.metrics
    overall = {b: metrics(rows, b) for b in branches}
    strata = {
        field: {
            str(v): {b: metrics([r for r in rows if r[field] == v], b) for b in branches}
            for v in sorted({r[field] for r in rows})
        }
        for field in ("quarter", "fund_code", "restored_question")
    }
    paired = sector.paired(rows, CANDIDATE, REFERENCE, spec)
    deltas = {
        k: {v: x[CANDIDATE]["weighted_accuracy"] - x[REFERENCE]["weighted_accuracy"] for v, x in strata[k].items()}
        for k in ("quarter", "fund_code")
    }
    conditions = {
        "overall_improved": paired["weighted_accuracy_difference"] > 0,
        "interval_positive": paired["block_bootstrap_interval_95"][0] > 0,
        "quarters_improved": sum(v > 0 for v in deltas["quarter"].values()) >= OBSERVATION_RULE["positive_quarters"],
        "funds_improved": sum(v > 0 for v in deltas["fund_code"].values()) >= OBSERVATION_RULE["positive_funds"],
        "fund_drop_controlled": min(deltas["fund_code"].values()) >= -OBSERVATION_RULE["maximum_fund_accuracy_drop"],
        "both_recalls_preserved": all(
            overall[CANDIDATE][k] >= overall[REFERENCE][k] for k in ("up_recall", "non_up_recall")
        ),
    }
    return {
        "kind": "DEVELOPMENT_BREADTH_COMPARISON_NOT_INDEPENDENT_TEST",
        "overall": overall,
        "strata": strata,
        "paired": paired,
        "conditions": conditions,
        "positive_quarters": sum(v > 0 for v in deltas["quarter"].values()),
        "positive_funds": sum(v > 0 for v in deltas["fund_code"].values()),
        "status": "DEVELOPMENT_OBSERVATION_CONDITIONS_MET" if all(conditions.values()) else "NO_STABLE_GAIN",
        "model_released": False,
        "runtime_model_changed": False,
        "limitations": [
            "2021—2024反复使用开发数据，区间仅作探索",
            "T日18:00行情与V2净值可用是假设，首版本未证明",
            "沪深有效成交普通A股比例不代表全市场全部股票",
            "2025/2026保护不变，真实效果仍待观察",
        ],
    }


def evaluate(root: Path) -> dict:
    verify(root)
    proof = read(root / "replay-proof.json")
    if proof != {
        "models_equal": True,
        "predictions_equal": True,
        "max_score_diff": 0.0,
        "successful_fits": 4,
        "prediction_count": read(root / "study.json")["exam_count"],
    }:
        raise ValueError("BREADTH1D_REPLAY_REQUIRED")
    activity.require_replay(root)
    result = comparison(read(root / "main/predictions.json"), read(root / "study.json"))
    write_new(root / "comparison.json", result)
    write_new(root / "comparison-receipt.json", {"at": datetime.now(ZONE).isoformat(), "hash": digest(result)})
    return result
