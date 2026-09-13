"""隔夜SPX单特征假设性探索：复用12维树，独立保留U08可用假设，不替代真实时间审计。"""

from datetime import date, datetime, time, timedelta
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from app.services import direction_1d_full_threshold as full
from app.services import direction_1d_overnight_audit as audit
from app.services.direction_1d_protocol import ZONE, digest
from app.services.direction_1d_training import read, weights, write_new
from app.services.direction_training_artifacts import file_hash

activity, tree, sector = full.activity, full.tree, full.sector
REFERENCE, CANDIDATE = "ACTIVITY12", "SPX_OVERNIGHT13"
QUARTERS = full.QUARTERS
FEATURES = [*activity.FEATURES, "spx_t15_to_u08_cumulative_return_assumed"]
FEATURE_RECIPE = {
    **audit.RECIPE,
    "availability": "U08_ASSUMED_EXPLORATION_V1",
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
    """旧研究源码不变；本轮源码与原日历分别绑定，不能把假设标签改为真实可用。"""
    result = full.fingerprint()
    project = Path(__file__).resolve().parents[2]
    for name in (
        "app/services/direction_1d_overnight_study.py",
        "scripts/direction_1d_overnight_study.py",
        "app/services/direction_1d_overnight_audit.py",
        "scripts/direction_1d_overnight_audit.py",
        "app/data/calendars/nyse_cash_2021_2024_v1.json",
    ):
        result[name] = file_hash(project / name)
    return result


def validate_source(source: dict) -> None:
    """原来源必须启用；新增SPX能力另由只读成功验权回执限定，不改原13项业务登记。"""
    if (
        source.get("source_code") != "TUSHARE_PRO_FUND"
        or source.get("enabled") is not True
        or not source.get("authorization_verified_at")
        or source.get("retention_days", 0) <= 0
    ):
        raise ValueError("OVERNIGHT1D_SOURCE_UNAVAILABLE")


def source_files(folder: Path) -> dict[str, str]:
    """复用四次固定年度请求，逐文件核对原采集回执，不追加网络请求。"""
    receipt = read(folder / "history-receipt.json")
    if receipt["status"] != "RECEIVED" or receipt["api_calls"] != 4 or len(receipt["requests"]) != 4:
        raise ValueError("OVERNIGHT1D_SOURCE_INCOMPLETE")
    proof = read(folder / "probe-result.json")
    if (
        proof["status"] != "PERMISSION_AVAILABLE_NOT_USAGE_OR_TIMING_PROOF"
        or proof["reservation"]["api_name"] != "index_global"
        or proof["reservation"]["params"]["ts_code"] != "SPX"
    ):
        raise ValueError("OVERNIGHT1D_CAPABILITY_INVALID")
    names = {"history-receipt.json", "history-reserved.json", "source-audit.json", "probe-result.json"}
    for year, item in zip(audit.YEARS, receipt["requests"], strict=True):
        name = f"history-{year}.json"
        receipt_name = f"history-{year}-receipt.json"
        if (
            item != read(folder / receipt_name)
            or item["year"] != year
            or file_hash(folder / name) != item["data_file_sha256"]
        ):
            raise ValueError("OVERNIGHT1D_SOURCE_HASH_INVALID")
        names.update((name, receipt_name))
    return {n: file_hash(folder / n) for n in sorted(names)}


def read_history(folder: Path) -> dict:
    source_files(folder)
    history = audit.validate_history([read(folder / f"history-{y}.json") for y in audit.YEARS])
    if audit.coverage(history, audit.sessions())["status"] != "COMPLETE":
        raise ValueError("OVERNIGHT1D_SOURCE_COVERAGE_INVALID")
    return history


def attach(rows: list[dict], history: dict) -> list[dict]:
    """追加明确标记为假设的第13维；原输入、标签、首版本未知信息不改写。"""
    identities = [full.identity(r) for r in rows]
    if len(identities) != len(set(identities)) or any("overnight_input" in r for r in rows):
        raise ValueError("OVERNIGHT1D_DUPLICATE")
    pairs = sorted({(r["t"], r["u"]) for r in rows})
    aligned = {(r["t"], r["u"]): r for r in audit.align_dates(pairs, history, audit.sessions())}
    result = []
    for row in rows:
        item = aligned[(row["t"], row["u"])]
        dates = [item["baseline_us_date"], *item["us_sessions"]]
        result.append(
            {
                **row,
                "overnight_input": {
                    "x": [item["diagnostic_return"]],
                    "baseline_us_date": item["baseline_us_date"],
                    "us_sessions": item["us_sessions"],
                    "source_hash": digest({d: history[d] for d in dates}),
                    "available_at_assumed": item["cutoff"],
                    "available_at_verified": None,
                    "availability_version": FEATURE_RECIPE["availability"],
                    "historical_first_version_verified": False,
                },
            }
        )
    return result


def controls(rows: list[dict]) -> list[dict]:
    return [{k: v for k, v in r.items() if k != "overnight_input"} for r in rows]


def vector(row: dict) -> list[float]:
    extra = row["overnight_input"]["x"]
    if len(extra) != 1 or not np.isfinite(extra).all() or extra[0] <= -1:
        raise ValueError("OVERNIGHT1D_FEATURE_INVALID")
    result = [*activity.vector(row), *extra]
    if len(result) != 13 or not np.isfinite(result).all():
        raise ValueError("OVERNIGHT1D_VECTOR_INVALID")
    return result


def available(rows: list[dict], cutoff: datetime, *, fit: bool) -> None:
    """仅在显式U08假设下检查时间隔离；不接受冒充已验证或平移到更早时间的输入。"""
    full.check_available(rows, cutoff, fit=fit)
    for row in rows:
        item = row["overnight_input"]
        expected = datetime.combine(date.fromisoformat(row["u"]), time(8), ZONE)
        if (
            item["availability_version"] != FEATURE_RECIPE["availability"]
            or item["available_at_verified"] is not None
            or item["historical_first_version_verified"] is not False
            or datetime.fromisoformat(item["available_at_assumed"]) != expected
        ):
            raise ValueError("OVERNIGHT1D_AVAILABILITY_HYPOTHESIS_CHANGED")
        if expected > cutoff:
            raise ValueError("OVERNIGHT1D_INPUT_TOO_LATE")


def freeze(base: Path, data: Path, plan: Path, scope_path: Path, source_path: Path, root: Path) -> dict:
    """验明旧模型及原始数据后固定4次主拟合和4次复现；此阶段不训练、不联网。"""
    full.verify(base)
    activity.require_replay(base)
    scope, source = read(scope_path), read(source_path)
    full.audit.validate_scope(scope)
    validate_source(source)
    old = read(base / "study.json")
    if old["owner_scope_hash"] != scope["scope_hash"] or not set(old["fund_codes"]) <= set(scope["codes"]):
        raise ValueError("OVERNIGHT1D_OWNER_SCOPE_CHANGED")
    data_names = source_files(data)
    source_snapshot = next(
        r for r in read(data / "source-audit.json")["sources"] if r["source_code"] == "TUSHARE_PRO_FUND"
    )
    if source_snapshot["source_id"] != source["source_id"]:
        raise ValueError("OVERNIGHT1D_SOURCE_ID_CHANGED")
    daily = read_history(data)
    acquired = datetime.fromisoformat(read(data / "history-reserved.json")["at"])
    expires = min(
        acquired + timedelta(days=source["retention_days"]),
        datetime.fromisoformat(read(data / "history-reserved.json")["retention_until"]),
        datetime.fromisoformat(old["source"]["source_expires_at"]),
    )
    if datetime.now(ZONE) >= expires:
        raise ValueError("OVERNIGHT1D_SOURCE_EXPIRED")
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
        "kind": "HISTORICAL_ONE_DAY_OVERNIGHT_ASSUMED_ONLY",
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
        "availability_hypothesis": FEATURE_RECIPE["availability"],
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
        "bootstrap_seed": 20260913,
        "bootstrap_repetitions": 2000,
        "bootstrap_block_days": 5,
    }
    spec["cohort_id"] = "D1-OVERNIGHT-" + digest({"files": files, "recipe": FEATURE_RECIPE})[:24]
    write_new(root / "study.json", spec)
    write_new(root / "study-receipt.json", {"at": spec["created_at"], "hash": digest(spec)})
    return {
        k: spec[k]
        for k in ("cohort_id", "created_at", "exam_count", "restored_count", "fit_counts", "source_daily_count")
    }


def verify_inputs(root: Path) -> dict:
    spec = read(root / "study.json")
    if digest(spec) != read(root / "study-receipt.json")["hash"] or spec["fingerprint"] != fingerprint():
        raise ValueError("OVERNIGHT1D_CODE_OR_SPEC_CHANGED")
    expected = {
        "kind": "HISTORICAL_ONE_DAY_OVERNIGHT_ASSUMED_ONLY",
        "historical_first_versions_verified": False,
        "availability_hypothesis": FEATURE_RECIPE["availability"],
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
        raise ValueError("OVERNIGHT1D_RECIPE_OR_BUDGET_CHANGED")
    if datetime.now(ZONE) >= datetime.fromisoformat(spec["source_expires_at"]):
        raise ValueError("OVERNIGHT1D_SOURCE_EXPIRED")
    for name, expected_hash in spec["input_files"].items():
        path = root / name
        if not path.resolve().is_relative_to(root.resolve()) or path.is_symlink() or file_hash(path) != expected_hash:
            raise ValueError("OVERNIGHT1D_INPUT_CHANGED")
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
        raise ValueError("OVERNIGHT1D_MODEL_PROTOCOL_OR_SCALE")
    for nodes in model["trees"]:
        if not 1 <= len(nodes) <= 2 * tree.TREE_RECIPE["max_leaf_nodes"] - 1:
            raise ValueError("OVERNIGHT1D_MODEL_NODES")
        seen, stack, leaves = set(), [(0, 0)], 0
        while stack:
            i, depth = stack.pop()
            if i in seen or not 0 <= i < len(nodes) or depth > tree.TREE_RECIPE["max_depth"]:
                raise ValueError("OVERNIGHT1D_MODEL_TOPOLOGY")
            seen.add(i)
            n = nodes[i]
            if n["depth"] != depth or not np.isfinite([n["value"], n["threshold"]]).all() or n["count"] <= 0:
                raise ValueError("OVERNIGHT1D_MODEL_NODE")
            if n["leaf"]:
                leaves += 1
            else:
                if not 0 <= n["feature"] < 13 or n["left"] <= i or n["right"] <= i:
                    raise ValueError("OVERNIGHT1D_MODEL_BRANCH")
                stack.extend(((n["left"], depth + 1), (n["right"], depth + 1)))
        if len(seen) != len(nodes) or leaves > tree.TREE_RECIPE["max_leaf_nodes"]:
            raise ValueError("OVERNIGHT1D_MODEL_UNREACHABLE")


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
        raise ValueError("OVERNIGHT1D_FIT_MEMBERS_OR_WEIGHTS")
    cutoff = datetime.fromisoformat(control["train_as_of"])
    available(rows, cutoff, fit=True)
    if {r["u"] for r in rows} & {r["u"] for r in exam}:
        raise ValueError("OVERNIGHT1D_FIT_EXAM_OVERLAP")
    for r in exam:
        deadline = datetime.combine(date.fromisoformat(r["u"]), time(8), ZONE)
        available([r], deadline, fit=False)
        if cutoff >= deadline:
            raise ValueError("OVERNIGHT1D_MODEL_TOO_LATE")
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
            raise ValueError("OVERNIGHT1D_RESTORE_MISMATCH")
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
                raise ValueError("OVERNIGHT1D_ANSWER_OR_QUESTION_CHANGED")
            result.append(
                {
                    **row,
                    "scores": {
                        REFERENCE: old["scores"][REFERENCE],
                        CANDIDATE: float(score),
                    },
                    "directions": {REFERENCE: old["directions"][REFERENCE], CANDIDATE: int(score > 0.5)},
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
                raise ValueError("OVERNIGHT1D_REPLAY_MISMATCH")
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
    """只读重建SPX累计变化、假设时间、尺度及分数；不联网、不拟合。"""
    spec = verify_inputs(root)
    if rebuild_source:
        full.verify(root / "control")
    daily = read_history(root / "data")
    universe, exam = (
        attach(read(root / "control/universe.json"), daily),
        attach(read(root / "control/exam.json"), daily),
    )
    schedule = {q: read(root / "control/schedule.json")[q] for q in QUARTERS}
    if any(
        value != read(root / name)
        for name, value in (("universe.json", universe), ("exam.json", exam), ("schedule.json", schedule))
    ):
        raise ValueError("OVERNIGHT1D_RECONSTRUCTION_CHANGED")
    if len(exam) != spec["exam_count"] or sorted({r["fund_code"] for r in exam}) != spec["fund_codes"]:
        raise ValueError("OVERNIGHT1D_SCOPE_CHANGED")
    old = read(root / "control/main/models.json")
    for mode in ("main", "replay"):
        out = root / mode
        if not (out / "completion.json").exists():
            if mode == "main":
                raise ValueError("OVERNIGHT1D_RUN_INCOMPLETE")
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
            raise ValueError("OVERNIGHT1D_BUDGET_OR_MODEL_CHANGED")
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
                raise ValueError("OVERNIGHT1D_MODEL_MEMBERS_SCALE_OR_RESERVATION_CHANGED")
        expected = predictions(root, exam, models)
        if expected != read(out / "predictions.json") or digest(expected) != complete["predictions_hash"]:
            raise ValueError("OVERNIGHT1D_SCORES_CHANGED")
    if (root / "comparison.json").exists():
        result = comparison(read(root / "main/predictions.json"), spec)
        if result != read(root / "comparison.json") or digest(result) != read(root / "comparison-receipt.json")["hash"]:
            raise ValueError("OVERNIGHT1D_EVALUATION_CHANGED")
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
        "kind": "DEVELOPMENT_OVERNIGHT_ASSUMED_COMPARISON_NOT_INDEPENDENT_TEST",
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
            "原T日18:00行情与V2净值可用假设继续保留",
            "SPX历史08:00可用及首版本均是假设，不能替代实际到达记录",
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
        raise ValueError("OVERNIGHT1D_REPLAY_REQUIRED")
    activity.require_replay(root)
    result = comparison(read(root / "main/predictions.json"), read(root / "study.json"))
    write_new(root / "comparison.json", result)
    write_new(root / "comparison-receipt.json", {"at": datetime.now(ZONE).isoformat(), "hash": digest(result)})
    return result
