"""冻结两种历史长度与两种更新频率，逐月成熟训练、同题评价和唯一复跑。"""

import json
import shutil
import sys
from collections import Counter
from copy import deepcopy
from pathlib import Path
from time import perf_counter

from app.schemas.direction_training import DirectionAnswer, DirectionInput
from app.services.direction_diagnostics import SOURCES, source_seals
from app.services.direction_linear_analysis import accuracy_delta, evaluate
from app.services.direction_linear_models import predict_model as predict_original
from app.services.direction_linear_protocol import BASELINES
from app.services.direction_linear_protocol import ROLLING_BRANCHES as BRANCHES
from app.services.direction_market_runner import window_predictions
from app.services.direction_rolling_models import (
    MONTHS,
    RECIPES,
    VERSION,
    fit_end,
    model_key,
    predict_model,
    validate_job,
)
from app.services.direction_training_artifacts import (
    digest,
    file_hash,
    new_folder,
    now,
    read_json,
    read_jsonl,
    read_seal,
    seal,
    write_json,
    write_jsonl,
)
from app.services.direction_training_dataset import FUNDS
from app.services.direction_training_evaluation import complete_blocks, grouped_metrics, paired_interval
from app.services.direction_training_protocol import runtime, source_fingerprint

STAGES = ("frozen", "predicted", "scored", "complete")
NUMERIC_FIELDS = (
    "mean",
    "scale",
    "coefficients",
    "intercept",
    "C",
    "penalty",
    "solver_iterations",
    "train_hash",
    "train_counts",
    "fit_end",
)


def merge_rows(rows):
    """同一基金同一天只保留一个原样输入与答案；任何冲突都拒绝。"""
    unique = {}
    for row in rows:
        if set(row) != {"input", "answer"}:
            raise ValueError("ROLLING_POOL_FIELDS")
        item = DirectionInput.model_validate(row["input"])
        answer = DirectionAnswer.model_validate(row["answer"])
        if item.key != answer.key:
            raise ValueError("ROLLING_POOL_IDENTITY")
        canonical = {"input": item.model_dump(mode="json"), "answer": answer.model_dump(mode="json")}
        if item.key in unique and unique[item.key] != canonical:
            raise ValueError("ROLLING_POOL_CONFLICT")
        unique[item.key] = canonical
    return [unique[k] for k in sorted(unique)]


def build_pool(folder, windows):
    """仅从已经封存的开发FIT与EXAM建立资料池；尚未成熟的行留待以后月份使用。"""
    answers = {(r["window"], r["sample_key"]): r["answer"] for r in read_jsonl(folder / "source-linear-answers.jsonl")}
    rows = []
    for window in windows:
        source = read_json(folder / f"source-linear-prepared-{window['name']}.json")
        rows.extend(source["complete"]["fit"])
        for item in source["complete"]["exam"]["CLEAN"]:
            rows.append({"input": item, "answer": answers[(window["name"], f"{item['fund']}:{item['cutoff']}")]})
    return merge_rows(rows)


def select_fit(pool, month, recipe):
    """按答案可用日期截断，再按每基金日期取全部或最近252条；不看答案涨跌。"""
    if recipe not in RECIPES:
        raise ValueError("ROLLING_RECIPE")
    boundary = str(fit_end(month))
    eligible = [r for r in pool if r["answer"]["available_at"] <= boundary]
    by_fund = {
        f: sorted((r for r in eligible if r["input"]["fund"] == f), key=lambda r: r["input"]["cutoff"]) for f in FUNDS
    }
    if sum(map(len, by_fund.values())) != len(eligible):
        raise ValueError("ROLLING_FIT_SCOPE")
    dates = [[r["input"]["cutoff"] for r in rs] for rs in by_fund.values()]
    if len(dates[0]) < 252 or len(set(dates[0])) != len(dates[0]) or any(ds != dates[0] for ds in dates):
        raise ValueError("ROLLING_COMMON_FIT_DATES")
    return [row for f in FUNDS for row in (by_fund[f][-252:] if recipe == "252" else by_fund[f])]


def make_jobs(folder, windows, pool):
    exams = [
        i
        for w in windows
        for i in read_json(folder / f"source-linear-prepared-{w['name']}.json")["complete"]["exam"]["CLEAN"]
    ]
    jobs = {}
    for month in MONTHS:
        for recipe in RECIPES:
            job = {
                "version": VERSION,
                "month": month,
                "recipe": recipe,
                "fit": select_fit(pool, month, recipe),
                "exam": [i for i in exams if i["cutoff"].startswith(month)],
            }
            validate_job(job)
            jobs[f"{month}_{recipe}"] = job
    return jobs


def freeze(design: Path):
    source, manifests = source_seals("linear")
    old = read_json(source / "linear-plan.json")
    folder, files = new_folder(), {}
    names = ["linear-plan.json", "linear-answers.jsonl"] + [
        f"linear-{kind}-{w['name']}.json" for w in old["windows"] for kind in ("prepared", "models")
    ]
    expected = {n: h for m in manifests.values() for n, h in m["files"].items()}
    for name in names:
        target = folder / f"source-{name}"
        shutil.copyfile(source / name, target)
        files[target.name] = file_hash(target)
        if files[target.name] != expected[name]:
            raise ValueError("ROLLING_SOURCE_COPY_CHANGED")
    shutil.copyfile(design, folder / "rolling-design.md")
    files["rolling-design.md"] = file_hash(folder / "rolling-design.md")
    binding = {"run": SOURCES["linear"][0], "stages": {s: m["manifest_hash"] for s, m in manifests.items()}}
    files["rolling-source-binding.json"] = write_json(folder / "rolling-source-binding.json", binding)
    pool = build_pool(folder, old["windows"])
    files["rolling-pool.jsonl"] = write_jsonl(folder / "rolling-pool.jsonl", pool)
    jobs = make_jobs(folder, old["windows"], pool)
    for key, job in jobs.items():
        files[f"rolling-job-{key}.json"] = write_json(folder / f"rolling-job-{key}.json", job)
    plan = {
        k: deepcopy(old[k])
        for k in ("windows", "funds", "bootstrap", "minimum_coverage", "minimum_windows", "selection")
    }
    plan.update(
        {
            "version": VERSION,
            "source_code_hash": source_fingerprint(),
            "runtime": runtime(),
            "design_hash": files["rolling-design.md"],
            "branches": list(BRANCHES),
            "baselines": list(BASELINES),
            "months": list(MONTHS),
            "recipes": list(RECIPES),
            "minimum": {"FIT": 252, "EXAM": 40},
            "main_fits": 48,
            "replay_fits": 48,
            "old_models_reused": 8,
            "pool_rows": len(pool),
            "worker_seconds": 120,
            "memory_bytes": 4 * 1024**3,
            "total_fit_seconds": 1800,
            "training_asof": "PREVIOUS_MONTH_LAST_NATURAL_DAY",
            "label_rule": "AVAILABLE_AT_LE_TRAINING_ASOF",
            "old_unused_calibration_gap_applies": False,
            "independent_test": False,
            "model_released": False,
            "network_calls": 0,
            "database_writes": 0,
        }
    )
    files["rolling-plan.json"] = write_json(folder / "rolling-plan.json", plan)
    frozen = seal(folder, "rolling-frozen.json", files, status="FROZEN_BEFORE_FIT")
    return {
        "run": folder.name.removeprefix("direction-training-"),
        "folder": str(folder),
        "code_hash": plan["source_code_hash"],
        **frozen,
    }


def load_plan(folder):
    frozen = read_seal(folder, "rolling-frozen.json")
    plan = read_json(folder / "rolling-plan.json")
    if plan["version"] != VERSION or plan["source_code_hash"] != source_fingerprint() or plan["runtime"] != runtime():
        raise ValueError("ROLLING_CODE_OR_RUNTIME_CHANGED")
    if (
        plan["branches"] != list(BRANCHES)
        or plan["months"] != list(MONTHS)
        or plan["recipes"] != list(RECIPES)
        or plan["model_released"]
        or plan["independent_test"]
    ):
        raise ValueError("ROLLING_SCOPE_CHANGED")
    binding = read_json(folder / "rolling-source-binding.json")
    if binding["run"] != SOURCES["linear"][0] or binding["stages"]["complete"] != SOURCES["linear"][1]:
        raise ValueError("ROLLING_SOURCE_CHANGED")
    return plan, frozen


def original_parity(folder, plan, model):
    """新旧FIT完全相同时，参数及训练行摘要必须逐值一致。"""
    if model["recipe"] != "ALL":
        return []
    matched = []
    for w in plan["windows"]:
        if w["fit_end"] == model["fit_end"]:
            old = read_json(folder / f"source-linear-models-{w['name']}.json")["REFERENCE"]["models"]["POOLED"]
            for field in NUMERIC_FIELDS:
                if old[field] != model[field]:
                    raise ValueError(f"ROLLING_ORIGINAL_NUMERIC_CHANGED:{field}")
            matched.append(w["name"])
    return matched


def rebuild_predictions(folder, plan, models):
    """所有分支使用原考试输入，按日期路由模型；原参照和五个简单对照保持原样。"""
    predictions = {}
    for w in plan["windows"]:
        name = w["name"]
        source = read_json(folder / f"source-linear-prepared-{name}.json")
        exam = source["complete"]["exam"]["CLEAN"]
        typed = [DirectionInput.model_validate(i) for i in exam]
        old = read_json(folder / f"source-linear-models-{name}.json")["REFERENCE"]
        if predict_original(old["models"]["POOLED"], typed) != old["scores"]:
            raise ValueError("ROLLING_OLD_SCORE_CHANGED")
        outputs = {"REFERENCE": old}
        for branch in BRANCHES[1:]:
            scores = []
            for item in typed:
                model = models[model_key(branch, item.cutoff)]
                if model["fit_end"] >= str(item.cutoff):
                    raise ValueError("ROLLING_ROUTE_TIME")
                scores.extend(predict_model(model, [item]))
            outputs[branch] = {"status": "PREDICTED", "scores": scores}
        bundle = {
            "window": w,
            "planned": source["planned"],
            "jobs": {b: {"fit": source["complete"]["fit"], "exam": exam} for b in BRANCHES},
        }
        predictions[name] = window_predictions(bundle, outputs, branches=BRANCHES)
    return predictions


def predict(folder):
    from app.services.direction_training_process import run_process

    plan, frozen = load_plan(folder)
    write_json(folder / "rolling-started.json", {"started_at": now(), "maximum_fits": 48})
    files, execution, models, parity = {}, {}, {}, {}
    started = perf_counter()
    for month in MONTHS:
        for recipe in RECIPES:
            key = f"{month}_{recipe}"
            if perf_counter() - started > plan["total_fit_seconds"]:
                raise ValueError("ROLLING_TOTAL_TIMEOUT")
            result = run_process(
                read_json(folder / f"rolling-job-{key}.json"),
                seconds=plan["worker_seconds"],
                memory_bytes=plan["memory_bytes"],
                command=[sys.executable, "-m", "scripts.direction_rolling_worker"],
            )
            if result["status"] != "PREDICTED":
                write_json(folder / f"rolling-failed-{key}.json", result)
                raise ValueError("ROLLING_WORKER_FAILED")
            files[f"rolling-model-{key}.json"] = write_json(folder / f"rolling-model-{key}.json", result)
            models[key] = result["model"]
            parity[key] = original_parity(folder, plan, result["model"])
            execution[key] = {k: result[k] for k in ("status", "model_fit_count", "elapsed_seconds")}
            print(
                json.dumps(
                    {
                        "model": key,
                        "status": result["status"],
                        "fit_per_fund": result["model"]["train_counts"][FUNDS[0]],
                    }
                ),
                flush=True,
            )
    for window, rows in rebuild_predictions(folder, plan, models).items():
        files[f"rolling-predictions-{window}.jsonl"] = write_jsonl(folder / f"rolling-predictions-{window}.jsonl", rows)
    files["rolling-parity.json"] = write_json(folder / "rolling-parity.json", parity)
    files["rolling-execution.json"] = write_json(folder / "rolling-execution.json", execution)
    return seal(folder, "rolling-predicted.json", files, frozen_hash=frozen["manifest_hash"], fitted=48, reused=8)


def load_models(folder):
    return {f"{m}_{r}": read_json(folder / f"rolling-model-{m}_{r}.json")["model"] for m in MONTHS for r in RECIPES}


def all_predictions(folder, plan):
    return [r for w in plan["windows"] for r in read_jsonl(folder / f"rolling-predictions-{w['name']}.jsonl")]


def diagnostics(folder, plan, predictions, answers):
    """拆开历史长度、更新频率及最新FIT类别频率；均为已观察开发资料的描述。"""
    labels = {(r["window"], r["sample_key"]): r["answer"]["y"] for r in answers}
    models = load_models(folder)
    rows = {b: [] for b in BRANCHES}
    for row in predictions:
        if row["branch"] in rows:
            y = labels[(row["window"], row["sample_key"])]
            if row["score"] is None:
                raise ValueError("ROLLING_DIAGNOSTIC_MISSING_SCORE")
            rows[row["branch"]].append({**row, "y": y, "correct": int(row["predicted_up"] == y)})
    for branch in BRANCHES[1:]:
        rates = []
        for row in rows[branch]:
            score = models[model_key(branch, row["cutoff"])]["train_up_rates"][row["fund"]]
            rates.append(
                {
                    **row,
                    "branch": branch + "_FIT_FREQUENCY",
                    "score": score,
                    "predicted_up": int(score > 0.5),
                    "correct": int(int(score > 0.5) == row["y"]),
                }
            )
        rows[branch + "_FIT_FREQUENCY"] = rates
    maps = {b: {(r["window"], r["sample_key"]): r for r in rs} for b, rs in rows.items()}
    keys = set(maps["REFERENCE"])
    if len(keys) != 1389 or any(set(rs) != keys for rs in maps.values()):
        raise ValueError("ROLLING_COMMON_EXAM_KEYS")
    planned = {
        w["name"]: read_json(folder / f"source-linear-prepared-{w['name']}.json")["planned"] for w in plan["windows"]
    }
    blocks, excluded = complete_blocks(plan, keys, planned_by_window=planned)
    summary = {b: grouped_metrics(rs, FUNDS) for b, rs in rows.items()}
    pairs = (
        ("QUARTER_252", "QUARTER_ALL"),
        ("MONTH_252", "MONTH_ALL"),
        ("MONTH_ALL", "QUARTER_ALL"),
        ("MONTH_252", "QUARTER_252"),
    )
    comparisons = {}
    for left, right in (*pairs, *((b, b + "_FIT_FREQUENCY") for b in BRANCHES[1:])):
        comparisons[f"{left}_vs_{right}"] = {
            "accuracy_delta": accuracy_delta(summary[left], summary[right]),
            "time_blocks": paired_interval(plan, blocks, maps[left], maps[right]),
            "descriptive_only": True,
        }
    training = {}
    for key, model in models.items():
        job = read_json(folder / f"rolling-job-{key}.json")
        fit, _ = validate_job(job)
        scores = predict_model(model, [i for i, _ in fit])
        training[key] = {
            "fit_end": model["fit_end"],
            "fit_first": min(str(i.cutoff) for i, _ in fit),
            "fit_last": max(str(i.cutoff) for i, _ in fit),
            "latest_answer_available_at": max(str(a.available_at) for _, a in fit),
            "train_counts": model["train_counts"],
            "train_up_rates": model["train_up_rates"],
            "metrics": grouped_metrics(
                [{"fund": i.fund, "y": a.y, "score": s} for (i, a), s in zip(fit, scores, strict=True)], FUNDS
            ),
        }
    return {
        "summary": summary,
        "comparisons": comparisons,
        "training": training,
        "per_year": {
            year: {b: grouped_metrics([r for r in rs if r["cutoff"].startswith(year)], FUNDS) for b, rs in rows.items()}
            for year in ("2023", "2024")
        },
        "per_window": {
            w["name"]: {
                b: grouped_metrics([r for r in rs if r["window"] == w["name"]], FUNDS) for b, rs in rows.items()
            }
            for w in plan["windows"]
        },
        "time_blocks": {"count": len(blocks), "exclusions": excluded},
        "same_question_count": len(keys),
        "independent_test": False,
        "model_released": False,
    }


def analyze(folder, plan):
    predictions = all_predictions(folder, plan)
    answers = read_jsonl(folder / "source-linear-answers.jsonl")
    return evaluate(plan, predictions, answers), diagnostics(folder, plan, predictions, answers)


def score(folder):
    plan, _ = load_plan(folder)
    predicted = read_seal(folder, "rolling-predicted.json")
    metrics, details = analyze(folder, plan)
    files = {
        "rolling-metrics.json": write_json(folder / "rolling-metrics.json", metrics),
        "rolling-diagnostics.json": write_json(folder / "rolling-diagnostics.json", details),
    }
    return seal(folder, "rolling-scored.json", files, predicted_hash=predicted["manifest_hash"])


def decision(metrics):
    return {
        "status": "QUALIFIED_DEVELOPMENT_CANDIDATE" if metrics["selected_candidate"] else "NO_STABLE_GAIN",
        "selected_candidate": metrics["selected_candidate"],
        "independent_test": False,
        "model_released": False,
        "automatic_followup_training": False,
    }


def candidate(folder, plan, metrics):
    selected = metrics["selected_candidate"]
    models = load_models(folder)
    needed = {model_key(selected, i["cutoff"]) for i in all_predictions(folder, plan) if i["branch"] == selected}
    return {
        "branch": selected,
        "plan_hash": file_hash(folder / "rolling-plan.json"),
        "model_hashes": {k: models[k]["hash"] for k in sorted(needed)},
        "model_released": False,
    }


def finalize(folder):
    plan, _ = load_plan(folder)
    scored = read_seal(folder, "rolling-scored.json")
    metrics = read_json(folder / "rolling-metrics.json")
    files = {"rolling-decision.json": write_json(folder / "rolling-decision.json", decision(metrics))}
    if metrics["selected_candidate"]:
        files["rolling-candidate.json"] = write_json(
            folder / "rolling-candidate.json", candidate(folder, plan, metrics)
        )
    return seal(folder, "rolling-complete.json", files, scored_hash=scored["manifest_hash"])


def verify(folder):
    """不重新拟合：来源重建、逐模型成熟性、路由分数与全部评分决定都独立复核。"""
    plan, _ = load_plan(folder)
    manifests = {s: read_seal(folder, f"rolling-{s}.json") for s in STAGES}
    for left, right in zip(STAGES[:-1], STAGES[1:], strict=True):
        if manifests[right][f"{left}_hash"] != manifests[left]["manifest_hash"]:
            raise ValueError("ROLLING_STAGE_CHAIN")
    pool = build_pool(folder, plan["windows"])
    if pool != read_jsonl(folder / "rolling-pool.jsonl") or len(pool) != plan["pool_rows"]:
        raise ValueError("ROLLING_POOL_CHANGED")
    jobs, models, parity = make_jobs(folder, plan["windows"], pool), load_models(folder), {}
    for key, job in jobs.items():
        if job != read_json(folder / f"rolling-job-{key}.json"):
            raise ValueError("ROLLING_JOB_CHANGED")
        fit, exam = validate_job(job)
        model = models[key]
        rates = {f: sum(a.y for i, a in fit if i.fund == f) / sum(i.fund == f for i, _ in fit) for f in FUNDS}
        if (
            model["train_hash"] != digest(job["fit"])
            or model["month"] != job["month"]
            or model["recipe"] != job["recipe"]
            or model["train_counts"] != dict(Counter(i.fund for i, _ in fit))
            or model["train_up_rates"] != rates
        ):
            raise ValueError("ROLLING_MODEL_TRAIN_BINDING")
        output = read_json(folder / f"rolling-model-{key}.json")
        if (
            output["status"] != "PREDICTED"
            or output["model_fit_count"] != 1
            or predict_model(model, exam) != output["scores"]
        ):
            raise ValueError("ROLLING_MODEL_SCORE_CHANGED")
        parity[key] = original_parity(folder, plan, model)
    if parity != read_json(folder / "rolling-parity.json"):
        raise ValueError("ROLLING_PARITY_CHANGED")
    for name, predictions in rebuild_predictions(folder, plan, models).items():
        if predictions != read_jsonl(folder / f"rolling-predictions-{name}.jsonl"):
            raise ValueError("ROLLING_PREDICTIONS_CHANGED")
    execution = read_json(folder / "rolling-execution.json")
    if (
        set(execution) != set(jobs)
        or any(r["status"] != "PREDICTED" or r["model_fit_count"] != 1 for r in execution.values())
        or manifests["predicted"]["fitted"] != 48
        or manifests["predicted"]["reused"] != 8
    ):
        raise ValueError("ROLLING_FIT_ACCOUNTING")
    metrics, details = analyze(folder, plan)
    if metrics != read_json(folder / "rolling-metrics.json") or details != read_json(
        folder / "rolling-diagnostics.json"
    ):
        raise ValueError("ROLLING_RESULTS_CHANGED")
    if decision(metrics) != read_json(folder / "rolling-decision.json"):
        raise ValueError("ROLLING_DECISION_CHANGED")
    if metrics["selected_candidate"]:
        if candidate(folder, plan, metrics) != read_json(folder / "rolling-candidate.json"):
            raise ValueError("ROLLING_CANDIDATE_CHANGED")
    elif (folder / "rolling-candidate.json").exists():
        raise ValueError("ROLLING_UNQUALIFIED_CANDIDATE")
    return manifests["complete"]


def replay(folder):
    if (folder / "rolling-replay-origin.json").exists():
        raise ValueError("ROLLING_REPLAY_OF_REPLAY")
    if (folder / "rolling-replay-started.json").exists():
        raise ValueError("ROLLING_REPLAY_ALREADY_STARTED")
    verify(folder)
    target = new_folder()
    origin = {"original": folder.name, "replay": target.name}
    write_json(folder / "rolling-replay-started.json", origin)
    frozen = read_seal(folder, "rolling-frozen.json")
    for name in ("rolling-frozen.json", *frozen["files"]):
        shutil.copyfile(folder / name, target / name)
    write_json(target / "rolling-replay-origin.json", origin)
    predict(target)
    score(target)
    finalize(target)
    verified = verify(target)
    plan = read_json(folder / "rolling-plan.json")
    if load_models(folder) != load_models(target) or all_predictions(folder, plan) != all_predictions(target, plan):
        raise ValueError("ROLLING_REPLAY_CHANGED")
    for name in ("rolling-metrics.json", "rolling-diagnostics.json", "rolling-decision.json"):
        if read_json(folder / name) != read_json(target / name):
            raise ValueError("ROLLING_REPLAY_RESULTS_CHANGED")
    receipt = {
        "run": target.name.removeprefix("direction-training-"),
        "scores": sum(r["branch"] in BRANCHES for r in all_predictions(folder, plan)),
        "maximum_difference": 0,
        "models_identical": True,
        "results_identical": True,
        "new_fits": 48,
        "reused_models": 8,
        "complete_hash": verified["manifest_hash"],
    }
    seal(
        target,
        "rolling-replay.json",
        {"rolling-replay-receipt.json": write_json(target / "rolling-replay-receipt.json", receipt)},
    )
    return receipt
