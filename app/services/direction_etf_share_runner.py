"""份额数据通过后固定A/B同题实验；不联网训练、不写正式表、不按结果追加方案。"""

import json
import shutil
import sys
from collections import Counter
from copy import deepcopy
from pathlib import Path
from time import perf_counter
from uuid import UUID

from app.services.direction_etf_share_data import ETF_CODES, add_feature, feature, load_history, quarter_month
from app.services.direction_etf_share_models import BRANCHES, VERSION, predict_model, validate_job
from app.services.direction_linear_analysis import evaluate
from app.services.direction_linear_protocol import BASELINES, study_windows
from app.services.direction_market_runner import window_predictions
from app.services.direction_rolling_runner import select_fit
from app.services.direction_training_artifacts import (
    digest,
    file_hash,
    new_folder,
    now,
    read_json,
    read_jsonl,
    read_seal,
    run_folder,
    seal,
    write_json,
    write_jsonl,
)
from app.services.direction_training_dataset import FUNDS
from app.services.direction_training_evaluation import grouped_metrics
from app.services.direction_training_protocol import runtime, source_fingerprint

SOURCE_RUN = "0352b89a-4b5a-453a-92c8-0a2da0103360"
SOURCE_HASH = "4f5554b4bd4b95f2ecad39d8cb3a2d61891658773f5865bedbb168baefd1d742"
STAGES = ("frozen", "predicted", "scored", "complete")


def build_bundles(folder):
    """先独立计算份额可用性，再同步截取三基金和两组；不接触评价标签来选样。"""
    history, audit = load_history(folder)
    pool = read_jsonl(folder / "rolling-pool.jsonl")
    cache = {
        f"{r['input']['fund']}:{r['input']['cutoff']}": feature(history, r["input"]["fund"], r["input"]["cutoff"])
        for r in pool
    }
    bundles = {}
    for window in study_windows(VERSION):
        full = select_fit(pool, quarter_month(window), "ALL")
        dates = sorted({r["input"]["cutoff"] for r in full})
        available = {d for d in dates if all(cache[f"{f}:{d}"][0] is not None for f in FUNDS)}
        fit = [r for r in full if r["input"]["cutoff"] in available]
        old = read_json(folder / f"source-linear-prepared-{window['name']}.json")
        exam = [
            r
            for r in old["complete"]["exam"]["CLEAN"]
            if all(cache[f"{f}:{r['cutoff']}"][0] is not None for f in FUNDS)
        ]
        jobs = {}
        for branch in BRANCHES:

            def enriched(item, branch=branch):
                return (
                    deepcopy(item)
                    if branch == "REFERENCE"
                    else add_feature(item, cache[f"{item['fund']}:{item['cutoff']}"][0])
                )

            jobs[branch] = {
                "version": VERSION,
                "branch": branch,
                "window": window,
                "fit": [{"input": enriched(r["input"]), "answer": r["answer"]} for r in fit],
                "exam": [enriched(i) for i in exam],
            }
            validate_job(jobs[branch])
        excluded = Counter(
            reason for d in dates if d not in available for f in FUNDS if (reason := cache[f"{f}:{d}"][1])
        )
        coverage = {
            "original_fit_per_fund": len(dates),
            "matched_fit_per_fund": len(available),
            "excluded_fit_dates": sorted(set(dates) - available),
            "exclusion_causes": dict(excluded),
            "planned_exam_per_fund": len(old["planned"]),
            "matched_exam_per_fund": len(exam) // len(FUNDS),
            "first_fit": min(available),
            "last_fit": max(available),
        }
        if len(exam) / len(FUNDS) < 40 or len(exam) / (len(FUNDS) * len(old["planned"])) < 0.90:
            raise ValueError("ETF_SHARE_EXAM_COVERAGE")
        bundles[window["name"]] = {"window": window, "planned": old["planned"], "jobs": jobs, "coverage": coverage}
    return bundles, audit


def freeze(design: Path, data: Path):
    """仅复制已核验的来源和旧封存资料，先锁定作业与代码再允许worker拟合。"""
    source = run_folder(UUID(SOURCE_RUN))
    source_seals = {s: read_seal(source, f"rolling-{s}.json") for s in STAGES}
    if source_seals["complete"]["manifest_hash"] != SOURCE_HASH:
        raise ValueError("ETF_SHARE_ORIGINAL_SOURCE_CHANGED")
    for a, b in zip(STAGES[:-1], STAGES[1:], strict=True):
        if source_seals[b][f"{a}_hash"] != source_seals[a]["manifest_hash"]:
            raise ValueError("ETF_SHARE_ORIGINAL_CHAIN")
    acquired = read_seal(data, "etf-share-acquired.json")
    probed = read_seal(data, "etf-share-probed.json")
    probe_frozen = read_seal(data, "etf-share-probe-frozen.json")
    if acquired["probe_hash"] != probed["manifest_hash"] or probed["frozen_hash"] != probe_frozen["manifest_hash"]:
        raise ValueError("ETF_SHARE_ACQUISITION_CHAIN")
    mapping = read_json(data / "etf-share-probe-plan.json")
    if {f: v["etf"] for f, v in mapping["funds"].items()} != ETF_CODES:
        raise ValueError("ETF_SHARE_MAPPING_CHANGED")
    folder, files = new_folder(), {}
    old = read_json(source / "rolling-plan.json")
    source_names = [
        "rolling-pool.jsonl",
        "source-linear-answers.jsonl",
        "rolling-plan.json",
        "rolling-metrics.json",
    ] + [f"source-linear-prepared-{w['name']}.json" for w in old["windows"]]
    expected = {n: h for s in source_seals.values() for n, h in s["files"].items()}
    for src, names, hashes in (
        (source, source_names, expected),
        (data, list(acquired["files"]), acquired["files"]),
        (data, list(probed["files"]), probed["files"]),
        (data, list(probe_frozen["files"]), probe_frozen["files"]),
    ):
        for name in names:
            if name in files:
                continue
            shutil.copyfile(src / name, folder / name)
            files[name] = file_hash(folder / name)
            if files[name] != hashes[name]:
                raise ValueError("ETF_SHARE_SOURCE_COPY_CHANGED")
    shutil.copyfile(design, folder / "etf-share-design.md")
    files["etf-share-design.md"] = file_hash(folder / "etf-share-design.md")
    binding = {
        "source_run": SOURCE_RUN,
        "source_complete": SOURCE_HASH,
        "acquired": acquired["manifest_hash"],
        "probed": probed["manifest_hash"],
    }
    files["etf-share-source-binding.json"] = write_json(folder / "etf-share-source-binding.json", binding)
    bundles, audit = build_bundles(folder)
    for name, bundle in bundles.items():
        files[f"etf-share-prepared-{name}.json"] = write_json(folder / f"etf-share-prepared-{name}.json", bundle)
    files["etf-share-data-audit.json"] = write_json(folder / "etf-share-data-audit.json", audit)
    plan = {k: deepcopy(old[k]) for k in ("funds", "bootstrap", "minimum_coverage", "minimum_windows", "selection")}
    plan.update(
        {
            "version": VERSION,
            "windows": study_windows(VERSION),
            "branches": list(BRANCHES),
            "baselines": list(BASELINES),
            "minimum": {"FIT": 252, "EXAM": 40},
            "source_code_hash": source_fingerprint(),
            "runtime": runtime(),
            "design_hash": files["etf-share-design.md"],
            "main_fits": 16,
            "replay_fits": 16,
            "worker_seconds": 120,
            "memory_bytes": 4 * 1024**3,
            "total_fit_seconds": 1800,
            "independent_test": False,
            "model_released": False,
            "history_first_publication_proven": False,
            "feature": "PREVIOUS_SECOND_SESSION_SHARE_DIV_FIVE_SESSIONS_EARLIER_MINUS_ONE",
            "auc_gate_added": False,
        }
    )
    files["etf-share-plan.json"] = write_json(folder / "etf-share-plan.json", plan)
    frozen = seal(folder, "etf-share-frozen.json", files, status="READY_FROZEN_BEFORE_FIT")
    return {
        "run": folder.name.removeprefix("direction-training-"),
        "folder": str(folder),
        "code_hash": plan["source_code_hash"],
        "frozen_hash": frozen["manifest_hash"],
        "data_audit": audit,
        "coverage": {n: b["coverage"] for n, b in bundles.items()},
    }


def load_plan(folder):
    frozen = read_seal(folder, "etf-share-frozen.json")
    plan = read_json(folder / "etf-share-plan.json")
    if (
        plan["version"] != VERSION
        or plan["windows"] != study_windows(VERSION)
        or plan["branches"] != list(BRANCHES)
        or plan["source_code_hash"] != source_fingerprint()
        or plan["runtime"] != runtime()
    ):
        raise ValueError("ETF_SHARE_PROTOCOL_CODE_OR_RUNTIME_CHANGED")
    binding = read_json(folder / "etf-share-source-binding.json")
    if (
        binding["source_run"] != SOURCE_RUN
        or binding["source_complete"] != SOURCE_HASH
        or plan["independent_test"]
        or plan["model_released"]
    ):
        raise ValueError("ETF_SHARE_SOURCE_OR_RELEASE_CHANGED")
    return plan, frozen


def predict(folder):
    from app.services.direction_training_process import run_process

    plan, frozen = load_plan(folder)
    write_json(folder / "etf-share-started.json", {"started_at": now(), "maximum_fits": 16})
    files, execution, started = {}, {}, perf_counter()
    for window in plan["windows"]:
        name = window["name"]
        bundle = read_json(folder / f"etf-share-prepared-{name}.json")
        outputs, execution[name] = {}, {}
        for branch in BRANCHES:
            if perf_counter() - started > plan["total_fit_seconds"]:
                raise ValueError("ETF_SHARE_TOTAL_TIMEOUT")
            result = run_process(
                bundle["jobs"][branch],
                seconds=plan["worker_seconds"],
                memory_bytes=plan["memory_bytes"],
                command=[sys.executable, "-m", "scripts.direction_etf_share_worker"],
            )
            if result["status"] != "PREDICTED":
                write_json(folder / f"etf-share-failed-{name}-{branch}.json", result)
                raise ValueError("ETF_SHARE_WORKER_FAILED")
            outputs[branch] = result
            execution[name][branch] = {k: result[k] for k in ("status", "model_fit_count", "elapsed_seconds")}
            print(json.dumps({"window": name, "branch": branch, "status": result["status"]}), flush=True)
        files[f"etf-share-models-{name}.json"] = write_json(folder / f"etf-share-models-{name}.json", outputs)
        files[f"etf-share-predictions-{name}.jsonl"] = write_jsonl(
            folder / f"etf-share-predictions-{name}.jsonl", window_predictions(bundle, outputs, branches=BRANCHES)
        )
    files["etf-share-execution.json"] = write_json(folder / "etf-share-execution.json", execution)
    return seal(folder, "etf-share-predicted.json", files, frozen_hash=frozen["manifest_hash"], fitted=16)


def diagnostics(folder, plan, predictions, answers):
    from sklearn.metrics import roc_auc_score

    labels = {(r["window"], r["sample_key"]): r["answer"]["y"] for r in answers}
    rows = {
        b: [
            {**r, "y": labels[(r["window"], r["sample_key"])]}
            for r in predictions
            if r["branch"] == b and r["score"] is not None
        ]
        for b in BRANCHES
    }
    windows = {}
    for window in plan["windows"]:
        name = window["name"]
        bundle = read_json(folder / f"etf-share-prepared-{name}.json")
        outputs = read_json(folder / f"etf-share-models-{name}.json")
        windows[name] = {}
        for branch in BRANCHES:
            fit, _ = validate_job(bundle["jobs"][branch])
            rs = [r for r in rows[branch] if r["window"] == name]
            auc = {}
            for fund in FUNDS:
                fs = [r for r in rs if r["fund"] == fund]
                auc[fund] = (
                    float(roc_auc_score([r["y"] for r in fs], [r["score"] for r in fs]))
                    if len({r["y"] for r in fs}) == 2
                    else None
                )
            training = [
                {"fund": i.fund, "y": a.y, "score": s}
                for (i, a), s in zip(fit, predict_model(outputs[branch]["model"], [i for i, _ in fit]), strict=True)
            ]
            windows[name][branch] = {
                "fit": grouped_metrics(training, FUNDS),
                "exam": grouped_metrics(rs, FUNDS),
                "auc_per_fund": auc,
                "auc_macro": sum(auc.values()) / len(FUNDS) if all(a is not None for a in auc.values()) else None,
            }
    return {
        "per_window": windows,
        "per_year": {
            year: {b: grouped_metrics([r for r in rs if r["cutoff"].startswith(year)], FUNDS) for b, rs in rows.items()}
            for year in ("2023", "2024")
        },
        "auc_mean_over_quarters": {
            b: sum(w[b]["auc_macro"] for w in windows.values()) / len(windows)
            if all(w[b]["auc_macro"] is not None for w in windows.values())
            else None
            for b in BRANCHES
        },
        "auc_descriptive_only": True,
        "independent_test": False,
        "model_released": False,
    }


def analyze(folder, plan):
    predictions = [r for w in plan["windows"] for r in read_jsonl(folder / f"etf-share-predictions-{w['name']}.jsonl")]
    answers = read_jsonl(folder / "source-linear-answers.jsonl")
    return evaluate(plan, predictions, answers), diagnostics(folder, plan, predictions, answers)


def score(folder):
    plan, _ = load_plan(folder)
    predicted = read_seal(folder, "etf-share-predicted.json")
    metrics, details = analyze(folder, plan)
    files = {
        "etf-share-metrics.json": write_json(folder / "etf-share-metrics.json", metrics),
        "etf-share-diagnostics.json": write_json(folder / "etf-share-diagnostics.json", details),
    }
    return seal(folder, "etf-share-scored.json", files, predicted_hash=predicted["manifest_hash"])


def decision(metrics):
    return {
        "status": "QUALIFIED_DEVELOPMENT_CANDIDATE" if metrics["selected_candidate"] else "NO_STABLE_GAIN",
        "selected_candidate": metrics["selected_candidate"],
        "independent_test": False,
        "model_released": False,
        "automatic_followup_training": False,
    }


def candidate(folder, plan, metrics):
    return {
        "branch": metrics["selected_candidate"],
        "plan_hash": file_hash(folder / "etf-share-plan.json"),
        "models": {
            w["name"]: read_json(folder / f"etf-share-models-{w['name']}.json")[metrics["selected_candidate"]]["model"][
                "hash"
            ]
            for w in plan["windows"]
        },
        "model_released": False,
    }


def finalize(folder):
    plan, _ = load_plan(folder)
    scored = read_seal(folder, "etf-share-scored.json")
    metrics = read_json(folder / "etf-share-metrics.json")
    files = {"etf-share-decision.json": write_json(folder / "etf-share-decision.json", decision(metrics))}
    if metrics["selected_candidate"]:
        files["etf-share-candidate.json"] = write_json(
            folder / "etf-share-candidate.json", candidate(folder, plan, metrics)
        )
    return seal(folder, "etf-share-complete.json", files, scored_hash=scored["manifest_hash"])


def verify(folder):
    plan, _ = load_plan(folder)
    manifests = {s: read_seal(folder, f"etf-share-{s}.json") for s in STAGES}
    for a, b in zip(STAGES[:-1], STAGES[1:], strict=True):
        if manifests[b][f"{a}_hash"] != manifests[a]["manifest_hash"]:
            raise ValueError("ETF_SHARE_STAGE_CHAIN")
    bundles, audit = build_bundles(folder)
    if audit != read_json(folder / "etf-share-data-audit.json"):
        raise ValueError("ETF_SHARE_SOURCE_AUDIT_CHANGED")
    execution = read_json(folder / "etf-share-execution.json")
    for name, bundle in bundles.items():
        if bundle != read_json(folder / f"etf-share-prepared-{name}.json"):
            raise ValueError("ETF_SHARE_FEATURE_OR_SAMPLING_CHANGED")
        outputs = read_json(folder / f"etf-share-models-{name}.json")
        if set(outputs) != set(BRANCHES):
            raise ValueError("ETF_SHARE_MODEL_BRANCHES")
        for branch, output in outputs.items():
            fit, exam = validate_job(bundle["jobs"][branch])
            model = output["model"]
            if (
                model["train_hash"] != digest(bundle["jobs"][branch]["fit"])
                or model["train_counts"] != dict(Counter(i.fund for i, _ in fit))
                or model["window"] != name
                or model["branch"] != branch
            ):
                raise ValueError("ETF_SHARE_MODEL_TRAIN_BINDING")
            if (
                predict_model(model, exam) != output["scores"]
                or output["status"] != "PREDICTED"
                or output["model_fit_count"] != 1
            ):
                raise ValueError("ETF_SHARE_MODEL_REPLAY_CHANGED")
        if window_predictions(bundle, outputs, branches=BRANCHES) != read_jsonl(
            folder / f"etf-share-predictions-{name}.jsonl"
        ):
            raise ValueError("ETF_SHARE_PREDICTIONS_CHANGED")
    if (
        set(execution) != set(bundles)
        or any(
            set(jobs) != set(BRANCHES)
            or any(j["status"] != "PREDICTED" or j["model_fit_count"] != 1 for j in jobs.values())
            for jobs in execution.values()
        )
        or manifests["predicted"]["fitted"] != 16
    ):
        raise ValueError("ETF_SHARE_FIT_ACCOUNTING")
    metrics, details = analyze(folder, plan)
    if (
        metrics != read_json(folder / "etf-share-metrics.json")
        or details != read_json(folder / "etf-share-diagnostics.json")
        or decision(metrics) != read_json(folder / "etf-share-decision.json")
    ):
        raise ValueError("ETF_SHARE_RESULTS_OR_DECISION_CHANGED")
    if metrics["selected_candidate"]:
        if candidate(folder, plan, metrics) != read_json(folder / "etf-share-candidate.json"):
            raise ValueError("ETF_SHARE_CANDIDATE_CHANGED")
    elif (folder / "etf-share-candidate.json").exists():
        raise ValueError("ETF_SHARE_UNQUALIFIED_CANDIDATE")
    return manifests["complete"]


def replay(folder):
    if (folder / "etf-share-replay-origin.json").exists() or (folder / "etf-share-replay-started.json").exists():
        raise ValueError("ETF_SHARE_REPLAY_ALREADY_USED")
    verify(folder)
    target = new_folder()
    origin = {"original": folder.name, "replay": target.name}
    write_json(folder / "etf-share-replay-started.json", origin)
    frozen = read_seal(folder, "etf-share-frozen.json")
    for name in ("etf-share-frozen.json", *frozen["files"]):
        shutil.copyfile(folder / name, target / name)
    write_json(target / "etf-share-replay-origin.json", origin)
    predict(target)
    score(target)
    finalize(target)
    verified = verify(target)
    count = 0
    for window in read_json(folder / "etf-share-plan.json")["windows"]:
        name = window["name"]
        a, b = (read_json(f / f"etf-share-models-{name}.json") for f in (folder, target))
        for branch in BRANCHES:
            if a[branch]["model"] != b[branch]["model"] or a[branch]["scores"] != b[branch]["scores"]:
                raise ValueError("ETF_SHARE_REFIT_CHANGED")
            count += len(a[branch]["scores"])
        if read_jsonl(folder / f"etf-share-predictions-{name}.jsonl") != read_jsonl(
            target / f"etf-share-predictions-{name}.jsonl"
        ):
            raise ValueError("ETF_SHARE_REPLAY_BASELINE_CHANGED")
    for name in ("etf-share-metrics.json", "etf-share-diagnostics.json", "etf-share-decision.json"):
        if read_json(folder / name) != read_json(target / name):
            raise ValueError("ETF_SHARE_REFIT_RESULTS_CHANGED")
    receipt = {
        "run": target.name.removeprefix("direction-training-"),
        "new_fits": 16,
        "scores": count,
        "maximum_difference": 0,
        "models_identical": True,
        "results_identical": True,
        "complete_hash": verified["manifest_hash"],
    }
    seal(
        target,
        "etf-share-replay.json",
        {"etf-share-replay-receipt.json": write_json(target / "etf-share-replay-receipt.json", receipt)},
    )
    return receipt
