"""固定样本量与时间跨度诊断；复用旧模型，不选候选、不发布、不连接来源。"""

import heapq
import json
import shutil
import statistics
import sys
from collections import Counter
from copy import deepcopy
from pathlib import Path
from time import perf_counter
from uuid import UUID

from app.services.direction_linear_models import predict_model, validate_job
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
)
from app.services.direction_training_dataset import FUNDS
from app.services.direction_training_evaluation import complete_blocks, grouped_metrics, paired_interval
from app.services.direction_training_protocol import runtime, source_fingerprint

VERSION = "DIRECTION_DIAGNOSTIC_SIZE_TIME_V1"
MODES = ("SPREAD_252", "SPREAD_MID", "RECENT_252")
ALL_MODES = ("FULL_REUSED", *MODES)
STAGES = ("frozen", "predicted", "scored", "complete")
SOURCES = {
    "linear": (
        "0c0e06a9-725e-4b68-b813-de6ff5124b29",
        "812c0f12064641f7d3e50653c2e27e28e235b430af544c74aaa25170e2caef6e",
    ),
    "algorithm": (
        "f212aa57-1c2c-4e13-9b8d-b44f8797090b",
        "1ee485b33b850916e450fab6c971e85b81ad3bb846fd1216ddc068b87a1894c8",
    ),
}


def spread_indices(total, count):
    """首尾固定、优先平分最大日期索引间隔；不同规模嵌套，选择不接触答案。"""
    if not 2 <= count <= total:
        raise ValueError("DIAGNOSTIC_SAMPLE_SIZE")
    selected, gaps = {0, total - 1}, [(-(total - 1), 0, total - 1)]
    while len(selected) < count:
        _, left, right = heapq.heappop(gaps)
        middle = (left + right) // 2
        selected.add(middle)
        for a, b in ((left, middle), (middle, right)):
            if b - a > 1:
                heapq.heappush(gaps, (-(b - a), a, b))
    return selected


def select_rows(rows, mode):
    """三基金按共同日期选样，原行序、每行内容和已成熟标签保持原样。"""
    if mode not in ALL_MODES:
        raise ValueError("DIAGNOSTIC_MODE")
    by_fund = {f: [r["input"]["cutoff"] for r in rows if r["input"]["fund"] == f] for f in FUNDS}
    dates = sorted(by_fund[FUNDS[0]])
    if len(dates) < 252 or len(set(dates)) != len(dates) or any(sorted(ds) != dates for ds in by_fund.values()):
        raise ValueError("DIAGNOSTIC_COMMON_FIT_DATES")
    if len(rows) != len(dates) * len(FUNDS):
        raise ValueError("DIAGNOSTIC_FIT_SCOPE")
    size = len(dates) if mode == "FULL_REUSED" else (252 + len(dates)) // 2 if mode == "SPREAD_MID" else 252
    positions = set(range(len(dates) - size, len(dates))) if mode == "RECENT_252" else spread_indices(len(dates), size)
    chosen = {dates[i] for i in positions}
    return [r for r in rows if r["input"]["cutoff"] in chosen]


def make_jobs(bundle, version):
    return {
        mode: {
            "version": version,
            "branch": "REFERENCE",
            "window": bundle["window"],
            "fit": select_rows(bundle["complete"]["fit"], mode),
            "exam": deepcopy(bundle["complete"]["exam"]["CLEAN"]),
        }
        for mode in ALL_MODES
    }


def source_seals(prefix):
    identifier, expected = SOURCES[prefix]
    folder = run_folder(UUID(identifier))
    stages = ("frozen", "prepared", "predicted", "scored", "complete")
    manifests = {s: read_seal(folder, f"{prefix}-{s}.json") for s in stages}
    for a, b in zip(stages[:-1], stages[1:], strict=True):
        if manifests[b][f"{a}_hash"] != manifests[a]["manifest_hash"]:
            raise ValueError("DIAGNOSTIC_SOURCE_CHAIN")
    if manifests["complete"]["manifest_hash"] != expected:
        raise ValueError("DIAGNOSTIC_SOURCE_CHANGED")
    return folder, manifests


def freeze(design: Path):
    """先验证已有来源封存，再把必要证据与不依赖答案的采样方案独占封存。"""
    linear, linear_seals = source_seals("linear")
    algorithm, algorithm_seals = source_seals("algorithm")
    old = read_json(linear / "linear-plan.json")
    folder, files = new_folder(), {}
    names = ["linear-plan.json", "linear-answers.jsonl"]
    names += [f"linear-{kind}-{w['name']}.json" for w in old["windows"] for kind in ("prepared", "models")]
    for source, names_here, manifests in (
        (linear, names, linear_seals),
        (algorithm, ["algorithm-diagnostics.json", "algorithm-execution.json"], algorithm_seals),
    ):
        expected_files = {name: value for manifest in manifests.values() for name, value in manifest["files"].items()}
        for name in names_here:
            target = folder / f"source-{name}"
            shutil.copyfile(source / name, target)
            files[target.name] = file_hash(target)
            if files[target.name] != expected_files[name]:
                raise ValueError("DIAGNOSTIC_SOURCE_COPY_CHANGED")
    shutil.copyfile(design, folder / "diagnostic-design.md")
    files["diagnostic-design.md"] = file_hash(folder / "diagnostic-design.md")
    binding = {
        p: {s: m["manifest_hash"] for s, m in ms.items()}
        for p, ms in (("linear", linear_seals), ("algorithm", algorithm_seals))
    }
    files["diagnostic-source-binding.json"] = write_json(folder / "diagnostic-source-binding.json", binding)
    plan = {
        "version": VERSION,
        "source_code_hash": source_fingerprint(),
        "runtime": runtime(),
        "source_runs": SOURCES,
        "design_hash": files["diagnostic-design.md"],
        "windows": old["windows"],
        "funds": list(FUNDS),
        "bootstrap": old["bootstrap"],
        "modes": list(ALL_MODES),
        "main_fits": 24,
        "replay_fits": 24,
        "selection": "DATE_ONLY_NESTED_LARGEST_GAP_BISECTION_252_MID_FULL_AND_RECENT_252",
        "independent_test": False,
        "model_released": False,
        "candidate_selection": False,
        "total_fit_seconds": 1800,
        "worker_seconds": 120,
        "memory_bytes": 4 * 1024**3,
        "network_calls": 0,
        "database_writes": 0,
    }
    files["diagnostic-plan.json"] = write_json(folder / "diagnostic-plan.json", plan)
    for w in old["windows"]:
        jobs = make_jobs(read_json(folder / f"source-linear-prepared-{w['name']}.json"), old["version"])
        for job in jobs.values():
            validate_job(job)
        name = f"diagnostic-jobs-{w['name']}.json"
        files[name] = write_json(folder / name, jobs)
    frozen = seal(folder, "diagnostic-frozen.json", files, status="DIAGNOSTIC_ONLY_BEFORE_FIT")
    return {
        "run": folder.name.removeprefix("direction-training-"),
        "folder": str(folder),
        "code_hash": plan["source_code_hash"],
        **frozen,
    }


def load_plan(folder):
    frozen = read_seal(folder, "diagnostic-frozen.json")
    plan = read_json(folder / "diagnostic-plan.json")
    if plan["version"] != VERSION or plan["source_code_hash"] != source_fingerprint() or plan["runtime"] != runtime():
        raise ValueError("DIAGNOSTIC_CODE_OR_RUNTIME_CHANGED")
    if plan["modes"] != list(ALL_MODES) or plan["model_released"] or plan["candidate_selection"]:
        raise ValueError("DIAGNOSTIC_SCOPE_CHANGED")
    return plan, frozen


def predict(folder):
    from app.services.direction_training_process import run_process

    plan, frozen = load_plan(folder)
    write_json(folder / "diagnostic-started.json", {"started_at": now(), "maximum_fits": 24})
    files, execution, started = {}, {}, perf_counter()
    for w in plan["windows"]:
        name = w["name"]
        jobs = read_json(folder / f"diagnostic-jobs-{name}.json")
        old = read_json(folder / f"source-linear-models-{name}.json")["REFERENCE"]
        outputs = {"FULL_REUSED": {**old, "model_fit_count": 0}}
        execution[name] = {}
        for mode in MODES:
            if perf_counter() - started > plan["total_fit_seconds"]:
                raise ValueError("DIAGNOSTIC_TOTAL_TIMEOUT")
            result = run_process(
                jobs[mode],
                seconds=plan["worker_seconds"],
                memory_bytes=plan["memory_bytes"],
                command=[sys.executable, "-m", "scripts.direction_linear_worker"],
            )
            if result["status"] != "PREDICTED":
                write_json(folder / f"diagnostic-failed-{name}-{mode}.json", result)
                raise ValueError("DIAGNOSTIC_WORKER_FAILED")
            outputs[mode] = result
            execution[name][mode] = {
                k: result[k] for k in ("status", "model_fit_count", "elapsed_seconds", "train_counts")
            }
            print(json.dumps({"window": name, "mode": mode, "status": result["status"]}), flush=True)
        file = f"diagnostic-models-{name}.json"
        files[file] = write_json(folder / file, outputs)
    files["diagnostic-execution.json"] = write_json(folder / "diagnostic-execution.json", execution)
    return seal(folder, "diagnostic-predicted.json", files, frozen_hash=frozen["manifest_hash"], fitted=24, reused=8)


def correlation(x, y):
    """仅描述特征与标签的线性关系；常量或过少观测保留未知。"""
    if len(x) < 3 or len(set(x)) < 2 or len(set(y)) < 2:
        return None
    return statistics.correlation(x, y)


def records(items, scores, labels, window):
    return [
        {
            "window": window,
            "sample_key": i.key,
            "fund": i.fund,
            "cutoff": str(i.cutoff),
            "y": y,
            "score": s,
            "correct": int(int(s > 0.5) == y),
        }
        for i, s, y in zip(items, scores, labels, strict=True)
    ]


def analyze(folder):
    """不拟合地重放全部保存模型，拆解样本量、时段和基金误差。"""
    plan, _ = load_plan(folder)
    answers = {
        (r["window"], r["sample_key"]): r["answer"]["y"] for r in read_jsonl(folder / "source-linear-answers.jsonl")
    }
    all_records = {m: [] for m in (*ALL_MODES, "TRAIN_FUND_FREQUENCY", "ALWAYS_NON_UP")}
    by_window, temporal, planned = {}, {}, {}
    old = read_json(folder / "source-linear-plan.json")
    for w in plan["windows"]:
        name = w["name"]
        source = read_json(folder / f"source-linear-prepared-{name}.json")
        jobs = make_jobs(source, old["version"])
        if jobs != read_json(folder / f"diagnostic-jobs-{name}.json"):
            raise ValueError("DIAGNOSTIC_SAMPLING_CHANGED")
        outputs = read_json(folder / f"diagnostic-models-{name}.json")
        if set(outputs) != set(ALL_MODES):
            raise ValueError("DIAGNOSTIC_MODEL_MODES")
        planned[name] = source["planned"]
        by_window[name], temporal[name] = {}, {}
        for mode in ALL_MODES:
            fit, exam = validate_job(jobs[mode])
            output, model = outputs[mode], outputs[mode]["models"]["POOLED"]
            if (
                output["status"] != "PREDICTED"
                or model["train_hash"] != digest(jobs[mode]["fit"])
                or model["fit_end"] != w["fit_end"]
            ):
                raise ValueError("DIAGNOSTIC_MODEL_TRAIN_BINDING")
            if model["train_counts"] != dict(Counter(i.fund for i, _ in fit)):
                raise ValueError("DIAGNOSTIC_MODEL_TRAIN_COUNTS")
            if (
                mode == "FULL_REUSED"
                and model != read_json(folder / f"source-linear-models-{name}.json")["REFERENCE"]["models"]["POOLED"]
            ):
                raise ValueError("DIAGNOSTIC_ORIGINAL_MODEL_CHANGED")
            scores = predict_model(model, exam)
            if scores != output["scores"]:
                raise ValueError("DIAGNOSTIC_MODEL_SCORE_CHANGED")
            ys = [answers[(name, i.key)] for i in exam]
            rs = records(exam, scores, ys, name)
            all_records[mode].extend(rs)
            training = records(
                [i for i, _ in fit], predict_model(model, [i for i, _ in fit]), [a.y for _, a in fit], name
            )
            dates = sorted({str(i.cutoff) for i, _ in fit})
            by_window[name][mode] = {
                "FIT": grouped_metrics(training, FUNDS),
                "EXAM": grouped_metrics(rs, FUNDS),
                "n_per_fund": len(dates),
                "fit_first": dates[0],
                "fit_last": dates[-1],
                "solver_iterations": model["solver_iterations"],
            }
            if mode != "FULL_REUSED":
                continue
            for fund in FUNDS:
                fi = [(i, a) for i, a in fit if i.fund == fund]
                ei = [(i, y) for i, y in zip(exam, ys, strict=True) if i.fund == fund]
                frequency = sum(a.y for _, a in fi) / len(fi)
                temporal[name][fund] = {
                    "fit_up_rate": frequency,
                    "exam_up_rate": sum(y for _, y in ei) / len(ei),
                    "correlation_fit": [correlation([i.x[j] for i, _ in fi], [a.y for _, a in fi]) for j in range(7)],
                    "correlation_exam": [correlation([i.x[j] for i, _ in ei], [y for _, y in ei]) for j in range(7)],
                    "mean_shift_in_fit_sd": [
                        (statistics.mean(i.x[j] for i, _ in ei) - statistics.mean(i.x[j] for i, _ in fi))
                        / model["scale"][j]
                        for j in range(7)
                    ],
                }
                all_records["TRAIN_FUND_FREQUENCY"].extend(
                    records([i for i, _ in ei], [frequency] * len(ei), [y for _, y in ei], name)
                )
                all_records["ALWAYS_NON_UP"].extend(
                    records([i for i, _ in ei], [0.0] * len(ei), [y for _, y in ei], name)
                )
    maps = {m: {(r["window"], r["sample_key"]): r for r in rs} for m, rs in all_records.items()}
    keys = set(maps["FULL_REUSED"])
    if len(keys) != 1389 or any(set(rows) != keys for rows in maps.values()):
        raise ValueError("DIAGNOSTIC_COMMON_EXAM_KEYS")
    blocks, excluded = complete_blocks(plan, keys, planned_by_window=planned)
    comparisons = {}
    for left, right in (("FULL_REUSED", "SPREAD_252"), ("FULL_REUSED", "SPREAD_MID"), ("RECENT_252", "SPREAD_252")):
        comparisons[f"{left}_vs_{right}"] = {
            "full_accuracy_delta": sum(maps[left][k]["correct"] - maps[right][k]["correct"] for k in keys) / len(keys),
            "time_blocks": paired_interval(plan, blocks, maps[left], maps[right]),
        }
    simultaneous = Counter()
    for name, day in sorted({(r["window"], r["cutoff"]) for r in all_records["FULL_REUSED"]}):
        simultaneous[str(sum(1 - maps["FULL_REUSED"][(name, f"{f}:{day}")]["correct"] for f in FUNDS))] += 1
    return {
        "version": VERSION,
        "same_questions": len(keys),
        "new_fits": 24,
        "reused_models": 8,
        "common": {m: grouped_metrics(rs, FUNDS) for m, rs in all_records.items()},
        "year_2024": {
            m: grouped_metrics([r for r in rs if r["cutoff"].startswith("2024")], FUNDS)
            for m, rs in all_records.items()
        },
        "per_window": by_window,
        "temporal": temporal,
        "comparisons": comparisons,
        "time_blocks": {"count": len(blocks), "exclusions": excluded},
        "simultaneous_errors_per_date": dict(sorted(simultaneous.items())),
        "old_algorithm_evidence": read_json(folder / "source-algorithm-diagnostics.json"),
        "interpretation": "EXPLORATORY_DIAGNOSTIC_NOT_CAUSAL_NOT_INDEPENDENT_NOT_CANDIDATE_SELECTION",
        "selected_candidate": None,
        "model_released": False,
        "independent_test": False,
        "automatic_followup_training": False,
    }


def score(folder):
    load_plan(folder)
    predicted = read_seal(folder, "diagnostic-predicted.json")
    files = {"diagnostic-results.json": write_json(folder / "diagnostic-results.json", analyze(folder))}
    return seal(folder, "diagnostic-scored.json", files, predicted_hash=predicted["manifest_hash"])


def finalize(folder):
    load_plan(folder)
    scored = read_seal(folder, "diagnostic-scored.json")
    result = {
        "status": "DIAGNOSTIC_COMPLETE_NOT_OPTIMIZED_MODEL",
        "model_released": False,
        "independent_test": False,
        "selected_candidate": None,
        "automatic_followup_training": False,
    }
    files = {"diagnostic-decision.json": write_json(folder / "diagnostic-decision.json", result)}
    return seal(folder, "diagnostic-complete.json", files, scored_hash=scored["manifest_hash"])


def verify(folder):
    load_plan(folder)
    seals = {s: read_seal(folder, f"diagnostic-{s}.json") for s in STAGES}
    for a, b in zip(STAGES[:-1], STAGES[1:], strict=True):
        if seals[b][f"{a}_hash"] != seals[a]["manifest_hash"]:
            raise ValueError("DIAGNOSTIC_STAGE_CHAIN")
    if analyze(folder) != read_json(folder / "diagnostic-results.json"):
        raise ValueError("DIAGNOSTIC_RESULTS_CHANGED")
    execution = read_json(folder / "diagnostic-execution.json")
    jobs = [job for window in execution.values() for job in window.values()]
    if len(jobs) != 24 or any(j["status"] != "PREDICTED" or j["model_fit_count"] != 1 for j in jobs):
        raise ValueError("DIAGNOSTIC_FIT_ACCOUNTING")
    if any(
        read_json(folder / "diagnostic-decision.json")[k]
        for k in ("model_released", "independent_test", "selected_candidate", "automatic_followup_training")
    ):
        raise ValueError("DIAGNOSTIC_RELEASE_FORBIDDEN")
    return seals["complete"]


def replay(folder):
    verify(folder)
    if (folder / "diagnostic-replay-origin.json").exists():
        raise ValueError("DIAGNOSTIC_REPLAY_OF_REPLAY")
    if (folder / "diagnostic-replay-started.json").exists():
        raise ValueError("DIAGNOSTIC_REPLAY_ALREADY_STARTED")
    target = new_folder()
    origin = {"original": folder.name, "replay": target.name}
    write_json(folder / "diagnostic-replay-started.json", origin)
    frozen = read_seal(folder, "diagnostic-frozen.json")
    for name in ("diagnostic-frozen.json", *frozen["files"]):
        shutil.copyfile(folder / name, target / name)
    write_json(target / "diagnostic-replay-origin.json", origin)
    predict(target)
    score(target)
    finalize(target)
    verified = verify(target)
    scores = 0
    for w in read_json(folder / "diagnostic-plan.json")["windows"]:
        a = read_json(folder / f"diagnostic-models-{w['name']}.json")
        b = read_json(target / f"diagnostic-models-{w['name']}.json")
        for mode in ALL_MODES:
            if a[mode]["models"] != b[mode]["models"] or a[mode]["scores"] != b[mode]["scores"]:
                raise ValueError("DIAGNOSTIC_REPLAY_CHANGED")
            scores += len(a[mode]["scores"])
    if read_json(folder / "diagnostic-results.json") != read_json(target / "diagnostic-results.json"):
        raise ValueError("DIAGNOSTIC_REPLAY_RESULTS_CHANGED")
    receipt = {
        "run": target.name.removeprefix("direction-training-"),
        "scores": scores,
        "maximum_difference": 0,
        "models_identical": True,
        "results_identical": True,
        "new_fits": 24,
        "reused_models": 8,
        "complete_hash": verified["manifest_hash"],
    }
    seal(
        target,
        "diagnostic-replay.json",
        {"diagnostic-replay-receipt.json": write_json(target / "diagnostic-replay-receipt.json", receipt)},
    )
    return receipt
