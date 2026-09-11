"""仅增加一个市场广度输入的两组研究：事前封存、受限采集、拟合和离线复核。"""

import json
import shutil
import sys
from collections import Counter
from copy import deepcopy
from time import perf_counter
from uuid import UUID

from app.services import direction_breadth_data as data
from app.services.direction_linear_analysis import evaluate
from app.services.direction_linear_models import predict_model, validate_job
from app.services.direction_linear_protocol import BASELINES, BREADTH_BRANCHES, BREADTH_VERSION, study_windows
from app.services.direction_market_runner import PRIOR_HASH, PRIOR_RUN, prior, reference_parity, window_predictions
from app.services.direction_training_artifacts import (
    ROOT,
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
from app.services.direction_training_protocol import runtime, source_fingerprint
from app.services.trading_calendar import load_calendar

STAGES = ("frozen", "acquired", "prepared", "predicted", "scored", "complete")
VOLUME_RUN = "9e89b98a-898f-4518-8e6b-e9f2ca9cfbe1"


def specification(old, probe):
    """原目标、同题窗口和门槛原样保留；新增信息只有已声明的五日上涨比例。"""
    plan = {
        k: deepcopy(old[k])
        for k in (
            "windows",
            "funds",
            "start",
            "end",
            "minimum",
            "minimum_windows",
            "minimum_coverage",
            "bootstrap",
            "budget",
            "selection",
            "independent",
            "features",
            "logistic",
            "threshold",
        )
    }
    plan.update(
        version=BREADTH_VERSION,
        branches=list(BREADTH_BRANCHES),
        baselines=list(BASELINES),
        source_code_hash=source_fingerprint(),
        runtime=runtime(),
        dependency_hash=file_hash(ROOT / "requirements.txt"),
        calendar_hash=load_calendar().content_hash,
        source_id=probe["metadata"]["source_id"],
        prior_run=PRIOR_RUN,
        prior_complete_hash=PRIOR_HASH,
        breadth_rules=deepcopy(data.RULES),
        extra_feature="mean_daily_up_fraction_5d",
        dimensions={"REFERENCE": 7, "MARKET_BREADTH": 8},
        C=1.0,
        parameter_search=False,
        calibration="NONE",
        request_budget={
            "unique_daily_requests": len(data.days()),
            "probe_reused": len(data.PROBE_DAYS),
            "maximum_retries": 10,
            "maximum_attempts_per_day": 2,
            "replay_network_calls": 0,
        },
        fit_budget={"main_maximum": 16, "replay_maximum": 16},
        training_scope="POOLED_EQUAL_FUND_SAME_FIT_AND_EXAM_KEYS_NO_PER_FUND_MODEL_SELECTION",
        snapshot_search="ONE_FIXED_A_B_BATCH_AND_ONE_REPLAY_NO_ADAPTIVE_FEATURE_OR_PARAMETER_CHANGES",
        historical_first_versions_verified=False,
        independent_test=False,
        model_released=False,
        database_written=False,
    )
    return plan


def freeze():
    previous = prior()
    probe, manifest = data.validate_probe(data.EVIDENCE)
    if shutil.disk_usage(ROOT).free < 4 * 1024**3:
        raise ValueError("BREADTH_LOCAL_ARTIFACT_SPACE_BUDGET")
    folder, files = new_folder(), {}
    for name in ("breadth-probe-seal.json", *manifest["files"]):
        shutil.copyfile(data.EVIDENCE / name, folder / name)
        files[name] = file_hash(folder / name)
    for name in (
        "volume-independence-audit.json",
        "volume-independence-source-5.json",
        "volume-independence-source-6.json",
        "volume-independence-source-7.json",
    ):
        source = run_folder(UUID(VOLUME_RUN)) / name
        shutil.copyfile(source, folder / name)
        files[name] = file_hash(folder / name)
    protocol = specification(read_json(previous / "linear-plan.json"), probe)
    files["breadth-plan.json"] = write_json(folder / "breadth-plan.json", protocol)
    frozen = seal(folder, "breadth-frozen.json", files, status="FROZEN_BEFORE_FULL_DATA_AND_FITS")
    return {"folder": str(folder), "run": folder.name.removeprefix("direction-training-"), **frozen}


def load_plan(folder):
    frozen = read_seal(folder, "breadth-frozen.json")
    probe, _ = data.validate_probe(folder)
    previous = prior()
    protocol = read_json(folder / "breadth-plan.json")
    if protocol != specification(read_json(previous / "linear-plan.json"), probe) or protocol[
        "windows"
    ] != study_windows(BREADTH_VERSION):
        raise ValueError("BREADTH_FROZEN_CODE_OR_PROTOCOL_CHANGED")
    return protocol, frozen


def acquire(folder):
    protocol, frozen = load_plan(folder)
    if (folder / "breadth-acquired.json").exists():
        return read_seal(folder, "breadth-acquired.json")
    result, files = data.acquire(folder, protocol)
    return seal(
        folder,
        "breadth-acquired.json",
        files,
        frozen_hash=frozen["manifest_hash"],
        status="ACQUIRED" if result["coverage"]["ready"] == result["coverage"]["planned"] else "INCOMPLETE_DATA",
    )


def build_window(old, daily, protocol, source_coverage):
    """两组仅使用相同的可用输入；缺失保留计划分母，不能删除难题掩盖覆盖损失。"""
    fit, exam = {b: [] for b in BREADTH_BRANCHES}, {b: [] for b in BREADTH_BRANCHES}
    excluded = []
    for phase, rows in (("FIT", old["complete"]["fit"]), ("EXAM", old["complete"]["exam"]["CLEAN"])):
        for row in rows:
            item = row["input"] if phase == "FIT" else row
            extra, reason = data.augment(item, daily)
            if reason:
                excluded.append({"phase": phase, "fund": item["fund"], "cutoff": item["cutoff"], "reason": reason})
                continue
            for branch, value in zip(BREADTH_BRANCHES, (item, extra), strict=True):
                if phase == "FIT":
                    fit[branch].append({"input": value, "answer": row["answer"]})
                else:
                    exam[branch].append(value)
    counts, exams = Counter(r["input"]["fund"] for r in fit["REFERENCE"]), Counter(r["fund"] for r in exam["REFERENCE"])
    ready = all(
        counts[f] >= protocol["minimum"]["FIT"]
        and exams[f] >= protocol["minimum"]["EXAM"]
        and exams[f] / len(old["planned"]) >= protocol["minimum_coverage"]
        and source_coverage["source"][f]["unused_gap_mature_count"] >= protocol["minimum"]["CAL"]
        for f in FUNDS
    )
    jobs = {
        b: {"version": BREADTH_VERSION, "branch": b, "window": old["window"], "fit": fit[b], "exam": exam[b]}
        for b in BREADTH_BRANCHES
    }
    if ready:
        for job in jobs.values():
            validate_job(job)
    keys = {
        b: {
            "fit": [(r["input"]["fund"], r["input"]["cutoff"]) for r in fit[b]],
            "exam": [(r["fund"], r["cutoff"]) for r in exam[b]],
        }
        for b in BREADTH_BRANCHES
    }
    if len({digest(k) for k in keys.values()}) != 1:
        raise ValueError("BREADTH_COMMON_KEYS_CHANGED")
    coverage = {
        "status": "READY" if ready else "INSUFFICIENT_DATA",
        "planned_per_fund": len(old["planned"]),
        "fit_counts": dict(counts),
        "exam_counts": dict(exams),
        "excluded": excluded,
        "same_keys_hash": digest(keys["REFERENCE"]),
        "fit_unchanged": fit["REFERENCE"] == old["complete"]["fit"],
        "exam_unchanged": exam["REFERENCE"] == old["complete"]["exam"]["CLEAN"],
    }
    return {"window": old["window"], "planned": old["planned"], "jobs": jobs}, coverage


def prepare(folder):
    protocol, frozen = load_plan(folder)
    acquired = read_seal(folder, "breadth-acquired.json")
    if acquired["frozen_hash"] != frozen["manifest_hash"] or acquired["status"] != "ACQUIRED":
        raise ValueError("BREADTH_DATA_NOT_READY")
    daily, previous = data.rebuild_daily(folder), prior()
    source_coverage = read_json(previous / "linear-coverage.json")
    files, coverage = {}, {}
    for window in protocol["windows"]:
        name = window["name"]
        bundle, coverage[name] = build_window(
            read_json(previous / f"linear-prepared-{name}.json"), daily, protocol, source_coverage[name]
        )
        filename = f"breadth-prepared-{name}.json"
        files[filename] = write_json(folder / filename, bundle)
    files["breadth-coverage.json"] = write_json(folder / "breadth-coverage.json", coverage)
    return seal(
        folder, "breadth-prepared.json", files, acquired_hash=acquired["manifest_hash"], worker_exam_answers=False
    )


def predict(folder):
    protocol, _ = load_plan(folder)
    prepared = read_seal(folder, "breadth-prepared.json")
    coverage, previous = read_json(folder / "breadth-coverage.json"), prior()
    write_json(folder / "breadth-predict-started.json", {"started_at": now(), "maximum_fits": 16})
    from app.services.direction_training_process import run_process

    files, execution, parity, fits = {}, {}, {}, 0
    started = perf_counter()
    for window in protocol["windows"]:
        name, outputs = window["name"], {}
        bundle = read_json(folder / f"breadth-prepared-{name}.json")
        for branch in BREADTH_BRANCHES:
            if coverage[name]["status"] != "READY":
                result = {"status": "INSUFFICIENT_DATA"}
            elif fits >= 16 or perf_counter() - started > protocol["budget"]["total_seconds"]:
                result = {"status": "FAILED", "reason": "BREADTH_TRAINING_BUDGET"}
            else:
                fits += 1
                result = run_process(
                    bundle["jobs"][branch],
                    seconds=protocol["budget"]["candidate_seconds"],
                    memory_bytes=protocol["budget"]["process_memory_bytes"],
                    command=[sys.executable, "-m", "scripts.direction_linear_worker"],
                )
            outputs[branch] = result
            print(json.dumps({"window": name, "branch": branch, "status": result["status"]}), flush=True)
        parity[name] = reference_parity(
            bundle, outputs["REFERENCE"], read_json(previous / f"linear-models-{name}.json"), coverage[name]
        )
        for filename, value, writer in (
            (f"breadth-models-{name}.json", outputs, write_json),
            (
                f"breadth-predictions-{name}.jsonl",
                window_predictions(bundle, outputs, branches=BREADTH_BRANCHES),
                write_jsonl,
            ),
        ):
            files[filename] = writer(folder / filename, value)
        execution[name] = {b: {k: v for k, v in o.items() if k not in ("models", "scores")} for b, o in outputs.items()}
    for name, value in (("breadth-execution.json", execution), ("breadth-reference-parity.json", parity)):
        files[name] = write_json(folder / name, value)
    files["breadth-predict-started.json"] = file_hash(folder / "breadth-predict-started.json")
    return seal(
        folder,
        "breadth-predicted.json",
        files,
        prepared_hash=prepared["manifest_hash"],
        model_fit_count=fits,
        status="PREDICTIONS_SEALED_BEFORE_EXAM_ANSWERS",
    )


def all_predictions(folder, protocol):
    return [r for w in protocol["windows"] for r in read_jsonl(folder / f"breadth-predictions-{w['name']}.jsonl")]


def score(folder):
    protocol, _ = load_plan(folder)
    predicted = read_seal(folder, "breadth-predicted.json")
    answers = read_jsonl(prior() / "linear-answers.jsonl")
    result = evaluate(protocol, all_predictions(folder, protocol), answers)
    files = {
        "breadth-answers.jsonl": write_jsonl(folder / "breadth-answers.jsonl", answers),
        "breadth-metrics.json": write_json(folder / "breadth-metrics.json", result),
    }
    return seal(
        folder,
        "breadth-scored.json",
        files,
        predicted_hash=predicted["manifest_hash"],
        selected_candidate=result["selected_candidate"],
        independent_test=False,
        model_released=False,
    )


def decision(result):
    return {
        "version": BREADTH_VERSION,
        "selected_candidate": result["selected_candidate"],
        "candidate_status": result["candidate_status"],
        "current_snapshot_search": "CLOSED",
        "automatic_followup_training": False,
        "model_released": False,
        "independent_test_run": False,
        "independent_status": "AWAITING_INDEPENDENCE_AND_FIRST_VERSION_EVIDENCE"
        if result["selected_candidate"]
        else "NOT_RUN_NO_QUALIFIED_CANDIDATE",
    }


def finalize(folder):
    protocol, _ = load_plan(folder)
    scored, result = read_seal(folder, "breadth-scored.json"), read_json(folder / "breadth-metrics.json")
    files = {"breadth-decision.json": write_json(folder / "breadth-decision.json", decision(result))}
    if result["selected_candidate"]:
        branch = result["selected_candidate"]
        files["breadth-frozen-candidate.json"] = write_json(
            folder / "breadth-frozen-candidate.json",
            {
                "branch": branch,
                "protocol_hash": digest(protocol),
                "scored_hash": scored["manifest_hash"],
                "models": {
                    w["name"]: read_json(folder / f"breadth-models-{w['name']}.json")[branch]["models"]
                    for w in protocol["windows"]
                },
                "release_authorized": False,
            },
        )
    return seal(
        folder,
        "breadth-complete.json",
        files,
        scored_hash=scored["manifest_hash"],
        status="FINITE_BATCH_COMPLETE",
        model_released=False,
        independent_test=False,
    )


def verify(folder):
    """从逐日原始股票行情复核全部派生结果；不联网、不连接数据库、不拟合。"""
    protocol, _ = load_plan(folder)
    stages = {s: read_seal(folder, f"breadth-{s}.json") for s in STAGES}
    for left, right in zip(STAGES[:-1], STAGES[1:], strict=True):
        if stages[right][f"{left}_hash"] != stages[left]["manifest_hash"]:
            raise ValueError("BREADTH_STAGE_CHAIN")
    daily, previous = data.rebuild_daily(folder), prior()
    coverage, parity = {}, {}
    source_coverage = read_json(previous / "linear-coverage.json")
    for window in protocol["windows"]:
        name = window["name"]
        bundle, coverage[name] = build_window(
            read_json(previous / f"linear-prepared-{name}.json"), daily, protocol, source_coverage[name]
        )
        if bundle != read_json(folder / f"breadth-prepared-{name}.json"):
            raise ValueError("BREADTH_INPUT_REBUILD_CHANGED")
        outputs = read_json(folder / f"breadth-models-{name}.json")
        if set(outputs) != set(BREADTH_BRANCHES):
            raise ValueError("BREADTH_MODEL_BRANCHES")
        for branch, output in outputs.items():
            if output["status"] == "PREDICTED":
                fit, exam = validate_job(bundle["jobs"][branch])
                model = output["models"]["POOLED"]
                if (
                    model["version"] != BREADTH_VERSION
                    or model["branch"] != branch
                    or model["fit_end"] != window["fit_end"]
                    or model["train_hash"] != digest(bundle["jobs"][branch]["fit"])
                    or model["train_counts"] != dict(Counter(i.fund for i, _ in fit))
                ):
                    raise ValueError("BREADTH_MODEL_TRAIN_BINDING")
                if predict_model(model, exam) != output["scores"]:
                    raise ValueError("BREADTH_STORED_MODEL_SCORE_CHANGED")
        if window_predictions(bundle, outputs, branches=BREADTH_BRANCHES) != read_jsonl(
            folder / f"breadth-predictions-{name}.jsonl"
        ):
            raise ValueError("BREADTH_PREDICTIONS_REBUILD_CHANGED")
        parity[name] = reference_parity(
            bundle, outputs["REFERENCE"], read_json(previous / f"linear-models-{name}.json"), coverage[name]
        )
    if coverage != read_json(folder / "breadth-coverage.json") or parity != read_json(
        folder / "breadth-reference-parity.json"
    ):
        raise ValueError("BREADTH_COVERAGE_OR_PARITY_CHANGED")
    answers = read_jsonl(folder / "breadth-answers.jsonl")
    if answers != read_jsonl(previous / "linear-answers.jsonl"):
        raise ValueError("BREADTH_ANSWERS_CHANGED")
    result = evaluate(protocol, all_predictions(folder, protocol), answers)
    if result != read_json(folder / "breadth-metrics.json") or decision(result) != read_json(
        folder / "breadth-decision.json"
    ):
        raise ValueError("BREADTH_RESULT_REBUILD_CHANGED")
    return stages["complete"]


def replay(folder):
    verify(folder)
    protocol, _ = load_plan(folder)
    if (folder / "breadth-replay-origin.json").exists():
        raise ValueError("BREADTH_REPLAY_OF_REPLAY")
    target = new_folder()
    origin = {"original_run": folder.name, "replay_run": target.name, "created_at": now()}
    write_json(folder / "breadth-replay-started.json", origin)
    for stage in STAGES[:3]:
        manifest = f"breadth-{stage}.json"
        names = {manifest, *read_seal(folder, manifest)["files"]}
        if stage == "acquired":
            names.update(
                n
                for month in sorted({d[:7] for d in data.days()})
                for n in read_seal(folder, f"breadth-source-month-{month}.json")["files"]
            )
        for name in sorted(names):
            if not (target / name).exists():
                shutil.copyfile(folder / name, target / name)
    write_json(target / "breadth-replay-origin.json", origin)
    predict(target)
    score(target)
    finalize(target)
    verify(target)
    differences, identical = [], True
    for w in protocol["windows"]:
        a = read_json(folder / f"breadth-models-{w['name']}.json")
        b = read_json(target / f"breadth-models-{w['name']}.json")
        for branch in BREADTH_BRANCHES:
            if a[branch]["status"] != b[branch]["status"]:
                raise ValueError("BREADTH_REPLAY_STATUS_CHANGED")
            if a[branch]["status"] == "PREDICTED":
                differences.extend(abs(x - y) for x, y in zip(a[branch]["scores"], b[branch]["scores"], strict=True))
                identical &= a[branch]["models"] == b[branch]["models"]
    if max(differences, default=0) != 0 or not identical:
        raise ValueError("BREADTH_REPLAY_CHANGED")
    receipt = {
        "run": target.name.removeprefix("direction-training-"),
        "original_run": folder.name,
        "scores": len(differences),
        "maximum_difference": max(differences, default=0),
        "models_identical": identical,
        "status": "REPRODUCED_NOT_NEW_EFFECT_EVIDENCE",
    }
    seal(
        target,
        "breadth-replay.json",
        {"breadth-replay-receipt.json": write_json(target / "breadth-replay-receipt.json", receipt)},
    )
    return receipt
