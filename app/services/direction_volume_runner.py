"""固定量价三组实验：阶段封存、原模型同题对照和有限复跑。"""

import json
import shutil
import sys
from copy import deepcopy
from time import perf_counter
from uuid import UUID

from app.services import direction_volume_data as data
from app.services.direction_linear_analysis import evaluate
from app.services.direction_linear_models import predict_model, validate_job
from app.services.direction_linear_protocol import BASELINES, VOLUME_BRANCHES, VOLUME_VERSION, study_windows
from app.services.direction_market_runner import (
    build_window,
    diagnostics,
    prior,
    reference_parity,
    validate_mapping,
    window_predictions,
)
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
from app.services.direction_training_protocol import runtime, source_fingerprint
from app.services.trading_calendar import load_calendar

MARKET_RUN = "00b782c0-21a1-4ab0-88ea-1a18b8e9c2cb"
MARKET_HASH = "2615993c3b2866343290f82ae029592cef328e6680d9dfa5177a48a4277c21d8"
EVIDENCE = ROOT / ".local-runs/direction-volume-evidence-20260911"
STAGES = ("frozen", "acquired", "prepared", "predicted", "scored", "complete")


def market_prior():
    """读取旧市场包的封存链，不用新代码重定义旧模型，也不提前解析评价答案。"""
    folder = run_folder(UUID(MARKET_RUN))
    previous = None
    for stage in STAGES:
        manifest = read_seal(folder, f"market-{stage}.json")
        if previous and manifest[f"{previous[0]}_hash"] != previous[1]["manifest_hash"]:
            raise ValueError("VOLUME_MARKET_PRIOR_CHAIN")
        previous = stage, manifest
    if manifest["manifest_hash"] != MARKET_HASH:
        raise ValueError("VOLUME_MARKET_PRIOR_CHANGED")
    return folder


def freeze():
    previous = market_prior()
    protocol = deepcopy(read_json(previous / "market-plan.json"))
    for key in ("market_features", "matching_claim", "market_information_rule", "original_shared_source_hash"):
        protocol.pop(key, None)
    protocol.update(
        version=VOLUME_VERSION,
        branches=list(VOLUME_BRANCHES),
        baselines=list(BASELINES),
        source_code_hash=source_fingerprint(),
        runtime=runtime(),
        dependency_hash=file_hash(ROOT / "requirements.txt"),
        calendar_hash=load_calendar().content_hash,
        market_prior_run=MARKET_RUN,
        market_prior_hash=MARKET_HASH,
        C_by_branch={b: 1.0 for b in VOLUME_BRANCHES},
        volume_features={
            "AMOUNT_ACTIVITY": ["amount_mean_5d_over_mean_20d"],
            "AMOUNT_INTERACTION": ["amount_mean_5d_over_mean_20d", "fund_return_20d_times_activity_minus_one"],
        },
        activity_formula="MEAN_AMOUNT_LAST_5/MEAN_AMOUNT_LAST_20_BOTH_END_PREVIOUS_SESSION",
        interaction_formula="ORIGINAL_FUND_RETURN_20D*(ACTIVITY-1)",
        volume_information_rule="PREVIOUS_SESSION_AMOUNT_AVAILABLE_BY_CUTOFF_END_NO_FILL_OR_SHIFT",
        database_scope="READ_EXISTING_SOURCE_METADATA_ONLY_ALL_RESPONSES_LOCAL_NO_WRITES",
        snapshot_search="ONE_FIXED_A_B_C_VOLUME_BATCH_NO_ADAPTIVE_FEATURES_OR_PARAMETERS",
        interaction_claim="REQUIRES_C_VS_B_INCREMENT_NOT_JUST_C_VS_A",
        request_budget={"probe": 3, "annual_main": 12, "retries": 0, "replay_network_calls": 0},
    )
    mapping = read_json(previous / "market-mapping.json")
    validate_mapping(mapping)
    probe, independence = read_json(EVIDENCE / "probe.json"), read_json(EVIDENCE / "independence-audit.json")
    if probe["status"] != "PASSED" or independence["independent_period_proven"]:
        raise ValueError("VOLUME_PREFLIGHT_SCOPE_CHANGED")
    folder, files = new_folder(), {}
    for name, value in {
        "volume-plan.json": protocol,
        "volume-mapping.json": mapping,
        "volume-price-reference.json": read_json(previous / "market-data.json")["prices"],
        "volume-probe.json": probe,
        "volume-independence-audit.json": independence,
    }.items():
        files[name] = write_json(folder / name, value)
    for entry in independence["evidence_files"]:
        source = ROOT / entry["path"]
        if file_hash(source) != entry["sha256"]:
            raise ValueError("VOLUME_INDEPENDENCE_EVIDENCE_CHANGED")
        name = "volume-independence-source-" + str(len(files)) + ".json"
        shutil.copyfile(source, folder / name)
        files[name] = file_hash(folder / name)
    result = seal(folder, "volume-frozen.json", files, status="FROZEN_BEFORE_ANNUAL_DATA_AND_FIT")
    return {"run": folder.name.removeprefix("direction-training-"), "folder": str(folder), **result}


def load_plan(folder):
    frozen = read_seal(folder, "volume-frozen.json")
    protocol, mapping = read_json(folder / "volume-plan.json"), read_json(folder / "volume-mapping.json")
    validate_mapping(mapping)
    if (
        protocol["version"] != VOLUME_VERSION
        or protocol["branches"] != list(VOLUME_BRANCHES)
        or protocol["windows"] != study_windows(VOLUME_VERSION)
        or protocol["source_code_hash"] != source_fingerprint()
        or protocol["runtime"] != runtime()
        or protocol["dependency_hash"] != file_hash(ROOT / "requirements.txt")
        or protocol["calendar_hash"] != load_calendar().content_hash
        or protocol["mapping_hash"] != digest(mapping)
    ):
        raise ValueError("VOLUME_FROZEN_CODE_OR_PROTOCOL_CHANGED")
    prior()
    return protocol, mapping, frozen


def acquire(folder):
    _, mapping, frozen = load_plan(folder)
    if (folder / "volume-acquired.json").exists():
        return read_seal(folder, "volume-acquired.json")
    result, files = data.acquire(folder, mapping, read_json(folder / "volume-price-reference.json"))
    return seal(
        folder,
        "volume-acquired.json",
        files,
        frozen_hash=frozen["manifest_hash"],
        status="ACQUIRED"
        if all(v["all_requests_succeeded"] for v in result["coverage"].values())
        else "DATA_SOURCE_FAILED",
    )


def prepare(folder):
    protocol, mapping, frozen = load_plan(folder)
    acquired = read_seal(folder, "volume-acquired.json")
    if acquired["frozen_hash"] != frozen["manifest_hash"] or acquired["status"] != "ACQUIRED":
        raise ValueError("VOLUME_DATA_NOT_READY")
    activity, previous = read_json(folder / "volume-data.json"), prior()
    old_coverage = read_json(previous / "linear-coverage.json")
    files, coverage = {}, {}
    for window in protocol["windows"]:
        name = window["name"]
        bundle, coverage[name] = build_window(
            read_json(previous / f"linear-prepared-{name}.json"),
            mapping,
            activity,
            protocol,
            old_coverage[name],
            augment_inputs=data.augment_inputs,
        )
        path = folder / f"volume-prepared-{name}.json"
        files[path.name] = write_json(path, bundle)
    files["volume-coverage.json"] = write_json(folder / "volume-coverage.json", coverage)
    return seal(
        folder, "volume-prepared.json", files, acquired_hash=acquired["manifest_hash"], worker_exam_answers=False
    )


def predict(folder):
    """一个研究包只允许一次受限拟合阶段；中断不会被静默重跑成另一次挑选机会。"""
    protocol, _, _ = load_plan(folder)
    prepared = read_seal(folder, "volume-prepared.json")
    coverage, previous = read_json(folder / "volume-coverage.json"), prior()
    write_json(folder / "volume-predict-started.json", {"started_at": now(), "maximum_fits": 24})
    from app.services.direction_training_process import run_process

    files, execution, parity, jobs_run = {}, {}, {}, 0
    started = perf_counter()
    for window in protocol["windows"]:
        name, outputs = window["name"], {}
        bundle = read_json(folder / f"volume-prepared-{name}.json")
        for branch in VOLUME_BRANCHES:
            if coverage[name]["status"] != "READY":
                result = {"status": "INSUFFICIENT_DATA"}
            elif (
                jobs_run >= protocol["fit_budget"]["main_maximum"]
                or perf_counter() - started > protocol["budget"]["total_seconds"]
            ):
                result = {"status": "FAILED", "reason": "VOLUME_TRAINING_BUDGET"}
            else:
                jobs_run += 1
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
        for file, value, writer in (
            (f"volume-models-{name}.json", outputs, write_json),
            (
                f"volume-predictions-{name}.jsonl",
                window_predictions(bundle, outputs, branches=VOLUME_BRANCHES),
                write_jsonl,
            ),
        ):
            files[file] = writer(folder / file, value)
        execution[name] = {b: {k: v for k, v in o.items() if k not in ("models", "scores")} for b, o in outputs.items()}
    for name, value in (("volume-execution.json", execution), ("volume-reference-parity.json", parity)):
        files[name] = write_json(folder / name, value)
    files["volume-predict-started.json"] = file_hash(folder / "volume-predict-started.json")
    return seal(
        folder,
        "volume-predicted.json",
        files,
        prepared_hash=prepared["manifest_hash"],
        model_fit_count=jobs_run,
        status="PREDICTIONS_SEALED_BEFORE_EXAM_ANSWERS",
    )


def all_predictions(folder, protocol):
    return [r for w in protocol["windows"] for r in read_jsonl(folder / f"volume-predictions-{w['name']}.jsonl")]


def report_diagnostics(protocol, predictions, answers):
    report = diagnostics(protocol, predictions, answers)
    report["interaction_stable_increment"] = report.pop("matching_stable_increment")
    report["interaction_error_transitions"] = report.pop("matched_error_transitions")
    old = read_json(prior() / "linear-metrics.json")
    report.update(previous_full_A=old["common"]["REFERENCE"], previous_full_question_count=old["same_question_count"])
    return report


def score(folder):
    protocol, _, _ = load_plan(folder)
    predicted = read_seal(folder, "volume-predicted.json")
    predictions = all_predictions(folder, protocol)
    answers = read_jsonl(prior() / "linear-answers.jsonl")
    result = evaluate(protocol, predictions, answers)
    files = {
        "volume-answers.jsonl": write_jsonl(folder / "volume-answers.jsonl", answers),
        "volume-metrics.json": write_json(folder / "volume-metrics.json", result),
        "volume-diagnostics.json": write_json(
            folder / "volume-diagnostics.json", report_diagnostics(protocol, predictions, answers)
        ),
    }
    return seal(
        folder,
        "volume-scored.json",
        files,
        predicted_hash=predicted["manifest_hash"],
        selected_candidate=result["selected_candidate"],
        independent_test=False,
        model_released=False,
    )


def decision(result):
    return {
        "version": VOLUME_VERSION,
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
    protocol, _, _ = load_plan(folder)
    scored, result = read_seal(folder, "volume-scored.json"), read_json(folder / "volume-metrics.json")
    files = {"volume-decision.json": write_json(folder / "volume-decision.json", decision(result))}
    if result["selected_candidate"]:
        b = result["selected_candidate"]
        files["volume-frozen-candidate.json"] = write_json(
            folder / "volume-frozen-candidate.json",
            {
                "branch": b,
                "protocol_hash": digest(protocol),
                "scored_hash": scored["manifest_hash"],
                "models": {
                    w["name"]: read_json(folder / f"volume-models-{w['name']}.json")[b]["models"]
                    for w in protocol["windows"]
                },
                "release_authorized": False,
            },
        )
    return seal(
        folder,
        "volume-complete.json",
        files,
        scored_hash=scored["manifest_hash"],
        status="FINITE_BATCH_COMPLETE",
        model_released=False,
        independent_test=False,
    )


def verify(folder):
    """从年度响应重建成交额、输入、模型分数和评价，无数据库、网络或fit。"""
    protocol, mapping, _ = load_plan(folder)
    stages = {s: read_seal(folder, f"volume-{s}.json") for s in STAGES}
    for left, right in zip(STAGES[:-1], STAGES[1:], strict=True):
        if stages[right][f"{left}_hash"] != stages[left]["manifest_hash"]:
            raise ValueError("VOLUME_STAGE_CHAIN")
    activity, prices = read_json(folder / "volume-data.json"), read_json(folder / "volume-price-reference.json")
    for code in activity["amounts"]:
        raw = [r for year in data.YEARS for r in read_json(folder / f"volume-response-{code}-{year}.json")["rows"]]
        amounts, cov = data.validate_rows(raw, prices[code])
        if amounts != activity["amounts"][code] or any(activity["coverage"][code][k] != v for k, v in cov.items()):
            raise ValueError("VOLUME_RAW_REBUILD_CHANGED")
    previous = prior()
    old_coverage, coverage, parity = read_json(previous / "linear-coverage.json"), {}, {}
    for window in protocol["windows"]:
        name = window["name"]
        bundle, coverage[name] = build_window(
            read_json(previous / f"linear-prepared-{name}.json"),
            mapping,
            activity,
            protocol,
            old_coverage[name],
            augment_inputs=data.augment_inputs,
        )
        if bundle != read_json(folder / f"volume-prepared-{name}.json"):
            raise ValueError("VOLUME_INPUT_REBUILD_CHANGED")
        outputs = read_json(folder / f"volume-models-{name}.json")
        for branch, output in outputs.items():
            if output["status"] == "PREDICTED":
                _, exam = validate_job(bundle["jobs"][branch])
                if predict_model(output["models"]["POOLED"], exam) != output["scores"]:
                    raise ValueError("VOLUME_STORED_MODEL_CHANGED")
        if window_predictions(bundle, outputs, branches=VOLUME_BRANCHES) != read_jsonl(
            folder / f"volume-predictions-{name}.jsonl"
        ):
            raise ValueError("VOLUME_PREDICTIONS_CHANGED")
        parity[name] = reference_parity(
            bundle, outputs["REFERENCE"], read_json(previous / f"linear-models-{name}.json"), coverage[name]
        )
    if coverage != read_json(folder / "volume-coverage.json") or parity != read_json(
        folder / "volume-reference-parity.json"
    ):
        raise ValueError("VOLUME_COVERAGE_OR_PARITY_CHANGED")
    answers = read_jsonl(folder / "volume-answers.jsonl")
    if answers != read_jsonl(previous / "linear-answers.jsonl"):
        raise ValueError("VOLUME_ANSWERS_CHANGED")
    predictions = all_predictions(folder, protocol)
    result = evaluate(protocol, predictions, answers)
    if (
        result != read_json(folder / "volume-metrics.json")
        or decision(result) != read_json(folder / "volume-decision.json")
        or report_diagnostics(protocol, predictions, answers) != read_json(folder / "volume-diagnostics.json")
    ):
        raise ValueError("VOLUME_RESULT_REBUILD_CHANGED")
    return stages["complete"]


def replay(folder):
    verify(folder)
    protocol, _, _ = load_plan(folder)
    target = new_folder()
    for stage in STAGES[:3]:
        manifest = f"volume-{stage}.json"
        for name in (manifest, *read_seal(folder, manifest)["files"]):
            if not (target / name).exists():
                shutil.copyfile(folder / name, target / name)
    write_json(target / "volume-replay-origin.json", {"original_run": folder.name, "created_at": now()})
    predict(target)
    score(target)
    finalize(target)
    verify(target)
    differences, identical = [], True
    for w in protocol["windows"]:
        a = read_json(folder / f"volume-models-{w['name']}.json")
        b = read_json(target / f"volume-models-{w['name']}.json")
        for branch in VOLUME_BRANCHES:
            if a[branch]["status"] != b[branch]["status"]:
                raise ValueError("VOLUME_REPLAY_STATUS_CHANGED")
            if a[branch]["status"] == "PREDICTED":
                differences.extend(abs(x - y) for x, y in zip(a[branch]["scores"], b[branch]["scores"], strict=True))
                identical &= a[branch]["models"] == b[branch]["models"]
    if max(differences, default=0) != 0 or not identical:
        raise ValueError("VOLUME_REPLAY_CHANGED")
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
        "volume-replay.json",
        {"volume-replay-receipt.json": write_json(target / "volume-replay-receipt.json", receipt)},
    )
    return receipt
