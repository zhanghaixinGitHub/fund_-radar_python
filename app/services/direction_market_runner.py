"""基金对应指数的有限实验：固定映射、补数、同题训练、评分和复现。"""

import json
import shutil
import sys
from collections import Counter
from copy import deepcopy
from decimal import Decimal
from time import perf_counter
from uuid import UUID

from app.services import direction_market_data as data
from app.services.direction_linear_analysis import evaluate
from app.services.direction_linear_models import predict_model, validate_job
from app.services.direction_linear_protocol import (
    BASELINES,
    MARKET_BRANCHES,
    MARKET_VERSION,
    evaluation_asof,
    planned_dates,
    study_windows,
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
from app.services.direction_training_dataset import FUNDS
from app.services.direction_training_evaluation import complete_blocks, grouped_metrics, paired_interval
from app.services.direction_training_process import run_process
from app.services.direction_training_protocol import runtime, source_fingerprint
from app.services.historical_nav_evaluation import fixed_momentum_score
from app.services.trading_calendar import load_calendar

PRIOR_RUN = "0c0e06a9-725e-4b68-b813-de6ff5124b29"
PRIOR_HASH = "812c0f12064641f7d3e50653c2e27e28e235b430af544c74aaa25170e2caef6e"
BASELINE_RUN = "feff6919-beca-437a-9b44-5478f61aad44"
MAPPING_FILE = ROOT / "app/data/direction_market_mapping_v1.json"
EVIDENCE = ROOT / ".local-runs/direction-market-evidence-20260911"
NUMERIC_FIELDS = ("mean", "scale", "coefficients", "intercept", "train_hash", "train_counts", "fit_end")


def prior():
    """校验旧研究全部阶段摘要；不会在训练前解析旧EXAM答案。"""
    folder = run_folder(UUID(PRIOR_RUN))
    seals = {
        s: read_seal(folder, f"linear-{s}.json") for s in ("frozen", "prepared", "predicted", "scored", "complete")
    }
    if seals["complete"]["manifest_hash"] != PRIOR_HASH:
        raise ValueError("MARKET_PRIOR_COMPLETE_CHANGED")
    names = list(seals)
    for left, right in zip(names[:-1], names[1:], strict=True):
        if seals[right][f"{left}_hash"] != seals[left]["manifest_hash"]:
            raise ValueError("MARKET_PRIOR_CHAIN")
    return folder


def validate_mapping(mapping):
    expected = {"001632": "930653.CSI", "006730": "000905.SH", "008888": "980017.SZ"}
    if (
        mapping["version"] != "DIRECTION_MARKET_MAPPING_V1"
        or mapping["shared_index"] != "000300.SH"
        or {f: v["index_code"] for f, v in mapping["funds"].items()} != expected
        or mapping["research_start"] != "2021-01-01"
        or mapping["research_end"] != "2024-12-31"
    ):
        raise ValueError("MARKET_MAPPING_SCOPE_CHANGED")


def freeze():
    """映射、日期、算法、评价门槛和来源证据在新增指数日线采集前封存。"""
    previous = prior()
    old = read_json(previous / "linear-plan.json")
    mapping = read_json(MAPPING_FILE)
    validate_mapping(mapping)
    if old["windows"] != study_windows(MARKET_VERSION):
        raise ValueError("MARKET_WINDOWS_CHANGED")
    folder, files = new_folder(), {}
    evidence = read_json(EVIDENCE / "retrieval.json")
    for entry in evidence:
        if entry["status"] != "SAVED" or file_hash(EVIDENCE / entry["file"]) != entry["sha256"]:
            raise ValueError("MARKET_EVIDENCE_MISSING_OR_CHANGED")
        target = folder / ("evidence-" + entry["file"])
        if target.resolve().parent != folder.resolve():
            raise ValueError("MARKET_EVIDENCE_PATH")
        shutil.copyfile(EVIDENCE / entry["file"], target)
        files[target.name] = file_hash(target)
    for name in ("retrieval.json", "food-index-catalog.json"):
        target = folder / ("evidence-" + name)
        files[target.name] = write_json(target, read_json(EVIDENCE / name))
    files["market-mapping.json"] = write_json(folder / "market-mapping.json", mapping)
    baseline = run_folder(UUID(BASELINE_RUN)) / "market.json"
    files["shared-source.json"] = write_json(folder / "shared-source.json", read_json(baseline))
    protocol = {
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
        )
    }
    protocol.update(
        version=MARKET_VERSION,
        branches=list(MARKET_BRANCHES),
        baselines=list(BASELINES),
        threshold=0.5,
        source_code_hash=source_fingerprint(),
        runtime=runtime(),
        dependency_hash=file_hash(ROOT / "requirements.txt"),
        calendar_hash=load_calendar().content_hash,
        prior_run=PRIOR_RUN,
        prior_complete_hash=PRIOR_HASH,
        mapping_hash=digest(mapping),
        original_shared_source_hash=file_hash(baseline),
        market_features=["market_return_20d", "market_volatility_20d", "fund_minus_market_return_20d"],
        C_by_branch={b: 1.0 for b in MARKET_BRANCHES},
        penalty="L2",
        training_scope="POOLED_EQUAL_FUND_SAME_FIT_KEYS_AND_EXAM_KEYS_ALL_THREE_BRANCHES",
        market_information_rule="PREVIOUS_SESSION_CLOSE_AVAILABLE_BY_CUTOFF_END_NO_FILL_NO_DATE_SHIFT",
        fit_budget={"main_maximum": 24, "replay_maximum": 24},
        snapshot_search="ONE_PREDECLARED_A_B_C_BATCH_NO_ADAPTIVE_INDEX_OR_PARAMETER_CHANGES",
        matching_claim="REQUIRES_C_VS_B_PAIRED_INCREMENT_NOT_JUST_C_VS_A",
        database_scope="INSERT_MISSING_TUSHARE_INDEX_CATALOG_ONLY_PRICES_IN_LOCAL_RESEARCH_PACKAGE",
        independent_test=False,
        model_released=False,
    )
    files["market-plan.json"] = write_json(folder / "market-plan.json", protocol)
    frozen = seal(folder, "market-frozen.json", files, status="FROZEN_BEFORE_NEW_INDEX_DAILY_REQUESTS")
    return {"folder": str(folder), "run": folder.name.removeprefix("direction-training-"), **frozen}


def load_plan(folder):
    frozen = read_seal(folder, "market-frozen.json")
    protocol = read_json(folder / "market-plan.json")
    mapping = read_json(folder / "market-mapping.json")
    validate_mapping(mapping)
    if (
        protocol["version"] != MARKET_VERSION
        or protocol["mapping_hash"] != digest(mapping)
        or protocol["source_code_hash"] != source_fingerprint()
        or protocol["runtime"] != runtime()
        or protocol["dependency_hash"] != file_hash(ROOT / "requirements.txt")
        or protocol["calendar_hash"] != load_calendar().content_hash
    ):
        raise ValueError("MARKET_FROZEN_CODE_RUNTIME_OR_MAPPING_CHANGED")
    prior()
    return protocol, mapping, frozen


def acquire(folder):
    _, mapping, frozen = load_plan(folder)
    if (folder / "market-acquired.json").exists():
        return read_seal(folder, "market-acquired.json")
    result, files = data.acquire(folder, mapping, folder / "shared-source.json")
    return seal(
        folder,
        "market-acquired.json",
        files,
        frozen_hash=frozen["manifest_hash"],
        status="ACQUIRED"
        if all(v.get("all_requests_succeeded", True) for v in result["coverage"].values())
        else "DATA_SOURCE_FAILED",
    )


def build_window(old_bundle, mapping, market, protocol, old_coverage, *, augment_inputs=None):
    """仅处理原先已封存的训练标签和考试输入，三个分支严格使用相同题键。"""
    window = old_bundle["window"]
    branches = tuple(protocol.get("branches", MARKET_BRANCHES))
    fit = {b: [] for b in branches}
    exam = {b: [] for b in branches}
    excluded = []
    for phase, rows in (("FIT", old_bundle["complete"]["fit"]), ("EXAM", old_bundle["complete"]["exam"]["CLEAN"])):
        for original in rows:
            item = original["input"] if phase == "FIT" else original
            if augment_inputs is None:
                shared, error_b = data.augment(item, market["prices"][mapping["shared_index"]])
                matched, error_c = data.augment(item, market["prices"][mapping["funds"][item["fund"]]["index_code"]])
            else:
                shared, matched, error_b, error_c = augment_inputs(item, mapping, market)
            if error_b or error_c:
                excluded.append(
                    {
                        "phase": phase,
                        "fund": item["fund"],
                        "cutoff": item["cutoff"],
                        branches[1]: error_b,
                        branches[2]: error_c,
                    }
                )
                continue
            inputs = dict(zip(branches, (item, shared, matched), strict=True))
            for branch, value in inputs.items():
                if phase == "FIT":
                    fit[branch].append({"input": value, "answer": original["answer"]})
                else:
                    exam[branch].append(value)
    counts = Counter(r["input"]["fund"] for r in fit["REFERENCE"])
    exam_counts = Counter(r["fund"] for r in exam["REFERENCE"])
    planned = old_bundle["planned"]
    ready = all(
        counts[f] >= protocol["minimum"]["FIT"]
        and exam_counts[f] >= protocol["minimum"]["EXAM"]
        and exam_counts[f] / len(planned) >= protocol["minimum_coverage"]
        and old_coverage["source"][f]["unused_gap_mature_count"] >= protocol["minimum"]["CAL"]
        for f in FUNDS
    )
    jobs = {
        b: {
            "version": protocol.get("version", MARKET_VERSION),
            "branch": b,
            "window": window,
            "fit": fit[b],
            "exam": exam[b],
        }
        for b in branches
    }
    if ready:
        for job in jobs.values():
            validate_job(job)
    keys = {
        b: {
            "fit": [(r["input"]["fund"], r["input"]["cutoff"]) for r in fit[b]],
            "exam": [(r["fund"], r["cutoff"]) for r in exam[b]],
        }
        for b in branches
    }
    if len({digest(v) for v in keys.values()}) != 1:
        raise ValueError("MARKET_COMMON_INPUT_KEYS_CHANGED")
    report = {
        "status": "READY" if ready else "INSUFFICIENT_DATA",
        "planned_per_fund": len(planned),
        "fit_counts": dict(counts),
        "exam_counts": dict(exam_counts),
        "excluded": excluded,
        "same_keys_hash": digest(keys["REFERENCE"]),
        "fit_unchanged": fit["REFERENCE"] == old_bundle["complete"]["fit"],
        "exam_unchanged": exam["REFERENCE"] == old_bundle["complete"]["exam"]["CLEAN"],
        "source_coverage": old_coverage["source"],
    }
    return {"window": window, "planned": planned, "jobs": jobs}, report


def prepare(folder):
    protocol, mapping, frozen = load_plan(folder)
    acquired = read_seal(folder, "market-acquired.json")
    if acquired["frozen_hash"] != frozen["manifest_hash"] or acquired["status"] != "ACQUIRED":
        raise ValueError("MARKET_DATA_NOT_READY")
    market = read_json(folder / "market-data.json")
    previous = prior()
    coverage_old = read_json(previous / "linear-coverage.json")
    files, coverage = {}, {}
    for window in protocol["windows"]:
        name = window["name"]
        bundle, coverage[name] = build_window(
            read_json(previous / f"linear-prepared-{name}.json"), mapping, market, protocol, coverage_old[name]
        )
        filename = f"market-prepared-{name}.json"
        files[filename] = write_json(folder / filename, bundle)
    files["market-coverage.json"] = write_json(folder / "market-coverage.json", coverage)
    return seal(
        folder,
        "market-prepared.json",
        files,
        acquired_hash=acquired["manifest_hash"],
        worker_exam_answers=False,
        status="PREPARED",
    )


def window_predictions(bundle, outputs, *, branches=MARKET_BRANCHES):
    """完整计划日期保留；缺输入或作业失败不删题，简单对照使用相同训练样本。"""
    index = {}
    for branch, result in outputs.items():
        if result["status"] == "PREDICTED":
            for item, score in zip(bundle["jobs"][branch]["exam"], result["scores"], strict=True):
                index[(branch, item["fund"], item["cutoff"])] = (score, item["input_hash"])
    fit = bundle["jobs"]["REFERENCE"]["fit"]
    counts = Counter(r["input"]["fund"] for r in fit)
    rates = {f: sum(r["answer"]["y"] for r in fit if r["input"]["fund"] == f) / counts[f] for f in FUNDS if counts[f]}
    for item in bundle["jobs"]["REFERENCE"]["exam"]:
        if item["fund"] not in rates:
            continue
        scores = {
            "ALWAYS_UP": 1.0,
            "ALWAYS_NON_UP": 0.0,
            "TRAIN_UP_FREQUENCY": rates[item["fund"]],
            "MOMENTUM_20D": float(item["x"][1] > 0),
            "FIXED_MOMENTUM_SCORE": float(fixed_momentum_score(Decimal(str(item["x"][1])))),
        }
        for branch, score in scores.items():
            index[(branch, item["fund"], item["cutoff"])] = (score, item["input_hash"])
    result = []
    for branch in (*branches, *BASELINES):
        for fund in FUNDS:
            for cutoff in bundle["planned"]:
                value, identity = index.get((branch, fund, cutoff), (None, None))
                result.append(
                    {
                        "window": bundle["window"]["name"],
                        "branch": branch,
                        "fund": fund,
                        "cutoff": cutoff,
                        "sample_key": f"{fund}:{cutoff}",
                        "score": value,
                        "input_hash": identity,
                        "predicted_up": int(value > 0.5) if value is not None else None,
                        "status": "PREDICTED"
                        if value is not None
                        else (
                            outputs[branch]["status"]
                            if branch in outputs and outputs[branch]["status"] != "PREDICTED"
                            else "INPUT_UNAVAILABLE"
                        ),
                    }
                )
    return result


def reference_parity(bundle, output, old_models, coverage):
    if not coverage["fit_unchanged"]:
        return {"status": "NOT_COMPARABLE_CHANGED_FIT_POPULATION"}
    if output["status"] != "PREDICTED":
        return {"status": "NOT_COMPARABLE_JOB_NOT_PREDICTED"}
    old = old_models["REFERENCE"]["models"]["POOLED"]
    new = output["models"]["POOLED"]
    for field in NUMERIC_FIELDS:
        if old[field] != new[field]:
            raise ValueError(f"MARKET_REFERENCE_NUMERIC_CHANGED:{field}")
    from app.schemas.direction_training import DirectionInput

    items = [DirectionInput.model_validate(r) for r in bundle["jobs"]["REFERENCE"]["exam"]]
    scores = predict_model(old, items)
    differences = [abs(a - b) for a, b in zip(scores, output["scores"], strict=True)]
    if max(differences, default=0) != 0:
        raise ValueError("MARKET_REFERENCE_SCORE_CHANGED")
    return {"status": "IDENTICAL_NUMERIC_MODEL_AND_SCORES", "scores": len(scores), "maximum_difference": 0}


def predict(folder):
    protocol, _, _ = load_plan(folder)
    prepared = read_seal(folder, "market-prepared.json")
    coverage = read_json(folder / "market-coverage.json")
    files, execution, parity, jobs_run = {}, {}, {}, 0
    started = perf_counter()
    previous = prior()
    for window in protocol["windows"]:
        name = window["name"]
        bundle, outputs = read_json(folder / f"market-prepared-{name}.json"), {}
        for branch in MARKET_BRANCHES:
            if coverage[name]["status"] != "READY":
                result = {"status": "INSUFFICIENT_DATA"}
            elif (
                jobs_run >= protocol["fit_budget"]["main_maximum"]
                or perf_counter() - started > protocol["budget"]["total_seconds"]
            ):
                result = {"status": "FAILED", "reason": "MARKET_TRAINING_BUDGET"}
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
        for filename, value, writer in (
            (f"market-models-{name}.json", outputs, write_json),
            (f"market-predictions-{name}.jsonl", window_predictions(bundle, outputs), write_jsonl),
        ):
            files[filename] = writer(folder / filename, value)
        execution[name] = {b: {k: v for k, v in o.items() if k not in ("models", "scores")} for b, o in outputs.items()}
    files["market-execution.json"] = write_json(folder / "market-execution.json", execution)
    files["market-reference-parity.json"] = write_json(folder / "market-reference-parity.json", parity)
    return seal(
        folder,
        "market-predicted.json",
        files,
        prepared_hash=prepared["manifest_hash"],
        model_fit_count=jobs_run,
        status="PREDICTIONS_SEALED_BEFORE_EXAM_ANSWERS",
    )


def all_predictions(folder, protocol):
    return [r for w in protocol["windows"] for r in read_jsonl(folder / f"market-predictions-{w['name']}.jsonl")]


def diagnostics(protocol, predictions, answers):
    answer_map = {(r["window"], r["sample_key"]): r["answer"] for r in answers}
    windows = {w["name"]: w for w in protocol["windows"]}
    branches = tuple(protocol["branches"])
    rows = {b: {} for b in branches}
    for p in predictions:
        a = answer_map[(p["window"], p["sample_key"])]
        if (
            p["branch"] in rows
            and p["score"] is not None
            and a is not None
            and a["available_at"] <= evaluation_asof(windows[p["window"]])
        ):
            rows[p["branch"]][(p["window"], p["sample_key"])] = {
                **p,
                "y": a["y"],
                "correct": int(p["predicted_up"] == a["y"]),
            }
    common = set.intersection(*(set(v) for v in rows.values()))
    blocks, _ = complete_blocks(
        protocol,
        common,
        planned_by_window={w["name"]: planned_dates(protocol["version"], w) for w in protocol["windows"]},
    )
    errors, changes = {}, {}
    for branch, values in rows.items():
        errors[branch] = {}
        for fund in FUNDS:
            group = [values[k] for k in sorted(common) if values[k]["fund"] == fund]
            errors[branch][fund] = {
                "count": len(group),
                "correct": sum(r["correct"] for r in group),
                "missed_up": sum(r["y"] == 1 and r["predicted_up"] == 0 for r in group),
                "false_up": sum(r["y"] == 0 and r["predicted_up"] == 1 for r in group),
            }
    for right in branches[:2]:
        changes[right] = {
            f: {
                "wrong_to_right": sum(
                    rows[right][k]["correct"] == 0 and rows[branches[2]][k]["correct"] == 1
                    for k in common
                    if rows[right][k]["fund"] == f
                ),
                "right_to_wrong": sum(
                    rows[right][k]["correct"] == 1 and rows[branches[2]][k]["correct"] == 0
                    for k in common
                    if rows[right][k]["fund"] == f
                ),
            }
            for f in FUNDS
        }
    cb = paired_interval(protocol, blocks, rows[branches[2]], rows[branches[1]])
    left = grouped_metrics([rows[branches[2]][k] for k in sorted(common)], FUNDS)
    right = grouped_metrics([rows[branches[1]][k] for k in sorted(common)], FUNDS)
    return {
        "same_questions": len(common),
        "errors": errors,
        "matched_error_transitions": changes,
        "C_vs_B": {"time_blocks": cb, "C": left, "B": right},
        "matching_stable_increment": bool(cb["interval"] and cb["interval"][0] > 1e-12),
        "interpretation": "DEVELOPMENT_COMPARISON_NOT_INDEPENDENT_VALIDATION",
    }


def score(folder):
    protocol, _, _ = load_plan(folder)
    predicted = read_seal(folder, "market-predicted.json")
    predictions = all_predictions(folder, protocol)
    # 此处才解析本轮EXAM答案；过去已观察题始终是开发资料，不能叫独立考试。
    previous = prior()
    answers = read_jsonl(previous / "linear-answers.jsonl")
    result = evaluate(protocol, predictions, answers)
    report = diagnostics(protocol, predictions, answers)
    old_metrics = read_json(previous / "linear-metrics.json")
    report["previous_full_A"] = old_metrics["common"]["REFERENCE"]
    report["previous_full_question_count"] = old_metrics["same_question_count"]
    files = {
        "market-answers.jsonl": write_jsonl(folder / "market-answers.jsonl", answers),
        "market-metrics.json": write_json(folder / "market-metrics.json", result),
        "market-diagnostics.json": write_json(folder / "market-diagnostics.json", report),
    }
    return seal(
        folder,
        "market-scored.json",
        files,
        predicted_hash=predicted["manifest_hash"],
        selected_candidate=result["selected_candidate"],
        independent_test=False,
        model_released=False,
    )


def decision(protocol, result):
    selected = result["selected_candidate"]
    return {
        "version": MARKET_VERSION,
        "selected_candidate": selected,
        "candidate_status": result["candidate_status"],
        "current_snapshot_search": "CLOSED",
        "automatic_followup_training": False,
        "model_released": False,
        "independent_test_run": False,
        "independent_plan": protocol["independent"],
        "independent_status": "AWAITING_INDEPENDENCE_AND_FIRST_VERSION_EVIDENCE"
        if selected
        else "NOT_RUN_NO_QUALIFIED_CANDIDATE",
    }


def finalize(folder):
    protocol, _, _ = load_plan(folder)
    scored = read_seal(folder, "market-scored.json")
    result = read_json(folder / "market-metrics.json")
    files = {"market-decision.json": write_json(folder / "market-decision.json", decision(protocol, result))}
    if result["selected_candidate"]:
        branch = result["selected_candidate"]
        files["market-frozen-candidate.json"] = write_json(
            folder / "market-frozen-candidate.json",
            {
                "branch": branch,
                "protocol_hash": digest(protocol),
                "scored_hash": scored["manifest_hash"],
                "models": {
                    w["name"]: read_json(folder / f"market-models-{w['name']}.json")[branch]["models"]
                    for w in protocol["windows"]
                },
                "release_authorized": False,
            },
        )
    return seal(
        folder,
        "market-complete.json",
        files,
        scored_hash=scored["manifest_hash"],
        status="FINITE_BATCH_COMPLETE",
        model_released=False,
        independent_test=False,
    )


def verify(folder):
    """只复算封存文件：不联网、不连库、不训练，任一阶段或数值改变即失败。"""
    protocol, mapping, frozen = load_plan(folder)
    stages = {
        "frozen": frozen,
        **{
            s: read_seal(folder, f"market-{s}.json")
            for s in ("acquired", "prepared", "predicted", "scored", "complete")
        },
    }
    names = list(stages)
    for left, right in zip(names[:-1], names[1:], strict=True):
        if stages[right][f"{left}_hash"] != stages[left]["manifest_hash"]:
            raise ValueError("MARKET_STAGE_CHAIN")
    previous, market = prior(), read_json(folder / "market-data.json")
    # 从年度原始响应重新拼接行情，核对汇总缓存，不只检验缓存自身的摘要。
    for code in market["prices"]:
        raw = (
            read_json(folder / "shared-source.json")["prices"]
            if code == mapping["shared_index"]
            else [
                row
                for year in data.YEARS
                for row in read_json(folder / f"market-response-{code}-{year}.json")["prices"]
            ]
        )
        rebuilt_prices, coverage = data.validate_prices(raw, code)
        if rebuilt_prices != market["prices"][code] or any(
            market["coverage"][code][k] != v for k, v in coverage.items()
        ):
            raise ValueError("MARKET_RAW_PRICE_REBUILD_CHANGED")
    previous_coverage = read_json(previous / "linear-coverage.json")
    expected_coverage, expected_parity = {}, {}
    for window in protocol["windows"]:
        name = window["name"]
        rebuilt, expected_coverage[name] = build_window(
            read_json(previous / f"linear-prepared-{name}.json"), mapping, market, protocol, previous_coverage[name]
        )
        if digest(rebuilt) != digest(read_json(folder / f"market-prepared-{name}.json")):
            raise ValueError("MARKET_INPUT_REBUILD_CHANGED")
        outputs = read_json(folder / f"market-models-{name}.json")
        for branch, output in outputs.items():
            if output["status"] == "PREDICTED":
                _, exam = validate_job(rebuilt["jobs"][branch])
                if predict_model(output["models"]["POOLED"], exam) != output["scores"]:
                    raise ValueError("MARKET_STORED_MODEL_SCORES_CHANGED")
        if window_predictions(rebuilt, outputs) != read_jsonl(folder / f"market-predictions-{name}.jsonl"):
            raise ValueError("MARKET_PREDICTIONS_CHANGED")
        expected_parity[name] = reference_parity(
            rebuilt, outputs["REFERENCE"], read_json(previous / f"linear-models-{name}.json"), expected_coverage[name]
        )
    if expected_coverage != read_json(folder / "market-coverage.json") or expected_parity != read_json(
        folder / "market-reference-parity.json"
    ):
        raise ValueError("MARKET_COVERAGE_OR_PARITY_CHANGED")
    predictions, answers = all_predictions(folder, protocol), read_jsonl(folder / "market-answers.jsonl")
    if answers != read_jsonl(previous / "linear-answers.jsonl"):
        raise ValueError("MARKET_EXAM_ANSWERS_CHANGED")
    result = evaluate(protocol, predictions, answers)
    if digest(result) != digest(read_json(folder / "market-metrics.json")) or decision(protocol, result) != read_json(
        folder / "market-decision.json"
    ):
        raise ValueError("MARKET_METRICS_OR_DECISION_CHANGED")
    report = diagnostics(protocol, predictions, answers)
    old_metrics = read_json(previous / "linear-metrics.json")
    report.update(
        previous_full_A=old_metrics["common"]["REFERENCE"],
        previous_full_question_count=old_metrics["same_question_count"],
    )
    if digest(report) != digest(read_json(folder / "market-diagnostics.json")):
        raise ValueError("MARKET_DIAGNOSTICS_CHANGED")
    return stages["complete"]


def replay(folder):
    verify(folder)
    protocol, _, _ = load_plan(folder)
    target = new_folder()
    for stage in ("frozen", "acquired", "prepared"):
        manifest = f"market-{stage}.json"
        for name in (manifest, *read_seal(folder, manifest)["files"]):
            if not (target / name).exists():
                shutil.copyfile(folder / name, target / name)
    write_json(
        target / "market-replay-origin.json",
        {"original": folder.name, "created_at": now(), "protocol_hash": digest(protocol)},
    )
    predict(target)
    score(target)
    finalize(target)
    verify(target)
    differences, model_identity = [], True
    for window in protocol["windows"]:
        a = read_json(folder / f"market-models-{window['name']}.json")
        b = read_json(target / f"market-models-{window['name']}.json")
        for branch in MARKET_BRANCHES:
            if a[branch]["status"] != b[branch]["status"]:
                raise ValueError("MARKET_REPLAY_JOB_STATUS_CHANGED")
            if a[branch]["status"] == "PREDICTED":
                differences.extend(abs(x - y) for x, y in zip(a[branch]["scores"], b[branch]["scores"], strict=True))
                model_identity &= a[branch]["models"] == b[branch]["models"]
    if max(differences, default=0) != 0 or not model_identity:
        raise ValueError("MARKET_REPLAY_CHANGED")
    receipt = {
        "run": target.name.removeprefix("direction-training-"),
        "original_run": folder.name,
        "scores": len(differences),
        "maximum_difference": max(differences, default=0),
        "models_identical": model_identity,
        "status": "REPRODUCED_NOT_NEW_EFFECT_EVIDENCE",
    }
    seal(
        target,
        "market-replay.json",
        {"market-replay-receipt.json": write_json(target / "market-replay-receipt.json", receipt)},
    )
    return receipt
