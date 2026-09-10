"""错误驱动的有限线性对照：冻结、训练、同题评价、条件验收、重放。"""

import json
import sys
from collections import Counter
from decimal import Decimal
from time import perf_counter
from uuid import UUID

from app.services.direction_linear_analysis import diagnose, evaluate
from app.services.direction_linear_models import validate_job
from app.services.direction_linear_protocol import (
    ABLATION_VERSION,
    BASELINES,
    RECENCY_VERSION,
    SOURCE_HASH,
    SOURCE_RUN,
    VERSION,
    specification_for,
)
from app.services.direction_training_artifacts import (
    digest,
    new_folder,
    read_json,
    read_jsonl,
    read_seal,
    run_folder,
    seal,
    write_json,
    write_jsonl,
)
from app.services.direction_training_dataset import FUNDS
from app.services.direction_training_process import run_process
from app.services.historical_nav_evaluation import fixed_momentum_score


def source():
    folder = run_folder(UUID(SOURCE_RUN))
    stages = {s: read_seal(folder, f"nav-{s}.json") for s in ("frozen", "prepared", "predicted", "scored")}
    if stages["scored"]["manifest_hash"] != SOURCE_HASH:
        raise ValueError("LINEAR_SOURCE_CHANGED")
    for current, previous in (("prepared", "frozen"), ("predicted", "prepared"), ("scored", "predicted")):
        if stages[current][f"{previous}_hash"] != stages[previous]["manifest_hash"]:
            raise ValueError("LINEAR_SOURCE_CHAIN")
    return folder


def freeze(version=VERSION):
    protocol = specification_for(version)
    original = source()
    diagnosis = diagnose(original)  # 已观察的开发集，用于提出假设，不能称为独立数据。
    folder, files = new_folder(), {}
    if version == ABLATION_VERSION:
        from app.services.direction_linear_evidence import independence_review, prior_evidence

        prior = prior_evidence()
        diagnosis = {
            **diagnosis,
            "version": version,
            "hypotheses": {b: text for b, text in protocol["changes"].items() if b != "REFERENCE"},
        }
        files["linear-hypothesis-evidence.json"] = write_json(folder / "linear-hypothesis-evidence.json", prior)
        files["linear-independence-review.json"] = write_json(
            folder / "linear-independence-review.json", independence_review(prior)
        )
        protocol["evidence_hashes"] = dict(files)
    if version == RECENCY_VERSION:
        from app.services.direction_linear_evidence import independence_review

        diagnosis = {
            **diagnosis,
            "version": version,
            "hypotheses": {b: text for b, text in protocol["changes"].items() if b != "REFERENCE"},
        }
        files["linear-independence-review.json"] = write_json(
            folder / "linear-independence-review.json",
            independence_review({"independent_plan": protocol["independent"]}),
        )
        protocol["evidence_hashes"] = dict(files)
    files["linear-diagnosis.json"] = write_json(folder / "linear-diagnosis.json", diagnosis)
    files["linear-plan.json"] = write_json(
        folder / "linear-plan.json",
        {
            **protocol,
            "diagnosis_hash": files["linear-diagnosis.json"],
        },
    )
    return folder, seal(folder, "linear-frozen.json", files, status="FROZEN", source_scored_hash=SOURCE_HASH)


def load_plan(folder):
    frozen = read_seal(folder, "linear-frozen.json")
    protocol = read_json(folder / "linear-plan.json")
    if any(protocol.get(k) != v for k, v in specification_for(protocol["version"]).items()):
        raise ValueError("LINEAR_PROTOCOL_CODE_OR_RUNTIME_CHANGED")
    if (
        frozen["source_scored_hash"] != SOURCE_HASH
        or protocol["diagnosis_hash"] != frozen["files"]["linear-diagnosis.json"]
    ):
        raise ValueError("LINEAR_PROTOCOL_LINK")
    if protocol["version"] == ABLATION_VERSION:
        expected = {
            n: frozen["files"][n] for n in ("linear-hypothesis-evidence.json", "linear-independence-review.json")
        }
        if protocol.get("evidence_hashes") != expected:
            raise ValueError("LINEAR_EVIDENCE_LINK")
    if protocol["version"] == RECENCY_VERSION and protocol.get("evidence_hashes") != {
        "linear-independence-review.json": frozen["files"]["linear-independence-review.json"]
    }:
        raise ValueError("LINEAR_EVIDENCE_LINK")
    return protocol, frozen


def make_job(bundle, branch, version=VERSION):
    base = bundle["complete"]
    if not base:
        return None
    return {
        "version": version,
        "window": bundle["window"],
        "branch": branch,
        "fit": bundle["recent_fit"][branch] if version == RECENCY_VERSION and branch != "REFERENCE" else base["fit"],
        "exam": base["exam"]["CLEAN"],
    }


def prepare(folder):
    protocol, frozen = load_plan(folder)
    original, files, coverage = source(), {}, {}
    old_coverage = read_json(original / "nav-coverage.json")
    recency_sources = None
    if protocol["version"] == RECENCY_VERSION:
        from app.services.direction_linear_recency import extend_fit, load_sources

        recency_sources = load_sources(original)
    for window in protocol["windows"]:
        name = window["name"]
        old = read_json(original / f"nav-prepared-{name}.json")
        bundle = {"window": window, "planned": old["planned"], "complete": old["jobs"].get("COMPLETE")}
        coverage[name] = {"source": old_coverage["windows"][name]["CLEAN"], "branches": {}}
        if recency_sources is not None and bundle["complete"]:
            bundle["recent_fit"], coverage[name]["training_audit"] = extend_fit(bundle, recency_sources)
        for branch in protocol["branches"]:
            job = make_job(bundle, branch, protocol["version"])
            if job is None:
                coverage[name]["branches"][branch] = {"status": "INSUFFICIENT_DATA"}
                continue
            try:
                fit, exam = validate_job(job)
                coverage[name]["branches"][branch] = {
                    "status": "READY",
                    "train_counts": dict(Counter(i.fund for i, a in fit)),
                    "exam_inputs": len(exam),
                    "fit_keys_hash": digest([i.key for i, a in fit]),
                    "exam_keys_hash": digest([i.key for i in exam]),
                }
            except ValueError as exc:
                coverage[name]["branches"][branch] = {"status": "FAILED", "reason": str(exc)[:240]}
        if protocol["version"] == ABLATION_VERSION:
            ready = [v for v in coverage[name]["branches"].values() if v["status"] == "READY"]
            if len({v["fit_keys_hash"] for v in ready}) > 1 or len({v["exam_keys_hash"] for v in ready}) > 1:
                raise ValueError("ABLATION_SAMPLE_KEYS_CHANGED")
        if protocol["version"] == RECENCY_VERSION:
            ready = [v for v in coverage[name]["branches"].values() if v["status"] == "READY"]
            if len({v["exam_keys_hash"] for v in ready}) > 1:
                raise ValueError("RECENCY_EXAM_KEYS_CHANGED")
        filename = f"linear-prepared-{name}.json"
        files[filename] = write_json(folder / filename, bundle)
    files["linear-coverage.json"] = write_json(folder / "linear-coverage.json", coverage)
    return seal(
        folder,
        "linear-prepared.json",
        files,
        status="PREPARED",
        frozen_hash=frozen["manifest_hash"],
        worker_exam_answers=False,
        previously_observed_development=True,
        database_connected=False,
    )


def predict(folder):
    protocol, frozen = load_plan(folder)
    prepared = read_seal(folder, "linear-prepared.json")
    if prepared["frozen_hash"] != frozen["manifest_hash"]:
        raise ValueError("LINEAR_STAGE_CHAIN")
    coverage = read_json(folder / "linear-coverage.json")
    files, executions, predictions, parity = {}, {}, [], []
    original, started, jobs_run = source(), perf_counter(), 0
    for window in protocol["windows"]:
        name, outputs, index = window["name"], {}, {}
        bundle = read_json(folder / f"linear-prepared-{name}.json")
        for branch in protocol["branches"]:
            readiness = coverage[name]["branches"][branch]
            if readiness["status"] != "READY":
                outputs[branch] = readiness
                continue
            if perf_counter() - started > protocol["budget"]["total_seconds"]:
                output = {"status": "FAILED", "reason": "EXPERIMENT_TIME_BUDGET"}
            elif jobs_run >= protocol.get("research_fit_budget", {}).get("maximum_jobs", 10000):
                output = {"status": "FAILED", "reason": "EXPERIMENT_JOB_BUDGET"}
            else:
                job = make_job(bundle, branch, protocol["version"])
                jobs_run += 1
                try:
                    output = run_process(
                        job,
                        seconds=protocol["budget"]["candidate_seconds"],
                        memory_bytes=protocol["budget"]["process_memory_bytes"],
                        command=[sys.executable, "-m", "scripts.direction_linear_worker"],
                    )
                except (ValueError, OSError) as exc:
                    output = {"status": "FAILED", "reason": str(exc)[:240], "error_type": type(exc).__name__}
            outputs[branch] = output
            if output.get("status") == "PREDICTED":
                for item, value in zip(job["exam"], output["scores"], strict=True):
                    index[(branch, item["fund"], item["cutoff"])] = (value, item["input_hash"])
            print(
                json.dumps({"stage": "TRAIN", "window": name, "branch": branch, "status": output.get("status")}),
                flush=True,
            )
        if bundle["complete"]:
            fit = bundle["complete"]["fit"]
            rates = {
                f: sum(r["answer"]["y"] for r in fit if r["input"]["fund"] == f)
                / sum(r["input"]["fund"] == f for r in fit)
                for f in FUNDS
            }
            for item in bundle["complete"]["exam"]["CLEAN"]:
                scores = {
                    "ALWAYS_UP": 1.0,
                    "ALWAYS_NON_UP": 0.0,
                    "TRAIN_UP_FREQUENCY": rates[item["fund"]],
                    "MOMENTUM_20D": float(item["x"][1] > 0),
                    "FIXED_MOMENTUM_SCORE": float(fixed_momentum_score(Decimal(str(item["x"][1])))),
                }
                for branch, value in scores.items():
                    index[(branch, item["fund"], item["cutoff"])] = (value, item["input_hash"])
            old = {
                r["sample_key"]: r["score"]
                for r in read_jsonl(original / f"nav-predictions-{name}.jsonl")
                if r["scenario"] == "COMPLETE_CLEAN" and r["score"] is not None
            }
            if outputs["REFERENCE"].get("status") == "PREDICTED":
                differences = [
                    abs(value - old[f"{fund}:{cutoff}"])
                    for (branch, fund, cutoff), (value, _) in index.items()
                    if branch == "REFERENCE"
                ]
                if len(differences) != len(old) or max(differences, default=0) > 1e-12:
                    raise ValueError("LINEAR_REFERENCE_NOT_REPRODUCED")
                parity.append(
                    {"window": name, "count": len(differences), "max_difference": max(differences, default=0)}
                )
        for branch in (*protocol["branches"], *BASELINES):
            for fund in FUNDS:
                for cutoff in bundle["planned"]:
                    value, input_hash = index.get((branch, fund, cutoff), (None, None))
                    status = (
                        "PREDICTED"
                        if value is not None
                        else (
                            outputs.get(branch, {}).get("status", "INSUFFICIENT_DATA")
                            if not bundle["complete"] or outputs.get(branch, {}).get("status") == "FAILED"
                            else "INPUT_UNAVAILABLE"
                        )
                    )
                    predictions.append(
                        {
                            "window": name,
                            "branch": branch,
                            "fund": fund,
                            "cutoff": cutoff,
                            "sample_key": f"{fund}:{cutoff}",
                            "score": value,
                            "predicted_up": int(value > 0.5) if value is not None else None,
                            "input_hash": input_hash,
                            "status": status,
                        }
                    )
        filename = f"linear-models-{name}.json"
        files[filename] = write_json(folder / filename, outputs)
        executions[name] = {
            b: {k: v for k, v in o.items() if k not in ("models", "scores")} for b, o in outputs.items()
        }
    for filename, value, writer in (
        ("linear-predictions.jsonl", predictions, write_jsonl),
        ("linear-execution.json", executions, write_json),
        ("linear-reference-parity.json", parity, write_json),
    ):
        files[filename] = writer(folder / filename, value)
    return seal(
        folder,
        "linear-predicted.json",
        files,
        status="PREDICTED",
        prepared_hash=prepared["manifest_hash"],
        model_fit_count=sum(o.get("model_fit_count", 0) for rows in executions.values() for o in rows.values()),
        worker_exam_answers=False,
    )


def score(folder):
    protocol, _ = load_plan(folder)
    predicted = read_seal(folder, "linear-predicted.json")  # 全部预测先封存，再生成本轮评分产物。
    original = source()
    answers = [
        {k: r[k] for k in ("window", "fund", "cutoff", "sample_key", "answer", "issues")}
        for r in read_jsonl(original / "nav-answers.jsonl")
    ]
    files = {"linear-answers.jsonl": write_jsonl(folder / "linear-answers.jsonl", answers)}
    result = evaluate(protocol, read_jsonl(folder / "linear-predictions.jsonl"), answers)
    files["linear-metrics.json"] = write_json(folder / "linear-metrics.json", result)
    if protocol["version"] == RECENCY_VERSION:
        from app.services.direction_linear_recency import diagnostic_report

        files["linear-time-diagnosis.json"] = write_json(
            folder / "linear-time-diagnosis.json", diagnostic_report(folder, original, protocol, answers)
        )
    return seal(
        folder,
        "linear-scored.json",
        files,
        status="SCORED",
        predicted_hash=predicted["manifest_hash"],
        selected_candidate=result["selected_candidate"],
        independent_test=False,
        model_released=False,
    )


def decision(protocol, result, scored_hash):
    candidate = result["selected_candidate"]
    return {
        "version": protocol["version"],
        "development_scored_hash": scored_hash,
        "selected_candidate": candidate,
        "candidate_status": result["candidate_status"],
        "candidate_rule_frozen": candidate is not None,
        "independent_status": "AWAITING_INDEPENDENCE_EVIDENCE_OR_FUTURE_OBSERVATION"
        if candidate
        else "NOT_RUN_NO_QUALIFIED_CANDIDATE",
        "independent_plan": protocol["independent"],
        "independent_test_run": False,
        "model_released": False,
        "reference_retained_for_research": "REFERENCE",
        "historical_2025_values_read": False,
        "future_predictions_created": 0,
    }


def finalize(folder):
    protocol, _ = load_plan(folder)
    scored = read_seal(folder, "linear-scored.json")
    result = read_json(folder / "linear-metrics.json")
    value = decision(protocol, result, scored["manifest_hash"])
    files = {"linear-decision.json": write_json(folder / "linear-decision.json", value)}
    if result["selected_candidate"]:
        branch = result["selected_candidate"]
        # 冻结候选规则及所有开发期拟合产物；不把某一历史窗口模型冒充当前可用模型。
        snapshot = {
            "branch": branch,
            "protocol": protocol,
            "development_scored_hash": scored["manifest_hash"],
            "historical_models": {
                w["name"]: read_json(folder / f"linear-models-{w['name']}.json")[branch] for w in protocol["windows"]
            },
        }
        files["linear-candidate.json"] = write_json(folder / "linear-candidate.json", snapshot)
    return seal(
        folder,
        "linear-complete.json",
        files,
        status="FINITE_BATCH_COMPLETE",
        scored_hash=scored["manifest_hash"],
        independent_status=value["independent_status"],
        model_released=False,
    )


def verify(folder):
    protocol, frozen = load_plan(folder)
    stages = {
        "frozen": frozen,
        **{s: read_seal(folder, f"linear-{s}.json") for s in ("prepared", "predicted", "scored", "complete")},
    }
    for current, previous in (
        ("prepared", "frozen"),
        ("predicted", "prepared"),
        ("scored", "predicted"),
        ("complete", "scored"),
    ):
        if stages[current][f"{previous}_hash"] != stages[previous]["manifest_hash"]:
            raise ValueError("LINEAR_STAGE_CHAIN")
    result = evaluate(
        protocol, read_jsonl(folder / "linear-predictions.jsonl"), read_jsonl(folder / "linear-answers.jsonl")
    )
    if digest(result) != digest(read_json(folder / "linear-metrics.json")):
        raise ValueError("LINEAR_METRICS_CHANGED")
    if decision(protocol, result, stages["scored"]["manifest_hash"]) != read_json(folder / "linear-decision.json"):
        raise ValueError("LINEAR_DECISION_CHANGED")
    if protocol["version"] == ABLATION_VERSION:
        from app.services.direction_linear_evidence import independence_review, prior_evidence

        prior = prior_evidence()
        if prior != read_json(folder / "linear-hypothesis-evidence.json"):
            raise ValueError("LINEAR_PRIOR_EVIDENCE_CHANGED")
        if independence_review(prior) != read_json(folder / "linear-independence-review.json"):
            raise ValueError("LINEAR_INDEPENDENCE_REVIEW_CHANGED")
    if protocol["version"] == RECENCY_VERSION:
        from app.services.direction_linear_evidence import independence_review
        from app.services.direction_linear_recency import diagnostic_report

        if independence_review({"independent_plan": protocol["independent"]}) != read_json(
            folder / "linear-independence-review.json"
        ):
            raise ValueError("LINEAR_INDEPENDENCE_REVIEW_CHANGED")
        report = diagnostic_report(folder, source(), protocol, read_jsonl(folder / "linear-answers.jsonl"))
        if report != read_json(folder / "linear-time-diagnosis.json"):
            raise ValueError("RECENCY_DIAGNOSTICS_CHANGED")
    source()
    return stages["complete"]


def replay(folder):
    complete = verify(folder)
    protocol, _ = load_plan(folder)
    target, rows, started = new_folder(), [], perf_counter()
    for window in protocol["windows"]:
        name = window["name"]
        bundle = read_json(folder / f"linear-prepared-{name}.json")
        outputs = read_json(folder / f"linear-models-{name}.json")
        for branch in protocol["branches"]:
            expected = outputs[branch]
            if expected.get("status") != "PREDICTED":
                rows.append(
                    {"window": name, "branch": branch, "status": "NOT_REPLAYED", "original_status": expected["status"]}
                )
                continue
            if perf_counter() - started > protocol["budget"]["total_seconds"]:
                raise TimeoutError("LINEAR_REPLAY_BUDGET")
            output = run_process(
                make_job(bundle, branch, protocol["version"]),
                command=[sys.executable, "-m", "scripts.direction_linear_worker"],
            )
            if output.get("status") != "PREDICTED":
                raise ValueError("LINEAR_REPLAY_STATUS")
            differences = [abs(a - b) for a, b in zip(output["scores"], expected["scores"], strict=True)]
            flips = sum((a > 0.5) != (b > 0.5) for a, b in zip(output["scores"], expected["scores"], strict=True))
            identical = output["models"] == expected["models"]
            if max(differences, default=0) > 1e-12 or flips or not identical:
                raise ValueError("LINEAR_REPLAY_MISMATCH")
            rows.append(
                {
                    "window": name,
                    "branch": branch,
                    "status": "REPLAY_VERIFIED",
                    "count": len(differences),
                    "max_difference": max(differences, default=0),
                    "direction_changes": flips,
                    "artifacts_identical": identical,
                    "model_fit_count": output["model_fit_count"],
                }
            )
    seal(
        target,
        "linear-replay.json",
        {"replay.json": write_json(target / "replay.json", rows)},
        status="REPLAY_VERIFIED",
        original_run=folder.name,
        original_complete_hash=complete["manifest_hash"],
    )
    return target
