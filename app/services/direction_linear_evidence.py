"""逐项删减的已观察依据与独立性条件核查，仅读取已封存研究及元数据。"""

from uuid import UUID

from app.services.direction_linear_protocol import (
    ABLATION_VERSION,
    PRIOR_HASH,
    PRIOR_RUN,
    RECENCY_HASH,
    RECENCY_RUN,
    REGULARIZATION_VERSION,
)
from app.services.direction_training_artifacts import read_json, read_seal, run_folder

P1_RUN = "feff6919-beca-437a-9b44-5478f61aad44"
P1_HASH = "8934dd2ba0fc610a5cfd45ad4a99d7d86c55abb0cce98a046289c83ed7df0c1c"


def prior_evidence():
    folder = run_folder(UUID(PRIOR_RUN))
    stages = {
        s: read_seal(folder, f"linear-{s}.json") for s in ("frozen", "prepared", "predicted", "scored", "complete")
    }
    if stages["complete"]["manifest_hash"] != PRIOR_HASH:
        raise ValueError("ABLATION_PRIOR_CHANGED")
    for current, previous in (
        ("prepared", "frozen"),
        ("predicted", "prepared"),
        ("scored", "predicted"),
        ("complete", "scored"),
    ):
        if stages[current][f"{previous}_hash"] != stages[previous]["manifest_hash"]:
            raise ValueError("ABLATION_PRIOR_CHAIN")
    metrics = read_json(folder / "linear-metrics.json")
    return {
        "version": ABLATION_VERSION,
        "prior_run": PRIOR_RUN,
        "prior_complete_hash": PRIOR_HASH,
        "status": "OBSERVED_DEVELOPMENT_HYPOTHESIS_ONLY",
        "same_question_count": metrics["same_question_count"],
        "previous_reference": metrics["common"]["REFERENCE"],
        "previous_group_removal": metrics["common"]["DROP_60D_GROUP"],
        "previous_candidate_status": metrics["candidate_status"],
        "independent_plan": read_json(folder / "linear-plan.json")["independent"],
    }


def independence_review(prior):
    folder = run_folder(UUID(P1_RUN))
    completed = read_seal(folder, "study-complete.json")
    if completed["manifest_hash"] != P1_HASH:
        raise ValueError("ABLATION_INDEPENDENCE_SOURCE_CHANGED")
    plan = read_json(folder / "independent-test-plan.json")
    return {
        "version": "DIRECTION_INDEPENDENCE_REVIEW_V2",
        "evidence_scope": "NAMED_SEALED_RUN_METADATA_ONLY_NOT_PROJECT_WIDE_ACCESS_ATTESTATION",
        "p1_run": P1_RUN,
        "p1_complete_hash": P1_HASH,
        "planned_historical_period": plan["planned_period"],
        "p1_independence_evidence": plan["independence_evidence"],
        "p1_source_versions_verified": plan["source_versions_verified"],
        "latest_historical_independence_attested": prior["independent_plan"]["historical_independence_attested"],
        "latest_source_first_versions_verified": prior["independent_plan"]["source_first_versions_verified"],
        "historical_test_status": "NOT_ESTABLISHED_AS_INDEPENDENT",
        "development_period_status": "2021_2024_OBSERVED_NOT_ELIGIBLE_AS_FINAL_TEST",
        "future_alternative": "FREEZE_QUALIFIED_CANDIDATE_THEN_RECORD_ACTUALLY_AVAILABLE_INPUTS_AND_PREDICTIONS",
        "future_start_date": None,
        "future_start_status": "AFTER_CANDIDATE_AND_LIVE_SOURCE_READINESS",
        "future_cutoffs_per_fund": 126,
        "horizon_sessions": 20,
        "require_labels_actually_available": True,
        "read_2025_values": False,
        "scheduler_created": False,
        "independence_passed": False,
    }


def regularization_evidence():
    folder = run_folder(UUID(RECENCY_RUN))
    stages = {
        s: read_seal(folder, f"linear-{s}.json") for s in ("frozen", "prepared", "predicted", "scored", "complete")
    }
    if stages["complete"]["manifest_hash"] != RECENCY_HASH:
        raise ValueError("REGULARIZATION_PRIOR_CHANGED")
    for current, previous in (
        ("prepared", "frozen"),
        ("predicted", "prepared"),
        ("scored", "predicted"),
        ("complete", "scored"),
    ):
        if stages[current][f"{previous}_hash"] != stages[previous]["manifest_hash"]:
            raise ValueError("REGULARIZATION_PRIOR_CHAIN")
    diagnosis = read_json(folder / "linear-time-diagnosis.json")
    metrics = read_json(folder / "linear-metrics.json")
    return {
        "version": REGULARIZATION_VERSION,
        "prior_run": RECENCY_RUN,
        "prior_complete_hash": RECENCY_HASH,
        "status": "OBSERVED_DEVELOPMENT_HYPOTHESIS_ONLY_NOT_PROVEN_OVERFITTING",
        "previous_common": metrics["common"],
        "previous_candidate_status": metrics["candidate_status"],
        "reference_temporal_diagnosis": {
            w: d["metrics"]["REFERENCE"] for w, d in diagnosis["windows"].items() if d["status"] == "DIAGNOSED"
        },
        "prior_unique_audited_inputs": diagnosis["unique_audited_input_count"],
        "prior_source_value_status": diagnosis["source_value_status"],
        "independent_plan": read_json(folder / "linear-plan.json")["independent"],
    }
