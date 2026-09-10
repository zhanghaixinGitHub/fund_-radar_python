"""四格有限对照：封存既有线索、核对旧组数值、解释新增组合的配对变化。"""

from collections import Counter
from uuid import UUID

from app.services.direction_linear_analysis import accuracy_delta, regularization_diagnostics
from app.services.direction_linear_evidence import prior_evidence
from app.services.direction_linear_protocol import (
    COMBINATION_VERSION,
    PRIOR_RUN,
    REGULARIZATION_HASH,
    REGULARIZATION_RUN,
)
from app.services.direction_training_artifacts import file_hash, read_json, read_jsonl, read_seal, run_folder
from app.services.direction_training_dataset import FUNDS
from app.services.direction_training_evaluation import complete_blocks, paired_interval

NEW_BRANCH = "DROP_60D_GROUP_L2"
CONTROLS = {"REFERENCE": REGULARIZATION_RUN, "DROP_60D_GROUP": PRIOR_RUN, "L2_STRONGER": REGULARIZATION_RUN}


def combination_evidence():
    removal = prior_evidence()
    folder = run_folder(UUID(REGULARIZATION_RUN))
    stages = {
        s: read_seal(folder, f"linear-{s}.json") for s in ("frozen", "prepared", "predicted", "scored", "complete")
    }
    if stages["complete"]["manifest_hash"] != REGULARIZATION_HASH:
        raise ValueError("COMBINATION_PRIOR_CHANGED")
    for previous, current in zip(tuple(stages)[:-1], tuple(stages)[1:], strict=True):
        if stages[current][f"{previous}_hash"] != stages[previous]["manifest_hash"]:
            raise ValueError("COMBINATION_PRIOR_CHAIN")
    metrics = read_json(folder / "linear-metrics.json")
    return {
        "version": COMBINATION_VERSION,
        "status": "TWO_OBSERVED_DEVELOPMENT_CLUES_NOT_INDEPENDENT_OR_ASSUMED_ADDITIVE",
        "group_removal_evidence": removal,
        "regularization_run": REGULARIZATION_RUN,
        "regularization_complete_hash": REGULARIZATION_HASH,
        "regularization_common": metrics["common"],
        "regularization_candidate_status": metrics["candidate_status"],
        "independent_plan": read_json(folder / "linear-plan.json")["independent"],
    }


def control_parity(folder, protocol):
    """拟合后、评分前核对三组对照；比较数值而非含版本名的模型摘要。"""
    if protocol["version"] != COMBINATION_VERSION:
        raise ValueError("COMBINATION_VERSION")
    rows = []
    for window in protocol["windows"]:
        name = window["name"]
        outputs = read_json(folder / f"linear-models-{name}.json")
        for branch, run_id in CONTROLS.items():
            prior = run_folder(UUID(run_id))
            if file_hash(folder / f"linear-prepared-{name}.json") != file_hash(prior / f"linear-prepared-{name}.json"):
                raise ValueError("COMBINATION_SAMPLE_CONTENT_CHANGED")
            current = outputs[branch]
            previous = read_json(prior / f"linear-models-{name}.json")[branch]
            if current["status"] != "PREDICTED":
                rows.append({"window": name, "branch": branch, "status": current["status"]})
                continue
            if previous["status"] != "PREDICTED":
                raise ValueError("COMBINATION_CONTROL_STATUS_CHANGED")
            model, old = current["models"]["POOLED"], previous["models"]["POOLED"]
            fields = set(old) - {"version", "hash"}
            if any(model[k] != old[k] for k in fields):
                raise ValueError("COMBINATION_CONTROL_MODEL_CHANGED")
            differences = [abs(a - b) for a, b in zip(current["scores"], previous["scores"], strict=True)]
            flips = sum((a > 0.5) != (b > 0.5) for a, b in zip(current["scores"], previous["scores"], strict=True))
            if max(differences, default=0) > 1e-12 or flips:
                raise ValueError("COMBINATION_CONTROL_PREDICTIONS_CHANGED")
            rows.append(
                {
                    "window": name,
                    "branch": branch,
                    "source_run": run_id,
                    "status": "VERIFIED",
                    "count": len(differences),
                    "max_difference": max(differences, default=0),
                    "direction_changes": flips,
                    "model_values_identical": True,
                }
            )
    return rows


def effect_summary(groups):
    """完整四格的描述性差值，不将组合收益假定为两个单项之和。"""
    effects = {
        "feature_effect_at_C1": accuracy_delta(groups["DROP_60D_GROUP"], groups["REFERENCE"]),
        "feature_effect_at_C001": accuracy_delta(groups[NEW_BRANCH], groups["L2_STRONGER"]),
        "penalty_effect_with_seven_features": accuracy_delta(groups["L2_STRONGER"], groups["REFERENCE"]),
        "penalty_effect_with_four_features": accuracy_delta(groups[NEW_BRANCH], groups["DROP_60D_GROUP"]),
    }
    a, b = effects["feature_effect_at_C001"], effects["feature_effect_at_C1"]
    return {**effects, "accuracy_interaction": a - b if a is not None and b is not None else None}


def combination_diagnostics(folder, protocol, answers, result):
    diagnosis = regularization_diagnostics(folder, protocol, answers)
    answer_map = {(r["window"], r["sample_key"]): r["answer"] for r in answers}
    exam_ends = {w["name"]: w["exam_end"] for w in protocol["windows"]}
    available = {b: {} for b in protocol["branches"]}
    for row in read_jsonl(folder / "linear-predictions.jsonl"):
        key = row["window"], row["sample_key"]
        answer = answer_map[key]
        if (
            row["branch"] in available
            and row["status"] == "PREDICTED"
            and answer
            and answer["available_at"] <= exam_ends[row["window"]]
        ):
            available[row["branch"]][key] = {
                **row,
                "y": answer["y"],
                "correct": int(row["predicted_up"] == answer["y"]),
            }
    common = set.intersection(*(set(rows) for rows in available.values()))
    # 简单对照共享输入；以主评价题键数作交叉检查，失败组导致的空交集照实保留。
    if len(common) != result["same_question_count"]:
        raise ValueError("COMBINATION_DIAGNOSTIC_POPULATION")
    blocks, _ = complete_blocks(protocol, common)
    paired, transitions = {}, {}
    for reference in CONTROLS:
        paired[reference] = {
            "accuracy_delta": accuracy_delta(result["common"][NEW_BRANCH], result["common"][reference]),
            "time_blocks": paired_interval(protocol, blocks, available[NEW_BRANCH], available[reference]),
        }
        counts = Counter()
        for key in sorted(common):
            old, new = available[reference][key], available[NEW_BRANCH][key]
            kind = (
                ("WRONG_TO_CORRECT" if new["correct"] else "CORRECT_TO_WRONG")
                if old["correct"] != new["correct"]
                else "UNCHANGED_CORRECT"
                if new["correct"]
                else "UNCHANGED_WRONG"
            )
            counts[(new["window"], new["fund"], kind)] += 1
        transitions[reference] = [
            {"window": w, "fund": f, "transition": k, "count": n} for (w, f, k), n in sorted(counts.items())
        ]
    return {
        **diagnosis,
        "purpose": "FIXED_FOUR_CELL_PAIRED_EFFECTS_NOT_CAUSAL_PROOF_OR_NEW_SELECTION_GATES",
        "same_question_count": len(common),
        "overall_effects": effect_summary(result["common"]),
        "per_fund_effects": {
            f: effect_summary({b: {"per_fund": {f: g["per_fund"][f]}} for b, g in result["common"].items()})
            for f in FUNDS
        },
        "per_window_effects": {w: effect_summary(groups) for w, groups in result["per_window"].items()},
        "new_combination_paired_comparisons": paired,
        "error_transitions": transitions,
        "search_closed": True,
    }
