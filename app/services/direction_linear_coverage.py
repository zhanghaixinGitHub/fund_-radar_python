"""固定模型扩大季度覆盖；季度预测期与事后标签成熟期分开，保留旧证据。"""

from collections import Counter
from datetime import date
from uuid import UUID

from app.services import direction_nav_data as data
from app.services.direction_linear_analysis import regularization_diagnostics
from app.services.direction_linear_models import validate_job
from app.services.direction_linear_protocol import (
    COMBINATION_HASH,
    COMBINATION_RUN,
    COVERAGE_VERSION,
    SOURCE_RUN,
    evaluation_asof,
    planned_dates,
)
from app.services.direction_linear_recency import audit_answer, audit_input, load_sources
from app.services.direction_training_artifacts import (
    digest,
    read_json,
    read_jsonl,
    read_seal,
    run_folder,
    seal,
    write_json,
)
from app.services.direction_training_dataset import FUNDS
from app.services.direction_training_evaluation import grouped_metrics
from app.services.trading_calendar import load_calendar


def prior_study():
    folder = run_folder(UUID(COMBINATION_RUN))
    stages = {
        s: read_seal(folder, f"linear-{s}.json") for s in ("frozen", "prepared", "predicted", "scored", "complete")
    }
    if stages["complete"]["manifest_hash"] != COMBINATION_HASH:
        raise ValueError("COVERAGE_PRIOR_CHANGED")
    for previous, current in zip(tuple(stages)[:-1], tuple(stages)[1:], strict=True):
        if stages[current][f"{previous}_hash"] != stages[previous]["manifest_hash"]:
            raise ValueError("COVERAGE_PRIOR_CHAIN")
    return folder


def evidence(protocol):
    prior = prior_study()
    previous = read_json(prior / "linear-plan.json")
    metrics = read_json(prior / "linear-metrics.json")
    return {
        "version": COVERAGE_VERSION,
        "prior_run": COMBINATION_RUN,
        "prior_complete_hash": COMBINATION_HASH,
        "locked_branches": {
            b: {"indices": previous["feature_indices"][b], "C": previous["C_by_branch"][b]}
            for b in protocol["branches"]
        },
        "previous_common": {b: metrics["common"][b] for b in protocol["branches"]},
        "previous_same_question_count": metrics["same_question_count"],
        "previous_selected_candidate": metrics["selected_candidate"],
        "interpretation": "BROADEN_OBSERVED_DEVELOPMENT_COVERAGE_NOT_NEW_INDEPENDENT_PERIOD",
        "independent_plan": protocol["independent"],
    }


def select_records(records, window, version=COVERAGE_VERSION):
    planned = [str(d) for d in planned_dates(version, window)]
    fit, exam, gap, failures = [], [], [], Counter()
    for cutoff, record in sorted(records.items()):
        item, label = record["inputs"]["CLEAN"], record["label"]
        if item is not None and label is not None and label["available_at"] <= window["fit_end"]:
            fit.append(record)
        if (
            window["fit_end"] < cutoff <= window["cal_end"]
            and item is not None
            and label is not None
            and label["available_at"] <= window["cal_end"]
        ):
            gap.append(record)
        if cutoff in planned:
            if item is not None:
                exam.append(record)
            else:
                failures.update(record["input_issues"]["CLEAN"])
            if label is None:
                failures.update(record["label_issues"])
            elif label["available_at"] > evaluation_asof(window):
                failures["ANSWER_NOT_MATURE_AT_EVALUATION"] += 1
    valid = [r for r in exam if r["label"] is not None and r["label"]["available_at"] <= evaluation_asof(window)]
    return (
        fit,
        exam,
        {
            "planned": len(planned),
            "input_count": len(exam),
            "scorable_count": len(valid),
            "fit_count": len(fit),
            "unused_gap_mature_count": len(gap),
            "coverage": len(valid) / len(planned),
            "issues": dict(failures),
            "prediction_start": planned[0],
            "prediction_last": planned[-1],
            "label_asof": evaluation_asof(window),
            "quarter_cutoffs_excluded_at_development_end": sum(
                window["cal_end"] < str(d) <= window["exam_end"] for d in load_calendar().sessions
            )
            - len(planned),
        },
    )


def prepare(folder, protocol, frozen, source):
    sources, files, coverage, audited_inputs, answer_cache = load_sources(source), {}, {}, set(), {}
    prior = prior_study()
    old_windows = {w["cal_end"]: w for w in read_json(prior / "linear-plan.json")["windows"]}
    for window in protocol["windows"]:
        name, fit, exam, details = window["name"], [], [], {}
        for fund in FUNDS:
            nav, events, records = sources[fund]
            fit_rows, exam_rows, details[fund] = select_records(records, window)
            for record in (*fit_rows, *exam_rows):
                item = record["inputs"]["CLEAN"]
                key = fund, record["cutoff"]
                if key not in audited_inputs:
                    audit_input(item, record, nav, events)
                    audited_inputs.add(key)
            for record in fit_rows:
                key = fund, record["cutoff"]
                if key not in answer_cache:
                    answer, issues = data.build_answer(
                        fund, date.fromisoformat(record["cutoff"]), nav, events, include_value=True
                    )
                    if issues or answer is None or str(answer.available_at) > window["fit_end"]:
                        raise ValueError("COVERAGE_FIT_ANSWER_BOUNDARY")
                    audit_answer(answer, nav, events)
                    answer_cache[key] = answer.model_dump(mode="json")
                if answer_cache[key]["available_at"] > window["fit_end"]:
                    raise ValueError("COVERAGE_CACHED_ANSWER_BOUNDARY")
                fit.append({"input": record["inputs"]["CLEAN"], "answer": answer_cache[key]})
            exam.extend(record["inputs"]["CLEAN"] for record in exam_rows)
        ready = all(
            d["fit_count"] >= protocol["minimum"]["FIT"]
            and d["unused_gap_mature_count"] >= protocol["minimum"]["CAL"]
            and d["scorable_count"] >= protocol["minimum"]["EXAM"]
            and d["coverage"] >= protocol["minimum_coverage"]
            for d in details.values()
        )
        if window["cal_end"] in old_windows:
            old = read_json(prior / f"linear-prepared-{old_windows[window['cal_end']]['name']}.json")["complete"]
            if old and fit != old["fit"]:
                raise ValueError("COVERAGE_OLD_TRAINING_CHANGED")
        bundle = {
            "window": window,
            "planned": [str(d) for d in planned_dates(protocol["version"], window)],
            "complete": {"fit": fit, "exam": {"CLEAN": exam}} if ready else None,
        }
        branches = {}
        for branch in protocol["branches"]:
            if not ready:
                branches[branch] = {"status": "INSUFFICIENT_DATA"}
                continue
            inputs, output = validate_job(
                {"version": protocol["version"], "branch": branch, "window": window, "fit": fit, "exam": exam}
            )
            branches[branch] = {
                "status": "READY",
                "train_counts": dict(Counter(i.fund for i, _ in inputs)),
                "exam_inputs": len(output),
                "fit_keys_hash": digest([i.key for i, _ in inputs]),
                "exam_keys_hash": digest([i.key for i in output]),
            }
        coverage[name] = {"source": details, "branches": branches}
        filename = f"linear-prepared-{name}.json"
        files[filename] = write_json(folder / filename, bundle)
    files["linear-coverage.json"] = write_json(folder / "linear-coverage.json", coverage)
    files["linear-input-audit.json"] = write_json(
        folder / "linear-input-audit.json",
        {
            "unique_input_count": len(audited_inputs),
            "unique_training_answer_count": len(answer_cache),
            "scope": "INDEPENDENT_ARITHMETIC_SAME_SEALED_SNAPSHOT_NOT_EXTERNAL_SOURCE_VALUE_ATTESTATION",
            "source_run": SOURCE_RUN,
            "source_first_versions_verified": False,
        },
    )
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


def export_answers(protocol, source):
    sources, result = load_sources(source), []
    for window in protocol["windows"]:
        for fund in FUNDS:
            nav, events, records = sources[fund]
            for cutoff in planned_dates(protocol["version"], window):
                record = records[str(cutoff)]
                answer, issues = data.build_answer(fund, cutoff, nav, events, include_value=True)
                if answer is not None:
                    audit_answer(answer, nav, events)
                    metadata = {k: answer.model_dump(mode="json")[k] for k in ("fund", "cutoff", "end", "available_at")}
                    if metadata != record["label"]:
                        raise ValueError("COVERAGE_LABEL_METADATA_CHANGED")
                result.append(
                    {
                        "window": window["name"],
                        "fund": fund,
                        "cutoff": str(cutoff),
                        "sample_key": f"{fund}:{cutoff}",
                        "answer": answer.model_dump(mode="json") if answer else None,
                        "issues": issues,
                    }
                )
    return result


def control_parity(folder, protocol):
    prior = prior_study()
    old_plan = read_json(prior / "linear-plan.json")
    old_windows = {w["cal_end"]: w for w in old_plan["windows"]}
    result = []
    for window in protocol["windows"]:
        old_window = old_windows.get(window["cal_end"])
        if old_window is None:
            continue
        old_bundle = read_json(prior / f"linear-prepared-{old_window['name']}.json")["complete"]
        if old_bundle is None:
            continue
        current_bundle = read_json(folder / f"linear-prepared-{window['name']}.json")["complete"]
        if current_bundle is None:
            result.append({"window": window["name"], "status": "INSUFFICIENT_DATA"})
            continue
        previous = read_json(prior / f"linear-models-{old_window['name']}.json")
        current = read_json(folder / f"linear-models-{window['name']}.json")
        for branch in protocol["branches"]:
            if current[branch]["status"] != "PREDICTED":
                result.append({"window": window["name"], "branch": branch, "status": current[branch]["status"]})
                continue
            model, old = current[branch]["models"]["POOLED"], previous[branch]["models"]["POOLED"]
            if any(model[k] != v for k, v in old.items() if k not in ("version", "hash")):
                raise ValueError("COVERAGE_MODEL_PARITY")

            def index(bundle, output):
                return {
                    f"{i['fund']}:{i['cutoff']}": s
                    for i, s in zip(bundle["exam"]["CLEAN"], output["scores"], strict=True)
                }

            before, after = index(old_bundle, previous[branch]), index(current_bundle, current[branch])
            common = set(before) & set(after)
            differences = [abs(before[k] - after[k]) for k in common]
            flips = sum((before[k] > 0.5) != (after[k] > 0.5) for k in common)
            if not common or max(differences) > 1e-12 or flips:
                raise ValueError("COVERAGE_SCORE_PARITY")
            result.append(
                {
                    "window": window["name"],
                    "prior_window": old_window["name"],
                    "branch": branch,
                    "status": "VERIFIED",
                    "count": len(common),
                    "max_difference": max(differences),
                    "direction_changes": flips,
                    "model_values_identical": True,
                    "prior_predictions_outside_new_plan": sorted(set(before) - set(after)),
                }
            )
    return result


def diagnostics(folder, protocol, answers):
    report = regularization_diagnostics(folder, protocol, answers)
    prior = prior_study()
    old_answers = {(r["window"], r["sample_key"]): r["answer"] for r in read_jsonl(prior / "linear-answers.jsonl")}
    old_ends = {w["name"]: w["exam_end"] for w in read_json(prior / "linear-plan.json")["windows"]}
    old_predictions = read_jsonl(prior / "linear-predictions.jsonl")
    old_scored_keys = {
        r["sample_key"]
        for r in old_predictions
        if r["branch"] == "REFERENCE"
        and r["score"] is not None
        and old_answers[(r["window"], r["sample_key"])] is not None
        and old_answers[(r["window"], r["sample_key"])]["available_at"] <= old_ends[r["window"]]
    }
    old_index = {
        (r["branch"], r["sample_key"]): r
        for r in old_predictions
        if r["branch"] in protocol["branches"] and r["sample_key"] in old_scored_keys
    }
    answer_map = {(r["window"], r["sample_key"]): r["answer"] for r in answers}
    evaluation_ends = {w["name"]: evaluation_asof(w) for w in protocol["windows"]}
    buckets = {
        name: {b: [] for b in protocol["branches"]} for name in ("PREVIOUSLY_SCORED_378", "ADDED_DEVELOPMENT_ROWS")
    }
    for row in read_jsonl(folder / "linear-predictions.jsonl"):
        answer = answer_map[(row["window"], row["sample_key"])]
        if (
            row["branch"] in protocol["branches"]
            and row["status"] == "PREDICTED"
            and answer is not None
            and answer["available_at"] <= evaluation_ends[row["window"]]
        ):
            name = "PREVIOUSLY_SCORED_378" if row["sample_key"] in old_scored_keys else "ADDED_DEVELOPMENT_ROWS"
            if name == "PREVIOUSLY_SCORED_378":
                old = old_index[(row["branch"], row["sample_key"])]
                if (
                    answer != old_answers[(old["window"], old["sample_key"])]
                    or abs(row["score"] - old["score"]) > 1e-12
                    or row["predicted_up"] != old["predicted_up"]
                ):
                    raise ValueError("COVERAGE_PREVIOUS_SCORED_ROW_CHANGED")
            buckets[name][row["branch"]].append({**row, "y": answer["y"]})
    return {
        **report,
        "purpose": "EXPANDED_DEVELOPMENT_AND_OLD_OVERLAP_SEPARATELY_NOT_INDEPENDENT_TEST",
        "cohorts": {
            name: {b: grouped_metrics(rows, FUNDS) for b, rows in groups.items()} for name, groups in buckets.items()
        },
        "current_snapshot_parameter_search": "CLOSED",
        "new_parameter_variants": 0,
        "previous_scored_parity": {
            b: {
                "verified_count": len(rows),
                "expected_count": len(old_scored_keys),
                "complete": len(rows) == len(old_scored_keys),
            }
            for b, rows in buckets["PREVIOUSLY_SCORED_378"].items()
        },
    }
