"""T05/T06/T07逐因素实施，T08准入结论和T09观察设计，不越过发布边界。"""

import subprocess
import sys
from datetime import date
from time import perf_counter

from app.schemas.direction_followup import StudyCohort, StudyInput
from app.services.direction_followup_data import (
    add_market,
    answer,
    feature_dataset,
    load_funds,
    market_prices,
    metadata,
    restore_fund,
    select_cohort,
    stage_cutoffs,
)
from app.services.direction_followup_protocol import FEATURE_MARKET, P0_NAME, code_hash, plan
from app.services.direction_training_artifacts import (
    ROOT,
    digest,
    new_folder,
    read_json,
    read_jsonl,
    read_seal,
    seal,
    write_json,
    write_jsonl,
)
from app.services.direction_training_dataset import FUNDS, exam_dates
from app.services.direction_training_evaluation import complete_blocks, grouped_metrics, paired_interval
from app.services.direction_training_process import run_process
from app.services.direction_training_protocol import BASELINES
from app.services.historical_nav_evaluation import FEATURE_NAMES, fixed_momentum_score


def freeze():
    original = read_seal(ROOT / ".local-runs" / P0_NAME, "complete.json")
    folder = new_folder()
    protocol = plan()
    protocol["p0_hash"] = original["manifest_hash"]
    protocol["calendar_hash"] = read_json(ROOT / ".local-runs" / P0_NAME / "source.json")["calendar_hash"]
    protocol["git_head"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    protocol["git_status"] = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()
    files = {"study-plan.json": write_json(folder / "study-plan.json", protocol)}
    manifest = seal(folder, "study-frozen.json", files, status="FROZEN", p0_hash=original["manifest_hash"])
    return folder, manifest


def load_plan(folder):
    frozen = read_seal(folder, "study-frozen.json")
    protocol = read_json(folder / "study-plan.json")
    current = plan()
    if any(protocol.get(k) != v for k, v in current.items()) or protocol["source_code_hash"] != code_hash():
        raise ValueError("STUDY_PLAN_CODE_OR_RUNTIME_CHANGED")
    return protocol, frozen


def prepare(folder):
    protocol, frozen = load_plan(folder)
    meta = metadata()
    cohort, decisions = select_cohort(meta, protocol)
    files = {
        "metadata.json": write_json(folder / "metadata.json", meta),
        "cohort-metadata.json": write_json(
            folder / "cohort-metadata.json", {"cohort": cohort.model_dump(mode="json"), "decisions": decisions}
        ),
    }
    sources, sql = load_funds(cohort)
    old = read_json(ROOT / ".local-runs" / P0_NAME / "source.json")["funds"]
    if any(digest(sources[f]) != digest(old[f]) for f in FUNDS):
        raise ValueError("P0_SOURCE_CHANGED_REQUIRES_SEPARATE_DATA_VERSION")
    inputs, availability = feature_dataset(sources, cohort)
    coverage = {}
    for window in protocol["windows"]:
        groups = {}
        for f in cohort.funds:
            groups[f] = {
                stage: stage_cutoffs(availability[f], lower, upper, exam=stage == "EXAM")
                for stage, lower, upper in (
                    ("FIT", "2020-12-31", window["fit_end"]),
                    ("CAL", window["fit_end"], window["cal_end"]),
                    ("EXAM", window["cal_end"], window["exam_end"]),
                )
            }
        ready = all(groups[f][s]["usable"] >= protocol["minimum"][s] for f in FUNDS for s in ("FIT", "CAL", "EXAM"))
        coverage[window["name"]] = {"funds": groups, "original_ready": ready}
    ready_windows = [w["name"] for w in protocol["windows"] if coverage[w["name"]]["original_ready"]]
    included = []
    for f in cohort.funds:
        eligible = bool(ready_windows) and all(
            coverage[w]["funds"][f][s]["usable"] >= protocol["minimum"][s]
            for w in ready_windows
            for s in ("FIT", "CAL", "EXAM")
        )
        if f in FUNDS or eligible:
            included.append(f)
        for r in decisions:
            if r["fund_code"] == f:
                r.update(
                    stage_eligible=eligible, final_status="INCLUDED" if f in included else "EXCLUDED_STAGE_COVERAGE"
                )
    cohort = StudyCohort(funds=tuple(included))
    files["cohort.json"] = write_json(
        folder / "cohort.json",
        {
            "cohort": cohort.model_dump(mode="json"),
            "decisions": decisions,
            "ready_windows": ready_windows,
            "historical_classification_verified": False,
            "shared_benchmark_is_not_independence": True,
        },
    )
    files["study-coverage.json"] = write_json(folder / "study-coverage.json", coverage)
    for f in sources:
        files[f"source-{f}.json"] = write_json(folder / f"source-{f}.json", sources[f])
        files[f"inputs-{f}.jsonl"] = write_jsonl(folder / f"inputs-{f}.jsonl", inputs[f])
        files[f"availability-{f}.jsonl"] = write_jsonl(folder / f"availability-{f}.jsonl", availability[f])
    market = market_prices(meta, protocol)
    files["market.json"] = write_json(folder / "market.json", market)
    prices = {r["date"]: r["close"] for r in market["prices"]}
    input_map = {f"{i['fund']}:{i['cutoff']}": StudyInput.model_validate(i) for f in cohort.funds for i in inputs[f]}
    for window in protocol["windows"]:
        name = window["name"]
        if name not in ready_windows:
            continue
        prepared = {
            "window": window,
            "cohort": cohort.model_dump(mode="json"),
            "FIT": [],
            "CAL": [],
            "EXAM": [],
            "market_inputs": {},
            "market_missing": [],
        }
        for f in cohort.funds:
            source, nav, events = restore_fund(sources[f])
            for stage in ("FIT", "CAL"):
                for cutoff in coverage[name]["funds"][f][stage]["cutoffs"]:
                    item = input_map[f"{f}:{cutoff}"]
                    label, issues = answer(f, item.cutoff, nav, events, cohort, value=True)
                    if issues:
                        raise ValueError("PREPARED_LABEL_CHANGED")
                    prepared[stage].append({"input": item.model_dump(mode="json"), "answer": label})
            _, dates = exam_dates(date.fromisoformat(window["cal_end"]), date.fromisoformat(window["exam_end"]))
            for cutoff in dates:
                if item := input_map.get(f"{f}:{cutoff}"):
                    prepared["EXAM"].append(item.model_dump(mode="json"))
        all_inputs = [r["input"] for stage in ("FIT", "CAL") for r in prepared[stage]] + prepared["EXAM"]
        for raw in all_inputs:
            item = StudyInput.model_validate(raw)
            if item.fund not in FUNDS:
                continue
            enlarged, reason = add_market(item, prices)
            if enlarged:
                prepared["market_inputs"][item.key] = enlarged.model_dump(mode="json")
            else:
                prepared["market_missing"].append({"key": item.key, "reason": reason})
        filename = f"study-prepared-{name}.json"
        files[filename] = write_json(folder / filename, prepared)
    return seal(
        folder,
        "study-prepared.json",
        files,
        status="PREPARED",
        frozen_hash=frozen["manifest_hash"],
        sql_statement_counts=sql,
        model_fitted=False,
        exam_answers_exported=False,
        market_status=market["status"],
    )


def build_jobs(data):
    cohort = StudyCohort.model_validate(data["cohort"])
    original = [r for r in data["FIT"] if r["input"]["fund"] in FUNDS]
    exam_original = [i for i in data["EXAM"] if i["fund"] in FUNDS]
    base = {
        "cohort": data["cohort"],
        "window": data["window"],
        "features": list(FEATURE_NAMES),
        "fit": original,
        "cal": [],
        "base": None,
        "train_funds": list(FUNDS),
        "exam": data["EXAM"],
    }
    jobs = {
        "ORIGINAL_7": {**base, "branch": "ORIGINAL_7"},
        "EXPANDED_7": {**base, "branch": "EXPANDED_7", "train_funds": list(cohort.funds), "fit": data["FIT"]},
        "CALIBRATED_6M": {
            **base,
            "branch": "CALIBRATED_6M",
            "exam": exam_original,
            "cal": [r for r in data["CAL"] if r["input"]["fund"] in FUNDS],
        },
    }
    matched = [r for r in original if f"{r['input']['fund']}:{r['input']['cutoff']}" in data["market_inputs"]]
    exam_matched = [i for i in exam_original if f"{i['fund']}:{i['cutoff']}" in data["market_inputs"]]

    def enlarged(i):
        return data["market_inputs"][f"{i['fund']}:{i['cutoff']}"]

    jobs["MARKET_MATCHED_7"] = {**base, "branch": "MARKET_MATCHED_7", "fit": matched, "exam": exam_matched}
    jobs["MARKET_10"] = {
        **base,
        "branch": "MARKET_10",
        "features": list((*FEATURE_NAMES, *FEATURE_MARKET)),
        "fit": [{"input": enlarged(r["input"]), "answer": r["answer"]} for r in matched],
        "exam": [enlarged(i) for i in exam_matched],
    }
    return jobs


def eligible_job(job):
    if any(sum(r["input"]["fund"] == f for r in job["fit"]) < 252 for f in job["train_funds"]):
        return "FIT_BELOW_MINIMUM"
    if job["branch"] == "CALIBRATED_6M" and any(
        sum(r["input"]["fund"] == f for r in job["cal"]) < 60 for f in job["train_funds"]
    ):
        return "CAL_BELOW_MINIMUM"
    if any(sum(r["fund"] == f for r in job["exam"]) < 40 for f in FUNDS):
        return "EXAM_INPUT_BELOW_MINIMUM"
    if job["branch"] == "EXPANDED_7" and len(job["train_funds"]) == 3:
        return "NO_ADDITIONAL_ELIGIBLE_FUND"
    return None


def predict(folder):
    protocol, _ = load_plan(folder)
    prepared = read_seal(folder, "study-prepared.json")
    cohort = StudyCohort.model_validate(read_json(folder / "cohort.json")["cohort"])
    rows, execution, files = [], {}, {}
    deadline = perf_counter() + protocol["budget"]["total_seconds"]
    coverage = read_json(folder / "study-coverage.json")
    for window in protocol["windows"]:
        name = window["name"]
        if not coverage[name]["original_ready"]:
            execution[name] = {"status": "NOT_RUN", "reason": "ORIGINAL_EXAM_COVERAGE_INSUFFICIENT"}
            continue
        data = read_json(folder / f"study-prepared-{name}.json")
        jobs = build_jobs(data)
        outputs = {}
        for branch in ("ORIGINAL_7", "EXPANDED_7", "MARKET_MATCHED_7", "MARKET_10", "CALIBRATED_6M"):
            job = jobs[branch]
            if branch == "CALIBRATED_6M":
                job["base"] = outputs["ORIGINAL_7"].get("artifact")
            reason = eligible_job(job)
            if reason:
                outputs[branch] = {"status": "NOT_RUN", "reason": reason}
                continue
            if deadline <= perf_counter():
                outputs[branch] = {"status": "FAILED", "reason": "TOTAL_TIME_BUDGET"}
                continue
            filename = f"study-job-{name}-{branch}.json"
            files[filename] = write_json(folder / filename, job)
            outputs[branch] = run_process(
                job,
                seconds=min(120, deadline - perf_counter()),
                command=[sys.executable, "-m", "scripts.direction_followup_worker"],
            )
        filename = f"study-models-{name}.json"
        files[filename] = write_json(folder / filename, outputs)
        execution[name] = {
            b: {k: v for k, v in r.items() if k not in ("scores", "artifact")} for b, r in outputs.items()
        }
        _, planned = exam_dates(date.fromisoformat(window["cal_end"]), date.fromisoformat(window["exam_end"]))
        for branch, job in jobs.items():
            item_map = {f"{r['fund']}:{r['cutoff']}": (i, r) for i, r in enumerate(job["exam"])}
            scores = outputs[branch].get("scores", [])
            eval_funds = cohort.funds if branch in ("ORIGINAL_7", "EXPANDED_7") else FUNDS
            for fund in eval_funds:
                for cutoff in planned:
                    key = f"{fund}:{cutoff}"
                    row = {
                        "window": name,
                        "sample_key": key,
                        "fund": fund,
                        "cutoff": str(cutoff),
                        "branch": branch,
                        "score": None,
                        "predicted_up": None,
                        "input_hash": None,
                        "status": outputs[branch]["status"],
                        "reason": outputs[branch].get("reason"),
                        "calibration_state": outputs[branch].get("calibration_state"),
                    }
                    if key not in item_map:
                        row.update(status="INPUT_UNAVAILABLE", reason="INPUT_OR_MARKET_MISSING")
                    else:
                        index, item = item_map[key]
                        row["input_hash"] = item["input_hash"]
                        if len(scores) == len(job["exam"]):
                            row.update(score=scores[index], predicted_up=int(scores[index] > 0.5))
                    rows.append(row)
        from decimal import Decimal

        history = data["FIT"] + data["CAL"]
        for item in data["EXAM"]:
            past = [r["answer"]["y"] for r in history if r["input"]["fund"] == item["fund"]]
            values = {
                "ALWAYS_UP": 1.0,
                "TRAIN_UP_FREQUENCY": sum(past) / len(past),
                "MOMENTUM_20D": float(item["x"][1] > 0),
                "FIXED_MOMENTUM_SCORE": float(fixed_momentum_score(Decimal(str(item["x"][1])))),
            }
            for branch, score in values.items():
                rows.append(
                    {
                        "window": name,
                        "sample_key": f"{item['fund']}:{item['cutoff']}",
                        "fund": item["fund"],
                        "cutoff": item["cutoff"],
                        "branch": branch,
                        "score": score,
                        "predicted_up": int(score > 0.5),
                        "input_hash": item["input_hash"],
                        "status": "PREDICTED",
                    }
                )
    files["study-predictions.jsonl"] = write_jsonl(folder / "study-predictions.jsonl", rows)
    files["study-execution.json"] = write_json(folder / "study-execution.json", execution)
    return seal(
        folder,
        "study-predicted.json",
        files,
        status="PREDICTED",
        prepared_hash=prepared["manifest_hash"],
        exam_answers_exported=False,
    )


def comparison(protocol, predictions, answers, branches, funds):
    answer_map = {(r["window"], r["sample_key"]): r["answer"] for r in answers}
    if len(answer_map) != len(answers):
        raise ValueError("DUPLICATE_STUDY_ANSWER")
    maps = {b: {} for b in branches}
    executed = {r["window"] for r in predictions}
    planned = set()
    for w in protocol["windows"]:
        if w["name"] in executed:
            _, dates = exam_dates(date.fromisoformat(w["cal_end"]), date.fromisoformat(w["exam_end"]))
            planned.update((w["name"], f"{f}:{d}") for f in funds for d in dates)
    seen = set()
    for row in predictions:
        if row["fund"] not in funds or row["branch"] not in branches:
            continue
        key = (row["window"], row["sample_key"])
        if key not in planned or (*key, row["branch"]) in seen:
            raise ValueError("PREDICTION_DATE_OR_DUPLICATE")
        seen.add((*key, row["branch"]))
        if row["score"] is not None and row["predicted_up"] != int(row["score"] > 0.5):
            raise ValueError("PREDICTION_DIRECTION")
        label = answer_map.get(key)
        if row["score"] is not None and label:
            maps[row["branch"]][key] = {**row, "y": label["y"], "correct": int(label["y"] == row["predicted_up"])}
    common = set.intersection(*(set(m) for m in maps.values()))
    results = {b: grouped_metrics([r for k, r in maps[b].items() if k in common], funds) for b in branches}
    statistical = {**protocol, "funds": list(funds)}
    blocks, exclusions = complete_blocks(statistical, common)
    windows = sorted({w for w, k in common})
    coverage = {
        f: {
            "planned": sum(k.split(":")[0] == f for w, k in planned),
            "common": sum(k.split(":")[0] == f for w, k in common),
        }
        for f in funds
    }
    per_window = {
        w: {b: grouped_metrics([r for k, r in maps[b].items() if k in common and k[0] == w], funds) for b in branches}
        for w in windows
    }
    delta = paired_interval(statistical, blocks, maps[branches[1]], maps[branches[0]]) if len(branches) > 1 else None
    candidate, reference = branches[1], branches[0]
    valid_windows = [
        w
        for w in windows
        if all(
            sum(ww == w and k.split(":")[0] == f for ww, k in common) >= protocol["minimum"]["EXAM"]
            and sum(ww == w and k.split(":")[0] == f for ww, k in common)
            / sum(ww == w and k.split(":")[0] == f for ww, k in planned)
            >= protocol["minimum_coverage"]
            for f in funds
        )
    ]
    sufficient = (
        len(valid_windows) >= protocol["minimum_windows"]
        and len(blocks) >= protocol["bootstrap"]["minimum_complete_blocks"]
        and all(c["planned"] and c["common"] / c["planned"] >= protocol["minimum_coverage"] for c in coverage.values())
        and not any(r["status"] == "FAILED" and r["branch"] in branches for r in predictions)
    )
    comparisons = {}
    for other in branches:
        if other == candidate:
            continue
        left, right = results[candidate]["equal_fund_macro"], results[other]["equal_fund_macro"]
        comparisons[other] = {
            "accuracy_delta": left["accuracy"] - right["accuracy"] if common else None,
            "balanced_accuracy_delta": left["balanced_accuracy"] - right["balanced_accuracy"]
            if left["balanced_accuracy"] is not None and right["balanced_accuracy"] is not None
            else None,
            "brier_delta": left["brier_score"] - right["brier_score"] if common else None,
            "block_interval": paired_interval(statistical, blocks, maps[candidate], maps[other]),
        }
    positive_funds = (
        sum(
            results[candidate]["per_fund"][f]["accuracy"] > results[reference]["per_fund"][f]["accuracy"] for f in funds
        )
        if common
        else 0
    )
    positive_windows = sum(
        per_window[w][candidate]["equal_fund_macro"]["accuracy"]
        > per_window[w][reference]["equal_fund_macro"]["accuracy"]
        for w in windows
    )
    stable = (
        sufficient
        and all(c["accuracy_delta"] > 0 and c["block_interval"]["interval"][0] > 0 for c in comparisons.values())
        and comparisons[reference]["balanced_accuracy_delta"] is not None
        and comparisons[reference]["balanced_accuracy_delta"] >= 0
        and positive_funds * 3 >= len(funds) * 2
        and positive_windows * 3 >= len(windows) * 2
        and all(p["interval"][1] >= 0 for p in delta["per_fund"].values())
    )
    return {
        "branches": list(branches),
        "funds": list(funds),
        "common_count": len(common),
        "coverage": coverage,
        "metrics": results,
        "window_count": len(windows),
        "valid_windows": valid_windows,
        "per_window": per_window,
        "window_equal_macro": {
            b: {
                field: sum(per_window[w][b]["equal_fund_macro"][field] for w in windows) / len(windows)
                if windows and all(per_window[w][b]["equal_fund_macro"][field] is not None for w in windows)
                else None
                for field in ("accuracy", "balanced_accuracy", "brier_score", "log_loss")
            }
            for b in branches
        },
        "branch_coverage": {
            b: {
                f: {
                    "planned": coverage[f]["planned"],
                    "predicted": sum(
                        r["fund"] == f and r["branch"] == b and r["score"] is not None for r in predictions
                    ),
                    "scored": sum(r["fund"] == f for r in maps[b].values()),
                    "common": coverage[f]["common"],
                }
                for f in funds
            }
            for b in branches
        },
        "complete_blocks_per_fund": len(blocks),
        "block_exclusions": exclusions,
        "candidate_minus_reference": delta,
        "comparisons": comparisons,
        "positive_funds": positive_funds,
        "positive_windows": positive_windows,
        "research_status": "INSUFFICIENT_DATA"
        if not sufficient
        else "RESEARCH_CANDIDATE"
        if stable
        else "NO_STABLE_GAIN",
        "independent_test": False,
    }


def score(folder):
    protocol, _ = load_plan(folder)
    predicted = read_seal(folder, "study-predicted.json")
    if predicted["prepared_hash"] != read_seal(folder, "study-prepared.json")["manifest_hash"]:
        raise ValueError("PREDICTION_LINK")
    cohort = StudyCohort.model_validate(read_json(folder / "cohort.json")["cohort"])
    sources = {f: restore_fund(read_json(folder / f"source-{f}.json")) for f in cohort.funds}
    predictions = read_jsonl(folder / "study-predictions.jsonl")
    windows = {w["name"]: w for w in protocol["windows"]}
    answers, seen = [], set()
    for row in predictions:
        identity = (row["window"], row["sample_key"])
        if identity in seen:
            continue
        seen.add(identity)
        f = row["fund"]
        label, issues = answer(f, date.fromisoformat(row["cutoff"]), sources[f][1], sources[f][2], cohort, value=True)
        if label and label["available_at"] > windows[row["window"]]["exam_end"]:
            label, issues = None, ["ANSWER_NOT_MATURE"]
        answers.append(
            {
                "window": row["window"],
                "sample_key": row["sample_key"],
                "answer": label,
                "issues": issues,
                "hash": digest(label),
            }
        )
    result = evaluate_study(protocol, predictions, answers, cohort)
    files = {
        "study-answers.jsonl": write_jsonl(folder / "study-answers.jsonl", answers),
        "study-metrics.json": write_json(folder / "study-metrics.json", result),
    }
    return seal(
        folder,
        "study-scored.json",
        files,
        status="SCORED",
        predicted_hash=predicted["manifest_hash"],
        test_scored=False,
    )


def evaluate_study(protocol, predictions, answers, cohort):
    result = {
        "T05_original": comparison(protocol, predictions, answers, ("ORIGINAL_7", "EXPANDED_7", *BASELINES), FUNDS),
        "T06": comparison(protocol, predictions, answers, ("MARKET_MATCHED_7", "MARKET_10", *BASELINES), FUNDS),
        "T07": comparison(protocol, predictions, answers, ("ORIGINAL_7", "CALIBRATED_6M"), FUNDS),
    }
    additional = tuple(f for f in cohort.funds if f not in FUNDS)
    if additional:
        result["T05_additional"] = comparison(
            protocol, predictions, answers, ("ORIGINAL_7", "EXPANDED_7", *BASELINES), additional
        )
    candidates = [k for k in ("T05_original", "T06") if result[k]["research_status"] == "RESEARCH_CANDIDATE"]
    result["qualified_candidate"] = candidates[0] if len(candidates) == 1 else None
    result["research_status"] = (
        "RESEARCH_CANDIDATE"
        if result["qualified_candidate"]
        else "INSUFFICIENT_DATA"
        if all(result[k]["research_status"] == "INSUFFICIENT_DATA" for k in ("T05_original", "T06"))
        else "COMBINED_CHANGE_NOT_TESTED"
        if len(candidates) > 1
        else "NO_STABLE_GAIN"
    )
    result["direction_reference"] = "ORIGINAL_7"
    result["publication_status"] = "MODEL_NOT_RELEASED"
    return result


def test_gate(qualified_candidate, independence_verified, historical_data_verified):
    reasons = []
    if qualified_candidate is None:
        reasons.append("NO_QUALIFIED_FROZEN_CANDIDATE")
    if not independence_verified:
        reasons.append("2025_INDEPENDENCE_NOT_FULLY_ATTESTED")
    if not historical_data_verified:
        reasons.append("HISTORICAL_FIRST_PUBLICATION_OR_EVENT_COMPLETENESS_UNVERIFIED")
    return {
        "status": "NOT_RUN_PREREQUISITE_FAILED" if reasons else "ELIGIBLE_FOR_SEPARATE_FROZEN_TEST",
        "reasons": reasons,
        "test_values_read": False,
        "test_scored": False,
    }


def finish(folder):
    protocol, frozen = load_plan(folder)
    if (folder / "study-complete.json").exists():
        return verify(folder)
    scored = read_seal(folder, "study-scored.json")
    prepared = read_seal(folder, "study-prepared.json")
    predicted = read_seal(folder, "study-predicted.json")
    if (
        prepared["frozen_hash"] != frozen["manifest_hash"]
        or predicted["prepared_hash"] != prepared["manifest_hash"]
        or scored["predicted_hash"] != predicted["manifest_hash"]
    ):
        raise ValueError("STUDY_STAGE_LINK_MISMATCH")
    predictions, answers = read_jsonl(folder / "study-predictions.jsonl"), read_jsonl(folder / "study-answers.jsonl")
    cohort = StudyCohort.model_validate(read_json(folder / "cohort.json")["cohort"])
    metrics = read_json(folder / "study-metrics.json")
    if digest(metrics) != digest(evaluate_study(protocol, predictions, answers, cohort)):
        raise ValueError("STUDY_METRICS_REPLAY")
    gate = test_gate(metrics["qualified_candidate"], False, False)
    test_protocol = {
        "version": "DIRECTION_INDEPENDENT_TEST_DESIGN_V1",
        "gate": gate,
        "development_run": folder.name,
        "candidate": metrics["qualified_candidate"],
        "planned_period": ["2025-01-01", "2025-12-31"],
        "funds": list(FUNDS),
        "horizon_sessions": 20,
        "policy": "FREEZE_CANDIDATE_AND_PASS_PRECHECK_BEFORE_ANY_TEST_BODY_READ",
        "execution_limit": 1,
        "after_failure": "MARK_PERIOD_OBSERVED_NO_RETUNING_ON_THIS_TEST",
        "minimum_coverage": 0.9,
        "minimum_complete_blocks_per_fund": 5,
        "threshold": 0.5,
        "success": "FUND_MACRO_DIRECTION_AND_BALANCED_ACCURACY_NOT_WORSE_THAN_"
        "FROZEN_REFERENCES_AND_PAIRED_INTERVAL_SUPPORT",
        "source_versions_verified": False,
        "independence_evidence": "P0_AND_THIS_RUN_MANIFESTS_TEST_SCORED_FALSE_ONLY_NOT_GLOBAL_ACCESS_ATTESTATION",
    }
    forward = {
        "version": "DIRECTION_FORWARD_OBSERVATION_DESIGN_V1",
        "status": "DESIGN_COMPLETE_NOT_STARTED",
        "funds": list(FUNDS),
        "start_rule": "FIRST_FUTURE_CN_TRADING_SESSION_AFTER_CANDIDATE_TEST_AND_SOURCE_PREREQUISITES_PASS",
        "schedule": "ONCE_PER_TRADING_DAY_AFTER_20_00_ASIA_SHANGHAI",
        "initial_observation_sessions": 126,
        "cutoff": "CURRENT_DAY_END_ONLY_WHEN_INPUT_SNAPSHOT_IS_ALREADY_KNOWN",
        "input_policy": "EXACT_61_SESSIONS_ANNOUNCED_NAV_MAX_ANCHOR_LAG_ONE_NO_FILL",
        "label_policy": "20_EXACT_FUTURE_SESSIONS_WAIT_FOR_ALL_REQUIRED_ANNOUNCEMENTS_NO_SHIFT",
        "prediction_storage": "EXCLUSIVE_JSONL_WITH_INPUT_HASH_MODEL_HASH_PROTOCOL_HASH_"
        "AND_ACTUAL_RECORDED_AT_BEFORE_LABEL",
        "scoring": "ONLY_MATURE_LABELS_REPORT_PLANNED_MISSING_AND_PENDING_SEPARATELY",
        "retraining": "NONE_DURING_FIRST_OBSERVATION_PROTOCOL_NEW_RUN_REQUIRED_FOR_CHANGE",
        "stop_conditions": [
            "SOURCE_AUTHORIZATION_REVOKED",
            "MODEL_OR_SNAPSHOT_HASH_CHANGED",
            "CALENDAR_GAP",
            "DATA_CONFLICT",
            "126_CUTOFFS_REACHED",
        ],
        "restart": "NEW_NUMBER_NO_BACKFILL_AS_IF_PREDICTED_IN_PAST",
        "scheduler_created": False,
        "notification_enabled": False,
        "private_holdings_used": False,
        "publication_status": "MODEL_NOT_RELEASED",
    }
    files = {
        "independent-test-plan.json": write_json(folder / "independent-test-plan.json", test_protocol),
        "forward-observation-plan.json": write_json(folder / "forward-observation-plan.json", forward),
    }
    return seal(
        folder,
        "study-complete.json",
        files,
        status="DOCUMENT_SCOPE_DELIVERED",
        scored_hash=scored["manifest_hash"],
        task_status={
            "T05": "EXPERIMENT_COMPLETED" if metrics["T05_original"]["common_count"] else "CONDITIONAL_NOT_RUN",
            "T06": "EXPERIMENT_COMPLETED" if metrics["T06"]["common_count"] else "CONDITIONAL_NOT_RUN",
            "T07": "EXPERIMENT_COMPLETED" if metrics["T07"]["common_count"] else "CONDITIONAL_NOT_RUN",
            "T08": "PLAN_AND_PRECISE_NONEXECUTION_REASON_DELIVERED",
            "T09": "DESIGN_COMPLETED",
        },
        research_status=metrics["research_status"],
        model_released=False,
        test_scored=False,
    )


def verify(folder):
    """只核验已封存链和重算指标，不连接来源或拟合。"""
    protocol, frozen = load_plan(folder)
    prepared = read_seal(folder, "study-prepared.json")
    predicted = read_seal(folder, "study-predicted.json")
    scored = read_seal(folder, "study-scored.json")
    completed = read_seal(folder, "study-complete.json")
    if (
        prepared["frozen_hash"] != frozen["manifest_hash"]
        or predicted["prepared_hash"] != prepared["manifest_hash"]
        or scored["predicted_hash"] != predicted["manifest_hash"]
        or completed["scored_hash"] != scored["manifest_hash"]
    ):
        raise ValueError("STUDY_STAGE_LINK_MISMATCH")
    cohort = StudyCohort.model_validate(read_json(folder / "cohort.json")["cohort"])
    answers = read_jsonl(folder / "study-answers.jsonl")
    if any(r["hash"] != digest(r["answer"]) for r in answers):
        raise ValueError("ANSWER_HASH_CHANGED")
    metrics = evaluate_study(protocol, read_jsonl(folder / "study-predictions.jsonl"), answers, cohort)
    if digest(metrics) != digest(read_json(folder / "study-metrics.json")):
        raise ValueError("STUDY_METRICS_CHANGED")
    return completed


def replay(folder):
    load_plan(folder)
    original = read_seal(folder, "study-predicted.json")
    target = new_folder()
    records = []
    for filename in sorted(original["files"]):
        if not filename.startswith("study-job-"):
            continue
        job = read_json(folder / filename)
        result = run_process(job, command=[sys.executable, "-m", "scripts.direction_followup_worker"])
        name = job["window"]["name"]
        expected = read_json(folder / f"study-models-{name}.json")[job["branch"]]
        if result.get("status") != expected["status"] or len(result.get("scores", [])) != len(
            expected.get("scores", [])
        ):
            raise ValueError("STUDY_REPLAY_STATUS")
        differences = [abs(a - b) for a, b in zip(result.get("scores", []), expected.get("scores", []), strict=True)]
        if any(d > 1e-12 for d in differences) or any(
            (a > 0.5) != (b > 0.5) for a, b in zip(result.get("scores", []), expected.get("scores", []), strict=True)
        ):
            raise ValueError("STUDY_REPLAY_NUMERIC_OR_DIRECTION")
        records.append(
            {
                "window": name,
                "branch": job["branch"],
                "scores": len(differences),
                "max_difference": max(differences, default=0),
            }
        )
    files = {
        "study-replay.json": write_json(
            target / "study-replay.json",
            {"origin": folder.name, "records": records, "new_effectiveness_evidence": False},
        )
    }
    return seal(
        target,
        "study-replay-complete.json",
        files,
        status="REPLAY_VERIFIED",
        replay_folder=str(target),
        origin_hash=original["manifest_hash"],
    )
