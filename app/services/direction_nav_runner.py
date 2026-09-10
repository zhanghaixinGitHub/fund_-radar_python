"""净值日期/缺失实验的离线阶段链；固定快照、不连接数据库、不覆盖旧产物。"""

import json
import sys
from collections import Counter
from datetime import date
from time import perf_counter
from uuid import UUID

from app.services import direction_nav_data as data
from app.services.direction_followup_data import restore_fund, stage_cutoffs
from app.services.direction_nav_protocol import SCENARIOS, SOURCE_HASH, SOURCE_RUN, VERSION, plan
from app.services.direction_training_artifacts import (
    digest,
    file_hash,
    new_folder,
    read_json,
    read_jsonl,
    read_seal,
    run_folder,
    seal,
    write_json,
    write_jsonl,
)
from app.services.direction_training_dataset import (
    END,
    FUNDS,
    START,
    exam_dates,
)
from app.services.direction_training_dataset import (
    build_answer as legacy_answer,
)
from app.services.direction_training_dataset import (
    build_input as legacy_input,
)
from app.services.direction_training_process import run_process
from app.services.historical_nav_evaluation import fixed_momentum_score
from app.services.trading_calendar import load_calendar


def freeze():
    original = run_folder(UUID(SOURCE_RUN))
    old = {s: read_seal(original, f"study-{s}.json") for s in ("frozen", "prepared", "predicted", "scored", "complete")}
    if old["complete"]["manifest_hash"] != SOURCE_HASH:
        raise ValueError("SOURCE_COMPLETE_CHANGED")
    for current, previous in (("prepared", "frozen"), ("predicted", "prepared"), ("scored", "predicted")):
        if old[current][f"{previous}_hash"] != old[previous]["manifest_hash"]:
            raise ValueError("SOURCE_STAGE_CHAIN")
    if old["complete"]["scored_hash"] != old["scored"]["manifest_hash"]:
        raise ValueError("SOURCE_STAGE_CHAIN")
    folder, files = new_folder(), {}
    for fund in FUNDS:
        name = f"source-{fund}.json"
        raw = read_json(original / name)
        restore_fund(raw)  # 日期/来源/数量保护先于使用正文。
        if old["prepared"]["files"].get(name) != file_hash(original / name):
            raise ValueError("SOURCE_NOT_SEALED")
        files[name] = write_json(folder / name, raw)
        if files[name] != file_hash(original / name):
            raise ValueError("SOURCE_COPY_CHANGED")
    files["nav-plan.json"] = write_json(folder / "nav-plan.json", {**plan(), "source_files": dict(files)})
    seal(folder, "nav-frozen.json", files, status="FROZEN", source_complete_hash=SOURCE_HASH)
    return folder


def load_plan(folder):
    frozen = read_seal(folder, "nav-frozen.json")
    protocol = read_json(folder / "nav-plan.json")
    if any(protocol.get(k) != v for k, v in plan().items()):
        raise ValueError("NAV_PROTOCOL_CODE_OR_RUNTIME_CHANGED")
    if frozen["source_complete_hash"] != SOURCE_HASH:
        raise ValueError("NAV_SOURCE_LINK")
    return protocol, frozen


def sources(folder):
    return {f: restore_fund(read_json(folder / f"source-{f}.json")) for f in FUNDS}


def _dump(item):
    return item.model_dump(mode="json") if item else None


def _prepare_fund(fund, source, nav, events):
    points = {p.nav_date: p for p in nav}
    records = []
    for cutoff in load_calendar().sessions:
        if not START <= cutoff <= END:
            continue
        old, old_issues = legacy_input(fund, cutoff, source, nav, events)
        clean, audit, issues = data.build_input(fund, cutoff, points, events)
        natural, natural_audit, natural_issues = data.build_input(fund, cutoff, points, events, tolerate=True)
        row = {
            "cutoff": str(cutoff),
            "inputs": {"LEGACY": _dump(old), "CLEAN": _dump(clean), "NATURAL": _dump(natural)},
            "input_issues": {"LEGACY": old_issues, "CLEAN": issues, "NATURAL": natural_issues},
            "audit": {"CLEAN": audit, "NATURAL": natural_audit},
        }
        for count in (1, 2):
            variant = f"DROP{count}"
            masked, mask_audit, mask_issues = (None, {}, ["COMPLETE_CONTROL_UNAVAILABLE"])
            if clean:
                hidden = data.stress_dates(fund, cutoff, events, count)
                masked, mask_audit, mask_issues = data.build_input(
                    fund, cutoff, points, events, tolerate=True, hidden=hidden
                )
            row["inputs"][variant] = _dump(masked)
            row["input_issues"][variant] = mask_issues
            row["audit"][variant] = mask_audit
        label, label_issues = data.build_answer(fund, cutoff, points, events)
        old_label, old_label_issues = legacy_answer(fund, cutoff, nav, events, include_value=False)
        row.update(label=label, label_issues=label_issues, legacy_label=old_label, legacy_label_issues=old_label_issues)
        records.append(row)
    return records


def _stage(records, kind, lower, upper, *, exam=False):
    legacy = kind == "LEGACY"
    rows = [
        {
            "cutoff": r["cutoff"],
            "input_issues": r["input_issues"][kind],
            "label": r["legacy_label" if legacy else "label"],
            "label_issues": r["legacy_label_issues" if legacy else "label_issues"],
        }
        for r in records
    ]
    return stage_cutoffs(rows, lower, upper, exam=exam)


def prepare(folder):
    protocol, frozen = load_plan(folder)
    raw_sources, files, records = sources(folder), {}, {}
    for fund, (source, nav, events) in raw_sources.items():
        records[fund] = _prepare_fund(fund, source, nav, events)
        name = f"nav-inputs-{fund}.jsonl"
        files[name] = write_jsonl(folder / name, records[fund])
        print(json.dumps({"stage": "INPUTS", "fund": fund, "rows": len(records[fund])}), flush=True)
    coverage = {"version": VERSION, "windows": {}, "natural_missing": {}, "ann_date_audit": {}}
    for fund, (_, nav, _) in raw_sources.items():
        coverage["natural_missing"][fund] = {
            "complete": sum(r["inputs"]["CLEAN"] is not None for r in records[fund]),
            "tolerant": sum(r["inputs"]["NATURAL"] is not None for r in records[fund]),
            "recovered": sum(
                r["inputs"]["CLEAN"] is None and r["inputs"]["NATURAL"] is not None for r in records[fund]
            ),
            "actual_gaps": [
                str(d) for d in load_calendar().sessions if START <= d <= END and d not in {p.nav_date for p in nav}
            ],
        }
        coverage["ann_date_audit"][fund] = {
            "missing_ann": sum(p.ann_date is None for p in nav),
            "delays_over_7_calendar_days": [
                {"nav_date": str(p.nav_date), "ann_date": str(p.ann_date)}
                for p in nav
                if p.ann_date and (p.ann_date - p.nav_date).days > 7
            ],
            "input_restored_cutoffs": [
                r["cutoff"] for r in records[fund] if r["inputs"]["LEGACY"] is None and r["inputs"]["CLEAN"] is not None
            ],
            "anchor_changed_on_common": sum(
                r["inputs"]["LEGACY"]["anchor"] != r["inputs"]["CLEAN"]["anchor"]
                for r in records[fund]
                if r["inputs"]["LEGACY"] and r["inputs"]["CLEAN"]
            ),
        }
    for window in protocol["windows"]:
        name, groups, matrix = window["name"], {}, {}
        for kind in ("LEGACY", "CLEAN", "NATURAL"):
            groups[kind] = {}
            for fund in FUNDS:
                groups[kind][fund] = {
                    "FIT": _stage(records[fund], kind, "2020-12-31", window["fit_end"]),
                    "CAL": _stage(records[fund], kind, window["fit_end"], window["cal_end"]),
                    "EXAM": _stage(records[fund], kind, window["cal_end"], window["exam_end"], exam=True),
                }
            ready = all(
                groups[kind][f][s]["usable"] >= protocol["minimum"][s] for f in FUNDS for s in ("FIT", "CAL", "EXAM")
            ) and all(
                groups[kind][f]["EXAM"]["usable"] / max(1, groups[kind][f]["EXAM"]["planned"]) >= 0.9 for f in FUNDS
            )
            matrix[kind] = {"ready": ready, "per_fund": groups[kind]}
        coverage["windows"][name] = matrix
        planned = [
            str(d) for d in exam_dates(date.fromisoformat(window["cal_end"]), date.fromisoformat(window["exam_end"]))[1]
        ]
        bundle = {"window": window, "planned": planned, "jobs": {}, "training_counts": {}}
        for branch, kind in (("LEGACY", "LEGACY"), ("COMPLETE", "CLEAN"), ("TOLERANT", "CLEAN")):
            if not matrix[kind]["ready"]:
                bundle["jobs"][branch] = None
                continue
            fit, exams, mask_counts = [], {"CLEAN": []}, Counter()
            if branch != "LEGACY":
                exams.update(DROP1=[], DROP2=[], NATURAL=[])
            for fund in FUNDS:
                _, nav, events = raw_sources[fund]
                points = {p.nav_date: p for p in nav}
                by_cutoff = {r["cutoff"]: r for r in records[fund]}
                for cutoff in groups[kind][fund]["FIT"]["cutoffs"]:
                    row = by_cutoff[cutoff]
                    if branch != "LEGACY" and any(row["inputs"][v] is None for v in ("CLEAN", "DROP1", "DROP2")):
                        continue  # 两训练分支使用相同题键，压力构造失败显式排除。
                    count = int(digest(["TRAIN_MASK_42", fund, cutoff])[:8], 16) % 3 if branch == "TOLERANT" else 0
                    variant = f"DROP{count}" if count else kind
                    item = row["inputs"][variant]
                    answer, issues = (
                        legacy_answer(fund, date.fromisoformat(cutoff), nav, events, include_value=True)
                        if branch == "LEGACY"
                        else data.build_answer(fund, date.fromisoformat(cutoff), points, events, include_value=True)
                    )
                    if issues or answer is None or str(answer.available_at) > window["fit_end"]:
                        raise ValueError("FIT_ANSWER_NOT_MATURE")
                    fit.append({"input": item, "answer": _dump(answer)})
                    mask_counts[str(count)] += 1
                for cutoff in planned:
                    row = by_cutoff[cutoff]
                    if branch == "LEGACY":
                        if row["inputs"]["LEGACY"]:
                            exams["CLEAN"].append(row["inputs"]["LEGACY"])
                    else:
                        for v in exams:
                            if row["inputs"][v]:
                                exams[v].append(row["inputs"][v])
            if any(sum(r["input"]["fund"] == f for r in fit) < 252 for f in FUNDS):
                raise ValueError("PAIRED_FIT_INSUFFICIENT")
            bundle["jobs"][branch] = {
                "version": VERSION,
                "window": window,
                "branch": branch,
                "fit": fit,
                "exam": exams,
            }
            bundle["training_counts"][branch] = {
                "funds": dict(Counter(r["input"]["fund"] for r in fit)),
                "mask_count": dict(mask_counts),
            }
        if bundle["jobs"].get("COMPLETE") and bundle["jobs"].get("TOLERANT"):
            clean, tolerant = (bundle["jobs"][b]["fit"] for b in ("COMPLETE", "TOLERANT"))
            if [r["answer"] for r in clean] != [r["answer"] for r in tolerant]:
                raise ValueError("PAIRED_TRAINING_LABELS_CHANGED")
        filename = f"nav-prepared-{name}.json"
        files[filename] = write_json(folder / filename, bundle)
    files["nav-coverage.json"] = write_json(folder / "nav-coverage.json", coverage)
    return seal(
        folder,
        "nav-prepared.json",
        files,
        status="PREPARED",
        frozen_hash=frozen["manifest_hash"],
        exam_answers_exported=False,
        database_connected=False,
        source_requests=0,
    )


def predict(folder):
    protocol, _ = load_plan(folder)
    prepared = read_seal(folder, "nav-prepared.json")
    started, files, executions = perf_counter(), {}, {}
    for window in protocol["windows"]:
        name = window["name"]
        bundle = read_json(folder / f"nav-prepared-{name}.json")
        outputs, predictions, index = {}, [], {}
        for branch, job in bundle["jobs"].items():
            if job is None:
                outputs[branch] = {"status": "INSUFFICIENT_DATA"}
                continue
            if perf_counter() - started > protocol["budget"]["total_seconds"]:
                raise TimeoutError("EXPERIMENT_TIME_BUDGET")
            output = run_process(job, command=[sys.executable, "-m", "scripts.direction_nav_worker"])
            outputs[branch] = output
            if output.get("status") != "PREDICTED":
                raise ValueError(f"NAV_JOB_FAILED:{name}:{branch}:{output.get('reason', output.get('error_type'))}")
            for variant, items in job["exam"].items():
                scenario = f"{branch}_{variant}"
                if scenario not in SCENARIOS:
                    continue
                for item, score in zip(items, output["scores"][variant], strict=True):
                    index[(scenario, item["fund"], item["cutoff"])] = (score, item["input_hash"])
            print(json.dumps({"stage": "PREDICTED", "window": name, "branch": branch}), flush=True)
        complete = bundle["jobs"].get("COMPLETE")
        if complete:
            from decimal import Decimal

            fit = complete["fit"]
            rates = {
                f: sum(r["answer"]["y"] for r in fit if r["input"]["fund"] == f)
                / sum(r["input"]["fund"] == f for r in fit)
                for f in FUNDS
            }
            for item in complete["exam"]["CLEAN"]:
                scores = {
                    "ALWAYS_UP": 1.0,
                    "ALWAYS_NON_UP": 0.0,
                    "TRAIN_UP_FREQUENCY": rates[item["fund"]],
                    "MOMENTUM_20D": float(item["x"][1] > 0),
                    "FIXED_MOMENTUM_SCORE": float(fixed_momentum_score(Decimal(str(item["x"][1])))),
                }
                for scenario, score in scores.items():
                    index[(scenario, item["fund"], item["cutoff"])] = (score, item["input_hash"])
        for scenario in SCENARIOS:
            branch = (
                "LEGACY"
                if scenario == "LEGACY_CLEAN"
                else "TOLERANT"
                if scenario.startswith("TOLERANT")
                else "COMPLETE"
            )
            for fund in FUNDS:
                for cutoff in bundle["planned"]:
                    score, input_hash = index.get((scenario, fund, cutoff), (None, None))
                    predictions.append(
                        {
                            "window": name,
                            "scenario": scenario,
                            "fund": fund,
                            "cutoff": cutoff,
                            "sample_key": f"{fund}:{cutoff}",
                            "score": score,
                            "predicted_up": int(score > 0.5) if score is not None else None,
                            "input_hash": input_hash,
                            "status": "PREDICTED"
                            if score is not None
                            else "INSUFFICIENT_DATA"
                            if not bundle["jobs"].get(branch)
                            else "INPUT_UNAVAILABLE",
                        }
                    )
        for filename, value, writer in (
            (f"nav-models-{name}.json", outputs, write_json),
            (f"nav-predictions-{name}.jsonl", predictions, write_jsonl),
        ):
            files[filename] = writer(folder / filename, value)
        executions[name] = {
            b: {k: v for k, v in o.items() if k not in ("artifact", "scores")} for b, o in outputs.items()
        }
    files["nav-execution.json"] = write_json(folder / "nav-execution.json", executions)
    return seal(
        folder,
        "nav-predicted.json",
        files,
        status="PREDICTED",
        prepared_hash=prepared["manifest_hash"],
        exam_answers_exported=False,
    )


def score(folder):
    from app.services.direction_nav_evaluation import evaluate

    protocol, _ = load_plan(folder)
    predicted = read_seal(folder, "nav-predicted.json")  # 必须先封存全部本轮预测。
    raw_sources, files = sources(folder), {}
    answers = []
    for window in protocol["windows"]:
        _, planned = exam_dates(date.fromisoformat(window["cal_end"]), date.fromisoformat(window["exam_end"]))
        for fund, (_, nav, events) in raw_sources.items():
            points = {p.nav_date: p for p in nav}
            for cutoff in planned:
                answer, issues = data.build_answer(fund, cutoff, points, events, include_value=True)
                old, old_issues = legacy_answer(fund, cutoff, nav, events, include_value=True)
                if answer and old and (answer.y, answer.future_return) != (old.y, old.future_return):
                    raise ValueError("NAV_POLICY_CHANGED_RETURN_MATH")
                answers.append(
                    {
                        "window": window["name"],
                        "fund": fund,
                        "cutoff": str(cutoff),
                        "sample_key": f"{fund}:{cutoff}",
                        "answer": _dump(answer),
                        "issues": issues,
                        "legacy_answer": _dump(old),
                        "legacy_issues": old_issues,
                    }
                )
    files["nav-answers.jsonl"] = write_jsonl(folder / "nav-answers.jsonl", answers)
    predictions = [r for w in protocol["windows"] for r in read_jsonl(folder / f"nav-predictions-{w['name']}.jsonl")]
    result = evaluate(protocol, predictions, answers)
    files["nav-metrics.json"] = write_json(folder / "nav-metrics.json", result)
    return seal(
        folder,
        "nav-scored.json",
        files,
        status="SCORED",
        predicted_hash=predicted["manifest_hash"],
        tolerance_status=result["tolerance_status"],
        model_released=False,
        test_scored=False,
    )


def verify(folder):
    from app.services.direction_nav_evaluation import evaluate

    protocol, frozen = load_plan(folder)
    prepared = read_seal(folder, "nav-prepared.json")
    predicted = read_seal(folder, "nav-predicted.json")
    scored = read_seal(folder, "nav-scored.json")
    if (
        prepared["frozen_hash"] != frozen["manifest_hash"]
        or predicted["prepared_hash"] != prepared["manifest_hash"]
        or scored["predicted_hash"] != predicted["manifest_hash"]
    ):
        raise ValueError("NAV_STAGE_CHAIN")
    predictions = [r for w in protocol["windows"] for r in read_jsonl(folder / f"nav-predictions-{w['name']}.jsonl")]
    result = evaluate(protocol, predictions, read_jsonl(folder / "nav-answers.jsonl"))
    if digest(result) != digest(read_json(folder / "nav-metrics.json")):
        raise ValueError("NAV_METRICS_CHANGED")
    original = run_folder(UUID(SOURCE_RUN))
    if any(file_hash(original / n) != h for n, h in protocol["source_files"].items()):
        raise ValueError("ORIGINAL_SOURCE_CHANGED")
    return scored


def replay(folder):
    protocol, _ = load_plan(folder)
    verify(folder)
    target, rows = new_folder(), []
    started = perf_counter()
    for window in protocol["windows"]:
        bundle = read_json(folder / f"nav-prepared-{window['name']}.json")
        outputs = read_json(folder / f"nav-models-{window['name']}.json")
        for branch, job in bundle["jobs"].items():
            if job is None:
                continue
            if perf_counter() - started > protocol["budget"]["total_seconds"]:
                raise TimeoutError("REPLAY_TIME_BUDGET")
            output = run_process(job, command=[sys.executable, "-m", "scripts.direction_nav_worker"])
            expected = outputs[branch]
            if output.get("status") != "PREDICTED" or set(output["scores"]) != set(expected["scores"]):
                raise ValueError("REPLAY_STATUS")
            differences, flips = [], 0
            for variant, actual in output["scores"].items():
                for a, b in zip(actual, expected["scores"][variant], strict=True):
                    differences.append(abs(a - b))
                    flips += (a > 0.5) != (b > 0.5)
            if max(differences, default=0) > 1e-12 or flips:
                raise ValueError("REPLAY_NUMBERS")
            rows.append(
                {
                    "window": window["name"],
                    "branch": branch,
                    "count": len(differences),
                    "max_difference": max(differences, default=0),
                    "direction_changes": flips,
                    "artifact_identical": output["artifact"] == expected["artifact"],
                }
            )
    seal(
        target,
        "nav-replay.json",
        {"replay.json": write_json(target / "replay.json", rows)},
        status="REPLAY_VERIFIED",
        original_run=folder.name,
        original_scored_hash=verify(folder)["manifest_hash"],
    )
    return target
