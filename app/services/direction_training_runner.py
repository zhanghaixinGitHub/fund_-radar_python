"""固定阶段执行：准备训练资料 → 全部预测封存 → 导出考试答案 → 评分。"""

import json
from datetime import date
from time import perf_counter

from app.schemas.direction_training import DirectionAnswer, DirectionInput
from app.services.direction_training_artifacts import (
    digest,
    read_json,
    read_jsonl,
    read_seal,
    seal,
    write_json,
    write_jsonl,
)
from app.services.direction_training_dataset import FUNDS, build_answer, exam_dates, restore_source
from app.services.direction_training_process import run_process
from app.services.direction_training_protocol import BASELINES, CANDIDATES, load_protocol


def prepare(folder):
    protocol, frozen = load_protocol(folder)
    snapshot, coverage = read_json(folder / "source.json"), read_json(folder / "coverage.json")
    inputs = {i.key: i for i in map(DirectionInput.model_validate, read_jsonl(folder / "inputs.jsonl"))}
    sources = {f: restore_source(snapshot, f) for f in FUNDS}
    files, summary = {}, {}
    for window in protocol["windows"]:
        name = window["name"]
        counts = coverage["windows"][name]
        summary[name] = {"status": counts["status"], "counts": counts["funds"]}
        if counts["status"] != "READY":
            continue
        stages = {"FIT": [], "CAL": [], "EXAM": []}
        for fund in FUNDS:
            for stage in ("FIT", "CAL"):
                for cutoff in counts["funds"][fund][stage]["cutoffs"]:
                    item = inputs[f"{fund}:{cutoff}"]
                    answer, problems = build_answer(
                        fund, item.cutoff, sources[fund][1], sources[fund][2], include_value=True
                    )
                    upper = date.fromisoformat(window["fit_end" if stage == "FIT" else "cal_end"])
                    if problems or answer.available_at > upper:
                        raise ValueError("PREPARATION_COVERAGE_CHANGED")
                    stages[stage].append(
                        {"input": item.model_dump(mode="json"), "answer": answer.model_dump(mode="json")}
                    )
            _, planned = exam_dates(date.fromisoformat(window["cal_end"]), date.fromisoformat(window["exam_end"]))
            for cutoff in planned:
                if item := inputs.get(f"{fund}:{cutoff}"):
                    stages["EXAM"].append(item.model_dump(mode="json"))
        filename = f"prepared-{name}.json"
        files[filename] = write_json(folder / filename, {"window": window, **stages})
    files["prepared.json"] = write_json(folder / "prepared.json", summary)
    return seal(
        folder,
        "prepared-manifest.json",
        files,
        status="PREPARED",
        protocol_hash=frozen["manifest_hash"],
        exam_answers_exported=False,
        model_fitted=False,
    )


def load_stage(folder, name, predecessor, expected_field):
    current, previous = read_seal(folder, name), read_seal(folder, predecessor)
    if current[expected_field] != previous["manifest_hash"]:
        raise ValueError("STAGE_LINK_MISMATCH")
    return current


def jobs_for_window(data):
    common = {"window": data["window"], "fit": data["FIT"], "exam": data["EXAM"], "base": None}
    return {
        b: {**common, "candidate": b, "cal": data["CAL"] if b in ("A_CAL", "BASELINES") else []}
        for b in (*CANDIDATES, "BASELINES")
    }


def predict(folder):
    protocol, _ = load_protocol(folder)
    prepared = load_stage(folder, "prepared-manifest.json", "protocol-manifest.json", "protocol_hash")
    summary = read_json(folder / "prepared.json")
    rows, results, files = [], {}, {}
    deadline = perf_counter() + protocol["budget"]["total_seconds"]
    for window in protocol["windows"]:
        name = window["name"]
        if summary[name]["status"] != "READY":
            results[name] = {b: {"status": "INSUFFICIENT_DATA"} for b in (*CANDIDATES, *BASELINES)}
            continue
        data = read_json(folder / f"prepared-{name}.json")
        jobs, outputs = jobs_for_window(data), {}
        for branch in ("A", "A_CAL", "B", "C", "BASELINES"):
            if branch == "A_CAL":
                base = outputs["A"].get("artifact")
                if not base:
                    outputs[branch] = {"status": "FAILED", "reason": "BASE_MODEL_FAILED"}
                    continue
                jobs[branch]["base"] = json.dumps(base)
            remaining = deadline - perf_counter()
            if remaining <= 0:
                outputs[branch] = {"status": "FAILED", "reason": "TOTAL_TIME_BUDGET"}
                continue
            job_name = f"job-{name}-{branch}.json"
            files[job_name] = write_json(folder / job_name, jobs[branch])
            outputs[branch] = run_process(jobs[branch], seconds=min(120, remaining))
            if branch == "B" and outputs[branch]["status"] == "NOT_DISTINCT":
                outputs[branch].update(
                    scores={"B": outputs["A"].get("scores", {}).get("A", [])}, artifact=outputs["A"].get("artifact")
                )
        output_name = f"models-{name}.json"
        files[output_name] = write_json(folder / output_name, outputs)
        results[name] = {
            b: {k: v for k, v in result.items() if k not in ("scores", "artifact")} for b, result in outputs.items()
        }
        input_map = {f"{i['fund']}:{i['cutoff']}": (index, i) for index, i in enumerate(data["EXAM"])}
        _, planned = exam_dates(date.fromisoformat(window["cal_end"]), date.fromisoformat(window["exam_end"]))
        for fund in FUNDS:
            for cutoff in planned:
                key = f"{fund}:{cutoff}"
                for branch in (*CANDIDATES, *BASELINES):
                    output = outputs["BASELINES" if branch in BASELINES else branch]
                    entry = {
                        "sample_key": key,
                        "fund": fund,
                        "cutoff": str(cutoff),
                        "window": name,
                        "candidate": branch,
                        "score": None,
                        "predicted_up": None,
                        "input_hash": None,
                        "status": output["status"],
                        "reason": output.get("reason"),
                    }
                    if key not in input_map:
                        entry.update(status="INPUT_UNAVAILABLE", reason="HISTORY_INPUT_UNAVAILABLE")
                    else:
                        index, item = input_map[key]
                        scores = output.get("scores", {}).get(branch, [])
                        entry["input_hash"] = item["input_hash"]
                        if len(scores) == len(data["EXAM"]):
                            entry.update(score=scores[index], predicted_up=int(scores[index] > 0.5))
                    rows.append(entry)
    files["predictions.jsonl"] = write_jsonl(folder / "predictions.jsonl", rows)
    files["execution.json"] = write_json(folder / "execution.json", results)
    return seal(
        folder,
        "predictions-manifest.json",
        files,
        status="PREDICTED",
        prepared_hash=prepared["manifest_hash"],
        all_candidate_attempts_finished=True,
        exam_answers_exported=False,
    )


def score(folder):
    protocol, _ = load_protocol(folder)
    load_stage(folder, "prepared-manifest.json", "protocol-manifest.json", "protocol_hash")
    predictions = load_stage(folder, "predictions-manifest.json", "prepared-manifest.json", "prepared_hash")
    snapshot = read_json(folder / "source.json")
    sources = {f: restore_source(snapshot, f) for f in FUNDS}
    rows = read_jsonl(folder / "predictions.jsonl")
    window_map = {w["name"]: w for w in protocol["windows"]}
    answers, seen = [], set()
    for row in rows:
        identity = (row["window"], row["sample_key"])
        if identity in seen:
            continue
        seen.add(identity)
        fund, cutoff = row["fund"], date.fromisoformat(row["cutoff"])
        answer, problems = build_answer(fund, cutoff, sources[fund][1], sources[fund][2], include_value=True)
        if answer and answer.available_at > date.fromisoformat(window_map[row["window"]]["exam_end"]):
            answer, problems = None, ["ANSWER_NOT_MATURE_BY_STAGE_END"]
        payload = answer.model_dump(mode="json") if answer else None
        answers.append(
            {
                "window": row["window"],
                "sample_key": row["sample_key"],
                "answer": payload,
                "answer_hash": digest(payload) if payload else None,
                "issues": problems,
            }
        )
    from app.services.direction_training_evaluation import evaluate

    metrics = evaluate(protocol, rows, answers, read_json(folder / "execution.json"))
    files = {
        "answers.exam.jsonl": write_jsonl(folder / "answers.exam.jsonl", answers),
        "metrics.json": write_json(folder / "metrics.json", metrics),
    }
    return seal(
        folder,
        "scored-manifest.json",
        files,
        status="SCORED",
        predictions_hash=predictions["manifest_hash"],
        test_scored=False,
        publication_status="MODEL_NOT_RELEASED",
    )


def replay(folder):
    """新编号重训重算已保存的题目；复现核验不算新一轮效果证据。"""
    from app.services.direction_training_artifacts import new_folder

    original = verify(folder)
    target = new_folder()
    names = set()
    for filename in ("coverage-manifest.json", "protocol-manifest.json", "prepared-manifest.json"):
        names.add(filename)
        names.update(read_seal(folder, filename)["files"])
    for filename in sorted(names):
        if filename.endswith(".jsonl"):
            write_jsonl(target / filename, read_jsonl(folder / filename))
        else:
            write_json(target / filename, read_json(folder / filename))
    predicted = predict(target)
    before, after = read_jsonl(folder / "predictions.jsonl"), read_jsonl(target / "predictions.jsonl")
    if len(before) != len(after):
        raise ValueError("REPLAY_COUNT_MISMATCH")
    maximum, compared = 0.0, 0
    for a, b in zip(before, after, strict=True):
        if {k: v for k, v in a.items() if k != "score"} != {k: v for k, v in b.items() if k != "score"}:
            raise ValueError("REPLAY_DIRECTION_OR_IDENTITY_MISMATCH")
        if a["score"] is not None:
            delta = abs(a["score"] - b["score"])
            maximum = max(maximum, delta)
            compared += 1
            if delta > 1e-12:
                raise ValueError("REPLAY_SCORE_MISMATCH")
    report = {
        "origin_folder": folder.name,
        "origin_hash": original["manifest_hash"],
        "prediction_count": len(after),
        "compared_score_count": compared,
        "max_absolute_score_delta": maximum,
        "tolerance": 1e-12,
        "directions_identical": True,
        "is_new_effectiveness_evidence": False,
    }
    files = {"replay.json": write_json(target / "replay.json", report)}
    return seal(
        target,
        "complete.json",
        files,
        status="REPLAY_VERIFIED",
        predictions_hash=predicted["manifest_hash"],
        replay_folder=str(target),
        model_refitted=True,
        database_written=False,
        test_scored=False,
        publication_status="MODEL_NOT_RELEASED",
    )


def verify(folder):
    protocol, _ = load_protocol(folder)
    load_stage(folder, "prepared-manifest.json", "protocol-manifest.json", "protocol_hash")
    load_stage(folder, "predictions-manifest.json", "prepared-manifest.json", "prepared_hash")
    scored = load_stage(folder, "scored-manifest.json", "predictions-manifest.json", "predictions_hash")
    from app.services.direction_training_evaluation import evaluate

    answers = read_jsonl(folder / "answers.exam.jsonl")
    for item in answers:
        if item["answer"]:
            answer = DirectionAnswer.model_validate(item["answer"])
            if answer.key != item["sample_key"] or digest(item["answer"]) != item["answer_hash"]:
                raise ValueError("ANSWER_IDENTITY_OR_HASH")
    metrics = evaluate(
        protocol, read_jsonl(folder / "predictions.jsonl"), answers, read_json(folder / "execution.json")
    )
    if digest(metrics) != digest(read_json(folder / "metrics.json")):
        raise ValueError("METRICS_REPLAY_MISMATCH")
    if (folder / "complete.json").exists():
        result = read_seal(folder, "complete.json")
        if result["scored_hash"] != scored["manifest_hash"]:
            raise ValueError("COMPLETE_LINK_MISMATCH")
        return result
    return seal(
        folder,
        "complete.json",
        {"metrics.json": scored["files"]["metrics.json"]},
        status="VERIFIED",
        scored_hash=scored["manifest_hash"],
        research_status=metrics["research_status"],
        model_refitted=False,
        database_written=False,
        test_scored=False,
        publication_status="MODEL_NOT_RELEASED",
    )
