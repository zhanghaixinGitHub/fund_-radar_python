"""离线分阶段执行：校准段建模，冻结所有考试预测，然后才能读取考试答案。"""

import importlib.metadata
import platform
from datetime import UTC, date, datetime
from pathlib import Path
from time import perf_counter

from app.services.calibration_policy import calibration_diagnostic
from app.services.cash_reinvestment_research import FUNDS, VERSIONS
from app.services.historical_nav_calibration import WINDOWS, restore_calibrated_artifact
from app.services.historical_nav_evaluation import FEATURE_NAMES
from app.services.historical_nav_training import restore_logistic_artifact
from app.services.model_comparison_adapters import (
    ChronosAdapter,
    calibrated_chronos_score,
    fit_chronos_calibrator,
    fit_self_trained,
    predict_self_trained,
)
from app.services.model_comparison_artifacts import (
    file_hash,
    fingerprint,
    read_json,
    read_jsonl,
    verify_files,
    write_json,
    write_jsonl,
)
from app.services.model_comparison_dataset import load_inputs, load_stage
from app.services.model_comparison_evaluation import evaluate_window
from app.services.model_comparison_protocol import validate_protocol
from app.services.trading_calendar import load_calendar

ROOT = Path(__file__).resolve().parents[2]
BASELINES = ("ALWAYS_UP", "TRAIN_UP_FREQUENCY", "MOMENTUM_20D", "FIXED_MOMENTUM_SCORE")


def checked_dataset(folder):
    protocol = validate_protocol(folder)
    manifest = read_json(folder / "dataset_manifest.json")
    if (
        manifest["manifest_hash"] != fingerprint({k: v for k, v in manifest.items() if k != "manifest_hash"})
        or manifest["protocol_hash"] != protocol["protocol_hash"]
    ):
        raise ValueError("dataset manifest mismatch")
    verify_files(folder, manifest["files"])
    return protocol, manifest, load_inputs(folder)


def freeze_runtime(folder):
    import psutil

    versions = dict(
        sorted((d.metadata["Name"].lower().replace("_", "-"), d.version) for d in importlib.metadata.distributions())
    )
    expected = {
        "chronos-forecasting": "2.3.2",
        "torch": "2.8.0+cpu",
        "transformers": "4.57.6",
        "huggingface-hub": "0.36.2",
        "numpy": "2.5.3",
        "scikit-learn": "1.9.0",
        "scipy": "1.18.1",
    }
    if any(versions.get(name) != version for name, version in expected.items()):
        raise ValueError("Use the pinned isolated Chronos environment")
    result = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "versions": versions,
        "physical_cores": psutil.cpu_count(logical=False),
        "logical_cores": psutil.cpu_count(),
        "total_memory_bytes": psutil.virtual_memory().total,
        "available_memory_bytes": psutil.virtual_memory().available,
        "source_files": {
            p.relative_to(ROOT).as_posix(): file_hash(p)
            for p in sorted(
                set((ROOT / "app" / "services").glob("model_comparison_*.py"))
                | {ROOT / "app" / "schemas" / name for name in ("model_comparison.py", "calibration_diagnostic.py")}
                | {
                    ROOT / "app" / "services" / name
                    for name in (
                        "historical_nav_training.py",
                        "calibration_policy.py",
                        "historical_nav_calibration.py",
                        "historical_nav_evaluation.py",
                        "cash_reinvestment_research.py",
                        "cash_reinvestment_storage.py",
                        "historical_nav_samples.py",
                        "trading_calendar.py",
                        "cash_exam_plan.py",
                    )
                }
            )
        },
    }
    if (folder / "environment.json").exists():
        old = read_json(folder / "environment.json")
        if old["versions"] != result["versions"] or old["python"] != result["python"]:
            raise ValueError("run environment changed")
        if old["source_files"] != result["source_files"]:
            raise ValueError("comparison source changed after model preparation")
    else:
        write_json(folder / "environment.json", result)
        with (folder / "requirements-resolved.txt").open("x", encoding="utf-8") as stream:
            stream.write("--extra-index-url https://download.pytorch.org/whl/cpu\n")
            stream.write("\n".join(f"{name}=={version}" for name, version in versions.items()) + "\n")
    return result


def _predict_many(adapter, items, deadline):
    results = {}
    # Batch members remain independent univariate series; never a multivariate fund group.
    for lag in (0, 1):
        group = [item for item in items if item.anchor_lag_sessions == lag]
        for offset in range(0, len(group), adapter.budget["batch_size"]):
            batch = group[offset : offset + adapter.budget["batch_size"]]
            try:
                if perf_counter() >= deadline:
                    raise TimeoutError("RUN_TIME_BUDGET")
                outputs = adapter.predict(batch)
            except (ValueError, RuntimeError, TimeoutError, MemoryError, OSError, EOFError) as error:
                outputs = [{"reason": type(error).__name__ + ":" + str(error)[:160], "raw_score": None}] * len(batch)
            for item, output in zip(batch, outputs, strict=True):
                results[item.key] = output
            if offset % 80 == 0:
                print(f"Chronos input progress {offset + len(batch)}/{len(group)}", flush=True)
    return results


def _model_paths(folder, window):
    model_dir = folder / "models"
    model_dir.mkdir(exist_ok=True)
    return {name: model_dir / f"{window.window_id}-{name}.json" for name in ("A-base", "A", "B", "binding")}


def prepare_models(folder, window, inputs, protocol, adapter, deadline):
    paths = _model_paths(folder, window)
    stage_hashes = {s: file_hash(folder / "answers" / f"{window.window_id}-{s}.jsonl") for s in ("FIT", "CALIBRATION")}
    binding = {
        "protocol_hash": protocol["protocol_hash"],
        "stage_hashes": stage_hashes,
        "chronos_base_hash": adapter.base_hash,
        "environment_hash": file_hash(folder / "environment.json"),
    }
    if paths["binding"].exists():
        saved = read_json(paths["binding"])
        if any(saved[key] != value for key, value in binding.items()):
            raise ValueError("model/data/protocol binding mismatch")
        verify_files(folder, saved["files"])
        base = restore_logistic_artifact(paths["A-base"].read_bytes())
        calibrated = restore_calibrated_artifact(paths["A"].read_bytes())
        return base, calibrated, read_json(paths["B"]), saved
    fit = load_stage(folder, window, "FIT", inputs)
    cal = load_stage(folder, window, "CALIBRATION", inputs)
    by_day = {(i.fund_code, i.cutoff_date): i for i in inputs.values()}
    cal_inputs = [by_day[r.fund_code, r.as_of_date] for r in cal]
    started = perf_counter()
    base, calibrated = fit_self_trained(fit, cal, window, VERSIONS)
    binding["A_fit_and_calibration_seconds"] = perf_counter() - started
    write_json(paths["A-base"], base.model_dump(mode="json"))
    write_json(paths["A"], calibrated.model_dump(mode="json"))
    started = perf_counter()
    forecasts = _predict_many(adapter, cal_inputs, deadline)
    write_jsonl(
        folder / "models" / f"{window.window_id}-B-calibration-predictions.jsonl",
        ({"key": i.key, "input_hash": i.input_hash, **forecasts[i.key]} for i in cal_inputs),
    )
    try:
        if any(forecasts[i.key]["raw_score"] is None for i in cal_inputs):
            raise ValueError("CALIBRATION_PREDICTION_FAILED")
        b_model = fit_chronos_calibrator(
            cal,
            [forecasts[i.key]["raw_score"] for i in cal_inputs],
            lower=window.fit_end_date,
            upper=window.calibration_end_date,
            base_hash=adapter.base_hash,
        )
    except (ValueError, RuntimeError) as error:
        b_model = {"rejected": str(error), "base_model_hash": adapter.base_hash}
    write_json(paths["B"], b_model)
    binding["B_calibration_and_forecast_seconds"] = perf_counter() - started
    binding["files"] = {
        p.relative_to(folder).as_posix(): file_hash(p) for name, p in paths.items() if name != "binding"
    }
    cal_path = folder / "models" / f"{window.window_id}-B-calibration-predictions.jsonl"
    binding["files"][cal_path.relative_to(folder).as_posix()] = file_hash(cal_path)
    write_json(paths["binding"], binding)
    return base, calibrated, b_model, binding


def _with_probability(output, b_model, base_hash, *, reject_nonpositive_slope=False):
    output = dict(output)
    output["probability"] = None
    if output.get("raw_score") is not None:
        try:
            if "rejected" in b_model:
                raise ValueError(b_model["rejected"])
            output["probability"] = float(
                calibrated_chronos_score(
                    b_model, output["raw_score"], base_hash, reject_nonpositive_slope=reject_nonpositive_slope
                )
            )
            if not reject_nonpositive_slope:
                output["calibration"] = calibration_diagnostic(b_model["slope"], b_model["intercept"]).model_dump(
                    mode="json"
                )
        except ValueError as error:
            output["reason"] = str(error)
    return output


def smoke(folder: Path, checkpoint: Path):
    protocol, manifest, inputs = checked_dataset(folder)
    policy = {"reject_nonpositive_slope": protocol["settings"]["calibration"]["reject_nonpositive_slope"]}
    freeze_runtime(folder)
    window = WINDOWS[-1]
    if manifest["windows"][window.window_id]["status"] != "READY":
        raise ValueError("MAIN_WINDOW_INSUFFICIENT_DATA")
    adapter = ChronosAdapter(checkpoint, protocol["settings"]["budget"])
    try:
        base, a_model, b_model, binding = prepare_models(
            folder, window, inputs, protocol, adapter, perf_counter() + protocol["settings"]["budget"]["total_seconds"]
        )
        cal = load_stage(folder, window, "CALIBRATION", inputs)
        by_day = {(i.fund_code, i.cutoff_date): i for i in inputs.values()}
        selected = []
        for fund in FUNDS:
            subset = [r for r in cal if r.fund_code == fund]
            selected.extend(by_day[r.fund_code, r.as_of_date] for r in (subset[0], subset[-1]))
        first = _predict_many(adapter, selected, perf_counter() + 300)
        second = _predict_many(adapter, selected, perf_counter() + 300)
        started = perf_counter()
        a = predict_self_trained(base, a_model, selected, **policy)
        a_seconds = perf_counter() - started
        rows = []
        for item, a_output in zip(selected, a, strict=True):
            if first[item.key]["raw_score"] is None or second[item.key]["raw_score"] is None:
                raise ValueError("SMOKE_FORECAST_FAILED")
            difference = abs(first[item.key]["raw_score"] - second[item.key]["raw_score"])
            if difference > protocol["settings"]["replay_tolerance"]["raw_score_absolute"]:
                raise ValueError("SMOKE_REPLAY_MISMATCH")
            rows.append(
                {
                    "key": item.key,
                    "input_hash": item.input_hash,
                    "A": a_output,
                    "B": _with_probability(first[item.key], b_model, adapter.base_hash, **policy),
                    "repeat_delta": difference,
                }
            )
        result = {
            "protocol_hash": protocol["protocol_hash"],
            "selected_from": "CALIBRATION_ONLY",
            "real_lags": sorted({i.anchor_lag_sessions for i in selected}),
            "rows": rows,
            "chronos_load": adapter.load,
            "chronos_batches": adapter.telemetry,
            "A_warm_predict_seconds": a_seconds,
            "model_binding": binding,
            "no_exam_answers_read": True,
        }
        write_json(folder / "smoke.json", result)
        return result
    finally:
        adapter.close()


def _baseline_outputs(item, histories):
    # Reuse the project's four-decimal fixed momentum rule, and all matured pre-exam history.
    from app.services.historical_nav_evaluation import fixed_momentum_score

    history = histories.get(item.fund_code, [])
    r = item.x[FEATURE_NAMES.index("return_20d")]
    probabilities = {
        "ALWAYS_UP": 1.0,
        "TRAIN_UP_FREQUENCY": sum(history) / len(history) if history else None,
        "MOMENTUM_20D": float(r > 0),
        "FIXED_MOMENTUM_SCORE": float(fixed_momentum_score(r)),
    }
    return {
        name: {
            "probability": p,
            "raw_direction": int(p > 0.5) if p is not None else None,
            "reason": None if p is not None else "HISTORY_UNAVAILABLE",
        }
        for name, p in probabilities.items()
    }


def run_comparison(folder: Path, checkpoint: Path):
    protocol, manifest, inputs = checked_dataset(folder)
    policy = {"reject_nonpositive_slope": protocol["settings"]["calibration"]["reject_nonpositive_slope"]}
    freeze_runtime(folder)
    if not (folder / "smoke.json").exists():
        raise ValueError("complete real development smoke before full comparison")
    adapter = ChronosAdapter(checkpoint, protocol["settings"]["budget"])
    started = perf_counter()
    deadline = started + protocol["settings"]["budget"]["total_seconds"]
    runtime = {"chronos_load": adapter.load, "windows": {}, "started_at": datetime.now(UTC).isoformat()}
    predictions = []
    by_day = {(i.fund_code, str(i.cutoff_date)): i for i in inputs.values()}
    calendar = load_calendar().sessions
    try:
        for window in WINDOWS:
            name = window.window_id
            dates = protocol["settings"]["planned_cutoffs"][name]
            eligible = manifest["windows"][name]["status"] == "READY"
            histories = {f: [] for f in FUNDS}
            for row in load_stage(folder, window, "HISTORY", inputs):
                histories[row.fund_code].append(row.y)
            items = [by_day[f, day] for f in FUNDS for day in dates if (f, day) in by_day]
            a_outputs, b_outputs, model_hashes = {}, {}, {"A": None, "B": None}
            if eligible:
                base, a_model, b_model, binding = prepare_models(folder, window, inputs, protocol, adapter, deadline)
                batch_start = perf_counter()
                a_outputs = dict(
                    zip((i.key for i in items), predict_self_trained(base, a_model, items, **policy), strict=True)
                )
                runtime["windows"][name] = {"A_warm_predict_seconds": perf_counter() - batch_start, "training": binding}
                batch_start = perf_counter()
                b_raw = _predict_many(adapter, items, deadline)
                b_outputs = {
                    key: _with_probability(value, b_model, adapter.base_hash, **policy) for key, value in b_raw.items()
                }
                runtime["windows"][name]["B_warm_predict_seconds"] = perf_counter() - batch_start
                model_hashes = {"A": a_model.model_hash, "B": b_model.get("model_hash")}
            for fund in FUNDS:
                for day in dates:
                    item = by_day.get((fund, day))
                    end = str(calendar[calendar.index(date.fromisoformat(day)) + 20])
                    key = f"{fund}:{day}:{end}"
                    reason = "INPUT_UNAVAILABLE" if item is None else "INSUFFICIENT_DATA"
                    missing = {"probability": None, "raw_score": None, "reason": reason}
                    outputs = {"A": a_outputs.get(key, missing), "B": b_outputs.get(key, missing)}
                    outputs.update(_baseline_outputs(item, histories) if item else {m: missing for m in BASELINES})
                    predictions.append(
                        {
                            "window": name,
                            "fund_code": fund,
                            "cutoff_date": day,
                            "label_end_date": end,
                            "key": key,
                            "input_hash": item.input_hash if item else None,
                            "anchor_nav_date": str(item.anchor_nav_date) if item else None,
                            "anchor_lag_sessions": item.anchor_lag_sessions if item else None,
                            "model_hashes": model_hashes,
                            "models": outputs,
                        }
                    )
        # This receipt is written before the first EXAM label file is parsed by this runner.
        write_jsonl(folder / "predictions.jsonl", predictions)
        write_json(
            folder / "predictions_frozen.json",
            {
                "protocol_hash": protocol["protocol_hash"],
                "predictions_sha256": file_hash(folder / "predictions.jsonl"),
                "created_at": datetime.now(UTC).isoformat(),
                "exam_answers_read": False,
            },
        )
        runtime["chronos_batches"] = adapter.telemetry
    finally:
        adapter.close()
    results = score_frozen_predictions(folder)
    runtime["total_seconds"] = perf_counter() - started
    runtime["finished_at"] = datetime.now(UTC).isoformat()
    write_json(folder / "metrics.json", results)
    write_json(folder / "runtime.json", runtime)
    files = {p.relative_to(folder).as_posix(): file_hash(p) for p in sorted(folder.rglob("*")) if p.is_file()}
    write_json(
        folder / "complete.json",
        {
            "protocol_hash": protocol["protocol_hash"],
            "files": files,
            "status": "COMPARISON_RECORDED",
            "publication_status": "MODEL_NOT_RELEASED",
        },
    )
    verify_files(folder, read_json(folder / "complete.json")["files"])
    return results


def score_frozen_predictions(folder):
    protocol, manifest, inputs = checked_dataset(folder)
    receipt = read_json(folder / "predictions_frozen.json")
    if (
        receipt["predictions_sha256"] != file_hash(folder / "predictions.jsonl")
        or receipt["protocol_hash"] != protocol["protocol_hash"]
    ):
        raise ValueError("predictions not frozen or changed")
    predictions = list(read_jsonl(folder / "predictions.jsonl"))
    by_day = {(i.fund_code, i.cutoff_date): i for i in inputs.values()}
    windows = {}
    for window in WINDOWS:
        rows = load_stage(folder, window, "EXAM", inputs)
        labels = {by_day[r.fund_code, r.as_of_date].key: r.y for r in rows}
        evaluated = evaluate_window(
            [p for p in predictions if p["window"] == window.window_id],
            labels,
            protocol["settings"]["planned_cutoffs"][window.window_id],
            protocol["settings"]["bootstrap"],
        )
        windows[window.window_id] = {"data_status": manifest["windows"][window.window_id]["status"], **evaluated}
        if not protocol["settings"]["calibration"]["reject_nonpositive_slope"]:
            windows[window.window_id]["calibration_diagnostics"] = {
                name: next(
                    (
                        p["models"][name]["calibration"]
                        for p in predictions
                        if p["window"] == window.window_id and "calibration" in p["models"][name]
                    ),
                    None,
                )
                for name in ("A", "B")
            }
    return {
        "protocol_hash": protocol["protocol_hash"],
        "manifest_hash": manifest["manifest_hash"],
        "windows": windows,
        "conclusion": windows["VALIDATION_2024"]["conclusion"],
        "scope": "Historical exploration; no publication; no 2025 values/answers/scores",
    }


def verify_run(folder):
    completion = read_json(folder / "complete.json")
    verify_files(folder, completion["files"])
    fresh = score_frozen_predictions(folder)
    if fingerprint(fresh) != fingerprint(read_json(folder / "metrics.json")):
        raise ValueError("metric replay mismatch")
    return {"verified_files": len(completion["files"]), "metrics_recomputed": True, "conclusion": fresh["conclusion"]}


def replay_smoke(folder: Path, checkpoint: Path):
    """Fresh environment replay; writes a separate receipt, never the sealed run."""
    from uuid import uuid4

    protocol, _, inputs = checked_dataset(folder)
    policy = {"reject_nonpositive_slope": protocol["settings"]["calibration"]["reject_nonpositive_slope"]}
    freeze_runtime(folder)
    verified = verify_run(folder)
    reference = read_json(folder / "smoke.json")
    adapter = ChronosAdapter(checkpoint, protocol["settings"]["budget"])
    try:
        selected = [inputs[r["key"]] for r in reference["rows"]]
        output = _predict_many(adapter, selected, perf_counter() + 300)
        model_paths = _model_paths(folder, WINDOWS[-1])
        base = restore_logistic_artifact(model_paths["A-base"].read_bytes())
        a_model = restore_calibrated_artifact(model_paths["A"].read_bytes())
        b_model = read_json(model_paths["B"])
        a_outputs = predict_self_trained(base, a_model, selected, **policy)
        deltas = []
        for old, item, a in zip(reference["rows"], selected, a_outputs, strict=True):
            b = _with_probability(output[item.key], b_model, adapter.base_hash, **policy)
            raw_delta = abs(old["B"]["raw_score"] - b["raw_score"])
            probability_deltas = []
            for name, current in (("A", a), ("B", b)):
                if (old[name]["probability"] is None) != (current["probability"] is None):
                    raise ValueError("replay probability availability changed")
                if current["probability"] is not None:
                    probability_deltas.append(abs(old[name]["probability"] - current["probability"]))
            max_probability_delta = max(probability_deltas, default=0.0)
            tolerance = protocol["settings"]["replay_tolerance"]
            if raw_delta > tolerance["raw_score_absolute"] or max_probability_delta > tolerance["probability_absolute"]:
                raise ValueError("clean environment replay outside frozen tolerance")
            deltas.append({"key": item.key, "raw_delta": raw_delta, "probability_delta": max_probability_delta})
        result = {
            "run": folder.name,
            "protocol_hash": protocol["protocol_hash"],
            "verified": verified,
            "rows": deltas,
            "load": adapter.load,
            "batches": adapter.telemetry,
            "created_at": datetime.now(UTC).isoformat(),
        }
        receipt = folder.parent / f"chronos2-replay-{uuid4()}.json"
        write_json(receipt, result)
        return {
            "receipt": str(receipt),
            "samples": len(deltas),
            "maximum_raw_delta": max(d["raw_delta"] for d in deltas),
            "maximum_probability_delta": max(d["probability_delta"] for d in deltas),
            **verified,
        }
    finally:
        adapter.close()
