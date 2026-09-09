"""负斜率诊断与固定模型2026新时段验证；不写数据库、不修改旧运行。"""

import importlib.metadata
import shutil
from datetime import UTC, date, datetime
from pathlib import Path
from time import perf_counter
from uuid import uuid4

from app.services.chronos_followup_data import (
    export_followup_answers,
    export_followup_inputs,
    load_followup_inputs,
)
from app.services.chronos_followup_diagnosis import diagnose_previous_run
from app.services.historical_nav_calibration import WINDOWS, restore_calibrated_artifact
from app.services.historical_nav_evaluation import FEATURE_NAMES, fixed_momentum_score
from app.services.historical_nav_training import predict_artifact_logits, predict_artifact_scores, sigmoid
from app.services.model_comparison_adapters import ChronosAdapter
from app.services.model_comparison_artifacts import (
    file_hash,
    fingerprint,
    read_json,
    read_jsonl,
    verify_files,
    write_json,
    write_jsonl,
)
from app.services.model_comparison_dataset import load_stage
from app.services.model_comparison_evaluation import evaluate_window
from app.services.model_comparison_protocol import REVISION, settings
from app.services.model_comparison_runner import checked_dataset
from app.services.trading_calendar import load_current_calendar

ROOT = Path(__file__).resolve().parents[2]
OLD_RUN = "model-comparison-4a329ccd-0cfc-4c81-9ecb-164e0fd9762d"
FUNDS = ("001632", "006730", "008888")
FIRST, LAST, OBSERVED = date(2026, 4, 9), date(2026, 8, 10), date(2026, 9, 8)


def freeze_followup() -> Path:
    import httpx

    old_folder = ROOT / ".local-runs" / OLD_RUN
    completion = read_json(old_folder / "complete.json")
    verify_files(old_folder, completion["files"])
    _, _, old_inputs = checked_dataset(old_folder)
    calendar = load_current_calendar()
    # Same tensor file was public in 2025; the current revision's later edits are documentation.
    with httpx.Client(timeout=30, follow_redirects=True) as client:
        response = client.get(
            "https://huggingface.co/api/models/amazon/chronos-2/revision/95a9710", params={"blobs": "true"}
        )
        response.raise_for_status()
        metadata = response.json()
    original_weight = next(f for f in metadata["siblings"] if f["rfilename"] == "model.safetensors")
    if original_weight["lfs"]["sha256"] != settings()["weight_sha256"]:
        raise ValueError("2025 public checkpoint differs from the frozen tensor file")
    run_id = uuid4()
    folder = ROOT / ".local-runs" / f"chronos-followup-{run_id}"
    folder.mkdir()
    (folder / "models").mkdir()
    for name in ("A-base", "A", "B"):
        shutil.copyfile(old_folder / "models" / f"VALIDATION_2024-{name}.json", folder / "models" / f"{name}.json")
    histories = load_stage(old_folder, WINDOWS[-1], "HISTORY", old_inputs)
    frequencies = {
        f: sum(r.y for r in histories if r.fund_code == f) / sum(r.fund_code == f for r in histories) for f in FUNDS
    }
    source_files = [*sorted((ROOT / "app/services").glob("chronos_followup*.py")), ROOT / "scripts/chronos_followup.py"]
    result = {
        "version": "CHRONOS_FIXED_MODELS_2026_V1",
        "run_id": str(run_id),
        "created_at": datetime.now(UTC).isoformat(),
        "old_run": OLD_RUN,
        "old_complete_sha256": file_hash(old_folder / "complete.json"),
        "funds": list(FUNDS),
        "start": str(FIRST),
        "end": str(LAST),
        "observation_date": str(OBSERVED),
        "planned_cutoffs": [str(d) for d in calendar.sessions if FIRST <= d <= LAST],
        "calendar_hash": calendar.content_hash,
        "revision": REVISION,
        "weight_sha256": settings()["weight_sha256"],
        "public_weight_evidence": {
            "original_revision": metadata["sha"],
            "weight_sha256": original_weight["lfs"]["sha256"],
            "commit_date": "2025-10-30",
            "source": "https://huggingface.co/amazon/chronos-2/commits/main",
        },
        "retrained": False,
        "new_calibration_fitted": False,
        "primary": "same-date raw direction accuracy and paired error rate; A raw probability > .5, B raw score > 0",
        "secondary": (
            "signed sigmoid using EXACT previously fitted a,b; negative slope permitted ONLY for this named "
            "offline diagnostic arm; old gate still rejected"
        ),
        "selection": (
            "full safe 2026 calendar window after 61 history sessions and matured 20-session answers; "
            "no selection by values or scores"
        ),
        "retrospective": True,
        "prospective": False,
        "publication_status": "MODEL_NOT_RELEASED",
        "baselines": settings()["baselines"],
        "baseline_up_frequencies": frequencies,
        "bootstrap": settings()["bootstrap"],
        "minimum_common_coverage": 0.9,
        "budget": settings()["budget"],
        "model_files": {
            p.relative_to(folder).as_posix(): file_hash(p) for p in sorted((folder / "models").glob("*.json"))
        },
        "source_files": {p.relative_to(ROOT).as_posix(): file_hash(p) for p in source_files},
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("chronos-forecasting", "torch", "transformers", "scikit-learn", "numpy", "scipy")
        },
        "unchanged_old_policy": (
            "No 2025 values/labels, no old protocol changes or successful status claims for negative slopes"
        ),
    }
    result["protocol_hash"] = fingerprint(result)
    write_json(folder / "protocol.json", result)
    write_json(folder / "diagnosis.json", diagnose_previous_run(old_folder))
    return folder


def validate_followup(folder: Path) -> dict:
    protocol = read_json(folder / "protocol.json")
    if fingerprint({k: v for k, v in protocol.items() if k != "protocol_hash"}) != protocol["protocol_hash"]:
        raise ValueError("followup protocol hash mismatch")
    calendar = load_current_calendar()
    if (
        protocol["version"] != "CHRONOS_FIXED_MODELS_2026_V1"
        or protocol["funds"] != list(FUNDS)
        or protocol["start"] != str(FIRST)
        or protocol["end"] != str(LAST)
        or protocol["observation_date"] != str(OBSERVED)
        or protocol["planned_cutoffs"] != [str(d) for d in calendar.sessions if FIRST <= d <= LAST]
        or protocol["calendar_hash"] != calendar.content_hash
        or protocol["retrained"]
        or protocol["new_calibration_fitted"]
        or protocol["weight_sha256"] != settings()["weight_sha256"]
    ):
        raise ValueError("2026 frozen followup scope mismatch")
    verify_files(folder, protocol["model_files"])
    verify_files(ROOT, protocol["source_files"])
    if any(importlib.metadata.version(k) != v for k, v in protocol["versions"].items()):
        raise ValueError("numerical environment differs from the frozen followup")
    return protocol


def signed_diagnostic_probability(slope: float, intercept: float, value: float) -> float:
    """Only the explicitly named offline arm; this does not grant publication approval."""
    import math

    if not all(math.isfinite(x) for x in (slope, intercept, value)):
        raise ValueError("nonfinite diagnostic mapping")
    return float(sigmoid(slope * value + intercept))


def forecast_followup(folder: Path, protocol: dict) -> dict:
    manifest = read_json(folder / "inputs_manifest.json")
    if (
        manifest["input_sha256"] != file_hash(folder / "inputs.jsonl")
        or manifest["protocol_hash"] != protocol["protocol_hash"]
    ):
        raise ValueError("input manifest mismatch")
    inputs = load_followup_inputs(folder)
    a = restore_calibrated_artifact((folder / "models/A.json").read_bytes())
    b = read_json(folder / "models/B.json")
    if b["model_hash"] != fingerprint({k: v for k, v in b.items() if k != "model_hash"}):
        raise ValueError("calibrator hash mismatch")
    adapter = ChronosAdapter(ROOT / ".local-runs/model-cache" / REVISION, protocol["budget"])
    if b["base_model_hash"] != adapter.base_hash:
        adapter.close()
        raise ValueError("calibrator bound to different Chronos weights")
    records = {}
    started = perf_counter()
    deadline = started + protocol["budget"]["total_seconds"]
    try:
        for lag in (0, 1):
            group = [i for i in inputs.values() if i.anchor_lag_sessions == lag]
            for offset in range(0, len(group), 8):
                batch = group[offset : offset + 8]
                logits = predict_artifact_logits(a.base_model, tuple(i.x for i in batch))
                raw_a = predict_artifact_scores(a.base_model, tuple(i.x for i in batch))
                try:
                    if perf_counter() >= deadline:
                        raise TimeoutError("followup budget exceeded")
                    outputs = adapter.predict(batch)
                except (ValueError, RuntimeError, OSError, TimeoutError, EOFError) as error:
                    outputs = [{"raw_score": None, "reason": type(error).__name__}] * len(batch)
                for item, z, raw, output in zip(batch, logits, raw_a, outputs, strict=True):
                    ap = signed_diagnostic_probability(a.calibrator.slope, a.calibrator.intercept, z)
                    bp = (
                        signed_diagnostic_probability(b["slope"], b["intercept"], output["raw_score"])
                        if output["raw_score"] is not None
                        else None
                    )
                    momentum = item.x[FEATURE_NAMES.index("return_20d")]
                    models = {
                        "A": {
                            "raw_score": float(raw),
                            "raw_direction": int(raw > 0.5),
                            "probability": ap,
                            "reason": None,
                            "old_gate_status": "CALIBRATOR_NONPOSITIVE_SLOPE",
                        },
                        "B": {**output, "probability": bp, "old_gate_status": "CALIBRATOR_NONPOSITIVE_SLOPE"},
                    }
                    baseline = {
                        "ALWAYS_UP": 1.0,
                        "TRAIN_UP_FREQUENCY": protocol["baseline_up_frequencies"][item.fund_code],
                        "MOMENTUM_20D": float(momentum > 0),
                        "FIXED_MOMENTUM_SCORE": float(fixed_momentum_score(momentum)),
                    }
                    models.update(
                        {
                            k: {"probability": v, "raw_direction": int(v > 0.5), "reason": None}
                            for k, v in baseline.items()
                        }
                    )
                    records[item.key] = {
                        "key": item.key,
                        "fund_code": item.fund_code,
                        "cutoff_date": str(item.cutoff_date),
                        "label_end_date": str(item.label_end_date),
                        "input_hash": item.input_hash,
                        "anchor_lag_sessions": lag,
                        "models": models,
                    }
                if offset % 80 == 0:
                    print(f"2026 prediction progress lag={lag}: {offset + len(batch)}/{len(group)}", flush=True)
        all_records = []
        calendar = load_current_calendar()
        for fund in FUNDS:
            for day in protocol["planned_cutoffs"]:
                end = str(calendar.future_sessions(date.fromisoformat(day))[-1])
                key = f"{fund}:{day}:{end}"
                missing = {"probability": None, "raw_direction": None, "reason": "INPUT_UNAVAILABLE"}
                all_records.append(
                    records.get(
                        key,
                        {
                            "key": key,
                            "fund_code": fund,
                            "cutoff_date": day,
                            "label_end_date": end,
                            "input_hash": None,
                            "models": {k: missing for k in ("A", "B", *protocol["baselines"])},
                        },
                    )
                )
        write_jsonl(folder / "predictions.jsonl", all_records)
        write_json(
            folder / "predictions_frozen.json",
            {
                "protocol_hash": protocol["protocol_hash"],
                "created_at": datetime.now(UTC).isoformat(),
                "predictions_sha256": file_hash(folder / "predictions.jsonl"),
                "answers_read": False,
                "retrospective": True,
            },
        )
        runtime = {"load": adapter.load, "batches": adapter.telemetry, "seconds": perf_counter() - started}
        write_json(folder / "runtime.json", runtime)
        return runtime
    finally:
        adapter.close()


def score_followup(folder: Path, protocol: dict) -> dict:
    predictions = list(read_jsonl(folder / "predictions.jsonl"))
    receipt = read_json(folder / "predictions_frozen.json")
    if receipt["predictions_sha256"] != file_hash(folder / "predictions.jsonl"):
        raise ValueError("predictions changed before scoring")
    answer_rows = list(read_jsonl(folder / "answers.jsonl"))
    labels = {r["key"]: r["y"] for r in answer_rows if r["y"] is not None}
    inputs = load_followup_inputs(folder)
    if len({r["key"] for r in answer_rows}) != len(answer_rows) or set(r["key"] for r in answer_rows) != set(inputs):
        raise ValueError("answer keys do not match exported inputs")
    for r in answer_rows:
        if r["y"] is not None and (
            r["input_hash"] != inputs[r["key"]].input_hash or r["label_available_at"] > protocol["observation_date"]
        ):
            raise ValueError("answer input or time boundary mismatch")
    signed = evaluate_window(predictions, labels, protocol["planned_cutoffs"], protocol["bootstrap"])
    raw_predictions = [
        {
            **p,
            "models": {
                m: {**v, "probability": float(v["raw_direction"]) if v.get("raw_direction") is not None else None}
                for m, v in p["models"].items()
            },
        }
        for p in predictions
    ]
    raw = evaluate_window(raw_predictions, labels, protocol["planned_cutoffs"], protocol["bootstrap"])
    raw["metric_semantics"] = "Brier on 0/1 directions equals direction error rate, not a probability-quality metric"
    return {
        "protocol_hash": protocol["protocol_hash"],
        "raw_direction": raw,
        "signed_calibration_diagnostic": signed,
        "planned": len(predictions),
        "input_count": len(inputs),
        "mature_answers": len(labels),
        "answer_failures": [r for r in answer_rows if r["y"] is None],
        "independence": (
            "2026 local-model holdout; retrospective current source versions, not genuinely prospective records"
        ),
        "publication_status": "MODEL_NOT_RELEASED",
        "model_selection_finalized": False,
    }


def execute_followup(folder: Path) -> dict:
    protocol = validate_followup(folder)
    export_followup_inputs(folder, protocol)
    forecast_followup(folder, protocol)
    export_followup_answers(folder, protocol, load_followup_inputs(folder))
    metrics = score_followup(folder, protocol)
    write_json(folder / "metrics.json", metrics)
    files = {p.relative_to(folder).as_posix(): file_hash(p) for p in sorted(folder.rglob("*")) if p.is_file()}
    write_json(folder / "complete.json", {"protocol_hash": protocol["protocol_hash"], "files": files})
    return metrics


def verify_followup(folder: Path) -> dict:
    protocol = validate_followup(folder)
    verify_files(folder, read_json(folder / "complete.json")["files"])
    metrics = score_followup(folder, protocol)
    if fingerprint(metrics) != fingerprint(read_json(folder / "metrics.json")):
        raise ValueError("followup score replay mismatch")
    return {"verified": True, "planned": metrics["planned"], "mature_answers": metrics["mature_answers"]}
