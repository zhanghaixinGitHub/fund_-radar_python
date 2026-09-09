"""明确现金批次的只读导出；模型输入与各段答案物理分离。"""

from collections import Counter
from datetime import date
from decimal import ROUND_HALF_UP, Decimal, localcontext
from pathlib import Path
from time import perf_counter
from uuid import UUID

from sqlalchemy import text
from sqlalchemy.orm import Session

from app.repositories.cash_reinvestment_research import iter_cash_batches
from app.schemas.model_comparison import ComparisonAnswer, ComparisonInput
from app.services.cash_reinvestment_research import cash_window_rows, prepare_cash_batches
from app.services.cash_reinvestment_storage import cash_hash, restore_cash_batch
from app.services.historical_nav_calibration import WINDOWS
from app.services.historical_nav_evaluation import FEATURE_NAMES, PreparedRow
from app.services.historical_nav_samples import _build_metrics
from app.services.model_comparison_artifacts import file_hash, fingerprint, read_jsonl, write_json, write_jsonl
from app.services.model_comparison_protocol import local_engine, validate_protocol
from app.services.trading_calendar import load_calendar


def validate_input(item: ComparisonInput) -> None:
    if fingerprint(item.model_dump(mode="json", exclude={"input_hash"})) != item.input_hash:
        raise ValueError("input hash mismatch")
    sessions = load_calendar().sessions
    anchor = sessions.index(item.anchor_nav_date)
    base = max(i for i, day in enumerate(sessions) if day <= item.cutoff_date)
    if (
        base - anchor != item.anchor_lag_sessions
        or sessions[base] != item.label_base_date
        or sessions[base + 20] != item.label_end_date
        or tuple(sessions[anchor - 60 : anchor + 1]) != item.history_dates
    ):
        raise ValueError("calendar/horizon alignment mismatch")
    with localcontext() as context:
        context.prec, context.rounding = 40, ROUND_HALF_UP
        metrics = _build_metrics(item.history_values)
    if metrics is None or tuple(Decimal(metrics[name]) for name in FEATURE_NAMES) != item.x:
        raise ValueError("seven features differ from the shared 61-point history")


def from_sample(item, batch_id: UUID) -> ComparisonInput:
    feature = item.feature_payload
    result = ComparisonInput(
        fund_code=item.fund_code,
        cutoff_date=item.cutoff_date,
        batch_id=batch_id,
        anchor_nav_date=item.anchor_nav_date,
        anchor_lag_sessions=item.anchor_lag_sessions,
        label_base_date=item.label_base_date,
        label_end_date=item.label_end_date,
        history_dates=tuple(p.nav_date for p in feature.history_series),
        history_values=tuple(Decimal(p.growth_index) for p in feature.history_series),
        history_available_at=tuple(p.available_at for p in feature.history_series),
        x=tuple(Decimal(feature.metrics[name]) for name in FEATURE_NAMES),
        source_feature_hash=item.feature_hash,
        input_hash="0" * 64,
    )
    result = result.model_copy(
        update={"input_hash": fingerprint(result.model_dump(mode="json", exclude={"input_hash"}))}
    )
    validate_input(result)
    return result


def export_dataset(folder: Path) -> dict:
    protocol = validate_protocol(folder)
    batches, read_only = [], False
    with Session(local_engine()) as session, session.begin():
        session.execute(text("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY"))
        read_only = session.scalar(text("SHOW transaction_read_only")) == "on"
        if not read_only:
            raise ValueError("database transaction not read only")
        for batch, rows in iter_cash_batches(session, tuple(UUID(i) for i in protocol["batch_ids"])):
            batches.append(restore_cash_batch(batch, rows))
    # No database connection is kept during validation, model loading or inference.
    data = prepare_cash_batches(batches, deadline=perf_counter() + 120)
    if data.report.dataset_hash != protocol["source_dataset_hash"]:
        raise ValueError("frozen source dataset mismatch")
    inputs, answers, seen, exclusions = {}, {}, {}, []
    for stored in batches:
        for sample in stored.preview.items:
            identity = f"{sample.fund_code}:{sample.cutoff_date}"
            content_hash = cash_hash(sample.model_dump(mode="json"))
            if identity in seen:
                if seen[identity] != content_hash:
                    raise ValueError("conflicting duplicate sample")
                continue
            seen[identity] = content_hash
            if not sample.feature_payload or sample.label_end_date > date(2024, 12, 31):
                exclusions.append(
                    {
                        "identity": identity,
                        "reason": "INPUT_UNAVAILABLE" if not sample.feature_payload else "TEST_PERIOD_PROTECTED",
                    }
                )
                continue
            item = from_sample(sample, stored.batch_id)
            inputs[item.key] = item
            if sample.offline_label:
                label = sample.offline_label
                answers[item.key] = ComparisonAnswer(
                    key=item.key,
                    input_hash=item.input_hash,
                    label_hash=sample.label_hash,
                    label_available_at=label.label_available_at,
                    y=label.label_up_20d,
                    future_return_20d=Decimal(label.future_return_20d),
                )
            else:
                exclusions.append({"identity": identity, "reason": "LABEL_UNAVAILABLE"})
    if len(inputs) > protocol["settings"]["budget"]["max_samples"]:
        raise ValueError("sample budget exceeded")
    write_jsonl(folder / "inputs.jsonl", (inputs[k].model_dump(mode="json") for k in sorted(inputs)))
    write_jsonl(folder / "labels.jsonl", (answers[k].model_dump(mode="json") for k in sorted(answers)))
    (folder / "answers").mkdir()
    windows = {}
    by_identity = {(i.fund_code, i.cutoff_date): i for i in inputs.values()}
    for window in WINDOWS:
        rows, counts = cash_window_rows(data, window)
        rows["HISTORY"] = tuple(
            r
            for r in data.rows
            if r.available_at <= window.calibration_end_date and r.label_available_at <= window.calibration_end_date
        )
        for stage, subset in rows.items():
            stage_answers = [
                answers[by_identity[r.fund_code, r.as_of_date].key].model_dump(mode="json") for r in subset
            ]
            write_jsonl(folder / "answers" / f"{window.window_id}-{stage}.jsonl", stage_answers)
        planned = protocol["settings"]["planned_cutoffs"][window.window_id]
        windows[window.window_id] = {
            "counts": [c.model_dump(mode="json") for c in counts],
            "status": "INSUFFICIENT_DATA" if any(any(c.missing.values()) for c in counts) else "READY",
            "planned_per_fund": len(planned),
            "input_available_per_fund": {
                f: sum(i.fund_code == f and str(i.cutoff_date) in planned for i in inputs.values())
                for f in protocol["settings"]["funds"]
            },
        }
    paths = [folder / "inputs.jsonl", folder / "labels.jsonl", *sorted((folder / "answers").glob("*.jsonl"))]
    manifest = {
        "protocol_hash": protocol["protocol_hash"],
        "database_read_only_verified": read_only,
        "preparation": data.report.model_dump(mode="json"),
        "windows": windows,
        "input_count": len(inputs),
        "answer_count": len(answers),
        "exclusions": exclusions,
        "anchor_lag_counts": dict(Counter(i.anchor_lag_sessions for i in inputs.values())),
        "files": {p.relative_to(folder).as_posix(): file_hash(p) for p in paths},
    }
    manifest["manifest_hash"] = fingerprint(manifest)
    write_json(folder / "dataset_manifest.json", manifest)
    return manifest


def load_inputs(folder: Path) -> dict[str, ComparisonInput]:
    result = {}
    for raw in read_jsonl(folder / "inputs.jsonl"):
        item = ComparisonInput.model_validate(raw)
        validate_input(item)
        if item.key in result:
            raise ValueError("duplicate input")
        result[item.key] = item
    return result


def load_stage(folder: Path, window, stage: str, inputs: dict[str, ComparisonInput]):
    bounds = {
        "FIT": (date(2021, 12, 31), window.fit_end_date),
        "HISTORY": (date(2021, 12, 31), window.calibration_end_date),
        "CALIBRATION": (window.fit_end_date, window.calibration_end_date),
        "EXAM": (window.calibration_end_date, window.evaluation_end_date),
    }
    lower, upper = bounds[stage]
    rows, seen = [], set()
    for raw in read_jsonl(folder / "answers" / f"{window.window_id}-{stage}.jsonl"):
        answer = ComparisonAnswer.model_validate(raw)
        item = inputs[answer.key]
        if answer.key in seen or answer.input_hash != item.input_hash:
            raise ValueError("answer join mismatch")
        if not lower < item.cutoff_date <= upper or not item.label_end_date <= answer.label_available_at <= upper:
            raise ValueError("answer time boundary mismatch")
        seen.add(answer.key)
        rows.append(
            PreparedRow(
                batch_id=item.batch_id,
                fund_code=item.fund_code,
                as_of_date=item.cutoff_date,
                available_at=item.cutoff_date,
                label_available_at=answer.label_available_at,
                x=item.x,
                y=int(answer.y),
                content_hash=fingerprint({"input": item.input_hash, "label": answer.label_hash}),
            )
        )
    return tuple(rows)
