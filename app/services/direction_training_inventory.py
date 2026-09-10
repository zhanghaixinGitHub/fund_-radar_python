"""T00：核验保留的自训练资料并重放现金模型，不拟合、不读保留测试。"""

import json
from collections import Counter
from contextlib import contextmanager
from decimal import Decimal
from importlib.metadata import version
from time import perf_counter
from uuid import UUID

from sqlalchemy import event, text
from sqlalchemy.orm import Session

from app.db.session import get_nav_sample_storage_engine
from app.repositories.cash_reinvestment_research import find_research
from app.schemas.cash_reinvestment_research import CashPrepareRequest
from app.schemas.historical_nav_calibration import HistoricalNavCalibrationResponse
from app.schemas.historical_nav_training import HistoricalNavTrainingResponse
from app.services.cash_reinvestment_research import (
    cash_window_rows,
    load_cash_dataset_in_session,
    restore_research,
)
from app.services.direction_training_artifacts import (
    ROOT,
    new_folder,
    read_json,
    read_seal,
    seal,
    verify_files,
    write_json,
    write_jsonl,
)
from app.services.historical_nav_calibration import predict_calibrated_scores, restore_calibrated_artifact
from app.services.historical_nav_evaluation import calculate_baseline_metrics
from app.services.historical_nav_training import predict_artifact_scores, restore_logistic_artifact

CASH_RUN = UUID("f70feb1a-129d-4482-b66d-f4e2e3a5425c")
LEGACY = {
    "candidate": "nav-candidate-6e25d583-5ab3-443c-bd65-8aa7ae2ce5d1",
    "calibration": "nav-calibration-f0f9f709-b82b-4af1-8326-f09b11abac50",
    "diagnostics": "nav-diagnostics-1054d538-301e-44a1-858c-0deca79a32f6",
}


@contextmanager
def local_snapshot():
    """使用项目配置，限制本地fund_ai；只记录SQL类别，不记录连接信息和参数。"""
    engine = get_nav_sample_storage_engine()
    if engine.url.host not in ("localhost", "127.0.0.1", "::1") or engine.url.database != "fund_ai":
        raise ValueError("LOCAL_FUND_AI_REQUIRED")
    statements = Counter()

    def capture(conn, cursor, statement, parameters, context, executemany):
        verb = statement.lstrip().split(None, 1)[0].upper()
        if verb not in ("SELECT", "SET", "SHOW"):
            raise ValueError("NON_READ_ONLY_STATEMENT")
        statements[verb] += 1

    # 连接级监听不会影响同进程其他调用者。
    with engine.connect() as connection:
        event.listen(connection, "before_cursor_execute", capture)
        try:
            with Session(bind=connection) as session, session.begin():
                session.execute(text("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY"))
                if session.scalar(text("SHOW transaction_read_only")) != "on":
                    raise ValueError("READ_ONLY_NOT_ENABLED")
                yield session, statements
        finally:
            event.remove(connection, "before_cursor_execute", capture)


def comparison_summary(comparison) -> dict:
    detail = {row.fund_code: row.metrics.model_dump(mode="json") for row in comparison.per_fund}
    macro = {}
    for field in ("accuracy", "balanced_accuracy", "brier_score"):
        values = [v[field] for v in detail.values()]
        macro[field] = (
            str(sum(Decimal(v) for v in values) / len(values)) if all(v is not None for v in values) else None
        )
    return {
        "sample_weighted": comparison.validation.model_dump(mode="json"),
        "equal_fund_macro": macro,
        "per_fund": detail,
    }


def verify_legacy(root=ROOT) -> dict:
    results = {}
    for kind, name in LEGACY.items():
        folder = root / ".local-runs" / name
        if not folder.exists():
            results[kind] = {"status": "MISSING", "folder": name}
            continue
        completion = read_json(folder / "complete.json")
        verify_files(folder, completion["files_sha256"])
        payload = read_json(folder / "report.json")
        if completion["database_written"] is not False or completion["publication_status"] != "MODEL_NOT_RELEASED":
            raise ValueError("LEGACY_MODE_MISMATCH")
        record = {"status": "VERIFIED", "folder": name, "files": completion["files_sha256"], "summary": {}}
        if kind == "candidate":
            report = HistoricalNavTrainingResponse.model_validate(payload)
            model = restore_logistic_artifact((folder / "model.json").read_bytes())
            if report.model != model or model.model_hash != completion["model_hash"]:
                raise ValueError("LEGACY_CANDIDATE_MISMATCH")
            record["summary"] = comparison_summary(report.candidate)
        elif kind == "calibration":
            report = HistoricalNavCalibrationResponse.model_validate(payload)
            models = read_json(folder / "models.json")
            for window in report.windows:
                if window.model:
                    model = restore_calibrated_artifact(json.dumps(models[window.window.window_id]))
                    if window.model != model or completion["model_hashes"][window.window.window_id] != model.model_hash:
                        raise ValueError("LEGACY_CALIBRATION_MISMATCH")
                    record["summary"][window.window.window_id] = {
                        "before": comparison_summary(window.before),
                        "after": comparison_summary(window.after),
                    }
        else:
            if payload["model_fitted"] or payload["test_scored"]:
                raise ValueError("LEGACY_DIAGNOSTIC_MODE")
            record["summary"] = {
                "raw_snapshot_hash": payload["raw_snapshot_hash"],
                "dataset_hash": payload["dataset_hash"],
            }
        results[kind] = record
    return results


def replay_cash(stored, dataset) -> tuple[dict, list[dict]]:
    if dataset.report.dataset_hash != stored.report.preparation.dataset_hash:
        raise ValueError("CASH_DATASET_CHANGED")
    windows, predictions = {}, []
    for window in stored.report.windows:
        rows, counts = cash_window_rows(dataset, window.window)
        details = {"status": window.status, "counts": [c.model_dump(mode="json") for c in counts]}
        if tuple(counts) != window.funds:
            raise ValueError("CASH_WINDOW_COUNTS_CHANGED")
        if window.model:
            exam = rows["EXAM"]
            raw = predict_artifact_scores(window.model.base_model, tuple(r.x for r in exam))
            calibrated = predict_calibrated_scores(window.model, tuple(r.x for r in exam))
            for reference, values in ((window.before, raw), (window.after, calibrated)):
                actual = calculate_baseline_metrics(tuple(r.y for r in exam), tuple(Decimal(str(v)) for v in values))
                if actual != reference.validation:
                    raise ValueError("CASH_SCORE_REPLAY_MISMATCH")
                for fund in reference.per_fund:
                    pairs = [
                        (r.y, Decimal(str(v)))
                        for r, v in zip(exam, values, strict=True)
                        if r.fund_code == fund.fund_code
                    ]
                    if (
                        calculate_baseline_metrics(tuple(p[0] for p in pairs), tuple(p[1] for p in pairs))
                        != fund.metrics
                    ):
                        raise ValueError("CASH_FUND_REPLAY_MISMATCH")
            details.update(
                before=comparison_summary(window.before),
                after=comparison_summary(window.after),
                baselines={b.baseline_id: comparison_summary(b) for b in window.baselines},
                model_hash=window.model.base_model.model_hash,
                calibration_hash=window.model.model_hash,
            )
            for row, before, after in zip(exam, raw, calibrated, strict=True):
                predictions.append(
                    {
                        "window": window.window.window_id,
                        "fund": row.fund_code,
                        "cutoff": str(row.available_at),
                        "input_hash": row.content_hash,
                        "raw_score": before,
                        "calibrated_score": after,
                        "y": row.y,
                    }
                )
        windows[window.window.window_id] = details
    return windows, predictions


def inventory() -> tuple[object, dict]:
    folder = new_folder(prefix="direction-inventory")
    legacy = verify_legacy()
    with local_snapshot() as (session, statements):
        stored = restore_research(find_research(session, run_id=CASH_RUN))
        data = load_cash_dataset_in_session(
            session, CashPrepareRequest(batchIds=stored.report.preparation.batch_ids), deadline=perf_counter() + 60
        )
    windows, predictions = replay_cash(stored, data)
    report = {
        "version": "DIRECTION_T00_V1",
        "cash_run_id": str(CASH_RUN),
        "cash_report_hash": stored.report.report_hash,
        "cash_dataset_hash": data.report.dataset_hash,
        "legacy": legacy,
        "cash_windows": windows,
        "cash_preparation": data.report.model_dump(mode="json"),
        "sql_statement_counts": dict(statements),
        "runtime": {n: version(n) for n in ("numpy", "scipy", "scikit-learn", "sqlalchemy", "pydantic")},
        "model_fitted": False,
        "test_scored": False,
        "database_written": False,
        "publication_status": "MODEL_NOT_RELEASED",
    }
    files = {
        "inventory.json": write_json(folder / "inventory.json", report),
        "cash-report.json": write_json(folder / "cash-report.json", stored.report.model_dump(mode="json")),
        "cash-predictions.jsonl": write_jsonl(folder / "cash-predictions.jsonl", predictions),
    }
    # 对账前后再次核对旧文件，只有新建本次产物。
    if verify_legacy() != legacy:
        raise ValueError("LEGACY_CHANGED_DURING_INVENTORY")
    seal(folder, "complete.json", files, status="INVENTORIED", model_fitted=False, test_scored=False)
    read_seal(folder, "complete.json")
    return folder, report
