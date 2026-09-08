"""隔离PostgreSQL真实保存/读取与训练；人工样本只存在随机测试schema，结束自动清理。"""

import os
from dataclasses import replace

import pytest
from app.schemas.historical_nav import HistoricalNavBatchPreviewResponse
from app.schemas.historical_nav_storage import HistoricalNavBatchSaveRequest
from app.services import historical_nav_evaluation as evaluation
from app.services import historical_nav_storage as storage
from app.services import historical_nav_training as training
from sqlalchemy import event
from tests.test_historical_nav_calibration import calibration_batches as calibration_batches
from tests.test_historical_nav_evaluation import batches as batches
from tests.test_historical_nav_evaluation import request_for
from tests.test_historical_nav_http import HEADERS  # noqa: F401
from tests.test_historical_nav_http import client as client
from tests.test_historical_nav_storage import SOURCE
from tests.test_historical_nav_storage_postgres import database as database
from tests.test_historical_nav_storage_postgres import row_counts
from tests.test_historical_nav_training import PATH, training_request

pytestmark = pytest.mark.skipif(os.getenv("RUN_NAV_STORAGE_PG_TESTS") != "1", reason="需要显式启用隔离PostgreSQL测试")


def persist_test_batches(engine, batches, monkeypatch):
    """两个真实读取测试共用相同隔离库种子保存，不向public写人工净值。"""
    source = replace(SOURCE, source_code="EVALUATION_TEST")
    by_start = {b.start_date: b for b in batches}

    def build(session, request, *, deadline):
        b = by_start[request.start_date]
        return source, HistoricalNavBatchPreviewResponse(
            fund_code=b.fund_code,
            start_date=b.start_date,
            end_date=b.end_date,
            page_size=30,
            page_count=2,
            sample_count=b.sample_count,
            scorable_count=b.scorable_count,
            data_insufficient_count=b.data_insufficient_count,
            label_not_matured_count=b.label_not_matured_count,
            unavailable_reasons=b.unavailable_reasons,
            items=b.items,
        )

    monkeypatch.setattr(storage, "build_stored_historical_nav_batch", build)
    stored = tuple(
        storage.save_historical_nav_batch(
            HistoricalNavBatchSaveRequest(
                fund_code=b.fund_code,
                start_date=b.start_date,
                end_date=b.end_date,
                request_key=b.request_key,
            )
        )[0]
        for b in batches
    )
    monkeypatch.setattr(evaluation, "get_nav_sample_storage_engine", lambda: engine)
    return stored


def test_real_read_only_candidate_and_connection_released_before_fit(database, batches, client, monkeypatch):
    engine, _ = database
    stored = persist_test_batches(engine, batches, monkeypatch)
    prepared = evaluation.load_historical_nav_dataset(request_for(stored))
    before = row_counts(engine)
    fitting = training._fit_training_rows

    def fit(rows):
        assert engine.pool.checkedout() == 0  # 训练不占用读取事务/连接。
        return fitting(rows)

    monkeypatch.setattr(training, "_fit_training_rows", fit)
    statements = []

    def record(conn, cursor, statement, parameters, context, executemany):
        statements.append(statement)

    event.listen(engine, "before_cursor_execute", record)
    try:
        response = client.post(
            PATH,
            json=training_request(stored, prepared).model_dump(mode="json", by_alias=True),
            headers=HEADERS,
        )
    finally:
        event.remove(engine, "before_cursor_execute", record)
    assert response.status_code == 200 and response.json()["status"] == "CANDIDATE_EVALUATED"
    assert response.json()["model"]["train_count"] == 252
    assert response.json()["candidate"]["validation"]["sample_count"] == 120
    assert statements[0] == "SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY"
    assert len(statements) == 3  # SET、一次封面、一次有界明细JOIN。
    assert all(sql.startswith(("SET TRANSACTION", "SELECT")) and "nav_daily" not in sql for sql in statements)
    assert before == row_counts(engine) and before[1] == 700


def test_real_read_only_calibration_reuses_one_snapshot(database, calibration_batches, client, monkeypatch):
    from tests.test_historical_nav_calibration import PATH as calibration_path
    from tests.test_historical_nav_calibration import request_for as calibration_request

    engine, _ = database
    stored = persist_test_batches(engine, calibration_batches, monkeypatch)
    prepared = evaluation.load_historical_nav_dataset(calibration_request(stored).evaluation_request())
    request = calibration_request(stored, prepared.report.dataset_hash)
    before = row_counts(engine)
    original_fit = training._fit_training_rows
    fit_count = []

    def fit(rows):
        assert engine.pool.checkedout() == 0
        fit_count.append(len(rows))
        return original_fit(rows)

    monkeypatch.setattr(training, "_fit_training_rows", fit)
    statements = []

    def record(conn, cursor, statement, parameters, context, executemany):
        statements.append(statement)

    event.listen(engine, "before_cursor_execute", record)
    try:
        response = client.post(calibration_path, json=request.model_dump(mode="json", by_alias=True), headers=HEADERS)
    finally:
        event.remove(engine, "before_cursor_execute", record)
    assert response.status_code == 200 and response.json()["status"] == "CALIBRATION_EVALUATED"
    assert len(fit_count) == 3 and response.json()["evaluated_window_count"] == 3
    assert statements[0] == "SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY"
    assert len(statements) == 4  # SET、一次封面、两页明细，3窗不各自读库。
    assert all(sql.startswith(("SET TRANSACTION", "SELECT")) and "nav_daily" not in sql for sql in statements)
    assert row_counts(engine) == before and before[1] == 1570
