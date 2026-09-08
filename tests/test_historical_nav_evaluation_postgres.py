"""显式启用的PostgreSQL读取验收，复用随机隔离schema；不写public业务表。"""

import os
from datetime import date
from uuid import uuid4

import pytest
from app.services import historical_nav_evaluation as evaluation
from app.services import historical_nav_storage as storage
from sqlalchemy import event
from tests.test_historical_nav_evaluation import PATH, HistoricalNavEvaluationRequest
from tests.test_historical_nav_http import HEADERS  # noqa: F401
from tests.test_historical_nav_http import client as client
from tests.test_historical_nav_storage import save_request
from tests.test_historical_nav_storage_postgres import database as database
from tests.test_historical_nav_storage_postgres import row_counts

pytestmark = pytest.mark.skipif(os.getenv("RUN_NAV_STORAGE_PG_TESTS") != "1", reason="需要显式启用隔离PostgreSQL测试")


def request_for_ids(ids):
    return HistoricalNavEvaluationRequest(
        batch_ids=ids,
        train_start_date=date(2025, 1, 1),
        train_end_date=date(2025, 6, 30),
        validation_end_date=date(2025, 12, 31),
        test_end_date=date(2026, 6, 30),
    )


def test_real_three_table_read_is_read_only_paged_and_deduplicated(database, monkeypatch, client):
    engine, _ = database
    batches = [storage.save_historical_nav_batch(save_request())[0] for _ in range(35)]
    request = request_for_ids(tuple(b.batch_id for b in batches))
    monkeypatch.setattr(evaluation, "get_nav_sample_storage_engine", lambda: engine)
    statements = []

    def record(conn, cursor, statement, parameters, context, executemany):
        statements.append(statement)

    before = row_counts(engine)
    event.listen(engine, "before_cursor_execute", record)
    try:
        response = client.post(PATH, json=request.model_dump(mode="json", by_alias=True), headers=HEADERS)
    finally:
        event.remove(engine, "before_cursor_execute", record)
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["status"] == "INSUFFICIENT_DATA" and result["baselines"] == []
    assert result["input_sample_count"] == 35 and result["duplicate_sample_count"] == 34
    assert result["funds"][0]["split_counts"] == {"TRAIN": 1, "VALIDATION": 0, "TEST": 0}
    assert statements[0] == "SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY"
    assert len(statements) == 4  # SET、一次批次查询、两页题目/答案查询，无N+1。
    assert not any("nav_daily" in sql or sql.startswith(("INSERT", "UPDATE", "DELETE")) for sql in statements)
    assert before == row_counts(engine) == [35, 35, 35]
    from app.repositories import historical_nav_evaluation as repository

    monkeypatch.setattr(repository, "BATCH_READ_PAGE_SIZE", 7)
    again = evaluation.evaluate_stored_historical_nav_batches(request)
    assert again.model_dump(mode="json") == result


def test_one_missing_batch_rejects_whole_selection(database, monkeypatch, client):
    engine, _ = database
    batch = storage.save_historical_nav_batch(save_request())[0]
    monkeypatch.setattr(evaluation, "get_nav_sample_storage_engine", lambda: engine)
    body = request_for_ids((batch.batch_id, uuid4())).model_dump(mode="json", by_alias=True)
    response = client.post(PATH, json=body, headers=HEADERS)
    assert response.status_code == 404 and row_counts(engine) == [1, 1, 1]


def test_pilot_execute_resumes_without_duplicate_inserts(database, monkeypatch):
    from scripts.historical_nav_baseline_pilot import execute_pilot

    engine, _ = database
    monkeypatch.setattr(evaluation, "get_nav_sample_storage_engine", lambda: engine)
    requests = (save_request(),)
    first, repeated = execute_pilot(requests), execute_pilot(requests)
    assert first["created_batches"] == 1 and first["reused_batches"] == 0
    assert repeated["created_batches"] == 0 and repeated["reused_batches"] == 1
    assert first["report"] == repeated["report"] and row_counts(engine) == [1, 1, 1]
