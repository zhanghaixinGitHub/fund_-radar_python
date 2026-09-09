"""真实PostgreSQL回执、约束、事务及并发；只写测试自己创建的随机schema。"""

import importlib.util
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path
from threading import Barrier
from types import SimpleNamespace
from uuid import uuid4

import pytest
from alembic.migration import MigrationContext
from alembic.operations import Operations
from app.schemas.cash_prediction_attempt import CashPredictionAttemptRequest
from app.schemas.cash_reinvestment_research import CashPrepareRequest, CashResearchRequest
from app.services import cash_prediction_attempt as attempts
from app.services import cash_prediction_check as check
from app.services import cash_reinvestment_research as research
from app.services import cash_reinvestment_storage as storage
from app.services.historical_nav_storage import HistoricalNavStorageError
from sqlalchemy import event, text
from sqlalchemy.exc import IntegrityError
from tests.test_cash_reinvestment_postgres import database as database
from tests.test_cash_reinvestment_storage import request

pytestmark = pytest.mark.skipif(os.getenv("RUN_NAV_STORAGE_PG_TESTS") != "1", reason="显式启用独立PostgreSQL测试")


def migration_module():
    path = Path(__file__).resolve().parents[1] / "alembic/versions/20260909_15_cash_prediction_attempt.py"
    spec = importlib.util.spec_from_file_location("cash_attempt_migration", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def attempts_db(database, monkeypatch):
    engine, _ = database
    with engine.begin() as conn:
        migration = migration_module()
        migration.op = Operations(MigrationContext.configure(conn))
        migration.upgrade()
    batch, _ = storage.save_cash_batch(request())
    data = research.load_cash_dataset(CashPrepareRequest(batchIds=[batch.batch_id]))
    run, _ = research.save_cash_research(
        CashResearchRequest(batchIds=[batch.batch_id], requestKey=uuid4(), expectedDatasetHash=data.report.dataset_hash)
    )
    monkeypatch.setattr(attempts, "get_nav_sample_storage_engine", lambda: engine)

    def metadata(session, fund_code, today, *, include_research):
        # 本schema只有基金外键，元数据明确用替身；研究读回、检查、INSERT及事务均为真实代码/SQL。
        assert not include_research
        assert session.execute(text("SHOW transaction_isolation")).scalar_one() == "repeatable read"
        return (
            SimpleNamespace(fund_code=fund_code, fund_type="STOCK", status="ACTIVE", source_code="TUSHARE_PRO_FUND"),
            SimpleNamespace(enabled=True),
            date(2026, 9, 4),
            None,
        )

    monkeypatch.setattr(check, "read_prediction_inputs", metadata)
    req = CashPredictionAttemptRequest(
        requestKey=uuid4(),
        fundCode="006730",
        cutoffDate="2026-09-04",
        researchRunId=run.run_id,
        expectedReportHash=run.report.report_hash,
    )
    return engine, req


def attempt_count(engine):
    with engine.connect() as conn:
        return conn.execute(text("SELECT count(*) FROM cash_prediction_attempt")).scalar_one()


def test_real_save_retry_read_and_no_inference(attempts_db, monkeypatch):
    from app.services import cash_prediction_features, cash_prediction_inference

    engine, req = attempts_db

    def forbidden(*a, **k):
        pytest.fail("rejection must not read history or calculate a score")

    monkeypatch.setattr(cash_prediction_features, "read_cash_prediction_feature", forbidden)
    monkeypatch.setattr(cash_prediction_inference, "calculate_cash_inference", forbidden)
    first, created = attempts.save_cash_prediction_attempt(req)
    assert created and first.check.status == "GENERATION_BLOCKED" and attempt_count(engine) == 1
    assert not first.forecast_created and first.check.up_probability is None
    monkeypatch.setattr(attempts, "check_cash_prediction_in_session", forbidden)
    assert attempts.save_cash_prediction_attempt(req) == (first, False)
    statements = []

    def capture(conn, cursor, statement, parameters, context, executemany):
        statements.append(statement)

    event.listen(engine, "before_cursor_execute", capture)
    try:
        assert attempts.get_cash_prediction_attempt(first.attempt_id) == first
    finally:
        event.remove(engine, "before_cursor_execute", capture)
    assert len(statements) == 2 and "READ ONLY" in statements[0]
    assert "cash_prediction_attempt.attempt_id =" in statements[1]
    assert all("cash_research_run" not in s and "nav_daily" not in s for s in statements)
    with engine.connect() as conn:
        assert conn.execute(text("SELECT count(*) FROM cash_research_run")).scalar_one() == 1
        missing_comments = conn.execute(
            text("""SELECT count(*) FROM pg_attribute
            WHERE attrelid='cash_prediction_attempt'::regclass AND attnum>0 AND NOT attisdropped
            AND col_description(attrelid,attnum) IS NULL""")
        ).scalar_one()
        assert missing_comments == 0


def test_key_conflict_and_new_key_does_not_overwrite(attempts_db):
    engine, req = attempts_db
    first, _ = attempts.save_cash_prediction_attempt(req)
    with pytest.raises(HistoricalNavStorageError) as error:
        attempts.save_cash_prediction_attempt(req.model_copy(update={"cutoff_date": date(2026, 9, 3)}))
    assert error.value.code == "REQUEST_KEY_CONFLICT"
    second, created = attempts.save_cash_prediction_attempt(req.model_copy(update={"request_key": uuid4()}))
    assert created and second.attempt_id != first.attempt_id and attempt_count(engine) == 2
    assert attempts.get_cash_prediction_attempt(first.attempt_id) == first


def test_wrong_report_leaves_no_receipt(attempts_db):
    engine, req = attempts_db
    with pytest.raises(HistoricalNavStorageError) as error:
        attempts.save_cash_prediction_attempt(req.model_copy(update={"expected_report_hash": "e" * 64}))
    assert error.value.code == "REPORT_HASH_MISMATCH" and attempt_count(engine) == 0


def test_failure_after_insert_rolls_back_whole_attempt(attempts_db, monkeypatch):
    engine, req = attempts_db

    def broken_receipt(row):
        raise ValueError("synthetic post-insert integrity failure")

    monkeypatch.setattr(attempts, "restore_attempt", broken_receipt)
    with pytest.raises(ValueError, match="post-insert"):
        attempts.save_cash_prediction_attempt(req)
    assert attempt_count(engine) == 0


def test_same_key_concurrent_requests_create_one_identical_receipt(attempts_db, monkeypatch):
    engine, req = attempts_db
    barrier = Barrier(2)
    original = attempts.check_cash_prediction_in_session

    def checked(*a, **k):
        result = original(*a, **k)
        barrier.wait(timeout=8)
        return result

    monkeypatch.setattr(attempts, "check_cash_prediction_in_session", checked)
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = tuple(pool.map(attempts.save_cash_prediction_attempt, (req, req)))
    assert results[0][0] == results[1][0]
    assert sorted(created for _, created in results) == [False, True] and attempt_count(engine) == 1


@pytest.mark.parametrize(
    "expression",
    [
        "jsonb_set(check_payload, '{up_probability}', '0.9')",
        "jsonb_set(check_payload, '{inference_executed}', 'true')",
        "jsonb_set(check_payload, '{blocking_codes}', '[]')",
        "check_payload - 'up_probability'",
        "'{}'::jsonb",
    ],
)
def test_database_cannot_hold_fake_success_or_missing_gate_fields(attempts_db, expression):
    engine, req = attempts_db
    first, _ = attempts.save_cash_prediction_attempt(req)
    with pytest.raises(IntegrityError, match="ck_cash_prediction_attempt_rejected"):
        with engine.begin() as conn:
            conn.exec_driver_sql(f"UPDATE cash_prediction_attempt SET check_payload={expression}")
    assert attempts.get_cash_prediction_attempt(first.attempt_id) == first


def test_downgrade_refuses_saved_receipts(attempts_db):
    engine, req = attempts_db
    attempts.save_cash_prediction_attempt(req)
    with pytest.raises(Exception, match="not empty"):
        with engine.begin() as conn:
            migration = migration_module()
            migration.op = Operations(MigrationContext.configure(conn))
            migration.downgrade()
    assert attempt_count(engine) == 1


def test_empty_table_can_downgrade_without_deleting_research(attempts_db):
    engine, _ = attempts_db
    with engine.begin() as conn:
        migration = migration_module()
        migration.op = Operations(MigrationContext.configure(conn))
        migration.downgrade()
        assert conn.execute(text("SELECT to_regclass('cash_prediction_attempt')")).scalar_one() is None
        assert conn.execute(text("SELECT count(*) FROM cash_research_run")).scalar_one() == 1
