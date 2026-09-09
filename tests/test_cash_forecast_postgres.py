"""隔离PG中的授权后完整工程链路；只有授权是明确替身，输入SQL、真实计算、保存与失效均走实代码。"""

import importlib.util
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from pathlib import Path
from threading import Barrier
from uuid import uuid4

import pytest
from alembic.migration import MigrationContext
from alembic.operations import Operations
from app.models.cash_reinvestment import CashResearchRun
from app.services import cash_forecast as service
from app.services.cash_publication import CashPublicationUnavailable
from app.services.historical_nav_storage import HistoricalNavStorageError
from sqlalchemy import event, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session
from tests.test_cash_forecast import CUTOFF, NOW, PATH
from tests.test_cash_forecast import bundle as bundle
from tests.test_cash_forecast import report as report
from tests.test_cash_prediction_features import SOURCE, history_rows
from tests.test_cash_prediction_features import request as input_request
from tests.test_cash_reinvestment_postgres import database as database
from tests.test_cash_reinvestment_research import data as data
from tests.test_historical_nav_http import HEADERS
from tests.test_historical_nav_http import client as client

pytestmark = pytest.mark.skipif(os.getenv("RUN_NAV_STORAGE_PG_TESTS") != "1", reason="显式启用隔离PostgreSQL测试")


def migration_module():
    spec = importlib.util.spec_from_file_location(
        "cash_forecast_migration",
        Path(__file__).resolve().parents[1] / "alembic/versions/20260909_16_cash_forecast_result.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def forecasts_db(database, bundle, report, monkeypatch):
    engine, _ = database
    req, grant, saved = bundle
    with engine.begin() as conn:
        migration = migration_module()
        migration.op = Operations(MigrationContext.configure(conn))
        migration.upgrade()
        # 限定在测试创建的随机schema内，原fund_share_class只是外键占位，补本链路实际SELECT需要的列。
        conn.exec_driver_sql(
            "ALTER TABLE fund_share_class ADD COLUMN fund_type text DEFAULT 'STOCK', "
            "ADD COLUMN status text DEFAULT 'ACTIVE', ADD COLUMN source_code text DEFAULT 'TUSHARE_PRO_FUND'"
        )
        conn.exec_driver_sql(
            "CREATE TABLE source_registry (source_id uuid, source_code text, "
            "enabled boolean, last_success_at timestamptz)"
        )
        conn.exec_driver_sql(
            "CREATE TABLE source_sync_run (sync_run_id uuid, source_id uuid, sync_type text, "
            "status text, finished_at timestamptz)"
        )
        conn.exec_driver_sql(
            "CREATE TABLE nav_daily (fund_code text, source_id uuid, nav_date date, "
            "ann_date date, unit_nav numeric(20,12))"
        )
        conn.exec_driver_sql(
            "CREATE TABLE fund_dividend (fund_code text, source_id uuid, source_event_key text, ann_date date, "
            "implementation_ann_date date, ex_date date, nav_ex_date date, "
            "cash_dividend numeric(20,12), process_status text)"
        )
        conn.execute(
            text("INSERT INTO source_registry VALUES (:id, :code, true, :finished)"),
            {"id": SOURCE.source_id, "code": SOURCE.source_code, "finished": SOURCE.source_sync_finished_at},
        )
        conn.execute(
            text("INSERT INTO source_sync_run VALUES (:run, :id, 'NAV_DAILY', 'SUCCEEDED', :finished)"),
            {"run": SOURCE.source_sync_run_id, "id": SOURCE.source_id, "finished": SOURCE.source_sync_finished_at},
        )
        conn.execute(
            text("INSERT INTO nav_daily VALUES (:fund, :source, :day, :ann, :value)"),
            [
                {
                    "fund": req.fund_code,
                    "source": SOURCE.source_id,
                    "day": p.nav_date,
                    "ann": p.ann_date,
                    "value": p.unit_nav,
                }
                for p in history_rows(input_request(CUTOFF))
            ],
        )
    with Session(engine) as session, session.begin():
        session.add(
            CashResearchRun(
                run_id=req.research_run_id,
                request_key=uuid4(),
                dataset_hash=report.preparation.dataset_hash,
                report=report.model_dump(mode="json"),
                publication_status="MODEL_NOT_RELEASED",
            )
        )
    monkeypatch.setattr(service, "get_nav_sample_storage_engine", lambda: engine)
    monkeypatch.setattr(service, "utc_now", lambda: NOW)
    monkeypatch.setattr(service, "resolve_cash_authorization", lambda *a, **k: grant)
    return engine, req, grant


def count(engine):
    with engine.connect() as conn:
        return conn.execute(text("SELECT count(*) FROM cash_forecast_result")).scalar_one()


def test_real_input_sql_numerical_model_persistence_retry_and_read(forecasts_db, monkeypatch):
    engine, req, _ = forecasts_db
    first = service.generate_cash_forecast(req)
    assert first.status == "AVAILABLE" and first.created and first.up_probability is not None
    assert count(engine) == 1

    def forbidden(*a, **k):
        pytest.fail("retry or GET must not read history values or re-execute the model")

    monkeypatch.setattr(service, "read_cash_prediction_feature_in_session", forbidden)
    monkeypatch.setattr(service, "calculate_cash_inference", forbidden)
    assert service.generate_cash_forecast(req) == first.model_copy(update={"created": False})
    sql = []

    def capture(conn, cursor, statement, parameters, context, executemany):
        sql.append(statement)

    event.listen(engine, "before_cursor_execute", capture)
    try:
        assert service.get_cash_forecast(first.forecast_id) == first.model_copy(update={"created": False})
    finally:
        event.remove(engine, "before_cursor_execute", capture)
    assert "READ ONLY" in sql[0]
    assert all("unit_nav" not in s and "fund_dividend" not in s for s in sql)
    assert count(engine) == 1


def test_http_full_post_get_retry_contract_with_explicit_artificial_authorization(forecasts_db, client):
    _, req, _ = forecasts_db
    first = client.post(PATH, json=req.model_dump(mode="json", by_alias=True), headers=HEADERS)
    assert first.status_code == 201 and first.json()["status"] == "AVAILABLE"
    read = client.get(f"{PATH}/{first.json()['forecast_id']}", headers=HEADERS)
    retry = client.post(PATH, json=req.model_dump(mode="json", by_alias=True), headers=HEADERS)
    assert read.status_code == retry.status_code == 200
    assert read.json() == retry.json() == {**first.json(), "created": False}
    assert all(r.headers["cache-control"] == "no-store" for r in (first, read, retry))


def test_publication_denial_precedes_input_and_model_and_creates_nothing(forecasts_db, monkeypatch):
    engine, req, _ = forecasts_db

    def denied(*a, **k):
        raise CashPublicationUnavailable(("INDEPENDENT_TEST_NOT_EVALUATED",))

    monkeypatch.setattr(service, "resolve_cash_authorization", denied)
    monkeypatch.setattr(
        service, "read_cash_prediction_feature_in_session", lambda *a, **k: pytest.fail("must not read inputs")
    )
    monkeypatch.setattr(service, "calculate_cash_inference", lambda *a, **k: pytest.fail("must not infer"))
    result = service.generate_cash_forecast(req)
    assert result.status == "MODEL_NOT_RELEASED" and result.up_probability is result.forecast_id is None
    assert not result.created and count(engine) == 0


def test_request_conflict_and_duplicate_business_rejected_before_repeat_inference(forecasts_db, monkeypatch):
    engine, req, _ = forecasts_db
    first = service.generate_cash_forecast(req)
    monkeypatch.setattr(service, "calculate_cash_inference", lambda *a, **k: pytest.fail("must not infer"))
    with pytest.raises(HistoricalNavStorageError) as error:
        service.generate_cash_forecast(req.model_copy(update={"expected_model_hash": "f" * 64}))
    assert error.value.code == "REQUEST_KEY_CONFLICT"
    with pytest.raises(HistoricalNavStorageError) as error:
        service.generate_cash_forecast(req.model_copy(update={"request_key": uuid4()}))
    assert error.value.code == "CASH_FORECAST_ALREADY_EXISTS"
    assert service.get_cash_forecast(first.forecast_id).status == "AVAILABLE" and count(engine) == 1


@pytest.mark.parametrize(
    "change,reason",
    [
        ("source", "SOURCE_REVISION_CHANGED"),
        ("latest", "LATEST_NAV_CHANGED"),
        ("disabled", "SOURCE_NOT_READY"),
        ("time", "FORECAST_WINDOW_ENDED"),
    ],
)
def test_changed_metadata_or_expiry_hides_old_probability_without_recalculation(
    forecasts_db, monkeypatch, change, reason
):
    engine, req, _ = forecasts_db
    first = service.generate_cash_forecast(req)
    if change == "time":
        monkeypatch.setattr(service, "utc_now", lambda: datetime(2026, 11, 1, 6, tzinfo=UTC))
    else:
        with engine.begin() as conn:
            if change == "source":
                conn.execute(text("UPDATE source_sync_run SET sync_run_id=:id"), {"id": uuid4()})
            elif change == "latest":
                conn.execute(text("UPDATE nav_daily SET ann_date=nav_date WHERE nav_date=:cutoff"), {"cutoff": CUTOFF})
            else:
                conn.exec_driver_sql("UPDATE source_registry SET enabled=false")
    monkeypatch.setattr(service, "calculate_cash_inference", lambda *a, **k: pytest.fail("GET must not infer"))
    result = service.get_cash_forecast(first.forecast_id)
    assert (
        result.status == "STALE" and result.up_probability is result.direction is None and reason in result.reason_codes
    )
    assert result.target_end_date == first.target_end_date and count(engine) == 1


def test_revocation_blocks_reads_and_same_key_retry(forecasts_db, monkeypatch):
    engine, req, _ = forecasts_db
    first = service.generate_cash_forecast(req)

    def revoked(*a, **k):
        raise CashPublicationUnavailable(("MODEL_SUSPENDED",))

    monkeypatch.setattr(service, "resolve_cash_authorization", revoked)
    for result in (service.get_cash_forecast(first.forecast_id), service.generate_cash_forecast(req)):
        assert result.status == "MODEL_NOT_RELEASED" and result.up_probability is result.direction is None
        assert result.forecast_id == first.forecast_id and not result.created
    assert count(engine) == 1


def test_missing_history_never_creates_partial_result(forecasts_db):
    engine, req, _ = forecasts_db
    with engine.begin() as conn:
        # 只删除本用例人工schema的一条输入，测试结束会整体清理；不碰public净值。
        conn.exec_driver_sql("DELETE FROM nav_daily WHERE fund_code='006730' AND nav_date=DATE '2026-09-07'")
    result = service.generate_cash_forecast(req)
    assert result.status == "DATA_INSUFFICIENT" and result.up_probability is None and count(engine) == 0


def test_error_after_insert_rolls_back_result(forecasts_db, monkeypatch):
    engine, req, _ = forecasts_db
    monkeypatch.setattr(
        service, "_view", lambda *a, **k: (_ for _ in ()).throw(ValueError("synthetic post-insert error"))
    )
    with pytest.raises(ValueError, match="post-insert"):
        service.generate_cash_forecast(req)
    assert count(engine) == 0


def test_same_key_concurrency_commits_one_result(forecasts_db, monkeypatch):
    engine, req, _ = forecasts_db
    original, barrier = service.calculate_cash_inference, Barrier(2)

    def synchronized(*a, **k):
        value = original(*a, **k)
        barrier.wait(timeout=10)
        return value

    monkeypatch.setattr(service, "calculate_cash_inference", synchronized)
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = tuple(pool.map(service.generate_cash_forecast, (req, req)))
    assert results[0].forecast_id == results[1].forecast_id
    assert sorted(r.created for r in results) == [False, True] and count(engine) == 1


def test_corrupt_snapshot_is_error_not_a_recomputed_success(forecasts_db):
    engine, req, _ = forecasts_db
    first = service.generate_cash_forecast(req)
    with engine.begin() as conn:
        conn.exec_driver_sql("UPDATE cash_forecast_result SET content_hash=repeat('f',64)")
    with pytest.raises(HistoricalNavStorageError) as error:
        service.get_cash_forecast(first.forecast_id)
    assert error.value.code == "CASH_FORECAST_CORRUPTED"


def test_non_nav_sync_invalidates_and_new_request_preserves_old_result(forecasts_db):
    engine, req, _ = forecasts_db
    old = service.generate_cash_forecast(req)
    revision = uuid4()
    with engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO source_sync_run VALUES (:run, :source, 'MARKET_FREE_DATA_COMPLETION', 'SUCCEEDED', :time)"
            ),
            {
                "run": revision,
                "source": SOURCE.source_id,
                "time": SOURCE.source_sync_finished_at + timedelta(seconds=1),
            },
        )
    stale = service.get_cash_forecast(old.forecast_id)
    assert stale.status == "STALE" and "SOURCE_REVISION_CHANGED" in stale.reason_codes
    new = service.generate_cash_forecast(req.model_copy(update={"request_key": uuid4()}))
    assert new.status == "AVAILABLE" and new.forecast_id != old.forecast_id and count(engine) == 2
    with engine.connect() as conn:
        rows = conn.execute(
            text("SELECT feature_hash, source_revision_id FROM cash_forecast_result ORDER BY created_at")
        ).all()
    assert rows[0].feature_hash == rows[1].feature_hash  # 没有新NAV，也必须识别分红/其他来源更新。
    assert {r.source_revision_id for r in rows} == {SOURCE.source_sync_run_id, revision}


@pytest.mark.parametrize("status", ["RUNNING", "FAILED"])
def test_unfinished_or_failed_source_update_blocks_old_read_and_new_calculation(forecasts_db, monkeypatch, status):
    engine, req, _ = forecasts_db
    old = service.generate_cash_forecast(req)
    with engine.begin() as conn:
        conn.execute(
            text("INSERT INTO source_sync_run VALUES (:run, :source, 'MARKET_FREE_DATA_COMPLETION', :status, :time)"),
            {
                "run": uuid4(),
                "source": SOURCE.source_id,
                "status": status,
                "time": None if status == "RUNNING" else SOURCE.source_sync_finished_at + timedelta(seconds=1),
            },
        )
    monkeypatch.setattr(service, "calculate_cash_inference", lambda *a, **k: pytest.fail("must not infer"))
    read = service.get_cash_forecast(old.forecast_id)
    assert read.status == "STALE" and read.reason_codes == ("SOURCE_UPDATE_NOT_READY",)
    new = service.generate_cash_forecast(req.model_copy(update={"request_key": uuid4()}))
    assert new.status == "DATA_INSUFFICIENT" and new.up_probability is None and count(engine) == 1


@pytest.mark.parametrize(
    "expression",
    [
        "jsonb_set(payload, '{value,up_score}', '1.1')",
        "jsonb_set(payload, '{value,up_score}', '-0.1')",
        "payload - 'value'",
    ],
)
def test_database_rejects_invalid_probability_or_absent_result(forecasts_db, expression):
    engine, req, _ = forecasts_db
    first = service.generate_cash_forecast(req)
    with pytest.raises(IntegrityError, match="ck_cash_forecast_payload"):
        with engine.begin() as conn:
            conn.exec_driver_sql(f"UPDATE cash_forecast_result SET payload={expression}")
    assert service.get_cash_forecast(first.forecast_id).up_probability == first.up_probability


def test_empty_downgrade_does_not_remove_old_research(forecasts_db):
    engine, _, _ = forecasts_db
    with engine.begin() as conn:
        migration = migration_module()
        migration.op = Operations(MigrationContext.configure(conn))
        migration.downgrade()
        assert conn.execute(text("SELECT to_regclass('cash_forecast_result')")).scalar_one() is None
        assert conn.execute(text("SELECT count(*) FROM cash_research_run")).scalar_one() == 1


def test_nonempty_downgrade_refused_and_all_columns_commented(forecasts_db):
    engine, req, _ = forecasts_db
    service.generate_cash_forecast(req)
    with pytest.raises(Exception, match="not empty"):
        with engine.begin() as conn:
            migration = migration_module()
            migration.op = Operations(MigrationContext.configure(conn))
            migration.downgrade()
    assert count(engine) == 1
    with engine.connect() as conn:
        assert (
            conn.execute(
                text(
                    "SELECT count(*) FROM pg_attribute WHERE attrelid='cash_forecast_result'::regclass "
                    "AND attnum>0 AND NOT attisdropped AND col_description(attrelid,attnum) IS NULL"
                )
            ).scalar_one()
            == 0
        )
