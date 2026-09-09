"""显式启用的本机隔离schema测试；不写public样本，不读取2025净值。"""

import importlib.util
import os
import re
from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path
from uuid import uuid4

import pytest
from alembic.migration import MigrationContext
from alembic.operations import Operations
from app.core.config import get_settings
from app.schemas.cash_reinvestment_research import CashPrepareRequest, CashResearchRequest
from app.services import cash_reinvestment_research as research
from app.services import cash_reinvestment_storage as storage
from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url
from sqlalchemy.pool import NullPool
from tests.test_cash_reinvestment_storage import batch_preview, request

pytestmark = pytest.mark.skipif(os.getenv("RUN_NAV_STORAGE_PG_TESTS") != "1", reason="显式启用独立PostgreSQL测试")


def migration_module():
    file = Path(__file__).resolve().parents[1] / "alembic/versions/20260908_14_cash_reinvestment_storage.py"
    spec = importlib.util.spec_from_file_location("cash_migration", file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def database(monkeypatch):
    url = make_url(get_settings().ai_database_url)
    assert url.host in {"localhost", "127.0.0.1", "::1"} and url.database == "fund_ai"
    schema = "cash_test_" + uuid4().hex
    control = create_engine(
        url,
        poolclass=NullPool,
        hide_parameters=True,
        connect_args={"connect_timeout": 5, "options": "-c statement_timeout=5000 -c lock_timeout=3000"},
    )
    engine = None
    try:
        with control.begin() as conn:
            conn.exec_driver_sql(f'CREATE SCHEMA "{schema}"')
            conn.exec_driver_sql(f'SET LOCAL search_path TO "{schema}", pg_catalog')
            conn.exec_driver_sql("CREATE TABLE fund_share_class (fund_code varchar(32) PRIMARY KEY)")
            m = migration_module()
            m.op = Operations(MigrationContext.configure(conn))
            m.upgrade()
            conn.exec_driver_sql("INSERT INTO fund_share_class VALUES ('006730')")
        engine = create_engine(
            url,
            pool_size=3,
            max_overflow=0,
            hide_parameters=True,
            connect_args={
                "connect_timeout": 5,
                "options": f"-c search_path={schema},pg_catalog -c statement_timeout=5000 -c lock_timeout=3000",
            },
        )
        monkeypatch.setattr(storage, "get_nav_sample_storage_engine", lambda: engine)
        monkeypatch.setattr(research, "get_nav_sample_storage_engine", lambda: engine)
        calls = []

        def build(session, req, *, deadline):
            assert session.execute(text("SHOW transaction_read_only")).scalar_one() == "off"
            calls.append(req)
            return batch_preview(req)

        monkeypatch.setattr(storage, "build_cash_batch_in_session", build)
        yield engine, calls
    finally:
        if engine:
            engine.dispose()
        assert re.fullmatch(r"cash_test_[0-9a-f]{32}", schema)
        with control.begin() as conn:
            owned = conn.execute(
                text("SELECT nspowner=current_user::regrole FROM pg_namespace WHERE nspname=:name"), {"name": schema}
            ).scalar_one_or_none()
            if owned is not None:
                assert owned
                conn.exec_driver_sql(f'DROP SCHEMA "{schema}" CASCADE')
        control.dispose()


def counts(engine):
    with engine.connect() as conn:
        return [
            conn.exec_driver_sql(f"SELECT count(*) FROM {t}").scalar_one()
            for t in ("cash_sample_batch", "cash_sample", "cash_sample_label", "cash_research_run")
        ]


def test_current_history_sql_masks_unannounced_prices_and_stays_readonly(database, monkeypatch):
    """只在本测试随机schema放人工源表；检验真实SQL，不将替身等同供应商现场数据。"""
    from app.repositories.cash_prediction_features import read_cash_history_inputs
    from app.services import cash_prediction_features as features
    from app.services.trading_calendar import load_current_calendar
    from sqlalchemy.orm import Session
    from tests.test_cash_prediction_features import SOURCE
    from tests.test_cash_prediction_features import request as feature_request

    engine, _ = database
    req = feature_request(date(2026, 9, 8))
    calendar = load_current_calendar()
    start, end = features.cash_history_bounds(calendar, req)
    rows = [
        {"day": d, "ann": d + timedelta(days=1), "value": Decimal(10) + Decimal(i) / 100}
        for i, d in enumerate(d for d in calendar.sessions if start <= d <= end)
    ]
    with engine.begin() as conn:
        conn.exec_driver_sql("""CREATE TABLE nav_daily (
            fund_code varchar(32), source_id uuid, nav_date date, ann_date date, unit_nav numeric(20,12))""")
        conn.exec_driver_sql("""CREATE TABLE fund_dividend (
            fund_code varchar(32), source_id uuid, source_event_key varchar(64), ann_date date,
            implementation_ann_date date, ex_date date, nav_ex_date date,
            cash_dividend numeric(20,8), process_status varchar(64))""")
        conn.execute(
            text("INSERT INTO nav_daily VALUES (:fund, :source, :day, :ann, :value)"),
            [dict(row, fund=req.fund_code, source=SOURCE.source_id) for row in rows],
        )
        for key, ann in (("known", rows[20]["day"]), ("late", end + timedelta(days=1))):
            conn.execute(
                text("INSERT INTO fund_dividend VALUES (:fund, :source, :key, :ann, NULL, :day, :day, 0.5, :status)"),
                {
                    "fund": req.fund_code,
                    "source": SOURCE.source_id,
                    "key": key,
                    "ann": ann,
                    "day": rows[20]["day"],
                    "status": "实施",
                },
            )
    before = counts(engine)
    with Session(engine) as session, session.begin():
        session.execute(text("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY"))
        nav, events = read_cash_history_inputs(
            session, fund_code=req.fund_code, source_id=SOURCE.source_id, start=start, cutoff=end
        )
        assert nav[-1].nav_date == end and nav[-1].unit_nav is None
        assert len(events) == 1 and events[0].event_key == "known"

    def read_source(session, **kwargs):
        assert session.execute(text("SHOW transaction_read_only")).scalar_one() == "on"
        assert session.execute(text("SHOW transaction_isolation")).scalar_one() == "repeatable read"
        return SOURCE

    monkeypatch.setattr(features, "get_nav_preview_engine", lambda: engine)
    monkeypatch.setattr(features, "read_historical_nav_source", read_source)
    result = features.read_cash_prediction_feature(req)
    assert result.status == "INPUT_READY" and result.anchor_nav_date == date(2026, 9, 7)
    assert {key for p in result.feature_payload.history_series for key in p.dividend_event_keys} == {"known"}
    assert result == features.build_cash_prediction_feature(req, SOURCE, nav, events, calendar)
    assert counts(engine) == before
    with engine.connect() as conn:
        assert conn.execute(text("SELECT count(*) FROM nav_daily")).scalar_one() == len(rows)
        assert conn.execute(text("SELECT count(*) FROM fund_dividend")).scalar_one() == 2


def test_save_read_retry_new_key_and_comments(database):
    engine, calls = database
    req = request()
    first, created = storage.save_cash_batch(req)
    assert created and first.preview == batch_preview(req)
    assert storage.get_cash_batch(first.batch_id) == first
    assert storage.save_cash_batch(req.model_copy(update={"page_size": 30})) == (first, False)
    assert len(calls) == 1 and counts(engine) == [1, 1, 1, 0]
    second, _ = storage.save_cash_batch(req.model_copy(update={"request_key": uuid4()}))
    assert second.batch_id != first.batch_id and storage.get_cash_batch(first.batch_id) == first
    with engine.connect() as conn:
        missing = conn.exec_driver_sql("""SELECT count(*) FROM pg_attribute a JOIN pg_class c ON c.oid=a.attrelid
            WHERE c.relnamespace=current_schema()::regnamespace AND left(c.relname,5) = 'cash_'
            AND c.relkind='r' AND a.attnum>0 AND NOT a.attisdropped
            AND col_description(c.oid,a.attnum) IS NULL""").scalar_one()
    assert missing == 0


def test_transaction_rollback_on_label_failure(database, monkeypatch):
    engine, _ = database
    original = storage.insert_rows

    def fail(session, batch, samples, labels):
        original(session, batch, samples, labels)
        raise ValueError("artificial after-insert error")

    monkeypatch.setattr(storage, "insert_rows", fail)
    with pytest.raises(ValueError):
        storage.save_cash_batch(request())
    assert counts(engine) == [0, 0, 0, 0]


def test_request_identity_conflict(database):
    engine, _ = database
    req = request()
    storage.save_cash_batch(req)
    with pytest.raises(storage.HistoricalNavStorageError) as error:
        storage.save_cash_batch(req.model_copy(update={"fund_code": "008888"}))
    assert error.value.code == "REQUEST_KEY_CONFLICT" and counts(engine) == [1, 1, 1, 0]


def test_deleted_label_fails_closed(database):
    engine, _ = database
    first, _ = storage.save_cash_batch(request())
    with engine.begin() as conn:
        conn.exec_driver_sql("DELETE FROM cash_sample_label")
    with pytest.raises(storage.HistoricalNavStorageError) as error:
        storage.get_cash_batch(first.batch_id)
    assert error.value.code == "CASH_BATCH_CORRUPTED"


def test_downgrade_refuses_material_data(database):
    engine, _ = database
    storage.save_cash_batch(request())
    with pytest.raises(Exception, match="not empty"):
        with engine.begin() as conn:
            m = migration_module()
            m.op = Operations(MigrationContext.configure(conn))
            m.downgrade()
    assert counts(engine) == [1, 1, 1, 0]


def test_research_prepare_save_get_and_retry(database, monkeypatch):
    engine, _ = database
    first, _ = storage.save_cash_batch(request())
    data = research.load_cash_dataset(CashPrepareRequest(batchIds=[first.batch_id]))
    assert data.report.usable_count == 1 and data.report.status == "INSUFFICIENT_DATA"
    req = CashResearchRequest(
        batchIds=[first.batch_id], requestKey=uuid4(), expectedDatasetHash=data.report.dataset_hash
    )
    run, created = research.save_cash_research(req)
    assert created and run.report.release_gate == "BLOCKED" and not run.report.model_fitted
    assert research.get_cash_research(run.run_id) == run
    monkeypatch.setattr(research, "evaluate_cash_dataset", lambda _: pytest.fail("retry must not fit"))
    assert research.save_cash_research(req) == (run, False)
    assert counts(engine) == [1, 1, 1, 1]
    with pytest.raises(research.HistoricalNavStorageError) as error:
        research.save_cash_research(req.model_copy(update={"expected_dataset_hash": "f" * 64}))
    assert error.value.code == "REQUEST_KEY_CONFLICT"


def test_model_release_cannot_be_enabled_by_sql_status(database):
    engine, _ = database
    first, _ = storage.save_cash_batch(request())
    data = research.load_cash_dataset(CashPrepareRequest(batchIds=[first.batch_id]))
    research.save_cash_research(
        CashResearchRequest(batchIds=[first.batch_id], requestKey=uuid4(), expectedDatasetHash=data.report.dataset_hash)
    )
    with pytest.raises(Exception, match="ck_cash_research_no_release"):
        with engine.begin() as conn:
            conn.exec_driver_sql("UPDATE cash_research_run SET publication_status='ACTIVE'")
    assert counts(engine) == [1, 1, 1, 1]


def test_concurrent_same_request_creates_one_complete_batch(database, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier

    engine, _ = database
    barrier = Barrier(2)
    original = storage.build_cash_batch_in_session

    def build(session, request, *, deadline):
        output = original(session, request, deadline=deadline)
        barrier.wait(timeout=8)
        return output

    monkeypatch.setattr(storage, "build_cash_batch_in_session", build)
    req = request()
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = tuple(pool.map(storage.save_cash_batch, (req, req)))
    assert results[0][0] == results[1][0]
    assert sorted(created for _, created in results) == [False, True]
    assert counts(engine) == [1, 1, 1, 0]


def test_prediction_precheck_reads_saved_report_in_readonly_transaction(database, monkeypatch):
    from datetime import date
    from types import SimpleNamespace

    from app.schemas.cash_prediction_check import CashPredictionCheckRequest
    from app.services import cash_prediction_check as check

    engine, _ = database
    batch, _ = storage.save_cash_batch(request())
    data = research.load_cash_dataset(CashPrepareRequest(batchIds=[batch.batch_id]))
    run, _ = research.save_cash_research(
        CashResearchRequest(batchIds=[batch.batch_id], requestKey=uuid4(), expectedDatasetHash=data.report.dataset_hash)
    )
    before = counts(engine)
    monkeypatch.setattr(check, "get_nav_preview_engine", lambda: engine)

    def source(session, fund_code, today, *, include_research):
        # 本隔离schema仅建最小基金FK表；源元数据用替身，报告及事务确实走PostgreSQL。
        assert not include_research
        assert session.execute(text("SHOW transaction_read_only")).scalar_one() == "on"
        assert session.execute(text("SHOW transaction_isolation")).scalar_one() == "repeatable read"
        return (
            SimpleNamespace(fund_code=fund_code, fund_type="STOCK", status="ACTIVE", source_code="TUSHARE_PRO_FUND"),
            SimpleNamespace(enabled=True),
            date(2024, 6, 20),
            None,
        )

    monkeypatch.setattr(check, "read_prediction_inputs", source)
    result = check.check_cash_prediction(
        CashPredictionCheckRequest(
            fundCode="006730",
            researchRunId=run.run_id,
            expectedReportHash=run.report.report_hash,
        )
    )
    assert result.status == "GENERATION_BLOCKED" and result.research_run_id == run.run_id
    assert result.incomplete_window_ids and not result.forecast_created and not result.database_written
    assert counts(engine) == before
