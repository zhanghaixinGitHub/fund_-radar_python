"""显式启用的真实PostgreSQL存储测试；仅在本机独立测试schema内写人工数据。

RUN_NAV_STORAGE_PG_TESTS=1时运行。正常pytest默认跳过，绝不自动连接真实库。
测试不使用public业务表：独立search_path、最小外键桩表；结束时只清理自己创建的随机schema。
"""

import os
import re
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from uuid import uuid4

import pytest
from alembic.migration import MigrationContext
from alembic.operations import Operations
from app.core.config import get_settings
from app.services import historical_nav_storage as storage
from sqlalchemy import create_engine, event, text
from sqlalchemy.engine import make_url
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.pool import NullPool
from tests.test_historical_nav_http import HEADERS  # noqa: F401
from tests.test_historical_nav_http import client as client
from tests.test_historical_nav_storage import PATH, SOURCE, save_request, synthetic_preview
from tests.test_historical_nav_storage_schema import load_migration

pytestmark = pytest.mark.skipif(os.getenv("RUN_NAV_STORAGE_PG_TESTS") != "1", reason="需要显式启用隔离PostgreSQL测试")


@pytest.fixture
def database(monkeypatch):
    url = make_url(get_settings().ai_database_url)
    assert url.host in {"localhost", "127.0.0.1", "::1"} and url.database == "fund_ai"
    schema = "nav_storage_test_" + uuid4().hex
    control = create_engine(
        url,
        poolclass=NullPool,
        hide_parameters=True,
        connect_args={"connect_timeout": 5, "options": "-c lock_timeout=3000 -c statement_timeout=5000"},
    )
    engine = None
    try:
        with control.begin() as conn:
            conn.exec_driver_sql(f'CREATE SCHEMA "{schema}"')
            conn.exec_driver_sql(f'SET LOCAL search_path TO "{schema}", pg_catalog')
            conn.exec_driver_sql("CREATE TABLE fund_share_class (fund_code varchar(32) PRIMARY KEY)")
            conn.exec_driver_sql("CREATE TABLE source_sync_run (sync_run_id uuid PRIMARY KEY)")
            migration = load_migration()
            migration.op = Operations(MigrationContext.configure(conn))
            migration.upgrade()
            conn.exec_driver_sql("INSERT INTO fund_share_class VALUES ('008888')")
            conn.execute(text("INSERT INTO source_sync_run VALUES (:run)"), {"run": SOURCE.source_sync_run_id})
        engine = create_engine(
            url,
            pool_size=4,
            max_overflow=0,
            pool_timeout=5,
            hide_parameters=True,
            connect_args={
                "connect_timeout": 5,
                "options": f"-c search_path={schema},pg_catalog -c lock_timeout=3000 -c statement_timeout=5000",
            },
        )
        with engine.connect() as conn:
            assert conn.execute(text("SELECT current_schema()")).scalar_one() == schema
        monkeypatch.setattr(storage, "get_nav_sample_storage_engine", lambda: engine)
        calls = []

        def build(session, request, *, deadline):
            calls.append(request)
            return synthetic_preview(request)

        monkeypatch.setattr(storage, "build_stored_historical_nav_batch", build)
        yield engine, calls
    finally:
        if engine is not None:
            engine.dispose()
        # 删除范围是本测试刚创建的随机命名空间，不允许替换成public或清理其他schema。
        assert re.fullmatch(r"nav_storage_test_[0-9a-f]{32}", schema)
        with control.begin() as conn:
            owned = conn.execute(
                text("SELECT nspowner=current_user::regrole FROM pg_namespace WHERE nspname=:name"), {"name": schema}
            ).scalar_one_or_none()
            if owned is not None:
                assert owned
                conn.exec_driver_sql(f'DROP SCHEMA "{schema}" CASCADE')
        control.dispose()


def row_counts(engine):
    with engine.connect() as conn:
        return [
            conn.exec_driver_sql(f"SELECT COUNT(*) FROM {name}").scalar_one()
            for name in (
                "historical_nav_sample_batch",
                "historical_nav_sample",
                "historical_nav_sample_label",
            )
        ]


def test_save_retry_get_and_page_independence(database, client):
    engine, calls = database
    request = save_request(60, 80)
    body = request.model_dump(mode="json", by_alias=True)
    first = client.post(PATH, json=body, headers=HEADERS)
    assert first.status_code == 201, first.text
    repeated = client.post(PATH, json={**body, "pageSize": 30}, headers=HEADERS)
    assert repeated.status_code == 200 and repeated.json() == first.json()
    readback = client.get(f"{PATH}/{first.json()['batch_id']}", headers=HEADERS)
    assert readback.status_code == 200 and readback.json() == first.json()
    assert len(calls) == 1 and row_counts(engine) == [1, 21, 21]
    assert first.json()["items"] == synthetic_preview(request)[1].model_dump(mode="json")["items"]


@pytest.mark.parametrize("first,last,expected", [(0, 20, [1, 21, 0]), (110, 130, [1, 21, 1]), (160, 165, [1, 0, 0])])
def test_rejected_pending_and_empty_batches_are_retained(database, first, last, expected):
    engine, _ = database
    request = save_request(first, last)
    stored, created = storage.save_historical_nav_batch(request)
    assert created and row_counts(engine) == expected
    assert stored.items == synthetic_preview(request)[1].items


def test_reuse_does_not_rebuild_and_explicit_recalculation_preserves_old_batch(database, monkeypatch):
    engine, _ = database
    request = save_request()
    first, _ = storage.save_historical_nav_batch(request)
    build = storage.build_stored_historical_nav_batch
    monkeypatch.setattr(
        storage, "build_stored_historical_nav_batch", lambda *args, **kwargs: pytest.fail("reused batch rebuilt")
    )
    assert storage.save_historical_nav_batch(request) == (first, False)
    monkeypatch.setattr(storage, "build_stored_historical_nav_batch", build)
    second, created = storage.save_historical_nav_batch(request.model_copy(update={"request_key": uuid4()}))
    assert created and second.batch_id != first.batch_id
    assert storage.get_historical_nav_batch(first.batch_id) == first
    assert row_counts(engine) == [2, 2, 2]


def test_same_request_key_different_range_conflicts(database):
    engine, _ = database
    request = save_request()
    storage.save_historical_nav_batch(request)
    with pytest.raises(storage.HistoricalNavStorageError) as error:
        storage.save_historical_nav_batch(save_request(60, 61, key=request.request_key))
    assert error.value.code == "REQUEST_KEY_CONFLICT" and row_counts(engine) == [1, 1, 1]


@pytest.mark.parametrize("table", ["historical_nav_sample", "historical_nav_sample_label"])
def test_insert_failure_rolls_back_whole_batch(database, client, table):
    engine, _ = database
    body = save_request().model_dump(mode="json", by_alias=True)

    def fail(conn, cursor, statement, parameters, context, executemany):
        if statement.startswith(f"INSERT INTO {table} "):
            raise SQLAlchemyError("synthetic storage failure")

    event.listen(engine, "before_cursor_execute", fail)
    try:
        response = client.post(PATH, json=body, headers=HEADERS)
        assert response.status_code == 503
    finally:
        event.remove(engine, "before_cursor_execute", fail)
    assert row_counts(engine) == [0, 0, 0]
    assert client.post(PATH, json=body, headers=HEADERS).status_code == 201
    assert row_counts(engine) == [1, 1, 1]


def test_commit_failure_returns_error_and_original_key_can_retry(database, client):
    engine, _ = database
    body = save_request().model_dump(mode="json", by_alias=True)

    def fail_commit(conn):
        raise SQLAlchemyError("synthetic commit failure")

    # 在真实事务提交前抛错，不能把已flush但尚未提交的数据当作保存成功。
    event.listen(engine, "commit", fail_commit)
    try:
        response = client.post(PATH, json=body, headers=HEADERS)
        assert response.status_code == 503
    finally:
        event.remove(engine, "commit", fail_commit)
    assert row_counts(engine) == [0, 0, 0]
    assert client.post(PATH, json=body, headers=HEADERS).status_code == 201
    assert row_counts(engine) == [1, 1, 1]


def test_concurrent_identical_requests_save_exactly_once(database, monkeypatch):
    engine, _ = database
    request = save_request()
    barrier = Barrier(2, timeout=5)

    def simultaneous_build(session, request, *, deadline):
        result = synthetic_preview(request)
        barrier.wait()
        return result

    monkeypatch.setattr(storage, "build_stored_historical_nav_batch", simultaneous_build)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(storage.save_historical_nav_batch, request) for _ in range(2)]
        results = [future.result(timeout=15) for future in futures]
    assert sorted(created for _, created in results) == [False, True]
    assert results[0][0] == results[1][0] and row_counts(engine) == [1, 1, 1]


def test_concurrent_different_ranges_with_same_key_conflict(database, monkeypatch):
    engine, _ = database
    request = save_request()
    requests = [request, save_request(60, 61, key=request.request_key)]
    barrier = Barrier(2, timeout=5)

    def simultaneous_build(session, request, *, deadline):
        result = synthetic_preview(request)
        barrier.wait()
        return result

    def save_or_conflict(request):
        try:
            return storage.save_historical_nav_batch(request)
        except storage.HistoricalNavStorageError as error:
            return error

    monkeypatch.setattr(storage, "build_stored_historical_nav_batch", simultaneous_build)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(save_or_conflict, item) for item in requests]
        results = [future.result(timeout=15) for future in futures]
    errors = [item for item in results if isinstance(item, storage.HistoricalNavStorageError)]
    successes = [item for item in results if isinstance(item, tuple)]
    assert len(errors) == len(successes) == 1
    assert errors[0].status_code == 409 and errors[0].code == "REQUEST_KEY_CONFLICT"
    batch, created = successes[0]
    assert created and row_counts(engine) == [1, batch.sample_count, batch.scorable_count]


def test_read_is_read_only_and_never_reads_nav(database):
    engine, _ = database
    batch, _ = storage.save_historical_nav_batch(save_request())
    statements = []

    def record(conn, cursor, statement, parameters, context, executemany):
        statements.append(statement)

    event.listen(engine, "before_cursor_execute", record)
    try:
        assert storage.get_historical_nav_batch(batch.batch_id) == batch
    finally:
        event.remove(engine, "before_cursor_execute", record)
    assert statements[0] == "SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY"
    assert len(statements) == 3 and not any("nav_daily" in statement for statement in statements)


def test_corrupt_stored_summary_fails_closed(database, client):
    engine, _ = database
    batch, _ = storage.save_historical_nav_batch(save_request())
    with engine.begin() as conn:
        conn.execute(text("UPDATE historical_nav_sample_batch SET sample_count=0,scorable_count=0"))
    response = client.get(f"{PATH}/{batch.batch_id}", headers=HEADERS)
    assert response.status_code == 503 and response.json()["detail"]["code"] == "STORED_BATCH_INCONSISTENT"


def test_unknown_batch_returns_404(database, client):
    response = client.get(f"{PATH}/{uuid4()}", headers=HEADERS)
    assert response.status_code == 404 and response.json()["detail"]["code"] == "BATCH_NOT_FOUND"
