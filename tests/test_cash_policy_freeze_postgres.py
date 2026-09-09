"""规则冻结真实SQL、不可变约束及并发；业务确认仅存在于隔离测试临时文件。"""

import copy
import importlib.util
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier
from uuid import uuid4

import pytest
from alembic.migration import MigrationContext
from alembic.operations import Operations
from app.models.cash_policy_freeze import CashPolicyFreezeRecord
from app.services import cash_policy_freeze as service
from app.services import cash_release_policy as policy_source
from app.services.historical_nav_storage import HistoricalNavStorageError
from sqlalchemy import event, text
from sqlalchemy.exc import DBAPIError, IntegrityError
from sqlalchemy.orm import Session
from tests.test_cash_policy_freeze import URL, approved_policy, request_for, row_for
from tests.test_cash_prediction_check import evaluated as evaluated
from tests.test_cash_reinvestment_postgres import database as database
from tests.test_cash_reinvestment_research import data as data
from tests.test_cash_release_review_postgres import review_db as review_db
from tests.test_historical_nav_http import HEADERS
from tests.test_historical_nav_http import client as client

pytestmark = pytest.mark.skipif(os.getenv("RUN_NAV_STORAGE_PG_TESTS") != "1", reason="显式启用隔离PG")


def migration_module():
    path = Path(__file__).resolve().parents[1] / "alembic/versions/20260909_17_cash_policy_freeze.py"
    spec = importlib.util.spec_from_file_location("cash_policy_migration", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def policy_db(review_db, monkeypatch, tmp_path):
    engine, research_request = review_db
    with engine.begin() as conn:
        module = migration_module()
        module.op = Operations(MigrationContext.configure(conn))
        module.upgrade()
    path = tmp_path / "approved-test-policy.json"
    path.write_text(approved_policy().model_dump_json(), encoding="utf-8")
    monkeypatch.setattr(policy_source, "POLICY_PATH", path)
    monkeypatch.setattr(service, "get_nav_sample_storage_engine", lambda: engine)
    return engine, request_for(), research_request, path


def count(engine):
    with engine.connect() as conn:
        return conn.execute(text("SELECT count(*) FROM cash_policy_freeze")).scalar_one()


def test_http_save_retry_get_and_review_bind_actual_snapshot(policy_db, client):
    from tests.test_cash_planned_research_schema import migration_module as planned_migration

    engine, req, research_request, _ = policy_db
    # 当前审查在已有规则匹配后按研究唯一键查新绑定；此旧报告没有绑定，不能补认。
    with engine.begin() as conn:
        module = planned_migration()
        module.op = Operations(MigrationContext.configure(conn))
        module.upgrade()
    payload = req.model_dump(mode="json", by_alias=True)
    first = client.post(URL + "/freezes", json=payload, headers=HEADERS)
    repeated = client.post(URL + "/freezes", json=payload, headers=HEADERS)
    assert first.status_code == 201 and repeated.status_code == 200
    a, b = first.json(), repeated.json()
    assert a["created"] and a["database_written"] and not b["created"] and not b["database_written"]
    assert a["freeze_id"] == b["freeze_id"] and a["content_hash"] == b["content_hash"]
    response = client.get(URL + "/freezes/" + a["freeze_id"], headers=HEADERS)
    assert response.status_code == 200 and response.json() == b and response.headers["cache-control"] == "no-store"
    assert count(engine) == 1
    review = client.post(
        "/internal/v1/predictions/release-review",
        json=research_request.model_dump(mode="json", by_alias=True),
        headers=HEADERS,
    )
    assert review.status_code == 200
    result = review.json()
    assert result["policy_persisted"] and result["policy_freeze_id"] == a["freeze_id"]
    assert not result["ex_ante_plan_verified"] and result["exam_coverage_evidence"] == []
    assert result["status"] == "BLOCKED" and not result["publication_allowed"] and not result["database_written"]
    assert (
        "POLICY_NOT_FROZEN" not in result["blocking_codes"]
        and "INDEPENDENT_TEST_NOT_EVALUATED" in result["blocking_codes"]
    )


def test_get_is_readonly_and_configuration_change_never_overwrites(policy_db, client, monkeypatch):
    engine, req, research_request, path = policy_db
    frozen = service.freeze_cash_policy(req)
    statements = []

    def capture(conn, cursor, statement, parameters, context, executemany):
        statements.append(statement)

    event.listen(engine, "before_cursor_execute", capture)
    try:
        read = service.get_cash_policy_freeze(frozen.freeze_id)
    finally:
        event.remove(engine, "before_cursor_execute", capture)
    assert len(statements) == 2 and "READ ONLY" in statements[0]
    assert read.content_hash == frozen.content_hash and not read.created
    assert all("cash_research_run" not in sql and "nav_daily" not in sql for sql in statements)
    changed = service.load_release_policy().model_dump(mode="json")
    changed["minimum_coverage"] = "0.81"
    path.write_text(json.dumps(changed), encoding="utf-8")
    result = client.post(
        "/internal/v1/predictions/release-review",
        json=research_request.model_dump(mode="json", by_alias=True),
        headers=HEADERS,
    )
    assert result.status_code == 409 and result.json()["detail"]["code"] == "CASH_POLICY_FREEZE_MISMATCH"
    with pytest.raises(HistoricalNavStorageError):
        service.freeze_cash_policy(request_for().model_copy(update={"request_key": req.request_key}))
    assert service.get_cash_policy_freeze(frozen.freeze_id).content_hash == frozen.content_hash and count(engine) == 1


@pytest.mark.parametrize(
    "statement",
    [
        "UPDATE cash_policy_freeze SET snapshot=snapshot",
        "DELETE FROM cash_policy_freeze",
        "TRUNCATE cash_policy_freeze",
    ],
)
def test_database_rejects_mutation(policy_db, statement):
    engine, req, _, _ = policy_db
    frozen = service.freeze_cash_policy(req)
    with pytest.raises(DBAPIError, match="immutable"):
        with engine.begin() as conn:
            conn.exec_driver_sql(statement)
    assert count(engine) == 1 and service.get_cash_policy_freeze(frozen.freeze_id).content_hash == frozen.content_hash


@pytest.mark.parametrize("same_key", [True, False])
def test_concurrent_freeze_is_unique(policy_db, monkeypatch, same_key):
    engine, req, _, _ = policy_db
    other = req if same_key else req.model_copy(update={"request_key": uuid4()})
    barrier, original = Barrier(2), service.find_policy_freeze

    def synchronized_read(session, **identity):
        row = original(session, **identity)
        if identity.get("policy_version") and row is None:
            barrier.wait(timeout=5)
        return row

    def save(request):
        try:
            return service.freeze_cash_policy(request)
        except HistoricalNavStorageError as error:
            return error.code

    monkeypatch.setattr(service, "find_policy_freeze", synchronized_read)
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(save, (req, other)))
    assert count(engine) == 1
    if same_key:
        assert results[0].freeze_id == results[1].freeze_id and sum(r.created for r in results) == 1
    else:
        assert sum(isinstance(r, str) and r == "CASH_POLICY_VERSION_FROZEN" for r in results) == 1


def test_failed_readback_rolls_back_insert(policy_db, monkeypatch):
    engine, req, _, _ = policy_db

    def fail(*args, **kwargs):
        raise ValueError("synthetic readback failed")

    monkeypatch.setattr(service, "restore_policy_freeze", fail)
    with pytest.raises(ValueError):
        service.freeze_cash_policy(req)
    assert count(engine) == 0


@pytest.mark.parametrize("mutation", ["draft", "missing_approval", "hash"])
def test_db_cannot_accept_draft_or_missing_approval(policy_db, mutation):
    engine, _, _, _ = policy_db
    row = copy.deepcopy(row_for())
    if mutation == "draft":
        row.snapshot["policy"]["approval_state"] = "DRAFT"
    elif mutation == "missing_approval":
        del row.snapshot["policy"]["approval_reference"]
    else:
        row.policy_hash = "fake"
    with pytest.raises(IntegrityError):
        with Session(engine) as session, session.begin():
            session.add(CashPolicyFreezeRecord(**vars(row)))
    assert count(engine) == 0


def test_nonempty_downgrade_refused_and_all_columns_commented(policy_db):
    engine, req, _, _ = policy_db
    service.freeze_cash_policy(req)
    with pytest.raises(DBAPIError, match="not empty"):
        with engine.begin() as conn:
            module = migration_module()
            module.op = Operations(MigrationContext.configure(conn))
            module.downgrade()
    with engine.connect() as conn:
        assert (
            conn.execute(
                text(
                    "SELECT count(*) FROM pg_attribute WHERE attrelid='cash_policy_freeze'::regclass "
                    "AND attnum>0 AND NOT attisdropped AND col_description(attrelid,attnum) IS NULL"
                )
            ).scalar_one()
            == 0
        )
    assert count(engine) == 1


def test_empty_downgrade_keeps_research_and_cleans_trigger_function(policy_db):
    engine, _, _, _ = policy_db
    with engine.begin() as conn:
        module = migration_module()
        module.op = Operations(MigrationContext.configure(conn))
        module.downgrade()
        assert conn.execute(text("SELECT to_regclass('cash_policy_freeze')")).scalar_one() is None
        assert conn.execute(text("SELECT to_regprocedure('cash_policy_freeze_immutable()')")).scalar_one() is None
        assert conn.execute(text("SELECT count(*) FROM cash_research_run")).scalar_one() == 1
