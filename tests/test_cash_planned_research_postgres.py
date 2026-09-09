"""隔离PG中的计划先行研究；确认资料/模型输入人工构造，不审批真实规则。"""

import json
import os
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from threading import Barrier
from uuid import uuid4

import pytest
from alembic.migration import MigrationContext
from alembic.operations import Operations
from app.schemas.cash_planned_research import CashPlannedResearchRequest
from app.schemas.cash_reinvestment_research import CashPrepareRequest
from app.services import cash_planned_research as service
from app.services import cash_policy_freeze as freeze
from app.services import cash_release_policy as policy_source
from app.services.cash_reinvestment_research import load_cash_dataset
from app.services.cash_reinvestment_storage import save_cash_batch
from app.services.historical_nav_storage import HistoricalNavStorageError
from sqlalchemy import event, text
from sqlalchemy.exc import DBAPIError
from tests.test_cash_planned_research import URL
from tests.test_cash_planned_research_schema import migration_module
from tests.test_cash_policy_freeze import approved_policy
from tests.test_cash_policy_freeze import request_for as policy_request
from tests.test_cash_policy_freeze_postgres import migration_module as policy_migration
from tests.test_cash_reinvestment_postgres import database as database
from tests.test_cash_reinvestment_research import data as data
from tests.test_cash_reinvestment_storage import request as batch_request
from tests.test_historical_nav_http import HEADERS
from tests.test_historical_nav_http import client as client

pytestmark = pytest.mark.skipif(os.getenv("RUN_NAV_STORAGE_PG_TESTS") != "1", reason="显式启用隔离PG")


@pytest.fixture
def planned_db(database, monkeypatch, tmp_path):
    engine, _ = database
    with engine.begin() as conn:
        for module in (policy_migration(), migration_module()):
            module.op = Operations(MigrationContext.configure(conn))
            module.upgrade()
    path = tmp_path / "approved-test-only.json"
    path.write_text(approved_policy().model_dump_json(), encoding="utf-8")
    monkeypatch.setattr(policy_source, "POLICY_PATH", path)
    monkeypatch.setattr(freeze, "get_nav_sample_storage_engine", lambda: engine)
    monkeypatch.setattr(service, "get_nav_sample_storage_engine", lambda: engine)
    frozen = freeze.freeze_cash_policy(policy_request())
    saved, _ = save_cash_batch(batch_request())
    prepared = load_cash_dataset(CashPrepareRequest(batchIds=(saved.batch_id,)))
    request = CashPlannedResearchRequest(
        requestKey=uuid4(),
        batchIds=prepared.report.batch_ids,
        expectedDatasetHash=prepared.report.dataset_hash,
        policyFreezeId=frozen.freeze_id,
        expectedPolicyFreezeHash=frozen.content_hash,
    )
    return engine, request, path


def totals(engine):
    with engine.connect() as conn:
        return tuple(
            conn.execute(text(f"SELECT count(*) FROM {table}")).scalar_one()
            for table in (
                "cash_policy_freeze",
                "cash_research_run",
                "cash_planned_research_binding",
            )
        )


def test_http_create_retry_get_and_durable_policy_before_evaluation(planned_db, client, monkeypatch):
    engine, req, _ = planned_db
    original = service.evaluate_cash_dataset
    calls = []

    def checked(data):
        assert engine.pool.checkedout() == 0  # 计算阶段已经释放资料读取事务/连接。
        assert totals(engine) == (1, 0, 0)  # 规则真的先保存；研究/绑定尚未保存，不把回执写入说成训练前。
        calls.append(data.report.dataset_hash)
        return original(data)

    monkeypatch.setattr(service, "evaluate_cash_dataset", checked)
    first = client.post(URL, headers=HEADERS, json=req.model_dump(mode="json", by_alias=True))
    assert first.status_code == 201
    second = client.post(URL, headers=HEADERS, json=req.model_dump(mode="json", by_alias=True))
    assert second.status_code == 200
    a, b = first.json(), second.json()
    assert a["created"] and not b["created"] and a["content_hash"] == b["content_hash"]
    fetched = client.get(URL + "/" + a["binding_id"], headers=HEADERS)
    assert fetched.status_code == 200 and fetched.json() == b and fetched.headers["cache-control"] == "no-store"
    assert totals(engine) == (1, 1, 1) and calls == [req.expected_dataset_hash]
    assert a["research"]["request_key"] == a["binding_id"] != a["request_key"]
    assert a["research"]["report"]["status"] == "INSUFFICIENT_DATA" and not a["research"]["report"]["model_fitted"]
    assert not a["publication_allowed"] and not a["independent_test_read"]


def test_real_numerical_evaluation_of_synthetic_matrix_is_bound_and_saved(planned_db, data, monkeypatch):
    engine, req, _ = planned_db
    # 此用例只替换资料读取为人工小矩阵，数值拟合与数据库保存实际执行；不称为真实数据准入。
    monkeypatch.setattr(service, "load_cash_dataset_in_session", lambda *a, **k: data)
    req = req.model_copy(update={"batch_ids": data.report.batch_ids, "expected_dataset_hash": data.report.dataset_hash})
    saved = service.save_planned_research(req)
    assert saved.research.report.status == "EVALUATED" and saved.research.report.model_fitted
    assert saved.plan_bound_before_evaluation and not saved.publication_allowed
    read = service.get_planned_research(saved.binding_id)
    assert [w.model.model_hash for w in read.research.report.windows] == [
        w.model.model_hash for w in saved.research.report.windows
    ]
    assert totals(engine) == (1, 1, 1)


@pytest.mark.parametrize("case", ["missing_freeze", "freeze_hash", "dataset_hash"])
def test_bad_proof_or_dataset_never_calls_evaluator(planned_db, monkeypatch, case):
    engine, req, _ = planned_db
    monkeypatch.setattr(service, "evaluate_cash_dataset", lambda _: pytest.fail("must not fit"))
    field, value, code = {
        "missing_freeze": ("policy_freeze_id", uuid4(), "CASH_POLICY_FREEZE_NOT_FOUND"),
        "freeze_hash": ("expected_policy_freeze_hash", "f" * 64, "CASH_POLICY_FREEZE_HASH_MISMATCH"),
        "dataset_hash": ("expected_dataset_hash", "f" * 64, "DATASET_HASH_MISMATCH"),
    }[case]
    with pytest.raises(HistoricalNavStorageError) as error:
        service.save_planned_research(req.model_copy(update={field: value}))
    assert error.value.code == code and totals(engine) == (1, 0, 0)


@pytest.mark.parametrize("phase", ["evaluation", "after_flush"])
def test_failures_leave_neither_report_nor_binding(planned_db, client, monkeypatch, phase):
    engine, req, _ = planned_db

    def fail(*a, **k):
        raise ValueError("synthetic rollback failure")

    monkeypatch.setattr(service, "evaluate_cash_dataset" if phase == "evaluation" else "restore_planned_research", fail)
    response = client.post(URL, headers=HEADERS, json=req.model_dump(mode="json", by_alias=True))
    assert response.status_code == 503 and "synthetic rollback failure" not in response.text
    assert totals(engine) == (1, 0, 0)


def test_retry_parameter_conflict_never_retrains(planned_db, monkeypatch):
    engine, req, _ = planned_db
    service.save_planned_research(req)
    monkeypatch.setattr(service, "evaluate_cash_dataset", lambda _: pytest.fail("no refit"))
    with pytest.raises(HistoricalNavStorageError) as error:
        service.save_planned_research(req.model_copy(update={"expected_dataset_hash": "f" * 64}))
    assert error.value.code == "REQUEST_KEY_CONFLICT" and totals(engine) == (1, 1, 1)


def test_evaluator_cannot_rewrite_the_pre_evaluation_preparation(planned_db, monkeypatch, client):
    engine, req, _ = planned_db
    original = service.evaluate_cash_dataset

    def mutate(data):
        data.report.input_count += 1  # 模拟内部算法意外修改元数据；计算前快照必须已经深复制隔离。
        return original(data)

    monkeypatch.setattr(service, "evaluate_cash_dataset", mutate)
    response = client.post(URL, headers=HEADERS, json=req.model_dump(mode="json", by_alias=True))
    assert response.status_code == 503 and totals(engine) == (1, 0, 0)


def test_get_is_readonly_and_does_not_require_current_approval_file(planned_db, monkeypatch, client):
    engine, req, path = planned_db
    saved = service.save_planned_research(req)
    monkeypatch.setattr(policy_source, "POLICY_PATH", path.with_name("missing.json"))
    monkeypatch.setattr(service, "evaluate_cash_dataset", lambda _: pytest.fail("GET no fit"))
    statements = []

    def capture(conn, cursor, statement, parameters, context, executemany):
        statements.append(statement)

    event.listen(engine, "before_cursor_execute", capture)
    try:
        response = client.get(URL + f"/{saved.binding_id}", headers=HEADERS)
    finally:
        event.remove(engine, "before_cursor_execute", capture)
    assert response.status_code == 200 and not response.json()["database_written"]
    assert len(statements) == 4 and "REPEATABLE READ, READ ONLY" in statements[0]
    assert all(sql.lstrip().upper().startswith(("SET", "SELECT")) for sql in statements)
    assert all("cash_sample" not in sql and "nav_daily" not in sql for sql in statements)


@pytest.mark.parametrize("mutation", ["UPDATE", "DELETE", "TRUNCATE"])
def test_database_rejects_binding_mutations(planned_db, mutation):
    engine, req, _ = planned_db
    saved = service.save_planned_research(req)
    sql = {
        "UPDATE": "UPDATE cash_planned_research_binding SET report_hash=report_hash",
        "DELETE": "DELETE FROM cash_planned_research_binding",
        "TRUNCATE": "TRUNCATE cash_planned_research_binding",
    }[mutation]
    with pytest.raises(DBAPIError, match="immutable"), engine.begin() as conn:
        conn.execute(text(sql))
    assert service.get_planned_research(saved.binding_id).content_hash == saved.content_hash


@pytest.mark.parametrize("same_data", [True, False])
def test_concurrent_worker_race_is_atomic_and_idempotent(planned_db, monkeypatch, same_data):
    engine, req, _ = planned_db
    other = req
    if not same_data:
        another, _ = save_cash_batch(batch_request())
        data = load_cash_dataset(CashPrepareRequest(batchIds=(another.batch_id,)))
        other = req.model_copy(
            update={"batch_ids": data.report.batch_ids, "expected_dataset_hash": data.report.dataset_hash}
        )
    # 模拟跨进程竞争：生产只有进程内计算槽，不能把它冒充跨进程互斥。
    monkeypatch.setattr(service, "training_slot", nullcontext)
    barrier = Barrier(2)
    original = service.evaluate_cash_dataset

    def concurrent(data):
        result = original(data)
        barrier.wait(timeout=15)
        return result

    monkeypatch.setattr(service, "evaluate_cash_dataset", concurrent)

    def worker(request):
        try:
            return service.save_planned_research(request)
        except HistoricalNavStorageError as error:
            return error.code

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(worker, (req, other)))
    assert totals(engine) == (1, 1, 1)
    saved = [r for r in results if not isinstance(r, str)]
    assert sum(r.created for r in saved) == 1
    if same_data:
        assert len(saved) == 2 and saved[0].binding_id == saved[1].binding_id
    else:
        assert len(saved) == 1 and "REQUEST_KEY_CONFLICT" in results


def test_changed_report_is_detected_even_if_its_own_hash_is_updated(planned_db):
    from app.services.cash_reinvestment_storage import cash_hash

    engine, req, _ = planned_db
    saved = service.save_planned_research(req)
    report = saved.research.report.model_dump(mode="json")
    report["release_blockers"].append("synthetic-new-blocker")
    report["report_hash"] = cash_hash({k: v for k, v in report.items() if k != "report_hash"})
    with engine.begin() as conn:
        conn.execute(
            text("UPDATE cash_research_run SET report=CAST(:r AS jsonb) WHERE run_id=:id"),
            {"r": json.dumps(report), "id": saved.research.run_id},
        )
    with pytest.raises(HistoricalNavStorageError) as error:
        service.get_planned_research(saved.binding_id)
    assert error.value.code == "CASH_PLANNED_RESEARCH_CORRUPTED"


def test_nonempty_downgrade_refused_and_comments_complete(planned_db):
    engine, req, _ = planned_db
    service.save_planned_research(req)
    with pytest.raises(DBAPIError, match="not empty"), engine.begin() as conn:
        migration = migration_module()
        migration.op = Operations(MigrationContext.configure(conn))
        migration.downgrade()
    with engine.connect() as conn:
        assert (
            conn.execute(
                text(
                    "SELECT count(*) FROM pg_attribute WHERE attrelid='cash_planned_research_binding'::regclass "
                    "AND attnum>0 AND NOT attisdropped AND col_description(attrelid,attnum) IS NULL"
                )
            ).scalar_one()
            == 0
        )
    assert totals(engine) == (1, 1, 1)


def test_empty_downgrade_preserves_research_and_policy_tables(planned_db):
    engine, _, _ = planned_db
    with engine.begin() as conn:
        migration = migration_module()
        migration.op = Operations(MigrationContext.configure(conn))
        migration.downgrade()
        assert conn.execute(text("SELECT to_regclass('cash_planned_research_binding')")).scalar_one() is None
        assert conn.execute(text("SELECT count(*) FROM cash_policy_freeze")).scalar_one() == 1
        assert conn.execute(text("SELECT count(*) FROM cash_research_run")).scalar_one() == 0
