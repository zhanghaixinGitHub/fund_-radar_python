"""经理目录的身份隔离、分页、任期修订、内部认证及任务幂等边界。"""

from datetime import UTC, date, datetime
from types import SimpleNamespace
from unittest.mock import Mock
from uuid import uuid4

import pytest
from app.api.dependencies import require_service_token
from app.core.config import get_settings
from app.integrations.fund_manager_source import ManagerSource
from app.integrations.tushare import TushareIntegrationError
from app.main import create_application
from app.schemas.fund_manager_directory import ManagerPage, ManagerPerformance, ManagerSyncRequest
from app.services import fund_manager_directory_read as read
from app.services.fund_manager_directory_sync import ManagerDirectorySync, _state, _team, day, normalize, stable_id
from app.services.sync_jobs import LocalSyncJobManager, SyncJobInProgressError
from fastapi.testclient import TestClient


def record(name="甲", code="000001.OF", begin="20240101", end=" ", **extra):
    return {
        "name": name,
        "ts_code": code,
        "begin_date": begin,
        "end_date": end,
        "ann_date": "20240101",
        "edu": "硕士",
        **extra,
    }


def test_empty_dates_are_unknown_and_same_name_has_different_fund_anchors():
    assert day(" ") is None
    assert day("20240229") == date(2024, 2, 29)
    with pytest.raises(ValueError):
        day("20240230")
    assert stable_id("source:000001:甲") != stable_id("source:000002:甲")
    assert stable_id("source:000001:甲") == stable_id("source:000001:甲")


def test_conflicting_end_dates_are_not_counted_as_definite_current():
    rows = normalize([record(), record(end="20250101")])
    assert len(rows) == 1
    assert not rows[0]["valid"]
    assert _state(rows[0], date(2026, 10, 26)) == "PENDING"
    assert _state(normalize([record(begin="20270101")])[0], date(2026, 10, 26)) == "FUTURE"
    assert _state(normalize([record(end="20261026")])[0], date(2026, 10, 26)) == "CURRENT"
    assert _state(normalize([record(end="20261025")])[0], date(2026, 10, 26)) == "HISTORY"


def test_coteam_requires_whole_fund_roster_and_overlap():
    manager = normalize([record()])[0]
    assert _team(manager, None) == "UNKNOWN"
    assert _team(manager, normalize([record(), record(name="乙", begin="20240102", end="20250101")])) == "CO_MANAGED"
    assert _team(manager, normalize([record(), record(name="乙", begin="20230101", end="20231231")])) == "SOLO"
    assert _team(manager, normalize([record(), record(name="乙", begin="")])) == "UNKNOWN"


def make_source(responses):
    source = ManagerSource.__new__(ManagerSource)
    source.client = SimpleNamespace(_query=Mock(side_effect=responses))
    source.pages = []
    return source


def test_source_reads_beyond_full_first_page_and_preserves_receipts():
    first = [record(code=f"{i:06d}.OF") for i in range(5000)]
    source = make_source([first, [record(code="999999.OF")]])
    assert len(source.managers(name="甲")) == 5001
    assert source.client._query.call_count == 2
    assert source.pages[1]["parameters"]["offset"] == 5000
    assert source.pages[0]["receivedAt"]


def test_source_repeated_page_wrong_person_and_partial_failure_are_rejected():
    page = [record()] * 5000
    with pytest.raises(TushareIntegrationError):
        make_source([page, page]).managers(name="甲")
    with pytest.raises(TushareIntegrationError):
        make_source([[record(name="乙")]]).managers(name="甲")
    with pytest.raises(TimeoutError):
        make_source([page, TimeoutError()]).managers(name="甲")
    assert make_source([[]]).managers(name="甲") == []


def test_same_name_and_same_company_do_not_automatically_accept_other_funds(tmp_path):
    primary = record(resume="甲的公开职业履历")
    other = record(code="000002.OF", resume="甲的公开职业履历")
    source = SimpleNamespace(pages=[], managers=lambda **kwargs: [primary, other] if kwargs.get("name") else [primary])
    sync = ManagerDirectorySync(engine=Mock(), source=source, root=tmp_path, official_reader=lambda *args: None)
    seed = {"name": "甲", "sourceCode": "000001.OF", "code": "000001", "company": "同一家基金公司", "rows": [primary]}
    prepared = sync.prepare(seed, datetime.now(UTC))
    assert {row["ts_code"] for row in prepared["rows"]} == {"000001.OF"}
    assert prepared["pending"] == 1
    assert prepared["profile"] == "甲的公开职业履历"


def test_complete_source_missing_old_relationship_is_pending_not_deleted_or_left_current(tmp_path):
    sync = ManagerDirectorySync(
        engine=Mock(), source=SimpleNamespace(pages=[], managers=lambda **kwargs: []), root=tmp_path
    )
    seed = {"name": "甲", "sourceCode": "000001.OF", "code": "000001", "company": "甲公司", "rows": [record()]}
    prepared = sync.prepare(seed, datetime.now(UTC))
    assert len(prepared["rows"]) == 1
    assert _state(prepared["rows"][0], date(2026, 10, 26)) == "PENDING"
    assert prepared["checked"] is None


def test_api_read_never_constructs_source_or_job_and_requires_internal_identity(monkeypatch):
    application = create_application()
    client = TestClient(application)
    assert client.get("/internal/v1/fund-managers").status_code in {403, 503}
    assert client.get("/internal/v1/fund-managers", headers={"Origin": "https://example.com"}).status_code == 403
    application.dependency_overrides[require_service_token] = lambda: None
    monkeypatch.setattr(
        read, "list_managers", lambda *args: ManagerPage(items=[], page=1, pageSize=20, totalCount=0, totalPages=0)
    )
    assert client.get("/internal/v1/fund-managers").json()["items"] == []
    assert client.get("/internal/v1/fund-managers?pageSize=5000").status_code == 422
    assert client.get("/internal/v1/fund-managers/not-a-uuid").status_code == 422


def test_expired_snapshot_is_not_silently_replaced(monkeypatch):
    application = create_application()
    application.dependency_overrides[require_service_token] = lambda: None

    def expired(*args):
        raise read.ManagerSnapshotExpired("资料版本已变化")

    monkeypatch.setattr(read, "list_managers", expired)
    response = TestClient(application).get("/internal/v1/fund-managers?snapshotId=" + str(uuid4()))
    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "SNAPSHOT_EXPIRED"


def test_unknown_values_remain_null_and_client_cannot_supply_source_or_paths():
    value = ManagerPerformance()
    assert value.returnRate is None and value.maxDrawdownRate is None and value.annualizedRate is None
    assert ManagerPerformance(returnRate="0.1234").returnRate.as_tuple().exponent == -4
    with pytest.raises(ValueError):
        ManagerSyncRequest(managerIds=[], requestId=uuid4(), url="http://internal")
    with pytest.raises(ValueError):
        ManagerSyncRequest(managerIds=[uuid4() for _ in range(21)], requestId=uuid4())


def test_manager_job_repeated_request_returns_same_id_and_keeps_global_mutex(tmp_path):
    manager = LocalSyncJobManager(state_path=tmp_path / "jobs.json")
    manager._executor.submit = Mock()
    try:
        request_id = str(uuid4())
        first = manager.start_fund_managers((), request_id)
        repeated = manager.start_fund_managers((), request_id)
        assert first.job_id == repeated.job_id
        assert manager._executor.submit.call_count == 1
        with pytest.raises(SyncJobInProgressError):
            manager.start_fund_managers((), str(uuid4()))
        assert request_id in (tmp_path / "jobs.json").read_text(encoding="utf-8")
    finally:
        manager.close()


def test_manager_job_restart_preserves_request_and_reports_interrupted(tmp_path):
    path = tmp_path / "jobs.json"
    manager = LocalSyncJobManager(state_path=path)
    manager._executor.submit = Mock()
    request_id = str(uuid4())
    original = manager.start_fund_managers((), request_id)
    manager.close()
    restored = LocalSyncJobManager(state_path=path)
    try:
        same = restored.start_fund_managers((), request_id)
        assert same.job_id == original.job_id
        assert same.status == "FAILED" and same.error_code == "SYNC_INTERRUPTED"
    finally:
        restored.close()


def test_manager_job_preserves_real_source_and_update_counts():
    run_id = uuid4()
    summary = {"status": "SUCCEEDED", "completed": 1, "fetchedRows": 20,
               "created": 0, "updated": 1, "skipped": 0, "syncRunId": str(run_id)}
    manager = LocalSyncJobManager(manager_synchronizer=lambda *args, **kwargs: summary)
    manager._executor.submit = Mock()
    try:
        request = str(uuid4())
        original = manager.start_fund_managers((), request)
        manager._run_fund_managers(original.job_id, (), request)
        value = manager.get_job(original.job_id)
        assert value.fetched_count == 20 and value.created_count == 0
        assert value.updated_count == 1 and value.sync_run_id == run_id
        assert manager.start_fund_managers((), request).job_id == original.job_id
    finally:
        manager.close()


def test_same_request_id_cannot_be_reused_for_a_different_scope():
    manager = LocalSyncJobManager()
    manager._executor.submit = Mock()
    try:
        request = str(uuid4())
        manager.start_fund_managers((), request)
        with pytest.raises(ValueError):
            manager.start_fund_managers((uuid4(),), request)
    finally:
        manager.close()


def test_snapshot_cannot_be_read_through_another_person(monkeypatch):
    from contextlib import contextmanager

    person, foreign = uuid4(), uuid4()
    session = Mock()
    session.get.side_effect = [
        SimpleNamespace(manager_id=person, redirect_to=None, current_snapshot_id=uuid4()),
        SimpleNamespace(manager_id=foreign, scope_kind="MANAGER"),
    ]

    @contextmanager
    def fake_reading():
        yield session

    monkeypatch.setattr(read, "reading", fake_reading)
    with pytest.raises(read.ManagerSnapshotExpired):
        read.detail(person, uuid4())


@pytest.fixture
def isolated_manager_database():
    """只在本机新建随机测试schema，七张表与公共数据隔离，结束后精确清理本测试schema。"""
    from app.db.base import Base
    from app.db.session import get_engine
    from sqlalchemy import create_engine, text
    from sqlalchemy.engine import make_url

    url = get_settings().ai_database_url
    if make_url(url).host not in {"localhost", "127.0.0.1", "::1"}:
        pytest.skip("数据库隔离测试仅允许本机")
    schema = "manager_test_" + uuid4().hex
    with get_engine().begin() as connection:
        connection.execute(text(f'CREATE SCHEMA "{schema}"'))
    engine = create_engine(url, connect_args={"options": f"-c search_path={schema},public"})
    names = [
        "fund_manager",
        "fund_manager_identity_evidence",
        "fund_manager_profile_version",
        "fund_manager_related_fund",
        "fund_manager_tenure_version",
        "fund_manager_coverage_snapshot",
        "fund_manager_metric_snapshot",
    ]
    try:
        # has_table会沿search_path找到public同名表；必须强制创建并在写入前验证本schema为空。
        with engine.connect() as connection:
            assert connection.scalar(text("SELECT current_schema()")) == schema
        Base.metadata.create_all(engine, tables=[Base.metadata.tables[name] for name in names], checkfirst=False)
        with engine.connect() as connection:
            assert connection.scalar(text("SELECT count(*) FROM fund_manager")) == 0
            assert connection.scalar(
                text("SELECT count(*) FROM pg_tables WHERE schemaname=:schema"), {"schema": schema}
            ) == 7
        yield engine
    finally:
        engine.dispose()
        assert schema.startswith("manager_test_") and len(schema) == 45
        with get_engine().begin() as connection:
            connection.execute(text(f'DROP SCHEMA "{schema}" CASCADE'))


def test_publish_is_idempotent_and_departure_replaces_current_version(isolated_manager_database, tmp_path):
    from app.models.fund_manager_directory import FundManager, FundManagerMetricSnapshot, FundManagerTenureVersion
    from sqlalchemy import func, select
    from sqlalchemy.orm import Session

    sync = ManagerDirectorySync(engine=isolated_manager_database, root=tmp_path)
    seed = sync.seeds()[0]
    prepared = sync.prepare(seed, datetime.now(UTC))
    assert sync.publish(seed, prepared, datetime.now(UTC)) is True
    assert sync.publish(seed, prepared, datetime.now(UTC)) is False
    with Session(isolated_manager_database) as session:
        assert session.scalar(select(func.count()).select_from(FundManager)) == 1
        assert session.scalar(select(func.count()).select_from(FundManagerMetricSnapshot)) == len(prepared["rows"])
    # 独立构造同一开始日从在任改为已离任，不能产生两个当前版本。
    original = next(row for row in prepared["rows"] if row["begin"])
    revised = {**original, "end_date": "20261001", "end": date(2026, 10, 1), "valid": True}
    sync.publish(seed, {**prepared, "rows": [revised]}, datetime.now(UTC))
    with Session(isolated_manager_database) as session:
        key = stable_id(f"{seed['id']}:{revised['ts_code']}:{revised['begin']}")
        current = session.scalars(
            select(FundManagerTenureVersion).where(
                FundManagerTenureVersion.tenure_key == key, FundManagerTenureVersion.is_current.is_(True)
            )
        ).all()
        assert len(current) == 1 and current[0].end_date == date(2026, 10, 1)


def test_failed_publication_keeps_previous_snapshot(isolated_manager_database, tmp_path, monkeypatch):
    from app.models.fund_manager_directory import FundManager
    from app.services import fund_manager_directory_sync as module
    from sqlalchemy.orm import Session

    sync = ManagerDirectorySync(engine=isolated_manager_database, root=tmp_path)
    seed = sync.seeds()[0]
    prepared = sync.prepare(seed, datetime.now(UTC))
    sync.publish(seed, prepared, datetime.now(UTC))
    with Session(isolated_manager_database) as session:
        before = session.get(FundManager, seed["id"]).current_snapshot_id
    monkeypatch.setattr(module, "_metric", Mock(side_effect=ValueError("模拟校验失败")))
    with pytest.raises(ValueError):
        sync.publish(seed, prepared, datetime.now(UTC))
    with Session(isolated_manager_database) as session:
        assert session.get(FundManager, seed["id"]).current_snapshot_id == before
