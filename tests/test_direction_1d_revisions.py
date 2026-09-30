"""追加版本的真实 PostgreSQL 验证在独立临时 schema 中运行，结束后清除测试 schema。"""

import importlib.util
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest
from app.services import direction_1d_jobs as jobs
from app.services import direction_1d_revisions as revisions
from app.services.direction_1d_protocol import ZONE, canonical, digest
from sqlalchemy import create_engine, text


def test_request_does_not_move_to_next_day(monkeypatch):
    monkeypatch.setattr(jobs.repo, "clock", lambda: datetime(2026, 9, 29, 15, 0, tzinfo=ZONE))
    monkeypatch.setattr(jobs, "submit", lambda *args: pytest.fail("截止后的旧目标不得登记新任务"))
    with pytest.raises(ValueError, match="WINDOW_CHANGED"):
        jobs.submit_forecast("002112", date(2026, 9, 29), uuid4())
    with pytest.raises(ValueError, match="EXPECTED_TARGET"):
        jobs.submit_forecast("002112", date(2026, 9, 30), None)


@pytest.fixture
def database(monkeypatch):
    if os.environ.get("FUND_RADAR_DB_TESTS") != "1":
        pytest.skip("本地独立 schema 的数据库验收需要显式开关")
    from app.core.config import get_settings

    admin = create_engine(get_settings().ai_database_url)
    if admin.url.host not in {"localhost", "127.0.0.1"}:
        pytest.fail("仅允许本地数据库测试")
    schema = "p06_test_" + uuid4().hex
    with admin.begin() as connection:
        connection.exec_driver_sql(f'CREATE SCHEMA "{schema}"')
    engine = create_engine(get_settings().ai_database_url, connect_args={"options": f"-c search_path={schema},public"})
    spec = importlib.util.spec_from_file_location(
        "revision_migration", Path("alembic/versions/20260929_31_direction_input_revision.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    try:
        with engine.begin() as connection:
            module.op = SimpleNamespace(execute=connection.exec_driver_sql)
            module.upgrade()
        monkeypatch.setattr(revisions, "get_engine", lambda: engine)
        yield engine
    finally:
        engine.dispose()
        with admin.begin() as connection:
            # schema 由本测试刚创建的随机十六进制标识组成，不接受配置或用户给定删除目标。
            connection.exec_driver_sql(f'DROP SCHEMA "{schema}" CASCADE')
        admin.dispose()


def reserve(database, token):
    data = {"fund_code": "123456", "target_nav_date": "2026-09-30", "protocol": "TEST_ONLY", "source": token}
    with revisions.scope_lock("123456", "2026-09-30", "TEST_ONLY"), database.begin() as connection:
        return revisions.select_revision(connection, data, datetime.now(ZONE) + timedelta(hours=1))


def test_same_input_and_reverted_input_have_correct_order(database):
    first = reserve(database, "A")
    assert reserve(database, "A")["revision_sequence"] == first["revision_sequence"]
    second, third = reserve(database, "B"), reserve(database, "A")
    assert first["revision_sequence"] < second["revision_sequence"] < third["revision_sequence"]


def test_concurrent_same_input_merges_before_assigning_sequence(database):
    with ThreadPoolExecutor(max_workers=2) as pool:
        values = list(pool.map(lambda _: reserve(database, "SAME"), range(2)))
    assert values[0]["revision_sequence"] == values[1]["revision_sequence"]


def test_success_keeps_original_bytes_and_cannot_be_overwritten(database):
    first = reserve(database, "A")
    body = {
        "revision_sequence": first["revision_sequence"],
        "input_identity": first["input_identity"],
        "kind": "SYNTHETIC_ROLLBACK_ONLY",
        "generated_at": datetime.now(ZONE).isoformat(),
    }
    result = {"payload_json": canonical(body), "content_hash": digest(body)}
    revisions.complete(first["revision_sequence"], result)
    assert reserve(database, "A")["result"] == result
    with database.connect() as connection, pytest.raises(Exception, match="immutable"):
        connection.execute(text("UPDATE direction_1d_input_revision SET result='{}'"))


def test_cutoff_is_checked_by_database_before_reserving(database):
    with database.begin() as connection, pytest.raises(ValueError, match="MISSED_DEADLINE"):
        revisions.select_revision(
            connection,
            {"fund_code": "123456", "target_nav_date": "2026-09-30", "protocol": "TEST_ONLY"},
            datetime.now(ZONE) - timedelta(seconds=1),
        )


def test_job_request_id_is_stable_and_target_specific(monkeypatch):
    monkeypatch.setattr(jobs.repo, "clock", lambda: datetime(2026, 9, 29, 14, 0, tzinfo=ZONE))
    monkeypatch.setattr(jobs, "submit", lambda key, kind, payload: (key, payload))
    request = uuid4()
    one = jobs.submit_forecast("002112", date(2026, 9, 29), request)
    assert one == jobs.submit_forecast("002112", date(2026, 9, 29), request)
    assert one[1]["revisions"] is True


def test_completion_after_cutoff_cannot_be_published(database):
    # 只在独立测试 schema 放置过期的待完成记录，模拟运算跨越截止；不修改系统时钟。
    with database.begin() as connection:
        sequence = connection.execute(text("""
            INSERT INTO direction_1d_input_revision(
                fund_code,target_nav_date,protocol,input_identity,identity_payload,deadline_at)
            VALUES('123456','2026-09-30','TEST_ONLY',repeat('a',64),'{}',clock_timestamp()-interval '1 second')
            RETURNING revision_sequence
        """)).scalar_one()
    with pytest.raises(ValueError, match="MISSED_DEADLINE"):
        revisions.complete(sequence, {"kind": "SYNTHETIC_ROLLBACK_ONLY"})
    with database.connect() as connection:
        assert connection.execute(text("SELECT result FROM direction_1d_input_revision")).scalar_one() is None
