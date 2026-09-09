"""真实PG源写入触发留档；随机自有schema和人工数值，不调用外部数据源。"""

import os
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from decimal import Decimal
from threading import Barrier
from uuid import uuid4

import pytest
from alembic.migration import MigrationContext
from alembic.operations import Operations
from app.models.fund import FundDividend, NavDaily
from app.repositories.cash_source_observation import expired_observation_ids, purge_expired_observations
from app.repositories.fund_sync import NavDailyUpsert, upsert_nav_daily_batch
from app.services import cash_source_observation as service
from sqlalchemy import event, text
from sqlalchemy.exc import DBAPIError, IntegrityError
from sqlalchemy.orm import Session
from tests.test_cash_reinvestment_postgres import database as database
from tests.test_cash_source_observation import PAYLOAD, URL
from tests.test_cash_source_observation_schema import migration_module
from tests.test_historical_nav_http import HEADERS
from tests.test_historical_nav_http import client as client

pytestmark = pytest.mark.skipif(os.getenv("RUN_NAV_STORAGE_PG_TESTS") != "1", reason="显式启用隔离PG")


def migrate(conn, action="upgrade"):
    module = migration_module()
    module.op = Operations(MigrationContext.configure(conn))
    getattr(module, action)()


@pytest.fixture
def observation_db(database, monkeypatch):
    engine, _ = database
    source_id = uuid4()
    with engine.begin() as conn:
        conn.exec_driver_sql(
            "ALTER TABLE fund_share_class ADD COLUMN fund_type text DEFAULT 'STOCK', "
            "ADD COLUMN status text DEFAULT 'ACTIVE', ADD COLUMN source_code text DEFAULT 'TUSHARE_PRO_FUND'"
        )
        conn.exec_driver_sql(
            "CREATE TABLE source_registry(source_id uuid PRIMARY KEY, "
            "source_code text, retention_days integer, enabled boolean)"
        )
        conn.execute(text("INSERT INTO source_registry VALUES (:id,'TUSHARE_PRO_FUND',365,true)"), {"id": source_id})
        NavDaily.__table__.create(conn)
        FundDividend.__table__.create(conn)
        conn.exec_driver_sql(
            "CREATE TABLE nav_daily_2024 PARTITION OF nav_daily FOR VALUES FROM ('2024-01-01') TO ('2025-01-01')"
        )
        conn.exec_driver_sql("CREATE TABLE nav_daily_default PARTITION OF nav_daily DEFAULT")
        # 已有旧值不回填：安装触发器之前就存在，后续修改才留下实际观察记录。
        conn.execute(
            text(
                "INSERT INTO nav_daily(fund_code,nav_date,source_id,unit_nav,ann_date,content_hash) "
                "VALUES ('006730','2024-01-02',:id,1,'2024-01-03',:hash)"
            ),
            {"id": source_id, "hash": "a" * 64},
        )
        migrate(conn)
    monkeypatch.setattr(service, "get_nav_preview_engine", lambda: engine)
    return engine, source_id


def entries(engine):
    with engine.connect() as conn:
        return (
            conn.execute(
                text(
                    "SELECT observation_id, kind, operation, before_payload, after_payload, observed_at, expires_at "
                    "FROM cash_source_observation ORDER BY observation_id"
                )
            )
            .mappings()
            .all()
        )


def write_nav(engine, source_id, *, value="2", day=date(2024, 1, 2), digest="b"):
    record = NavDailyUpsert(
        fund_code="006730",
        nav_date=day,
        unit_nav=Decimal(value),
        accumulated_nav=None,
        content_hash=digest * 64,
        ann_date=day.replace(day=3),
    )
    with Session(engine) as session, session.begin():
        return upsert_nav_daily_batch(session, source_id=source_id, records=(record,))


def test_real_upsert_preserves_old_and_new_values_and_reversions(observation_db):
    engine, source = observation_db
    assert entries(engine) == []
    assert write_nav(engine, source).updated_count == 1
    assert write_nav(engine, source).skipped_count == 1
    # 只有非现金相关字段的source hash变更，不凭空增加现金源值版本。
    assert write_nav(engine, source, digest="c").updated_count == 1
    assert len(entries(engine)) == 1
    write_nav(engine, source, value="1", digest="d")
    rows = entries(engine)
    assert [r["operation"] for r in rows] == ["UPDATE", "UPDATE"]
    assert [(r["before_payload"]["unit_nav"], r["after_payload"]["unit_nav"]) for r in rows] == [
        ("1.00000000", "2.00000000"),
        ("2.00000000", "1.00000000"),
    ]
    assert all(r["observed_at"].year >= 2026 and (r["expires_at"] - r["observed_at"]).days == 365 for r in rows)
    assert set(rows[0]["after_payload"]) == {"fund_code", "source_id", "nav_date", "ann_date", "unit_nav"}


def test_new_partitions_and_direct_child_writes_are_captured(observation_db):
    engine, source = observation_db
    with engine.begin() as conn:
        conn.exec_driver_sql(
            "CREATE TABLE nav_daily_2026 PARTITION OF nav_daily FOR VALUES FROM ('2026-01-01') TO ('2027-01-01')"
        )
        conn.execute(
            text(
                "INSERT INTO nav_daily_2026(fund_code,nav_date,source_id,unit_nav,ann_date,content_hash) "
                "VALUES ('006730','2026-09-08',:id,3,'2026-09-09',:hash)"
            ),
            {"id": source, "hash": "e" * 64},
        )
        conn.execute(text("UPDATE nav_daily_2024 SET unit_nav=4 WHERE fund_code='006730' AND nav_date='2024-01-02'"))
    assert [r["operation"] for r in entries(engine)] == ["INSERT", "UPDATE"]


def test_dividend_revision_and_deletion_preserve_before_values(observation_db):
    engine, source = observation_db
    with engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO fund_dividend(fund_code,source_id,source_event_key,ann_date,ex_date,"
                "cash_dividend,base_unit,content_hash) "
                "VALUES ('006730',:id,'event-one','2024-01-02','2024-01-03',0.1,1,:hash)"
            ),
            {"id": source, "hash": "a" * 64},
        )
        conn.exec_driver_sql(
            "UPDATE fund_dividend SET process_status='implemented',cash_dividend=0.2 WHERE source_event_key='event-one'"
        )
        conn.exec_driver_sql("DELETE FROM fund_dividend WHERE source_event_key='event-one'")
    rows = entries(engine)
    assert [r["operation"] for r in rows] == ["INSERT", "UPDATE", "DELETE"]
    assert rows[-1]["after_payload"] is None and rows[-1]["before_payload"]["cash_dividend"] == "0.20000000"
    assert all(r["kind"] == "DIVIDEND" for r in rows)


def test_source_and_observation_rollback_together(observation_db):
    engine, source = observation_db
    with pytest.raises(ValueError):
        with engine.begin() as conn:
            conn.exec_driver_sql("UPDATE nav_daily SET unit_nav=7 WHERE fund_code='006730'")
            assert conn.execute(text("SELECT count(*) FROM cash_source_observation")).scalar_one() == 1
            raise ValueError("synthetic source transaction failure")
    assert entries(engine) == []
    with engine.connect() as conn:
        assert conn.execute(text("SELECT unit_nav FROM nav_daily WHERE fund_code='006730'")).scalar_one() == 1


@pytest.mark.parametrize(
    "statement",
    [
        "UPDATE cash_source_observation SET observed_at=observed_at",
        "DELETE FROM cash_source_observation",
        "TRUNCATE cash_source_observation",
    ],
)
def test_active_observations_cannot_be_rewritten_or_removed(observation_db, statement):
    engine, source = observation_db
    write_nav(engine, source)
    with pytest.raises(DBAPIError, match="cannot be rewritten"):
        with engine.begin() as conn:
            conn.exec_driver_sql(statement)
    assert len(entries(engine)) == 1


def test_http_audit_reads_only_metadata_and_does_not_backdate(observation_db, client):
    engine, source = observation_db
    write_nav(engine, source)
    statements = []

    def capture(conn, cursor, statement, parameters, context, executemany):
        statements.append(statement)

    event.listen(engine, "before_cursor_execute", capture)
    try:
        result = client.post(URL, json=PAYLOAD, headers=HEADERS)
    finally:
        event.remove(engine, "before_cursor_execute", capture)
    assert result.status_code == 200 and result.headers["cache-control"] == "no-store"
    body = result.json()
    assert body["status"] == "ONLY_AFTER_CUTOFF" and body["active_record_count"] == 1
    assert body["recorded_before_cutoff_count"] == 0 and not body["training_eligible"]
    assert not body["historical_first_versions_verified"] and not body["source_payload_read"]
    assert len(statements) == 4 and "REPEATABLE READ, READ ONLY" in statements[0]
    assert all(s.lstrip().upper().startswith(("SET", "SELECT")) for s in statements)
    assert all(
        not any(k in s for k in ("before_payload", "after_payload", "unit_nav", "nav_daily")) for s in statements
    )


def test_non_pilot_source_changes_and_zero_retention_do_not_expand_archive(observation_db):
    engine, source = observation_db
    with engine.begin() as conn:
        conn.exec_driver_sql("INSERT INTO fund_share_class(fund_code) VALUES ('000001')")
        conn.execute(
            text(
                "INSERT INTO nav_daily(fund_code,nav_date,source_id,unit_nav,content_hash) "
                "VALUES ('000001','2024-01-02',:id,1,:hash)"
            ),
            {"id": source, "hash": "e" * 64},
        )
        conn.exec_driver_sql("UPDATE source_registry SET retention_days=0")
    write_nav(engine, source)
    assert entries(engine) == []


def test_empty_downgrade_preserves_source_values_and_nonempty_refuses(observation_db):
    engine, source = observation_db
    with engine.begin() as conn:
        migrate(conn, "downgrade")
        assert conn.execute(text("SELECT count(*) FROM nav_daily")).scalar_one() == 1
        migrate(conn)
    write_nav(engine, source)
    with pytest.raises(DBAPIError, match="not empty"):
        with engine.begin() as conn:
            migrate(conn, "downgrade")
    assert len(entries(engine)) == 1


def test_database_json_payload_constraint_rejects_both_null(observation_db):
    engine, source = observation_db
    with pytest.raises(IntegrityError):
        with engine.begin() as conn:
            conn.execute(
                text(
                    "INSERT INTO cash_source_observation(source_id,fund_code,kind,record_key,operation,"
                    "observed_at,expires_at,content_hash) "
                    "VALUES (:id,'006730','NAV','2024-01-02','UPDATE',now(),now()+interval '1 day',:hash)"
                ),
                {"id": source, "hash": "a" * 64},
            )


def test_concurrent_source_updates_keep_actual_predecessor_chain(observation_db):
    engine, source = observation_db
    barrier = Barrier(2)

    def update(value):
        with engine.begin() as conn:
            barrier.wait(timeout=5)
            conn.execute(
                text("UPDATE nav_daily SET unit_nav=:value WHERE fund_code='006730' AND nav_date='2024-01-02'"),
                {"value": value},
            )

    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(update, (2, 3)))
    rows = entries(engine)
    assert len(rows) == 2
    assert rows[0]["before_payload"]["unit_nav"] == "1.00000000"
    assert rows[1]["before_payload"]["unit_nav"] == rows[0]["after_payload"]["unit_nav"]
    assert {r["after_payload"]["unit_nav"] for r in rows} == {"2.00000000", "3.00000000"}


def test_expired_records_are_not_evidence_and_purge_is_bounded(observation_db, client):
    engine, source = observation_db
    write_nav(engine, source)
    # 测试自己插入过去的人工日志；真实触发器没有回填时间参数。
    with engine.begin() as conn:
        for _ in range(3):
            conn.exec_driver_sql(
                "INSERT INTO cash_source_observation(source_id,fund_code,kind,record_key,event_date,operation,"
                "observed_at,expires_at,before_payload,after_payload,content_hash) "
                "SELECT source_id,fund_code,kind,record_key,event_date,operation,"
                "'2023-01-01'::timestamptz,'2023-01-02'::timestamptz,before_payload,after_payload,content_hash "
                "FROM cash_source_observation ORDER BY observation_id LIMIT 1"
            )
    result = client.post(URL, json=PAYLOAD, headers=HEADERS).json()
    assert result["expired_record_count"] == 3 and result["active_record_count"] == 1
    assert result["recorded_before_cutoff_count"] == 0 and result["status"] == "ONLY_AFTER_CUTOFF"
    with Session(engine) as session, session.begin():
        assert len(expired_observation_ids(session, limit=2)) == 2
        assert purge_expired_observations(session, limit=2) == 2
    assert len(entries(engine)) == 2
    with Session(engine) as session, session.begin():
        assert purge_expired_observations(session, limit=1000) == 1
    assert len(entries(engine)) == 1


def test_source_identity_reassignment_is_rejected(observation_db):
    engine, source = observation_db
    with pytest.raises(DBAPIError, match="identity cannot be reassigned"):
        with engine.begin() as conn:
            conn.exec_driver_sql("UPDATE nav_daily SET nav_date='2024-01-04' WHERE fund_code='006730'")
    assert entries(engine) == []
