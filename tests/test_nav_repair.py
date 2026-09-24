"""净值补拉隔离库验收；外部来源用替身，不调用Tushare、不触碰业务schema。"""

import importlib.util
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from uuid import uuid4

import pytest
from alembic.migration import MigrationContext
from alembic.operations import Operations
from app.db.session import get_engine
from app.integrations.tushare import TushareFundNav, TushareIntegrationError
from app.repositories import nav_repair as repository
from app.repositories.fund_sync import MarketSyncTarget, WriteStats
from app.services import tushare_fund_sync as sync
from app.services.nav_repair import contiguous_ranges, expected_dates
from sqlalchemy import create_engine, text


def test_gaps_include_interior_but_not_weekend_or_unclosed_day():
    sessions = tuple(map(date.fromisoformat, ["2026-09-18", "2026-09-21", "2026-09-22", "2026-09-23", "2026-09-24"]))
    existing = {sessions[0], sessions[1], sessions[3]}
    wanted = expected_dates(sessions, existing, sessions[-1], now=datetime.fromisoformat("2026-09-24T14:59:59+08:00"))
    missing = tuple(d for d in wanted if d not in existing)
    assert missing == (date(2026, 9, 22),)
    assert contiguous_ranges(missing, sessions) == ((missing[0], missing[0]),)


@pytest.fixture
def isolated(monkeypatch):
    original = get_engine()
    if original.url.host not in {"localhost", "127.0.0.1", "::1"}:
        pytest.skip("仅在本机隔离schema验证")
    schema = "nav_repair_test_" + uuid4().hex
    with original.begin() as c:
        c.execute(text(f"CREATE SCHEMA {schema}"))
    engine = create_engine(original.url, connect_args={"options": f"-c search_path={schema}"}, hide_parameters=True)
    source, run = uuid4(), uuid4()
    try:
        with engine.begin() as c:
            c.execute(
                text("""
              CREATE TABLE source_registry(source_id uuid PRIMARY KEY);
              CREATE TABLE source_sync_run(sync_run_id uuid PRIMARY KEY);
              CREATE TABLE fund_share_class(fund_code varchar(6) PRIMARY KEY,fund_name text,fund_type text);
              CREATE TABLE fund_profile(fund_code text,source_id uuid,benchmark text,source_fund_type text,
                                        invest_type text,found_date date);
              CREATE TABLE nav_daily(fund_code text,source_id uuid,nav_date date,unit_nav numeric,
                                     PRIMARY KEY(fund_code,source_id,nav_date));
            """)
            )
            c.execute(text("INSERT INTO source_registry VALUES(:id)"), {"id": source})
            c.execute(text("INSERT INTO source_sync_run VALUES(:id)"), {"id": run})
        path = Path(__file__).parents[1] / "alembic/versions/20260924_29_nav_sync_state.py"
        spec = importlib.util.spec_from_file_location("nav_migration", path)
        migration = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(migration)
        with engine.begin() as c:
            monkeypatch.setattr(migration, "op", Operations(MigrationContext.configure(c)))
            migration.upgrade()
            for code in ("000001", "000002", "000003"):
                c.execute(text("INSERT INTO fund_share_class VALUES(:code,'境内测试基金','STOCK')"), {"code": code})
                for day in (date(2026, 9, 21), date(2026, 9, 23)):
                    c.execute(
                        text("INSERT INTO nav_daily VALUES(:code,:s,:day,1)"), {"code": code, "s": source, "day": day}
                    )
        yield engine, source, run
    finally:
        engine.dispose()
        with original.begin() as c:
            c.execute(text(f"DROP SCHEMA {schema} CASCADE"))


def test_one_failure_continues_commits_and_restart_retries_only_due(isolated, monkeypatch):
    engine, source, run = isolated
    calls, completed = [], []
    fail = True

    class Provider:
        def list_nav_history(self, code, *, start_date, end_date):
            calls.append((code, start_date, end_date))
            if code == "000001.OF" and fail:
                raise TushareIntegrationError("fund_nav", "synthetic timeout", retryable=True)
            if code == "000003.OF":
                return ()
            return (TushareFundNav(code, None, start_date, Decimal("1.01"), None),)

    def service():
        value = object.__new__(sync.TushareFundSyncService)
        value._engine, value._provider, value._batch_size = engine, Provider(), 100
        value._start_run = lambda **_: (source, run)
        value._complete_run = lambda _s, result, _stats: completed.append(result)
        return value

    monkeypatch.setattr(
        sync,
        "list_active_market_sync_targets",
        lambda _: tuple(MarketSyncTarget(code, code + ".OF") for code in ("000001", "000002", "000003")),
    )
    monkeypatch.setattr(
        sync, "get_latest_nav_dates", lambda *_, **__: dict.fromkeys(("000001", "000002", "000003"), date(2026, 9, 23))
    )

    def write(session, *, source_id, records):
        for r in records:
            session.execute(
                text("INSERT INTO nav_daily VALUES(:code,:s,:day,:nav) ON CONFLICT DO NOTHING"),
                {"code": r.fund_code, "s": source_id, "day": r.nav_date, "nav": r.unit_nav},
            )
        return WriteStats(created_count=len(records))

    monkeypatch.setattr(sync, "upsert_nav_daily_batch", write)
    result = service()._run_market_nav_incremental(target_date=date(2026, 9, 23), progress_reporter=None)
    assert result.status == "PARTIAL_SUCCESS" and result.created_count == 1 and len(result.issues) == 2
    assert [x[0] for x in calls] == ["000001.OF", "000002.OF", "000003.OF"]
    assert all(x[1:] == (date(2026, 9, 22), date(2026, 9, 22)) for x in calls)
    with engine.connect() as c:
        states = dict(c.execute(text("SELECT fund_code,status FROM nav_sync_state")).all())
    assert states == {"000001": "SYNC_FAILED", "000002": "SUCCEEDED", "000003": "STATUS_UNKNOWN"}
    # 模拟重启：新服务读取原持久状态，尚未到期不重新请求；成功基金的中间缺口已经提交。
    calls.clear()
    service()._run_market_nav_incremental(target_date=date(2026, 9, 23), progress_reporter=None, automatic=True)
    assert calls == []
    with engine.begin() as c:
        c.execute(
            text(
                "UPDATE nav_sync_state SET next_retry_at=clock_timestamp()-interval '1 second' WHERE fund_code='000001'"
            )
        )
    fail = False
    service()._run_market_nav_incremental(target_date=date(2026, 9, 23), progress_reporter=None, automatic=True)
    assert [x[0] for x in calls] == ["000001.OF"]
    with engine.connect() as c:
        assert c.execute(text("SELECT status FROM nav_sync_state WHERE fund_code='000001'")).scalar() == "SUCCEEDED"


def test_nonretryable_failure_persists_manual_repair_requirement(isolated):
    engine, source, run = isolated
    from sqlalchemy.orm import Session

    with Session(engine) as session, session.begin():
        repository.save_state(
            session,
            source_id=source,
            code="000001",
            run_id=run,
            status="SYNC_FAILED",
            missing=[date(2026, 9, 22)],
            reason="来源授权待修正",
            retryable=False,
        )
    with engine.connect() as c:
        row = c.execute(
            text("SELECT status,next_retry_at,missing_dates FROM nav_sync_state WHERE fund_code='000001'")
        ).one()
        assert row.status == "SYNC_FAILED" and row.next_retry_at is None and row.missing_dates == ["2026-09-22"]
