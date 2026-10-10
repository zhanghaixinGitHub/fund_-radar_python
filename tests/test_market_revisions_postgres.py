"""可选本机 PostgreSQL 临时表测试；无真实业务表读写，全部事务回滚。"""

import json
import os
from datetime import date
from uuid import uuid4

import pytest
from app.core.config import get_settings
from app.repositories.fund_sync import TUSHARE_SOURCE_CODE, WriteStats
from app.services.market_revisions import REVISION_SQL, revision_of
from app.services.tushare_fund_sync import TushareFundSyncService
from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_LOCAL_REVISION_SQL_TESTS") != "1", reason="需要显式启用本机临时表测试"
)


@pytest.fixture
def connection():
    url = get_settings().ai_database_url
    assert make_url(url).host in {"localhost", "127.0.0.1", "::1"}, "只允许本机数据库"
    engine = create_engine(url, hide_parameters=True, connect_args={"connect_timeout": 5})
    with engine.connect() as c, c.begin() as transaction:
        # search_path 不含 public：表缺失立即报错，绝不回落到真实业务表。
        c.execute(text("SET LOCAL search_path TO pg_temp"))
        c.execute(text("SET LOCAL statement_timeout TO '5s'"))
        tables = [
            "source_registry(source_id uuid,source_code text,enabled boolean,"
            "authorized_api_names jsonb,retention_days int)",
            "fund_share_class(fund_code text PRIMARY KEY,source_code text,fund_name text,source_fund_code text)",
            "simulation_market_refresh(fund_code text PRIMARY KEY,status text,attempted_at timestamptz,"
            "dividends_verified_at timestamptz,message text)",
            "nav_daily(fund_code text,source_id uuid,nav_date date,unit_nav numeric,accumulated_nav numeric,"
            "ann_date date,content_hash text)",
            "fund_dividend(fund_code text,source_id uuid,source_event_key text,record_date date,ex_date date,"
            "nav_ex_date date,pay_date date,cash_dividend numeric,process_status text,content_hash text)",
        ]
        for ddl in tables:
            c.execute(text("CREATE TEMP TABLE " + ddl))
        source = uuid4()
        c.execute(
            text("INSERT INTO source_registry VALUES(:id,:code,true,'[\"fund_nav\"]',365)"),
            {"id": source, "code": TUSHARE_SOURCE_CODE},
        )
        c.execute(
            text("INSERT INTO fund_share_class VALUES('002112',:code,'合成基金','002112.OF')"),
            {"code": TUSHARE_SOURCE_CODE},
        )
        for day in ("2026-09-14", "2026-09-15"):
            c.execute(
                text("INSERT INTO nav_daily VALUES('002112',:source,:day,1.2,1.2,:day,:day)"),
                {"source": source, "day": date.fromisoformat(day)},
            )
        yield c, source
        transaction.rollback()
    engine.dispose()


def snapshot(c, kind="LABEL"):
    return dict(
        c.execute(
            REVISION_SQL,
            {
                "queries": json.dumps(
                    [
                        {
                            "key": "test",
                            "fund_code": "002112",
                            "start_date": "2026-09-14",
                            "end_date": "2026-09-15",
                            "kind": kind,
                        }
                    ]
                ),
                "source": TUSHARE_SOURCE_CODE,
                "today": date(2026, 10, 9),
            },
        )
        .mappings()
        .one()
    )


def test_sql_limits_label_to_relevant_dates_and_detects_old_nav_changes(connection):
    c, source = connection
    first = revision_of(snapshot(c))
    assert snapshot(c)["ready"] is True
    c.execute(
        text("INSERT INTO nav_daily VALUES('002112',:source,'2026-09-16',1.3,1.3,'2026-09-16','new')"),
        {"source": source},
    )
    assert revision_of(snapshot(c)) == first  # 无关日期不会重查已核对的旧预测。
    c.execute(text("UPDATE nav_daily SET unit_nav=1.1 WHERE nav_date='2026-09-14'"))
    assert revision_of(snapshot(c)) != first  # 旧日修订不依赖最大日期或最大更新时间。
    c.execute(text("DELETE FROM nav_daily WHERE nav_date='2026-09-15'"))
    assert snapshot(c)["ready"] is False


def test_sql_future_announcement_and_dividend_revision(connection):
    c, source = connection
    first = revision_of(snapshot(c))
    c.execute(text("UPDATE nav_daily SET ann_date='2099-01-01' WHERE nav_date='2026-09-15'"))
    assert snapshot(c)["ready"] is False
    c.execute(text("UPDATE nav_daily SET ann_date=nav_date WHERE nav_date='2026-09-15'"))
    assert revision_of(snapshot(c)) == first
    c.execute(
        text(
            "INSERT INTO fund_dividend(fund_code,source_id,source_event_key,ex_date,content_hash) "
            "VALUES('002112',:source,'synthetic','2026-09-15','dividend-revision')"
        ),
        {"source": source},
    )
    assert revision_of(snapshot(c)) != first
    c.execute(text("UPDATE source_registry SET enabled=false"))
    assert snapshot(c)["ready"] is False


def test_manual_dividend_verification_advances_only_on_success_in_same_transaction(connection):
    c, source = connection
    worker = object.__new__(TushareFundSyncService)
    worker._engine, worker._batch_size = c, 1
    worker._start_run = lambda **kw: (source, uuid4())
    worker._complete_run = lambda *args: None
    first = revision_of(snapshot(c, "SIMULATION"))

    def run(invalid=0, write=lambda *args, **kw: WriteStats(), records=()):
        return worker._sync_detail_funds(
            ("002112.OF",),
            sync_type="MARKET_DETAIL_DIVIDEND",
            fetch=lambda _: [],
            normalize=lambda *_: (records, invalid),
            write=write,
        )

    assert run(invalid=1).status == "FAILED"
    assert c.execute(text("SELECT count(*) FROM simulation_market_refresh")).scalar() == 0
    assert run().status == "SUCCEEDED"  # 来源明确无分红，也构成已核验的证据。
    verified = c.execute(text("SELECT dividends_verified_at FROM simulation_market_refresh")).scalar()
    assert verified is not None and revision_of(snapshot(c, "SIMULATION")) != first

    def fail_write(session, **kwargs):
        # 在临时表里模拟写到一半后出错，保存点应撤回该次写入。
        session.execute(text("UPDATE simulation_market_refresh SET message='must rollback'"))
        raise ValueError("injected")

    # 让 ORM 子事务使用保存点，失败不得回滚测试夹具创建的临时表。
    from unittest.mock import patch

    from sqlalchemy.orm import Session

    with patch(
        "app.services.tushare_fund_sync.Session", lambda bind: Session(bind, join_transaction_mode="create_savepoint")
    ):
        assert run(write=fail_write, records=(1,)).status == "FAILED"
    row = c.execute(text("SELECT dividends_verified_at,message FROM simulation_market_refresh")).one()
    assert row.dividends_verified_at == verified and row.message is None
