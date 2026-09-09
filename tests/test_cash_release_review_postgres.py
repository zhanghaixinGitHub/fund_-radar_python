"""真实SQL/只读事务，人工报告；仅使用并清理测试拥有的随机schema。"""

import copy
import json
import os
from uuid import uuid4

import pytest
from app.models.cash_reinvestment import CashResearchRun
from app.services import cash_release_review as review
from app.services.cash_reinvestment_storage import cash_hash
from sqlalchemy import event, text
from sqlalchemy.orm import Session
from tests.test_cash_prediction_check import evaluated as evaluated
from tests.test_cash_prediction_check import request_for
from tests.test_cash_reinvestment_postgres import database as database
from tests.test_cash_reinvestment_research import data as data
from tests.test_historical_nav_http import HEADERS
from tests.test_historical_nav_http import client as client

pytestmark = pytest.mark.skipif(os.getenv("RUN_NAV_STORAGE_PG_TESTS") != "1", reason="显式启用隔离PG")
URL = "/internal/v1/predictions/release-review"


@pytest.fixture
def review_db(database, evaluated, monkeypatch):
    engine, _ = database
    source_id = uuid4()
    with engine.begin() as conn:
        conn.exec_driver_sql(
            "ALTER TABLE fund_share_class ADD COLUMN fund_type text DEFAULT 'STOCK', "
            "ADD COLUMN status text DEFAULT 'ACTIVE', ADD COLUMN source_code text DEFAULT 'TUSHARE_PRO_FUND'"
        )
        conn.exec_driver_sql("CREATE TABLE source_registry (source_id uuid, source_code text, enabled boolean)")
        conn.exec_driver_sql(
            "CREATE TABLE nav_daily (fund_code text, source_id uuid, nav_date date, ann_date date, unit_nav numeric)"
        )
        conn.execute(text("INSERT INTO source_registry VALUES (:id, 'TUSHARE_PRO_FUND', true)"), {"id": source_id})
        conn.execute(
            text("INSERT INTO nav_daily VALUES ('006730', :id, '2025-08-07', '2025-08-08', 999999)"), {"id": source_id}
        )
    with Session(engine) as session, session.begin():
        session.add(
            CashResearchRun(
                run_id=evaluated.run_id,
                request_key=evaluated.request_key,
                dataset_hash=evaluated.dataset_hash,
                report=evaluated.report,
                publication_status="MODEL_NOT_RELEASED",
            )
        )
    monkeypatch.setattr(review, "get_nav_preview_engine", lambda: engine)
    return engine, request_for(evaluated)


def snapshot(engine):
    with engine.connect() as conn:
        return conn.execute(text("SELECT run_id, md5(report::text) FROM cash_research_run ORDER BY run_id")).all()


def test_real_sql_preserves_report_and_never_selects_price_or_answers(review_db, monkeypatch, client):
    engine, req = review_db
    before = snapshot(engine)
    statements = []
    original = review.load_cash_prediction_research

    def checked_read(session, request, *, now):
        assert session.execute(text("SHOW transaction_read_only")).scalar_one() == "on"
        assert session.execute(text("SHOW transaction_isolation")).scalar_one() == "repeatable read"
        return original(session, request, now=now)

    def capture(conn, cursor, statement, parameters, context, executemany):
        statements.append(statement)

    monkeypatch.setattr(review, "load_cash_prediction_research", checked_read)
    event.listen(engine, "before_cursor_execute", capture)
    try:
        first = client.post(URL, json=req.model_dump(mode="json", by_alias=True), headers=HEADERS)
        second = client.post(URL, json=req.model_dump(mode="json", by_alias=True), headers=HEADERS)
    finally:
        event.remove(engine, "before_cursor_execute", capture)
    assert first.status_code == second.status_code == 200
    a, b = first.json(), second.json()
    a.pop("checked_at")
    b.pop("checked_at")
    assert a == b and a["status"] == "BLOCKED" and not a["database_written"]
    assert snapshot(engine) == before
    assert sum("WHERE cash_research_run.run_id =" in sql for sql in statements) == 2
    assert all(
        not any(t in sql for t in ("unit_nav", "cash_sample", "forecast_result", "INSERT", "UPDATE"))
        for sql in statements
    )


def test_database_itself_rejects_accidental_write_and_http_hides_error(review_db, monkeypatch, client):
    engine, req = review_db
    before = snapshot(engine)

    def accidental_write(session, request, *, now):
        # 故意在本测试自己的只读事务中尝试原值UPDATE，须被PostgreSQL阻止，而非仅靠布尔字段保证只读。
        session.execute(
            text("UPDATE cash_research_run SET report=report WHERE run_id=:id"), {"id": req.research_run_id}
        )
        pytest.fail("PostgreSQL must reject a write in the review transaction")

    monkeypatch.setattr(review, "load_cash_prediction_research", accidental_write)
    result = client.post(URL, json=req.model_dump(mode="json", by_alias=True), headers=HEADERS)
    assert result.status_code == 503 and "UPDATE" not in result.text
    assert snapshot(engine) == before


def test_sql_stored_corrupt_bins_are_not_published_or_repaired(review_db, client, evaluated):
    engine, req = review_db
    report = copy.deepcopy(evaluated.report)
    report["windows"][-1]["reliability_after"]["ece"] = "0"
    report["report_hash"] = cash_hash({k: v for k, v in report.items() if k != "report_hash"})
    with engine.begin() as conn:
        conn.execute(
            text("UPDATE cash_research_run SET report=CAST(:report AS jsonb) WHERE run_id=:id"),
            {"report": json.dumps(report), "id": req.research_run_id},
        )
    before = snapshot(engine)
    payload = {**req.model_dump(mode="json", by_alias=True), "expectedReportHash": report["report_hash"]}
    result = client.post(URL, json=payload, headers=HEADERS)
    assert result.status_code == 503 and result.json()["detail"]["code"] == "CASH_RESEARCH_CORRUPTED"
    assert snapshot(engine) == before
