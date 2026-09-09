"""隔离真实SQL验证日期先行、正文保护及分页；不接触public业务样本。"""

import os
from contextlib import contextmanager

import pytest
from app.schemas.cash_reinvestment_research import CashPrepareRequest
from app.services import cash_reinvestment_research as research
from app.services.cash_reinvestment_storage import save_cash_batch
from sqlalchemy import event, text
from tests.test_cash_exam_plan import URL, request_for
from tests.test_cash_reinvestment_postgres import counts
from tests.test_cash_reinvestment_postgres import database as database
from tests.test_cash_reinvestment_storage import request as batch_request
from tests.test_historical_nav_http import HEADERS
from tests.test_historical_nav_http import client as client

pytestmark = pytest.mark.skipif(os.getenv("RUN_NAV_STORAGE_PG_TESTS") != "1", reason="显式启用隔离PG")


@pytest.fixture
def prepared(database):
    engine, _ = database
    stored, _ = save_cash_batch(batch_request())
    data = research.load_cash_dataset(CashPrepareRequest(batchIds=(stored.batch_id,)))
    return engine, request_for(data)


@contextmanager
def captured(engine):
    statements = []

    def capture(conn, cursor, statement, parameters, context, executemany):
        statements.append(statement)

    event.listen(engine, "before_cursor_execute", capture)
    try:
        yield statements
    finally:
        event.remove(engine, "before_cursor_execute", capture)


def full_payload_queries(statements):
    # 日期预检只使用 ->> 取日期字符串；此标记对应随后全量读取X/y正文的SELECT。
    return [sql for sql in statements if "cash_sample.feature_payload," in sql]


def test_http_readonly_deterministic_and_dates_precede_payload(prepared, client):
    engine, request = prepared
    before = counts(engine)
    with captured(engine) as statements:
        first = client.post(URL, headers=HEADERS, json=request.model_dump(mode="json", by_alias=True))
        second = client.post(URL, headers=HEADERS, json=request.model_dump(mode="json", by_alias=True))
    assert first.status_code == second.status_code == 200
    assert first.json() == second.json() and first.headers["cache-control"] == "no-store"
    assert counts(engine) == before
    assert len(statements) == 8 and len(full_payload_queries(statements)) == 2
    assert "REPEATABLE READ, READ ONLY" in statements[0]
    assert "CAST(cash_sample_label.label_payload ->>" in statements[2]
    assert statements[3] == full_payload_queries(statements)[0]
    assert all(sql.lstrip().upper().startswith(("SET", "SELECT")) for sql in statements)
    assert all("nav_daily" not in sql and "cash_research_run" not in sql for sql in statements)


@pytest.mark.parametrize("target", ["batch", "sample", "label"])
def test_protected_dates_are_rejected_before_loading_values(prepared, client, target):
    engine, request = prepared
    with engine.begin() as conn:
        if target == "batch":
            conn.execute(text("UPDATE cash_sample_batch SET start_date='2025-01-02', end_date='2025-01-02'"))
        elif target == "sample":
            conn.execute(text("UPDATE cash_sample SET cutoff_date='2025-01-02'"))
        else:
            conn.execute(
                text(
                    "UPDATE cash_sample_label SET label_payload="
                    "jsonb_set(label_payload, '{label_available_at}', CAST(:v AS jsonb))"
                ),
                {"v": '"2025-01-02"'},
            )
    with captured(engine) as statements:
        response = client.post(URL, headers=HEADERS, json=request.model_dump(mode="json", by_alias=True))
    assert response.status_code == 409 and response.json()["detail"]["code"] == "TEST_PERIOD_PROTECTED"
    assert not full_payload_queries(statements)
    assert all("future_return_20d" not in sql and "label_up_20d" not in sql for sql in statements)


@pytest.mark.parametrize("bad_date", [None, "wrong-date", "2023-01-01"])
def test_invalid_label_date_stops_before_payload_and_is_redacted(prepared, client, bad_date):
    import json

    engine, request = prepared
    with engine.begin() as conn:
        conn.execute(
            text(
                "UPDATE cash_sample_label SET label_payload="
                "jsonb_set(label_payload, '{label_available_at}', CAST(:v AS jsonb))"
            ),
            {"v": json.dumps(bad_date)},
        )
    with captured(engine) as statements:
        response = client.post(URL, headers=HEADERS, json=request.model_dump(mode="json", by_alias=True))
    assert response.status_code == 503 and response.json()["detail"]["code"] == "CASH_BATCH_CORRUPTED"
    assert not full_payload_queries(statements) and "SELECT" not in response.text


def test_nine_batches_use_two_bounded_pages_and_do_not_inflate_coverage(database, client):
    engine, _ = database
    batches = tuple(save_cash_batch(batch_request())[0].batch_id for _ in range(9))
    data = research.load_cash_dataset(CashPrepareRequest(batchIds=batches))
    request = request_for(data)
    with captured(engine) as statements:
        response = client.post(URL, headers=HEADERS, json=request.model_dump(mode="json", by_alias=True))
    assert response.status_code == 200
    assert response.json()["preparation"]["duplicate_count"] == 8
    assert response.json()["preparation"]["usable_count"] == 1
    assert len(statements) == 7 and len(full_payload_queries(statements)) == 2
    assert sum("CAST(cash_sample_label.label_payload ->>" in sql for sql in statements) == 2
    assert [c["planned_count"] for c in response.json()["coverage"]] == [44] * 3 + [40] * 3 + [222] * 3 + [223] * 3


def test_database_rejects_accidental_write_in_preparation(prepared, client, monkeypatch):
    from app.repositories import cash_reinvestment_research as repository

    engine, request = prepared
    before = counts(engine)

    def mistaken_write(session, batches):
        assert session.execute(text("SHOW transaction_read_only")).scalar_one() == "on"
        assert session.execute(text("SHOW transaction_isolation")).scalar_one() == "repeatable read"
        session.execute(text("UPDATE cash_sample SET cutoff_date=cutoff_date"))

    monkeypatch.setattr(repository, "_check_page_dates", mistaken_write)
    response = client.post(URL, headers=HEADERS, json=request.model_dump(mode="json", by_alias=True))
    assert response.status_code == 503
    assert response.json()["detail"]["code"] == "CASH_RESEARCH_UNAVAILABLE"
    assert "UPDATE" not in response.text and counts(engine) == before
