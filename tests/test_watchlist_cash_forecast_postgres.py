"""产品GET贯通隔离PG结果；仅人工授权，真实SQL/计算/存储/失效，不冒称正式发布通过。"""

import os
from datetime import timedelta
from uuid import uuid4

import pytest
from app.models.cash_forecast import CashForecastRecord
from app.services import cash_forecast as generator
from app.services import watchlist_prediction as product
from app.services.cash_publication import CashPublicationUnavailable
from sqlalchemy import event, text
from sqlalchemy.orm import Session
from tests.test_cash_forecast import NOW
from tests.test_cash_forecast import bundle as bundle
from tests.test_cash_forecast import report as report
from tests.test_cash_forecast_postgres import forecasts_db as forecasts_db
from tests.test_cash_reinvestment_postgres import database as database
from tests.test_cash_reinvestment_research import data as data
from tests.test_historical_nav_http import HEADERS
from tests.test_historical_nav_http import client as client

pytestmark = pytest.mark.skipif(os.getenv("RUN_NAV_STORAGE_PG_TESTS") != "1", reason="显式启用隔离PG")


@pytest.fixture
def product_db(forecasts_db, monkeypatch):
    engine, req, grant = forecasts_db
    monkeypatch.setattr(product, "get_nav_preview_engine", lambda: engine)
    monkeypatch.setattr(product, "utc_now", lambda: NOW)
    return engine, req, grant


def test_saved_result_reaches_product_get_without_recalculation(product_db, client, monkeypatch):
    engine, req, _ = product_db
    first = generator.generate_cash_forecast(req)
    monkeypatch.setattr(generator, "calculate_cash_inference", lambda *a, **k: pytest.fail("GET cannot infer"))
    statements = []

    def capture(conn, cursor, statement, parameters, context, executemany):
        statements.append(statement)

    event.listen(engine, "before_cursor_execute", capture)
    try:
        response = client.get(f"/internal/v1/predictions/{req.fund_code}", headers=HEADERS)
    finally:
        event.remove(engine, "before_cursor_execute", capture)
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "AVAILABLE" and payload["forecast_id"] == str(first.forecast_id)
    assert payload["up_probability"] == str(first.up_probability)
    assert payload["cutoff_date"] == str(req.cutoff_date)
    assert payload["target_end_date"] == str(first.target_end_date)
    assert payload["reason_codes"] == [] and payload["reasons"] == []
    assert response.headers["cache-control"] == "no-store"
    assert not {"model", "artifact", "feature", "value", "payload", "authorization_id"} & payload.keys()
    assert "READ ONLY" in statements[0]
    assert all("unit_nav" not in sql and "INSERT" not in sql and "UPDATE" not in sql for sql in statements)


@pytest.mark.parametrize("change", ["source", "revoked", "expired"])
def test_product_hides_numbers_after_invalidation(product_db, client, monkeypatch, change):
    engine, req, _ = product_db
    first = generator.generate_cash_forecast(req)
    if change == "source":
        with engine.begin() as conn:
            conn.execute(text("UPDATE source_registry SET enabled=false"))
    elif change == "revoked":

        def denied(*a, **k):
            raise CashPublicationUnavailable(("PUBLICATION_AUTHORIZATION_CHANGED",))

        monkeypatch.setattr(generator, "resolve_cash_authorization", denied)
    else:
        monkeypatch.setattr(product, "utc_now", lambda: NOW + timedelta(days=50))
    result = client.get(f"/internal/v1/predictions/{req.fund_code}", headers=HEADERS).json()
    assert result["status"] != "AVAILABLE"
    assert result["up_probability"] is result["direction"] is None
    assert result["reason_codes"] and result["reasons"]
    if change != "source":
        assert result["forecast_id"] == str(first.forecast_id)
        assert result["target_end_date"] == str(first.target_end_date)


def test_corrupt_latest_result_never_falls_back_to_older_probability(product_db, client):
    engine, req, _ = product_db
    first = generator.generate_cash_forecast(req)
    with Session(engine) as session, session.begin():
        original = session.get(CashForecastRecord, first.forecast_id)
        values = {col.name: getattr(original, col.name) for col in CashForecastRecord.__table__.columns}
        values.update(
            forecast_id=uuid4(),
            request_key=uuid4(),
            source_revision_id=uuid4(),
            created_at=original.created_at + timedelta(seconds=1),
            content_hash="f" * 64,
        )
        session.add(CashForecastRecord(**values))
    response = client.get(f"/internal/v1/predictions/{req.fund_code}", headers=HEADERS)
    assert response.status_code == 503
    assert "up_probability" not in response.json()


def test_no_result_retains_real_research_refusal_branch(product_db, client):
    _, req, _ = product_db
    result = client.get(f"/internal/v1/predictions/{req.fund_code}", headers=HEADERS)
    assert result.status_code == 200
    payload = result.json()
    assert payload["status"] == "MODEL_NOT_RELEASED" and payload["forecast_id"] is None
    assert payload["up_probability"] is None and "INDEPENDENT_TEST_NOT_EVALUATED" in payload["reason_codes"]
