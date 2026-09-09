"""页面预测状态的失败关闭、内部鉴权和只读查询契约。"""

from datetime import UTC, date, datetime
from types import SimpleNamespace
from uuid import uuid4

import pytest
from app.api.routes import watchlist_prediction as api
from app.repositories.watchlist_prediction import read_prediction_inputs
from app.schemas.watchlist_prediction import WatchlistPrediction
from app.services.cash_reinvestment_research import evaluate_cash_dataset
from app.services.watchlist_prediction import build_prediction_status
from pydantic import ValidationError
from sqlalchemy.dialects import postgresql
from sqlalchemy.exc import SQLAlchemyError
from tests.test_cash_reinvestment_research import small_dataset
from tests.test_historical_nav_http import HEADERS
from tests.test_historical_nav_http import client as client


def inputs(**updates):
    fund = SimpleNamespace(fund_code="008888", fund_type="STOCK", status="ACTIVE", source_code="TUSHARE_PRO_FUND")
    for key, value in updates.items():
        setattr(fund, key, value)
    return fund, SimpleNamespace(enabled=True, source_id=uuid4()), date(2026, 9, 4), None


def run():
    report = evaluate_cash_dataset(small_dataset())
    return SimpleNamespace(
        run_id=uuid4(),
        request_key=uuid4(),
        created_at=datetime.now(UTC),
        report=report.model_dump(mode="json"),
        dataset_hash=report.preparation.dataset_hash,
        publication_status="MODEL_NOT_RELEASED",
    )


@pytest.mark.parametrize("kind", ["BOND", "MIXED", "INDEX", "MONEY", "QDII", "FOF", "OTHER"])
def test_only_stock_applies(kind):
    p = build_prediction_status(*inputs(fund_type=kind))
    assert p.status == "NOT_APPLICABLE" and p.up_probability is None and p.direction is None


def test_stock_outside_pilot_is_not_blindly_predicted():
    assert build_prediction_status(*inputs(fund_code="000001")).status == "DATA_INSUFFICIENT"


@pytest.mark.parametrize("enabled", [False, None])
def test_disabled_or_missing_source(enabled):
    fund, source, nav, _ = inputs()
    source.enabled = enabled
    assert (
        build_prediction_status(fund, source if enabled is not None else None, nav, None).status == "DATA_INSUFFICIENT"
    )


def test_missing_and_research_models_both_withhold_probability():
    fund, source, nav, _ = inputs()
    first = build_prediction_status(fund, source, nav, None)
    second = build_prediction_status(fund, source, nav, run())
    assert first.status == second.status == "MODEL_NOT_RELEASED"
    assert first.up_probability is second.up_probability is None
    assert second.research_run_id is not None
    assert "INDEPENDENT_TEST_NOT_EVALUATED" in second.reason_codes
    assert "model" not in second.model_dump() and "windows" not in second.model_dump()


def test_schema_cannot_emit_research_score_or_direction():
    value = build_prediction_status(*inputs()).model_dump()
    for update in ({"up_probability": 0.8}, {"direction": "UP"}, {"status": "AVAILABLE"}):
        with pytest.raises(ValidationError):
            WatchlistPrediction.model_validate({**value, **update})


def test_query_is_bounded_and_reads_no_nav_values_or_old_model_release():
    fund, source, day, _ = inputs()

    class Session:
        def __init__(self):
            self.results = iter((fund, source, day, None))
            self.sql = []

        def scalar(self, statement):
            self.sql.append(str(statement.compile(dialect=postgresql.dialect())))
            return next(self.results)

    session = Session()
    assert read_prediction_inputs(session, "008888", date(2026, 9, 8)) == (fund, source, day, None)
    nav = next(sql for sql in session.sql if "nav_daily" in sql)
    assert "unit_nav" not in nav and "accumulated_nav" not in nav and "LIMIT" in nav
    assert all("forecast_result" not in sql and "analysis_model_release" not in sql for sql in session.sql)
    assert "LIMIT" in session.sql[-1]


def test_http_status_trace_cache_and_error_redaction(client, monkeypatch):
    monkeypatch.setattr(api, "get_watchlist_prediction", lambda _: build_prediction_status(*inputs()))
    response = client.get("/internal/v1/predictions/008888", headers=HEADERS)
    assert response.status_code == 200 and response.json()["up_probability"] is None
    assert response.headers["cache-control"] == "no-store"

    def fail(_):
        raise SQLAlchemyError("synthetic-private-connection-details")

    monkeypatch.setattr(api, "get_watchlist_prediction", fail)
    response = client.get("/internal/v1/predictions/008888", headers=HEADERS)
    assert response.status_code == 503 and "synthetic-private" not in response.text


@pytest.mark.parametrize("headers", [{}, {"X-Service-Token": "bad"}, {**HEADERS, "Origin": "http://localhost"}])
def test_http_auth_before_read(client, monkeypatch, headers):
    monkeypatch.setattr(api, "get_watchlist_prediction", lambda _: pytest.fail("must not read"))
    assert client.get("/internal/v1/predictions/008888", headers=headers).status_code == 403


@pytest.mark.parametrize("fund", ["123", "abcdef", "1234567"])
def test_http_fund_validation(client, monkeypatch, fund):
    monkeypatch.setattr(api, "get_watchlist_prediction", lambda _: pytest.fail("must not read"))
    assert client.get(f"/internal/v1/predictions/{fund}", headers=HEADERS).status_code == 422
