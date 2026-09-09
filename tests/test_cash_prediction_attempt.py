"""回执完整性与内部HTTP契约；使用人工对象，不连接实际数据库。"""

import copy
from datetime import UTC, date, datetime
from types import SimpleNamespace
from uuid import uuid4

import pytest
from app.api.routes import watchlist_prediction as api
from app.schemas.cash_prediction_attempt import CashPredictionAttemptRequest
from app.schemas.cash_prediction_check import CashPredictionCheck
from app.services import cash_prediction_attempt as service
from app.services.cash_reinvestment_research import BLOCKERS
from app.services.cash_reinvestment_storage import cash_hash
from app.services.historical_nav_storage import HistoricalNavStorageError
from sqlalchemy.exc import SQLAlchemyError
from tests.test_historical_nav_http import HEADERS
from tests.test_historical_nav_http import client as client

PATH = "/internal/v1/predictions/generation-attempts"


def request():
    return CashPredictionAttemptRequest(
        requestKey=uuid4(),
        fundCode="006730",
        cutoffDate="2026-09-04",
        researchRunId=uuid4(),
        expectedReportHash="a" * 64,
    )


def stored(req=None):
    req = req or request()
    check = CashPredictionCheck(
        checked_at=datetime.now(UTC),
        fund_code=req.fund_code,
        research_run_id=req.research_run_id,
        report_hash=req.expected_report_hash,
        blocking_codes=BLOCKERS,
        comparisons=(),
        incomplete_window_ids=(),
    )
    fingerprint = service.attempt_request_hash(req)
    return SimpleNamespace(
        attempt_id=uuid4(),
        request_key=req.request_key,
        request_hash=fingerprint,
        fund_code=req.fund_code,
        cutoff_date=req.cutoff_date,
        research_run_id=req.research_run_id,
        report_hash=req.expected_report_hash,
        check_payload=check.model_dump(mode="json"),
        receipt_hash=cash_hash({"request_hash": fingerprint, "check": check.model_dump(mode="json")}),
        created_at=datetime.now(UTC),
    )


def test_receipt_is_not_a_prediction_and_request_hash_ignores_retry_key():
    req = request()
    row = stored(req)
    value = service.restore_attempt(row)
    assert value.database_written and value.historical_receipt and not value.forecast_created
    assert not value.check.database_written and not value.check.inference_executed
    assert value.check.up_probability is value.check.direction is None
    assert service.attempt_request_hash(req) == service.attempt_request_hash(
        req.model_copy(update={"request_key": uuid4()})
    )


@pytest.mark.parametrize("case", ["hash", "request", "fund", "report", "reasons", "probability", "inference", "status"])
def test_corrupt_receipt_is_not_silently_returned(case):
    row = copy.deepcopy(stored())
    if case == "hash":
        row.receipt_hash = "e" * 64
    elif case == "request":
        row.cutoff_date = date(2026, 9, 3)
    elif case == "fund":
        row.check_payload["fund_code"] = "008888"
    elif case == "report":
        row.check_payload["report_hash"] = "d" * 64
    else:
        key, value = {
            "reasons": ("blocking_codes", []),
            "probability": ("up_probability", 0.9),
            "inference": ("inference_executed", True),
            "status": ("status", "GENERATED"),
        }[case]
        row.check_payload[key] = value
        # 即使攻击者重算指纹，也不能绕过固定协议的语义校验。
        row.receipt_hash = cash_hash({"request_hash": row.request_hash, "check": row.check_payload})
    with pytest.raises(HistoricalNavStorageError) as error:
        service.restore_attempt(row)
    assert error.value.code == "CASH_ATTEMPT_CORRUPTED"


def test_future_cutoff_rejected_before_database(monkeypatch):
    monkeypatch.setattr(service, "get_nav_sample_storage_engine", lambda: pytest.fail("must not connect"))
    with pytest.raises(HistoricalNavStorageError) as error:
        service.save_cash_prediction_attempt(request().model_copy(update={"cutoff_date": date(2099, 1, 1)}))
    assert error.value.code == "CUTOFF_DAY_NOT_CLOSED"


@pytest.mark.parametrize("headers", [{}, {"X-Service-Token": "bad"}, {**HEADERS, "Origin": "http://localhost"}])
def test_http_auth_before_read_or_write(client, monkeypatch, headers):
    monkeypatch.setattr(api, "save_cash_prediction_attempt", lambda _: pytest.fail("must not write"))
    monkeypatch.setattr(api, "get_cash_prediction_attempt", lambda _: pytest.fail("must not read"))
    assert client.post(PATH, json=request().model_dump(mode="json", by_alias=True), headers=headers).status_code == 403
    assert client.get(f"{PATH}/{uuid4()}", headers=headers).status_code == 403


@pytest.mark.parametrize("extra", [{"force": True}, {"x": [1] * 7}, {"model": {}}, {"cutoffDate": "2025-08-07"}])
def test_http_cannot_bypass_or_upload_inputs(client, monkeypatch, extra):
    monkeypatch.setattr(api, "save_cash_prediction_attempt", lambda _: pytest.fail("must not write"))
    assert (
        client.post(
            PATH, json={**request().model_dump(mode="json", by_alias=True), **extra}, headers=HEADERS
        ).status_code
        == 422
    )


def test_http_create_retry_read_and_redacted_failure(client, monkeypatch):
    req = request()
    value = service.restore_attempt(stored(req))
    results = iter(((value, True), (value, False)))
    monkeypatch.setattr(api, "save_cash_prediction_attempt", lambda _: next(results))
    monkeypatch.setattr(api, "get_cash_prediction_attempt", lambda _: value)
    first = client.post(PATH, json=req.model_dump(mode="json", by_alias=True), headers=HEADERS)
    second = client.post(PATH, json=req.model_dump(mode="json", by_alias=True), headers=HEADERS)
    read = client.get(f"{PATH}/{value.attempt_id}", headers=HEADERS)
    assert (first.status_code, second.status_code, read.status_code) == (201, 200, 200)
    assert first.json() == second.json() == read.json()
    assert all(r.headers["cache-control"] == "no-store" for r in (first, second, read))

    def fail(_):
        raise SQLAlchemyError("synthetic-private-details")

    monkeypatch.setattr(api, "save_cash_prediction_attempt", fail)
    failed = client.post(PATH, json=req.model_dump(mode="json", by_alias=True), headers=HEADERS)
    assert failed.status_code == 503 and "synthetic-private" not in failed.text
