"""现金存储的不可变快照、独立答案、校验和HTTP边界。"""

from datetime import UTC, datetime
from types import SimpleNamespace
from uuid import uuid4

import pytest
from app.api.routes import cash_reinvestment_storage as api
from app.schemas.cash_reinvestment_storage import CashBatchSaveRequest
from app.services import cash_reinvestment_storage as service
from app.services.cash_reinvestment_batch import summarize_cash_batch
from app.services.trading_calendar import load_calendar
from tests.test_cash_reinvestment_samples import SOURCE, preview
from tests.test_historical_nav_http import HEADERS
from tests.test_historical_nav_http import client as client

PATH = "/internal/v1/features/cash-reinvestment/batches"


def request(**updates):
    return CashBatchSaveRequest(
        **{
            "fundCode": "006730",
            "startDate": "2023-06-20",
            "endDate": "2023-06-20",
            "requestKey": str(uuid4()),
            **updates,
        }
    )


def batch_preview(req=None):
    req = req or request()
    return summarize_cash_batch(req, SOURCE, load_calendar(), (preview(),), page_count=1)


def stored_parts(req=None):
    req = req or request()
    result = batch_preview(req)
    batch_id, sample_id = uuid4(), uuid4()
    batch = SimpleNamespace(
        batch_id=batch_id,
        request_key=req.request_key,
        fund_code=req.fund_code,
        start_date=req.start_date,
        end_date=req.end_date,
        summary=result.model_dump(mode="json", exclude={"items"}),
        batch_hash=result.batch_hash,
        created_at=datetime.now(UTC),
    )
    item = result.items[0]
    row = SimpleNamespace(
        sample_id=sample_id,
        batch_id=batch_id,
        cutoff_date=item.cutoff_date,
        context=item.model_dump(mode="json", exclude={"feature_payload", "offline_label"}),
        feature_payload=item.feature_payload.model_dump(mode="json"),
    )
    label = SimpleNamespace(sample_id=sample_id, label_payload=item.offline_label.model_dump(mode="json"))
    return batch, [(row, label)]


def test_preview_and_stored_readback_exactly_equal():
    batch, rows = stored_parts()
    restored = service.restore_cash_batch(batch, rows)
    assert restored.preview == batch_preview()
    assert restored.database_written and not restored.preview.database_written
    assert "offline_label" not in rows[0][0].feature_payload
    assert restored.preview.items[0].offline_label is not None


@pytest.mark.parametrize("corruption", ["hash", "label", "feature", "missing_row", "duplicate", "identity"])
def test_corruption_is_rejected_not_recalculated(corruption):
    batch, rows = stored_parts()
    if corruption == "hash":
        batch.batch_hash = "0" * 64
    if corruption == "label":
        rows[0][1].label_payload["label_up_20d"] = 0
    if corruption == "feature":
        rows[0][0].feature_payload["metrics"]["return_20d"] = "0.99000000"
    if corruption == "missing_row":
        rows.clear()
    if corruption == "duplicate":
        rows.append(rows[0])
    if corruption == "identity":
        batch.fund_code = "001632"
    with pytest.raises(service.HistoricalNavStorageError) as error:
        service.restore_cash_batch(batch, rows)
    assert error.value.code == "CASH_BATCH_CORRUPTED"


def test_missing_batch():
    with pytest.raises(service.HistoricalNavStorageError) as error:
        service.restore_cash_batch(None, ())
    assert error.value.status_code == 404


def test_http_save_retry_and_get(client, monkeypatch):
    batch, rows = stored_parts()
    result = service.restore_cash_batch(batch, rows)
    body = request(requestKey=str(batch.request_key)).model_dump(mode="json", by_alias=True)
    monkeypatch.setattr(api, "save_cash_batch", lambda _: (result, True))
    assert client.post(PATH, headers=HEADERS, json=body).status_code == 201
    monkeypatch.setattr(api, "save_cash_batch", lambda _: (result, False))
    repeated = client.post(PATH, headers=HEADERS, json=body)
    assert repeated.status_code == 200
    monkeypatch.setattr(api, "get_cash_batch", lambda _: result)
    assert client.get(f"{PATH}/{batch.batch_id}", headers=HEADERS).json() == repeated.json()


@pytest.mark.parametrize("headers", [{}, {"X-Service-Token": "wrong"}, {**HEADERS, "Origin": "http://localhost"}])
def test_auth_before_database(client, monkeypatch, headers):
    monkeypatch.setattr(api, "save_cash_batch", lambda _: pytest.fail("must not access storage"))
    assert client.post(PATH, headers=headers, json=request().model_dump(mode="json", by_alias=True)).status_code == 403


@pytest.mark.parametrize(
    "changes", [{"pageSize": True}, {"pageSize": 1.5}, {"trainingEligible": True}, {"fundCode": "000001"}]
)
def test_invalid_body(client, monkeypatch, changes):
    monkeypatch.setattr(api, "save_cash_batch", lambda _: pytest.fail("must not access storage"))
    body = request().model_dump(mode="json", by_alias=True)
    assert client.post(PATH, headers=HEADERS, json={**body, **changes}).status_code == 422


def test_2025_protection_before_opening_db(monkeypatch):
    monkeypatch.setattr(service, "get_nav_sample_storage_engine", lambda: pytest.fail("no database access"))
    with pytest.raises(Exception) as error:
        service.save_cash_batch(request(startDate="2024-12-20", endDate="2024-12-20"))
    assert error.value.code == "TEST_PERIOD_PROTECTED"
