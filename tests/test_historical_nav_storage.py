"""存储边界与HTTP契约的离线测试；所有样本均为人工数据，不连接项目数据库。"""

from collections import Counter
from copy import deepcopy
from dataclasses import replace
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from types import SimpleNamespace
from uuid import UUID, uuid4

import pytest
from app.api.routes import historical_nav_storage as api
from app.models.historical_nav_sample import HistoricalNavSampleBatch
from app.repositories.feature_snapshot import FeatureSourceReadiness
from app.schemas.historical_nav import HistoricalNavBatchPreviewResponse
from app.schemas.historical_nav_storage import HistoricalNavBatchSaveRequest, HistoricalNavStoredBatch
from app.services.historical_nav_samples import (
    HISTORICAL_NAV_FEATURE_VERSION,
    HISTORICAL_NAV_LABEL_VERSION,
    HISTORICAL_NAV_SAMPLE_RULE_VERSION,
    HistoricalNavPoint,
    HistoricalNavSampleInput,
    build_historical_nav_samples,
)
from app.services.historical_nav_storage import HistoricalNavStorageError, _is_retryable_conflict
from app.services.historical_nav_storage_validation import validate_batch_samples
from sqlalchemy.exc import DBAPIError, SQLAlchemyError
from tests.test_historical_nav_http import HEADERS  # noqa: F401
from tests.test_historical_nav_http import client as client

PATH = "/internal/v1/features/historical-nav-samples/batches"
START = date(2025, 1, 1)
SOURCE = FeatureSourceReadiness(UUID(int=1), "STORAGE_TEST", UUID(int=2), datetime(2025, 8, 1, tzinfo=UTC))


def synthetic_preview(request):
    """使用真实构建器制作包含不足/完整/未成熟三种状态的人工净值样本。"""
    points = tuple(
        HistoricalNavPoint(
            START + timedelta(days=i),
            START + timedelta(days=i + 1),
            Decimal("1") + Decimal(i) / 100,
            Decimal("1.5") + Decimal(i) / 100,
        )
        for i in range(131)
    )
    all_samples = build_historical_nav_samples(
        HistoricalNavSampleInput(
            fund_code=request.fund_code,
            fund_type="STOCK",
            source_code=SOURCE.source_code,
            source_sync_run_id=SOURCE.source_sync_run_id,
            nav_points=points,
        )
    )
    samples = tuple(s for s in all_samples if request.start_date <= s.as_of_date <= request.end_date)
    counts = Counter(s.eligibility_status for s in samples)
    return SOURCE, HistoricalNavBatchPreviewResponse(
        fund_code=request.fund_code,
        start_date=request.start_date,
        end_date=request.end_date,
        page_size=request.page_size,
        page_count=(len(samples) + request.page_size - 1) // request.page_size,
        sample_count=len(samples),
        scorable_count=counts["SCORABLE"],
        data_insufficient_count=counts["DATA_INSUFFICIENT"],
        label_not_matured_count=counts["LABEL_NOT_MATURED"],
        unavailable_reasons=dict(Counter(s.unavailable_reason for s in samples if s.unavailable_reason)),
        items=samples,
    )


def save_request(first=60, last=60, *, key=None, page_size=10):
    return HistoricalNavBatchSaveRequest(
        request_key=key or uuid4(),
        fund_code="008888",
        start_date=START + timedelta(days=first),
        end_date=START + timedelta(days=last),
        page_size=page_size,
    )


@pytest.fixture
def bundle():
    request = save_request()
    source, preview = synthetic_preview(request)
    batch = HistoricalNavSampleBatch(
        batch_id=uuid4(),
        request_key=request.request_key,
        fund_code=request.fund_code,
        fund_type="STOCK",
        source_code=source.source_code,
        source_sync_run_id=source.source_sync_run_id,
        start_date=request.start_date,
        end_date=request.end_date,
        feature_version=HISTORICAL_NAV_FEATURE_VERSION,
        sample_rule_version=HISTORICAL_NAV_SAMPLE_RULE_VERSION,
        label_version=HISTORICAL_NAV_LABEL_VERSION,
        purpose="LEARNING_ONLY",
        sample_count=1,
        scorable_count=1,
        data_insufficient_count=0,
        label_not_matured_count=0,
        unavailable_reasons={},
        created_at=datetime(2026, 9, 7, tzinfo=UTC),
    )
    return request, batch, preview.items


def test_valid_batch_and_stored_json_preserve_preview_and_timezone(bundle):
    request, batch, samples = bundle
    validate_batch_samples(batch, samples)
    stored = HistoricalNavStoredBatch(
        **{c.name: getattr(batch, c.name) for c in batch.__table__.columns}, items=samples
    )
    assert stored.model_dump(mode="json")["created_at"].endswith("+08:00")
    assert stored.items == samples
    assert stored.request_key == request.request_key


@pytest.mark.parametrize(
    "field,value",
    [
        ("fund_code", "000001"),
        ("feature_version", "WRONG"),
        ("sample_rule_version", "WRONG"),
        ("available_at", None),
        ("as_of_date", START),
        ("feature_hash", "0" * 64),
        ("offline_label", None),
        ("nav_value_basis", "UNIT_NAV"),
        ("unavailable_reason", "WRONG"),
    ],
)
def test_inconsistent_sample_is_rejected_before_writing(bundle, field, value):
    _, batch, samples = bundle
    with pytest.raises(ValueError):
        validate_batch_samples(batch, (replace(samples[0], **{field: value}),))


@pytest.mark.parametrize(
    "field,value",
    [
        ("label_version", "WRONG"),
        ("horizon_trading_days", 5),
        ("label_end_date", START),
        ("label_available_at", START),
        ("future_return_20d", Decimal("NaN")),
        ("label_up_20d", 0),
    ],
)
def test_invalid_future_answer_is_rejected(bundle, field, value):
    _, batch, samples = bundle
    bad = replace(samples[0], offline_label=replace(samples[0].offline_label, **{field: value}))
    with pytest.raises(ValueError):
        validate_batch_samples(batch, (bad,))


@pytest.mark.parametrize("location", ["top", "input", "source", "metrics", "quality"])
def test_answer_cannot_be_smuggled_into_feature_fields(bundle, location):
    _, batch, samples = bundle
    payload = deepcopy(samples[0].feature_payload)
    target = payload if location == "top" else payload[location]
    target["label_up_20d"] = 1
    with pytest.raises(ValueError):
        validate_batch_samples(batch, (replace(samples[0], feature_payload=payload),))


def test_counts_and_duplicate_dates_are_validated(bundle):
    _, batch, samples = bundle
    with pytest.raises(ValueError):
        validate_batch_samples(batch, samples + samples)
    batch.sample_count = 0
    batch.scorable_count = 0
    with pytest.raises(ValueError, match="summary"):
        validate_batch_samples(batch, samples)


@pytest.mark.parametrize("created,status", [(True, 201), (False, 200)])
def test_save_http_status_and_readback_contract(client, bundle, monkeypatch, created, status):
    request, batch, samples = bundle
    stored = HistoricalNavStoredBatch(
        **{c.name: getattr(batch, c.name) for c in batch.__table__.columns}, items=samples
    )
    monkeypatch.setattr(api, "save_historical_nav_batch", lambda request: (stored, created))
    monkeypatch.setattr(api, "get_historical_nav_batch", lambda batch_id: stored)
    response = client.post(
        PATH, json=request.model_dump(mode="json", by_alias=True), headers={**HEADERS, "X-Trace-Id": "storage-test"}
    )
    assert response.status_code == status
    assert response.headers["X-Trace-Id"] == "storage-test"
    assert response.json() == client.get(f"{PATH}/{batch.batch_id}", headers=HEADERS).json()


@pytest.mark.parametrize("headers", [{}, {"X-Service-Token": "wrong"}, {**HEADERS, "Origin": "https://example.com"}])
def test_storage_authentication_precedes_any_io(client, monkeypatch, headers):
    def forbidden(*args):
        pytest.fail("unauthenticated request reached storage")

    monkeypatch.setattr(api, "save_historical_nav_batch", forbidden)
    monkeypatch.setattr(api, "get_historical_nav_batch", forbidden)
    assert (
        client.post(PATH, json=save_request().model_dump(mode="json", by_alias=True), headers=headers).status_code
        == 403
    )
    assert client.get(f"{PATH}/{uuid4()}", headers=headers).status_code == 403


@pytest.mark.parametrize(
    "changes",
    [
        {"requestKey": None},
        {"requestKey": "bad-uuid"},
        {"fundCode": "88"},
        {"endDate": "2025-12-31"},
        {"endDate": "2024-01-01"},
        {"pageSize": 0},
        {"pageSize": 31},
        {"pageSize": True},
        {"pageSize": 1.5},
        {"items": []},
        {"user_id": "not-accepted"},
    ],
)
def test_invalid_save_requests_fail_before_io(client, monkeypatch, changes):
    monkeypatch.setattr(
        api, "save_historical_nav_batch", lambda request: pytest.fail("invalid request reached storage")
    )
    body = save_request().model_dump(mode="json", by_alias=True)
    assert client.post(PATH, json={**body, **changes}, headers=HEADERS).status_code == 422


@pytest.mark.parametrize(
    "error,status,code",
    [
        (HistoricalNavStorageError("REQUEST_KEY_CONFLICT", "请求冲突", 409), 409, "REQUEST_KEY_CONFLICT"),
        (SQLAlchemyError("driver details must not reach response"), 503, "BATCH_STORAGE_UNAVAILABLE"),
        (ValueError("invalid generated payload"), 422, "INVALID_NAV_SAMPLES"),
    ],
)
def test_storage_errors_have_safe_response(client, monkeypatch, error, status, code):
    def fail(request):
        raise error

    monkeypatch.setattr(api, "save_historical_nav_batch", fail)
    response = client.post(PATH, json=save_request().model_dump(mode="json", by_alias=True), headers=HEADERS)
    assert response.status_code == status
    assert response.json()["detail"]["code"] == code
    assert "driver details" not in response.text


@pytest.mark.parametrize(
    "state,constraint,expected",
    [
        ("40001", None, True),
        ("23505", "uq_historical_nav_batch_request", True),
        ("23505", "uq_historical_nav_sample_batch_date", False),
        ("23503", None, False),
        ("55P03", None, False),
    ],
)
def test_only_expected_concurrency_conflicts_are_retried(state, constraint, expected):
    original = Exception("synthetic database error")
    original.sqlstate = state
    original.diag = SimpleNamespace(constraint_name=constraint)
    assert _is_retryable_conflict(DBAPIError(None, None, original)) is expected
