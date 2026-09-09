"""观察日志诊断的HTTP边界；不能通过请求自报证明或读取2025正文。"""

from datetime import UTC, datetime

import pytest
from app.api.routes import cash_reinvestment_storage as api
from app.schemas.cash_source_observation import CashObservationCheck, CashObservationCheckRequest
from pydantic import ValidationError
from tests.test_historical_nav_http import HEADERS
from tests.test_historical_nav_http import client as client

URL = "/internal/v1/features/cash-reinvestment/source-observation-check"
PAYLOAD = {"fundCode": "006730", "startDate": "2022-01-01", "endDate": "2024-12-31", "cutoffDate": "2024-12-31"}


@pytest.mark.parametrize("headers", [{}, {"X-Service-Token": "bad"}, {**HEADERS, "Origin": "http://localhost"}])
def test_auth_precedes_any_read(client, monkeypatch, headers):
    monkeypatch.setattr(api, "check_cash_source_observations", lambda _: pytest.fail("no read"))
    assert client.post(URL, json=PAYLOAD, headers=headers).status_code == 403


@pytest.mark.parametrize(
    "extra",
    [
        {"fundCode": "000001"},
        {"endDate": "2025-01-01"},
        {"cutoffDate": "2025-01-01"},
        {"startDate": "2024-12-31", "endDate": "2024-01-01"},
        {"cutoffDate": "2023-01-01"},
        {"verified": True},
        {"includePayload": True},
        {"observedAt": "2023-01-01"},
    ],
)
def test_invalid_scope_and_self_reported_evidence_rejected(client, monkeypatch, extra):
    monkeypatch.setattr(api, "check_cash_source_observations", lambda _: pytest.fail("no read"))
    assert client.post(URL, json={**PAYLOAD, **extra}, headers=HEADERS).status_code == 422


@pytest.mark.parametrize(
    "extra",
    [
        {"training_eligible": True},
        {"historical_first_versions_verified": True},
        {"source_payload_read": True},
        {"status": "LOCAL_WRITE_RECORDS_ONLY"},
        {"recorded_before_cutoff_count": 1},
    ],
)
def test_metadata_never_claims_formal_evidence(extra):
    payload = dict(
        request=CashObservationCheckRequest.model_validate(PAYLOAD),
        checked_at=datetime.now(UTC),
        status="NO_LOCAL_RECORDS",
        active_record_count=0,
        recorded_before_cutoff_count=0,
        expired_record_count=0,
        first_observed_at=None,
    )
    with pytest.raises(ValidationError):
        CashObservationCheck(**{**payload, **extra})
