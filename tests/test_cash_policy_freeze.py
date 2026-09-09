"""规则冻结的人工契约测试；不替真实规则确认，不授予模型发布资格。"""

import copy
from datetime import UTC, datetime
from decimal import Decimal
from types import SimpleNamespace
from uuid import uuid4

import pytest
from app.api.routes import watchlist_prediction as api
from app.schemas.cash_policy_freeze import CashFrozenPolicy, CashPolicyFreezeRequest, CashPolicySnapshot
from app.services import cash_policy_freeze as service
from app.services.cash_reinvestment_research import restore_research
from app.services.cash_reinvestment_storage import cash_hash
from app.services.cash_release_review import review_cash_research
from app.services.historical_nav_storage import HistoricalNavStorageError
from pydantic import ValidationError
from tests.test_cash_prediction_check import evaluated as evaluated
from tests.test_cash_reinvestment_research import data as data
from tests.test_historical_nav_http import HEADERS
from tests.test_historical_nav_http import client as client

URL = "/internal/v1/predictions/release-policy"


def approved_policy():
    return service.load_release_policy().model_copy(
        update={"approval_state": "APPROVED", "approval_reference": "synthetic-test-only"}
    )


def row_for(policy=None):
    policy = policy or approved_policy()
    snapshot = CashPolicySnapshot(policy=policy, binding=service.current_rule_binding())
    row = SimpleNamespace(
        freeze_id=uuid4(),
        request_key=uuid4(),
        created_at=datetime.now(UTC),
        policy_version=policy.version,
        policy_hash=cash_hash(policy.model_dump(mode="json")),
        binding_hash=cash_hash(snapshot.binding.model_dump(mode="json")),
        snapshot=snapshot.model_dump(mode="json"),
    )
    row.content_hash = service._content_hash(row)
    return row


def request_for(descriptor=None):
    descriptor = descriptor or service.describe_cash_policy()
    return CashPolicyFreezeRequest(
        requestKey=uuid4(), expectedPolicyHash=descriptor.policy_hash, expectedBindingHash=descriptor.binding_hash
    )


def test_draft_descriptor_does_not_access_database_and_freeze_is_rejected(monkeypatch, client):
    monkeypatch.setattr(service, "get_nav_sample_storage_engine", lambda: pytest.fail("draft must not query or write"))
    result = client.get(URL, headers=HEADERS)
    assert result.status_code == 200 and result.headers["cache-control"] == "no-store"
    descriptor = service.describe_cash_policy()
    assert not descriptor.approval_ready and not descriptor.database_written and not descriptor.publication_allowed
    assert len(descriptor.binding.windows) == 3 and descriptor.binding.independent_test_required
    response = client.post(
        URL + "/freezes", json=request_for(descriptor).model_dump(mode="json", by_alias=True), headers=HEADERS
    )
    assert response.status_code == 409 and response.json()["detail"]["code"] == "CASH_POLICY_APPROVAL_REQUIRED"


@pytest.mark.parametrize("field", ["expected_policy_hash", "expected_binding_hash"])
def test_stale_hash_rejected_before_database(monkeypatch, field):
    policy = approved_policy()
    monkeypatch.setattr(service, "load_release_policy", lambda: policy)
    monkeypatch.setattr(service, "get_nav_sample_storage_engine", lambda: pytest.fail("hash mismatch must not query"))
    request = request_for().model_copy(update={field: "f" * 64})
    with pytest.raises(HistoricalNavStorageError) as error:
        service.freeze_cash_policy(request)
    assert error.value.code == "CASH_POLICY_HASH_MISMATCH"


def test_frozen_snapshot_identity_content_and_write_flags():
    row = row_for()
    saved = service.restore_policy_freeze(row, created=True)
    read = service.restore_policy_freeze(row)
    assert saved.created and saved.database_written and not read.created and not read.database_written
    assert read.freeze_id == saved.freeze_id and read.content_hash == saved.content_hash
    assert (
        read.snapshot.policy.approval_state == "APPROVED"
        and not read.publication_allowed
        and not read.independent_test_read
    )
    service.validate_policy_binding(read, approved_policy())
    with pytest.raises(ValidationError):
        CashFrozenPolicy.model_validate({**read.model_dump(mode="json"), "publication_allowed": True})


@pytest.mark.parametrize("mutation", ["id", "key", "time", "policy", "binding", "content", "draft", "version"])
def test_mutated_snapshot_rejected(mutation):
    row = copy.deepcopy(row_for())
    if mutation == "id":
        row.freeze_id = uuid4()
    elif mutation == "key":
        row.request_key = uuid4()
    elif mutation == "time":
        row.created_at = row.created_at.replace(year=2025)
    elif mutation == "policy":
        row.snapshot["policy"]["maximum_ece"] = "0.01"
    elif mutation == "binding":
        row.snapshot["binding"]["bin_edges"][1] = "0.1"
    elif mutation == "draft":
        row.snapshot["policy"]["approval_state"] = "DRAFT"
    elif mutation == "version":
        row.policy_version = "different"
    else:
        row.content_hash = "f" * 64
    with pytest.raises(HistoricalNavStorageError) as error:
        service.restore_policy_freeze(row)
    assert error.value.code == "CASH_POLICY_FREEZE_CORRUPTED"


def test_different_current_policy_cannot_borrow_old_freeze():
    frozen = service.restore_policy_freeze(row_for())
    changed = approved_policy().model_copy(update={"minimum_coverage": Decimal("0.81")})
    with pytest.raises(HistoricalNavStorageError) as error:
        service.validate_policy_binding(frozen, changed)
    assert error.value.code == "CASH_POLICY_FREEZE_MISMATCH"


def test_persisted_rules_remove_only_freeze_blocker_not_data_or_test_gates(evaluated):
    frozen = service.restore_policy_freeze(row_for())
    result = review_cash_research(
        restore_research(evaluated),
        frozen.snapshot.policy,
        fund_code="006730",
        checked_at=datetime.now(UTC),
        frozen_policy=frozen,
    )
    assert result.policy_persisted and result.policy_freeze_id == frozen.freeze_id
    assert next(c.status for c in result.checks if c.code == "POLICY_FREEZE") == "PASS"
    assert "POLICY_NOT_FROZEN" not in result.blocking_codes
    assert "INDEPENDENT_TEST_NOT_EVALUATED" in result.blocking_codes
    assert "HISTORICAL_FIRST_VERSIONS_UNVERIFIED" in result.blocking_codes
    assert not result.database_written and not result.publication_allowed and result.status == "BLOCKED"


@pytest.mark.parametrize(
    "extra",
    [
        {"approved": True},
        {"approvalReference": "pretend"},
        {"policy": {}},
        {"force": True},
        {"includeTest": True},
        {"frozenAt": "2024-01-01"},
    ],
)
def test_http_cannot_upload_approval_policy_or_backdate(client, monkeypatch, extra):
    monkeypatch.setattr(api, "freeze_cash_policy", lambda _: pytest.fail("invalid input must not reach storage"))
    payload = {**request_for().model_dump(mode="json", by_alias=True), **extra}
    assert client.post(URL + "/freezes", json=payload, headers=HEADERS).status_code == 422


@pytest.mark.parametrize("headers", [{}, {"X-Service-Token": "bad"}, {**HEADERS, "Origin": "http://localhost"}])
@pytest.mark.parametrize("endpoint", ["descriptor", "create", "read"])
def test_http_auth_before_any_policy_work(client, monkeypatch, headers, endpoint):
    monkeypatch.setattr(api, "describe_cash_policy", lambda: pytest.fail("no read"))
    monkeypatch.setattr(api, "freeze_cash_policy", lambda _: pytest.fail("no write"))
    monkeypatch.setattr(api, "get_cash_policy_freeze", lambda _: pytest.fail("no read"))
    if endpoint == "descriptor":
        response = client.get(URL, headers=headers)
    elif endpoint == "read":
        response = client.get(URL + f"/freezes/{uuid4()}", headers=headers)
    else:
        response = client.post(
            URL + "/freezes", json=request_for().model_dump(mode="json", by_alias=True), headers=headers
        )
    assert response.status_code == 403
