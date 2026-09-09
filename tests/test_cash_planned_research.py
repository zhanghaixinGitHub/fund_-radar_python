"""新研究绑定契约；真实规则不审批，数值与确认资料均为测试人工样本。"""

import copy
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from uuid import uuid4

import pytest
from app.api.routes import cash_reinvestment_storage as api
from app.schemas.cash_planned_research import CashPlannedResearch, CashPlannedResearchRequest
from app.services import cash_planned_research as service
from app.services.cash_exam_plan import build_cash_exam_plan
from app.services.cash_exam_preparation import summarize_cash_exam_preparation
from app.services.cash_reinvestment_research import evaluate_cash_dataset
from app.services.historical_nav_storage import HistoricalNavStorageError
from pydantic import ValidationError
from tests.test_cash_policy_freeze import row_for
from tests.test_cash_reinvestment_research import data as data
from tests.test_cash_reinvestment_research import small_dataset
from tests.test_historical_nav_http import HEADERS
from tests.test_historical_nav_http import client as client

URL = "/internal/v1/features/cash-reinvestment/planned-research-runs"


def request_for():
    data = small_dataset()
    return CashPlannedResearchRequest(
        requestKey=uuid4(),
        batchIds=data.report.batch_ids,
        expectedDatasetHash=data.report.dataset_hash,
        policyFreezeId=uuid4(),
        expectedPolicyFreezeHash="a" * 64,
    )


@pytest.fixture(scope="module")
def records(data):
    frozen = row_for()
    preparation = summarize_cash_exam_preparation(data, build_cash_exam_plan())
    started = datetime.now(UTC)
    report = evaluate_cash_dataset(data)
    completed = datetime.now(UTC)
    binding_id = uuid4()
    research = SimpleNamespace(
        run_id=uuid4(),
        request_key=binding_id,
        created_at=completed,
        dataset_hash=data.report.dataset_hash,
        report=report.model_dump(mode="json"),
        publication_status="MODEL_NOT_RELEASED",
    )
    binding = SimpleNamespace(
        binding_id=binding_id,
        request_key=uuid4(),
        research_run_id=research.run_id,
        policy_freeze_id=frozen.freeze_id,
        policy_freeze_hash=frozen.content_hash,
        report_hash=report.report_hash,
        preparation=preparation.model_dump(mode="json"),
        evaluation_started_at=started,
        completed_at=completed,
    )
    binding.content_hash = service.binding_content_hash(binding)
    return binding, research, frozen


def test_draft_cannot_open_database_or_train(client, monkeypatch):
    monkeypatch.setattr(service, "get_nav_sample_storage_engine", lambda: pytest.fail("no DB"))
    monkeypatch.setattr(service, "evaluate_cash_dataset", lambda _: pytest.fail("no fit"))
    response = client.post(URL, headers=HEADERS, json=request_for().model_dump(mode="json", by_alias=True))
    assert response.status_code == 409 and response.json()["detail"]["code"] == "CASH_POLICY_APPROVAL_REQUIRED"


def test_real_numerical_report_matches_bound_plan_but_stays_unpublished(records):
    result = service.restore_planned_research(*records, created=True)
    assert result.created and result.database_written and result.plan_bound_before_evaluation
    assert result.research.report.model_fitted
    assert result.research.report.status == "EVALUATED" and not result.publication_allowed
    assert not result.policy_freeze.database_written and not result.independent_test_read
    assert result.policy_freeze.frozen_at <= result.evaluation_started_at <= result.completed_at
    assert len(result.preparation.coverage) == 12
    assert all(group.usable_count is None for group in result.preparation.coverage[-3:])
    assert service.restore_planned_research(*records).created is False


@pytest.mark.parametrize(
    "field",
    [
        "binding_id",
        "request_key",
        "research_run_id",
        "policy_freeze_id",
        "policy_freeze_hash",
        "report_hash",
        "content_hash",
    ],
)
def test_changed_binding_identity_or_hash_is_rejected(records, field):
    row, research, frozen = copy.deepcopy(records)
    setattr(row, field, uuid4() if field.endswith("_id") or field == "request_key" else "f" * 64)
    with pytest.raises(HistoricalNavStorageError) as error:
        service.restore_planned_research(row, research, frozen)
    assert error.value.code == "CASH_PLANNED_RESEARCH_CORRUPTED"


@pytest.mark.parametrize(
    "case", ["before_freeze", "after_completion", "naive", "foreign_request", "changed_report", "changed_preparation"]
)
def test_even_rehashed_receipt_cannot_backdate_or_attach_wrong_report(records, case):
    row, research, frozen = copy.deepcopy(records)
    if case == "before_freeze":
        row.evaluation_started_at = frozen.created_at - timedelta(seconds=1)
    elif case == "after_completion":
        row.evaluation_started_at = row.completed_at + timedelta(seconds=1)
    elif case == "naive":
        row.evaluation_started_at = row.evaluation_started_at.replace(tzinfo=None)
    elif case == "foreign_request":
        research.request_key = uuid4()
    elif case == "changed_report":
        research.created_at -= timedelta(seconds=1)
    else:
        row.preparation["preparation"]["dataset_hash"] = "f" * 64
    row.content_hash = service.binding_content_hash(row)
    with pytest.raises(HistoricalNavStorageError) as error:
        service.restore_planned_research(row, research, frozen)
    assert error.value.code == "CASH_PLANNED_RESEARCH_CORRUPTED"


def test_missing_binding_is_404_but_broken_references_are_503(records):
    with pytest.raises(HistoricalNavStorageError) as error:
        service.restore_planned_research(None, None, None)
    assert error.value.status_code == 404
    with pytest.raises(HistoricalNavStorageError) as error:
        service.restore_planned_research(records[0], None, records[2])
    assert error.value.status_code == 503


@pytest.mark.parametrize(
    "extra",
    [
        {"researchRunId": str(uuid4())},
        {"report": {}},
        {"model": {}},
        {"force": True},
        {"includeTest": True},
        {"evaluationStartedAt": "2023-01-01T00:00:00Z"},
        {"approved": True},
    ],
)
def test_request_cannot_attach_old_report_or_supply_approval_or_times(client, monkeypatch, extra):
    monkeypatch.setattr(api, "save_planned_research", lambda _: pytest.fail("no work"))
    response = client.post(URL, headers=HEADERS, json={**request_for().model_dump(mode="json", by_alias=True), **extra})
    assert response.status_code == 422


@pytest.mark.parametrize("headers", [{}, {"X-Service-Token": "bad"}, {**HEADERS, "Origin": "http://localhost"}])
@pytest.mark.parametrize("read", [False, True])
def test_http_auth_precedes_work(client, monkeypatch, headers, read):
    monkeypatch.setattr(api, "save_planned_research", lambda _: pytest.fail("no write"))
    monkeypatch.setattr(api, "get_planned_research", lambda _: pytest.fail("no read"))
    response = (
        client.get(URL + f"/{uuid4()}", headers=headers)
        if read
        else client.post(URL, headers=headers, json=request_for().model_dump(mode="json", by_alias=True))
    )
    assert response.status_code == 403


@pytest.mark.parametrize("field", ["publication_allowed", "independent_test_read", "database_written"])
def test_read_response_cannot_claim_publication_test_or_write(records, field):
    result = service.restore_planned_research(*records).model_dump(mode="json")
    with pytest.raises(ValidationError):
        CashPlannedResearch.model_validate({**result, field: True})
