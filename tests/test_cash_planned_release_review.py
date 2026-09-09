"""计划绑定证据进入只读发布审查；人工成绩不能冒充真实数据或正式发布。"""

import copy
from datetime import UTC, datetime
from decimal import Decimal, localcontext
from uuid import uuid4

import pytest
from app.schemas.cash_release_review import CashReleaseReview
from app.services import cash_planned_research as planned
from app.services import cash_release_review as review
from pydantic import ValidationError
from tests.test_cash_planned_research import records as records
from tests.test_cash_reinvestment_research import data as data


def checked(records, *, with_binding=True):
    proof = planned.restore_planned_research(*records)
    return review.review_cash_research(
        proof.research,
        proof.policy_freeze.snapshot.policy,
        fund_code="006730",
        checked_at=datetime(2026, 9, 9, 7, tzinfo=UTC),
        frozen_policy=proof.policy_freeze,
        planned_research=proof if with_binding else None,
    )


def test_verified_binding_gives_scored_coverage_without_publication(records):
    result = checked(records)
    assert result.ex_ante_plan_verified and result.planned_research_binding_id == records[0].binding_id
    assert result.planned_research_binding_hash == records[0].content_hash
    assert result.exam_plan_hash == records[0].preparation["plan"]["plan_hash"]
    assert len(result.exam_coverage_evidence) == 9
    checks = {(c.window_id, c.scope): c for c in result.checks if c.code == "EXAM_COVERAGE"}
    denominators = {"DEV_2023_Q3": 44, "DEV_2023_Q4": 40, "VALIDATION_2024": 222}
    for item in result.exam_coverage_evidence:
        assert item.planned_count == denominators[item.window_id]
        assert checks[item.window_id, item.fund_code].actual == Decimal(item.scored_count) / item.planned_count
        assert checks[item.window_id, item.fund_code].status == "PASS"
    assert "EX_ANTE_COVERAGE_EVIDENCE_MISSING" not in result.blocking_codes
    assert {
        "HISTORICAL_FIRST_VERSIONS_UNVERIFIED",
        "DIVIDEND_COMPLETENESS_UNVERIFIED",
        "NON_CASH_ADJUSTMENTS_UNVERIFIED",
        "INDEPENDENT_TEST_NOT_EVALUATED",
        "MARKET_STATE_EVIDENCE_MISSING",
    }.issubset(result.blocking_codes)
    assert result.status == "BLOCKED" and not result.publication_allowed
    assert not result.inference_executed and not result.independent_test_read and not result.database_written
    payload = result.model_dump_json()
    assert all(f'"{key}"' not in payload for key in ("base_model", "up_probability", "coefficients", "offline_label"))


def test_freeze_without_research_binding_does_not_credit_old_report(records):
    result = checked(records, with_binding=False)
    assert result.policy_persisted and not result.ex_ante_plan_verified
    assert result.exam_coverage_evidence == () and result.planned_research_binding_id is None
    assert "EX_ANTE_COVERAGE_EVIDENCE_MISSING" in result.blocking_codes
    assert all(c.status == "MISSING" and c.actual is None for c in result.checks if c.code == "EXAM_COVERAGE")


def test_context_and_input_are_not_mutated(records):
    before = copy.deepcopy(records)
    expected = checked(records)
    with localcontext() as context:
        context.prec = 9
        actual = checked(records)
        CashReleaseReview.model_validate_json(actual.model_dump_json())
    assert actual == expected and records == before


@pytest.mark.parametrize("field", ["run_id", "request_key", "created_at"])
def test_binding_cannot_be_used_with_another_stored_research(records, field):
    proof = planned.restore_planned_research(*records)
    value = datetime(2024, 1, 1, tzinfo=UTC) if field == "created_at" else uuid4()
    with pytest.raises(ValueError, match="different policy or research"):
        review.review_cash_research(
            proof.research.model_copy(update={field: value}),
            proof.policy_freeze.snapshot.policy,
            fund_code="006730",
            checked_at=datetime.now(UTC),
            frozen_policy=proof.policy_freeze,
            planned_research=proof,
        )


@pytest.mark.parametrize(
    "case",
    [
        "unverified",
        "missing_binding",
        "missing_hash",
        "wrong_plan",
        "missing_item",
        "duplicate_item",
        "ratio",
        "test_period",
        "publication",
    ],
)
def test_response_rejects_false_evidence_or_flags(records, case):
    payload = checked(records).model_dump(mode="json")
    if case == "unverified":
        payload["ex_ante_plan_verified"] = False
    elif case == "missing_binding":
        payload["planned_research_binding_id"] = None
    elif case == "missing_hash":
        payload["planned_research_binding_hash"] = None
    elif case == "wrong_plan":
        payload["exam_plan_hash"] = "f" * 64
    elif case == "missing_item":
        payload["exam_coverage_evidence"].pop()
    elif case == "duplicate_item":
        payload["exam_coverage_evidence"].append(payload["exam_coverage_evidence"][0])
    elif case == "ratio":
        payload["exam_coverage_evidence"][0]["scored_count"] -= 1
    elif case == "test_period":
        payload["exam_coverage_evidence"][0]["window_id"] = "INDEPENDENT_TEST_2025"
    else:
        payload["publication_allowed"] = True
    with pytest.raises(ValidationError):
        CashReleaseReview.model_validate(payload)
