"""真实PG读取绑定并审查；来源和数值矩阵人工构造，不开放真实审批或测试期。"""

import json
import os
from dataclasses import replace
from datetime import date
from uuid import uuid4

import pytest
from alembic.migration import MigrationContext
from alembic.operations import Operations
from app.schemas.cash_planned_research import CashPlannedResearchRequest
from app.schemas.cash_prediction_check import CashPredictionCheckRequest
from app.schemas.cash_release_review import CashReleaseReview
from app.services import cash_planned_research as planned
from app.services import cash_policy_freeze as freeze
from app.services import cash_release_policy as policy_source
from app.services import cash_release_review as review
from app.services import historical_nav_training as training
from sqlalchemy import event, text
from tests.test_cash_planned_research_schema import migration_module
from tests.test_cash_policy_freeze_postgres import policy_db as policy_db
from tests.test_cash_prediction_check import evaluated as evaluated
from tests.test_cash_reinvestment_postgres import database as database
from tests.test_cash_reinvestment_research import data as data
from tests.test_cash_release_review_postgres import review_db as review_db
from tests.test_historical_nav_http import HEADERS
from tests.test_historical_nav_http import client as client

pytestmark = pytest.mark.skipif(os.getenv("RUN_NAV_STORAGE_PG_TESTS") != "1", reason="显式启用隔离PG")
URL = "/internal/v1/predictions/release-review"


@pytest.fixture
def binding_database(policy_db, monkeypatch):
    engine, req, old_request, path = policy_db
    with engine.begin() as conn:
        module = migration_module()
        module.op = Operations(MigrationContext.configure(conn))
        module.upgrade()
    frozen = freeze.freeze_cash_policy(req)
    monkeypatch.setattr(planned, "get_nav_sample_storage_engine", lambda: engine)
    return engine, frozen, old_request, path


def save_matrix(binding_database, data, monkeypatch):
    engine, frozen, _, _ = binding_database
    # 仅资料读取替换为人工矩阵，研究器、保存、关联读取与发布审查均实际执行。
    monkeypatch.setattr(planned, "load_cash_dataset_in_session", lambda *a, **k: data)
    saved = planned.save_planned_research(
        CashPlannedResearchRequest(
            requestKey=uuid4(),
            batchIds=data.report.batch_ids,
            expectedDatasetHash=data.report.dataset_hash,
            policyFreezeId=frozen.freeze_id,
            expectedPolicyFreezeHash=frozen.content_hash,
        )
    )
    return saved, CashPredictionCheckRequest(
        fundCode="006730",
        researchRunId=saved.research.run_id,
        expectedReportHash=saved.research.report.report_hash,
    )


def snapshot(engine):
    with engine.connect() as conn:
        return tuple(
            conn.execute(text(sql)).all()
            for sql in (
                "SELECT run_id, request_key, md5(report::text) FROM cash_research_run ORDER BY run_id",
                "SELECT binding_id, content_hash FROM cash_planned_research_binding ORDER BY binding_id",
                "SELECT freeze_id, content_hash FROM cash_policy_freeze ORDER BY freeze_id",
            )
        )


def test_real_readonly_lookup_credits_exact_report_and_no_other_report(binding_database, data, monkeypatch, client):
    engine, _, old_request, _ = binding_database
    saved, request = save_matrix(binding_database, data, monkeypatch)
    before = snapshot(engine)
    statements = []

    def forbidden(*a, **k):
        pytest.fail("review must not compute or train")

    monkeypatch.setattr(planned, "evaluate_cash_dataset", forbidden)
    monkeypatch.setattr(training, "fit_logistic_artifact", forbidden)
    monkeypatch.setattr(training, "predict_artifact_logits", forbidden)

    def capture(conn, cursor, statement, parameters, context, executemany):
        statements.append(statement)

    event.listen(engine, "before_cursor_execute", capture)
    try:
        responses = [
            client.post(URL, json=request.model_dump(mode="json", by_alias=True), headers=HEADERS) for _ in range(2)
        ]
    finally:
        event.remove(engine, "before_cursor_execute", capture)
    assert all(r.status_code == 200 and r.headers["cache-control"] == "no-store" for r in responses)
    a, b = [CashReleaseReview.model_validate(r.json()) for r in responses]
    assert a.model_dump(exclude={"checked_at"}) == b.model_dump(exclude={"checked_at"})
    assert a.ex_ante_plan_verified and a.planned_research_binding_id == saved.binding_id
    assert len(a.exam_coverage_evidence) == 9 and not a.publication_allowed
    assert "EX_ANTE_COVERAGE_EVIDENCE_MISSING" not in a.blocking_codes
    assert "INDEPENDENT_TEST_NOT_EVALUATED" in a.blocking_codes
    assert len(statements) == 18 and "REPEATABLE READ, READ ONLY" in statements[0]
    assert sum("WHERE cash_planned_research_binding.research_run_id =" in s for s in statements) == 2
    assert all(s.lstrip().upper().startswith(("SET", "SELECT")) for s in statements)
    assert all(
        not any(v in s for v in ("unit_nav", "cash_sample", "forecast_result", "analysis_model_release"))
        for s in statements
    )
    old = client.post(URL, json=old_request.model_dump(mode="json", by_alias=True), headers=HEADERS)
    assert old.status_code == 200 and not old.json()["ex_ante_plan_verified"]
    assert old.json()["exam_coverage_evidence"] == [] and snapshot(engine) == before


@pytest.mark.parametrize("start, scored_windows", [(date(2022, 5, 1), {"VALIDATION_2024"}), (date(2023, 1, 1), set())])
def test_partial_research_does_not_turn_prepared_population_into_scored_coverage(
    binding_database, data, monkeypatch, client, start, scored_windows
):
    # 缺早期FIT资料会导致部分或全部窗口不能拟合；EXAM资料本身仍有40/44/200题。
    partial = replace(data, rows=tuple(r for r in data.rows if r.available_at >= start))
    saved, request = save_matrix(binding_database, partial, monkeypatch)
    result = client.post(URL, json=request.model_dump(mode="json", by_alias=True), headers=HEADERS)
    assert result.status_code == 200
    result = CashReleaseReview.model_validate(result.json())
    assert result.ex_ante_plan_verified and len(result.exam_coverage_evidence) == 3 * len(scored_windows)
    assert {item.window_id for item in result.exam_coverage_evidence} == scored_windows
    pending = [c for c in result.checks if c.code == "EXAM_COVERAGE" and c.window_id not in scored_windows]
    assert len(pending) == 9 - 3 * len(scored_windows) and all(
        c.status == "MISSING" and c.actual is None for c in pending
    )
    assert any(g.usable_count for g in saved.preparation.coverage if g.window_id == "DEV_2023_Q4")
    assert "EX_ANTE_COVERAGE_EVIDENCE_MISSING" in result.blocking_codes


def test_completed_exam_with_low_coverage_is_failed_not_missing(binding_database, data, monkeypatch, client):
    subset = replace(
        data, rows=tuple(r for r in data.rows if r.available_at.year != 2024 or r.available_at >= date(2024, 4, 1))
    )
    _, request = save_matrix(binding_database, subset, monkeypatch)
    response = client.post(URL, json=request.model_dump(mode="json", by_alias=True), headers=HEADERS)
    assert response.status_code == 200
    result = CashReleaseReview.model_validate(response.json())
    assert len(result.exam_coverage_evidence) == 9
    annual = [c for c in result.checks if c.code == "EXAM_COVERAGE" and c.window_id == "VALIDATION_2024"]
    assert len(annual) == 3 and all(c.status == "FAIL" and c.actual < c.required for c in annual)
    assert "EX_ANTE_COVERAGE_EVIDENCE_MISSING" not in result.blocking_codes
    assert "RELEASE_RULES_NOT_MET" in result.blocking_codes and not result.publication_allowed


@pytest.mark.parametrize("case", ["report_key", "report_hash", "time"])
def test_broken_binding_is_rejected_not_treated_as_legacy(binding_database, data, monkeypatch, client, case):
    engine, _, _, _ = binding_database
    saved, request = save_matrix(binding_database, data, monkeypatch)
    with engine.begin() as conn:
        if case == "report_key":
            conn.execute(
                text("UPDATE cash_research_run SET request_key=:key WHERE run_id=:id"),
                {"key": uuid4(), "id": saved.research.run_id},
            )
        elif case == "time":
            conn.execute(
                text("UPDATE cash_research_run SET created_at=created_at-interval '1 second' WHERE run_id=:id"),
                {"id": saved.research.run_id},
            )
        else:
            # 报告自身仍有效且指纹重算，但已绑定的完整报告指纹不同，必须拒绝。
            from app.services.cash_reinvestment_storage import cash_hash

            report = saved.research.report.model_dump(mode="json")
            report["preparation"]["input_count"] += 1
            report["report_hash"] = cash_hash({k: v for k, v in report.items() if k != "report_hash"})
            conn.execute(
                text("UPDATE cash_research_run SET report=CAST(:report AS jsonb) WHERE run_id=:id"),
                {"report": json.dumps(report), "id": saved.research.run_id},
            )
            request = request.model_copy(update={"expected_report_hash": report["report_hash"]})
    before = snapshot(engine)
    result = client.post(URL, json=request.model_dump(mode="json", by_alias=True), headers=HEADERS)
    assert result.status_code == 503 and result.json()["detail"]["code"] == "CASH_PLANNED_RESEARCH_CORRUPTED"
    assert snapshot(engine) == before and str(saved.binding_id) not in result.text


@pytest.mark.parametrize("state", ["DRAFT", "CHANGED_APPROVED"])
def test_current_policy_cannot_borrow_a_historical_binding(binding_database, data, monkeypatch, client, state):
    engine, _, _, path = binding_database
    _, request = save_matrix(binding_database, data, monkeypatch)
    policy = policy_source.load_release_policy().model_dump(mode="json")
    if state == "DRAFT":
        policy.update(approval_state="DRAFT", approval_reference=None)
    else:
        policy["minimum_coverage"] = "0.81"
    path.write_text(json.dumps(policy), encoding="utf-8")
    before = snapshot(engine)
    result = client.post(URL, json=request.model_dump(mode="json", by_alias=True), headers=HEADERS)
    if state == "DRAFT":
        assert result.status_code == 200 and not result.json()["ex_ante_plan_verified"]
        assert result.json()["exam_coverage_evidence"] == []
    else:
        assert result.status_code == 409 and result.json()["detail"]["code"] == "CASH_POLICY_FREEZE_MISMATCH"
    assert snapshot(engine) == before


def test_planned_lookup_runs_in_actual_readonly_transaction(binding_database, data, monkeypatch, client):
    engine, _, _, _ = binding_database
    _, request = save_matrix(binding_database, data, monkeypatch)
    before = snapshot(engine)

    def accidental_write(session, **identity):
        assert session.execute(text("SHOW transaction_read_only")).scalar_one() == "on"
        session.execute(
            text("UPDATE cash_research_run SET report=report WHERE run_id=:id"), {"id": identity["research_run_id"]}
        )
        pytest.fail("PostgreSQL must reject the accidental write")

    monkeypatch.setattr(review, "find_planned_research", accidental_write)
    result = client.post(URL, json=request.model_dump(mode="json", by_alias=True), headers=HEADERS)
    assert result.status_code == 503 and "UPDATE" not in result.text and snapshot(engine) == before
