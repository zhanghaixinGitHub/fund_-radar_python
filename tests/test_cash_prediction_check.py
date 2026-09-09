"""人工资料只验证拒绝/比较语义，不伪造可发布模型或真实基金成绩。"""

from datetime import UTC, date, datetime
from types import SimpleNamespace
from uuid import uuid4

import pytest
from app.api.routes import watchlist_prediction as api
from app.schemas.cash_prediction_check import CashPredictionCheckRequest
from app.services import cash_prediction_check as check
from app.services.cash_reinvestment_research import evaluate_cash_dataset, restore_research
from app.services.cash_reinvestment_storage import cash_hash
from app.services.historical_nav_storage import HistoricalNavStorageError
from sqlalchemy.dialects import postgresql
from sqlalchemy.exc import SQLAlchemyError
from tests.test_cash_reinvestment_research import data as data
from tests.test_historical_nav_http import HEADERS
from tests.test_historical_nav_http import client as client
from tests.test_watchlist_prediction import run


@pytest.fixture(scope="module")
def evaluated(data):
    report = evaluate_cash_dataset(data)
    return SimpleNamespace(
        run_id=uuid4(),
        request_key=uuid4(),
        created_at=datetime.now(UTC),
        report=report.model_dump(mode="json"),
        dataset_hash=report.preparation.dataset_hash,
        publication_status="MODEL_NOT_RELEASED",
    )


def request_for(row):
    return CashPredictionCheckRequest(
        fundCode="006730", researchRunId=row.run_id, expectedReportHash=row.report["report_hash"]
    )


def db(monkeypatch, row, *, fund=None, source=None):
    """显式指定报告，不查询“最新报告”，捕获所有SQL以验证没有原始数值/预测写入。"""
    fund = fund or SimpleNamespace(
        fund_code="006730", fund_type="STOCK", status="ACTIVE", source_code="TUSHARE_PRO_FUND"
    )
    source = source or SimpleNamespace(enabled=True, source_id=uuid4())
    results = iter((fund, source, date(2026, 9, 4), row))
    sql = []

    class FakeSession:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def begin(self):
            return self

        def execute(self, statement):
            sql.append(str(statement.compile(dialect=postgresql.dialect())))
            return SimpleNamespace(one_or_none=lambda: next(results))

        def scalar(self, statement):
            self.execute(statement)
            return next(results)

    monkeypatch.setattr(check, "Session", lambda _: FakeSession())
    monkeypatch.setattr(check, "get_nav_preview_engine", lambda: None)
    return sql


def test_comparison_is_per_window_and_fund_without_selecting_winner(evaluated):
    blockers, comparisons, incomplete = check.inspect_cash_research(restore_research(evaluated))
    assert not incomplete and len(comparisons) == 3 * 4 * 4
    assert set(check.BLOCKERS).issubset(blockers)  # 人工数据全部做完，也不能授予发布资格。
    for c in comparisons:
        assert c.strict_gain_observed == (c.accuracy_delta > 0 and c.brier_delta < 0)


def test_missing_windows_do_not_invent_metrics():
    blockers, comparisons, incomplete = check.inspect_cash_research(restore_research(run()))
    assert len(incomplete) == 3 and not comparisons and "ROLLING_WINDOWS_INCOMPLETE" in blockers


@pytest.mark.parametrize("mutation", ["missing_window", "missing_baseline", "duplicate_fund", "population", "nan"])
def test_corrupt_comparison_evidence_rejected_even_after_rehash(evaluated, mutation):
    import copy

    row = copy.deepcopy(evaluated)
    window = row.report["windows"][0]
    if mutation == "missing_window":
        row.report["windows"].pop()
    elif mutation == "missing_baseline":
        window["baselines"].pop()
    elif mutation == "duplicate_fund":
        window["after"]["per_fund"][0]["fund_code"] = window["after"]["per_fund"][1]["fund_code"]
    elif mutation == "population":
        window["baselines"][0]["validation"]["actual_up_count"] -= 1
    else:
        window["after"]["validation"]["brier_score"] = "NaN"
    row.report["report_hash"] = cash_hash({k: v for k, v in row.report.items() if k != "report_hash"})
    with pytest.raises((ValueError, HistoricalNavStorageError)):
        check.inspect_cash_research(restore_research(row))


def test_precheck_is_readonly_and_never_calls_numeric_inference(monkeypatch, evaluated):
    from app.services import historical_nav_calibration, historical_nav_training

    def forbidden(*args, **kwargs):
        pytest.fail("rejected publication must not fit or infer")

    monkeypatch.setattr(historical_nav_calibration, "predict_calibrated_scores", forbidden)
    monkeypatch.setattr(historical_nav_training, "predict_artifact_logits", forbidden)
    monkeypatch.setattr(historical_nav_training, "fit_logistic_artifact", forbidden)
    sql = db(monkeypatch, evaluated)
    result = check.check_cash_prediction(request_for(evaluated))
    assert result.status == "GENERATION_BLOCKED" and result.up_probability is result.direction is None
    assert not result.forecast_created and not result.inference_executed and not result.database_written
    assert len(sql) == 5 and "READ ONLY" in sql[0] and "REPEATABLE READ" in sql[0]
    assert "WHERE cash_research_run.run_id =" in sql[-1]  # 主键等值至多一行，不要求冗余LIMIT。
    assert all(
        not any(t in s for t in ("unit_nav", "forecast_result", "analysis_model_release", "cash_sample")) for s in sql
    )


@pytest.mark.parametrize("failure", ["report_hash", "no_fund_samples", "source_disabled", "non_stock"])
def test_precheck_rejects_wrong_scope(monkeypatch, evaluated, failure):
    import copy

    row = copy.deepcopy(evaluated)
    request = request_for(row)
    fund, source = None, None
    if failure == "report_hash":
        request = request.model_copy(update={"expected_report_hash": "f" * 64})
    elif failure == "no_fund_samples":
        row.report["preparation"]["fund_counts"]["006730"] = {"TRAIN": 0, "VALIDATION": 0}
        row.report["report_hash"] = cash_hash({k: v for k, v in row.report.items() if k != "report_hash"})
        request = request_for(row)
    elif failure == "source_disabled":
        source = SimpleNamespace(enabled=False, source_id=uuid4())
    else:
        fund = SimpleNamespace(fund_code="006730", fund_type="BOND", status="ACTIVE", source_code="TUSHARE_PRO_FUND")
    db(monkeypatch, row, fund=fund, source=source)
    with pytest.raises(HistoricalNavStorageError) as error:
        check.check_cash_prediction(request)
    assert error.value.status_code == 409


@pytest.mark.parametrize("extra", [{"force": True}, {"includeTest": True}, {"trainingEligible": True}, {"x": [1] * 7}])
def test_http_cannot_pass_model_or_bypass_gate(client, monkeypatch, extra):
    monkeypatch.setattr(api, "check_cash_prediction", lambda _: pytest.fail("must not read"))
    request = request_for(run()).model_dump(mode="json", by_alias=True)
    assert (
        client.post("/internal/v1/predictions/generation-check", json={**request, **extra}, headers=HEADERS).status_code
        == 422
    )


@pytest.mark.parametrize("headers", [{}, {"X-Service-Token": "bad"}, {**HEADERS, "Origin": "http://localhost"}])
def test_http_requires_internal_auth(client, monkeypatch, headers):
    monkeypatch.setattr(api, "check_cash_prediction", lambda _: pytest.fail("must not read"))
    request = request_for(run()).model_dump(mode="json", by_alias=True)
    assert client.post("/internal/v1/predictions/generation-check", json=request, headers=headers).status_code == 403


def test_http_real_service_checks_no_store_and_redaction(client, monkeypatch, evaluated):
    db(monkeypatch, evaluated)
    request = request_for(evaluated).model_dump(mode="json", by_alias=True)
    response = client.post("/internal/v1/predictions/generation-check", json=request, headers=HEADERS)
    assert response.status_code == 200 and response.headers["cache-control"] == "no-store"
    assert response.json()["status"] == "GENERATION_BLOCKED" and response.json()["up_probability"] is None

    def fail(_):
        raise SQLAlchemyError("synthetic-private-details")

    monkeypatch.setattr(api, "check_cash_prediction", fail)
    response = client.post("/internal/v1/predictions/generation-check", json=request, headers=HEADERS)
    assert response.status_code == 503 and "synthetic-private" not in response.text
