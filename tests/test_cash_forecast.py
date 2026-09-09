"""结果投影/授权绑定的人工契约测试；人工授权仅是后续工程的替身，绝不证明正式发布成功。"""

from datetime import UTC, date, datetime
from decimal import Decimal
from types import SimpleNamespace
from uuid import uuid4

import pytest
from app.api.routes import watchlist_prediction as api
from app.schemas.cash_forecast import CashAuthorizedModel, CashForecastRequest, CashForecastSnapshot, CashForecastView
from app.schemas.cash_prediction_check import CashPredictionCheck
from app.services import cash_publication as publication
from app.services.cash_forecast import cash_forecast_stale_reasons
from app.services.cash_prediction_features import build_cash_prediction_feature
from app.services.cash_prediction_inference import calculate_cash_inference
from app.services.cash_reinvestment_research import BLOCKERS, evaluate_cash_dataset
from app.services.trading_calendar import load_prediction_calendar
from tests.test_cash_prediction_features import SOURCE, history_rows
from tests.test_cash_prediction_features import request as input_request
from tests.test_cash_reinvestment_research import data as data
from tests.test_historical_nav_http import HEADERS
from tests.test_historical_nav_http import client as client

PATH = "/internal/v1/predictions/cash-forecasts"
CUTOFF = date(2026, 9, 8)
NOW = datetime(2026, 9, 9, 6, tzinfo=UTC)


@pytest.fixture(scope="module")
def report(data):
    return evaluate_cash_dataset(data)


@pytest.fixture
def bundle(report):
    """与真实人工训练模型一致的授权替身；生产解析器没有签发它的接口。"""
    model = report.windows[2].model
    req = CashForecastRequest(
        fundCode="006730",
        cutoffDate=CUTOFF,
        requestKey=uuid4(),
        researchRunId=uuid4(),
        expectedReportHash=report.report_hash,
        expectedModelHash=model.model_hash,
    )
    calendar = load_prediction_calendar(CUTOFF)
    known = build_cash_prediction_feature(
        input_request(CUTOFF), SOURCE, history_rows(input_request(CUTOFF)), (), calendar
    )
    grant = CashAuthorizedModel(
        authorization_id=uuid4(),
        authorization_hash="0" * 64,
        research_run_id=req.research_run_id,
        report_hash=report.report_hash,
        artifact=model,
        calendar_hashes=(calendar.content_hash,),
    )
    grant = grant.model_copy(update={"authorization_hash": publication.authorization_hash(grant)})
    value = calculate_cash_inference(known, model, expected_model_hash=model.model_hash)
    saved = CashForecastSnapshot(
        request=req,
        authorization_id=grant.authorization_id,
        authorization_hash=grant.authorization_hash,
        source_revision_id=SOURCE.source_sync_run_id,
        feature=known,
        value=value,
    )
    return req, grant, saved


@pytest.mark.parametrize("field", ["expected_model_hash", "expected_report_hash", "research_run_id", "fund_code"])
def test_grant_cannot_authorize_another_request(bundle, field):
    req, grant, _ = bundle
    changed = uuid4() if field == "research_run_id" else "000001" if field == "fund_code" else "f" * 64
    with pytest.raises(publication.HistoricalNavStorageError, match="不匹配"):
        publication.validate_cash_authorization(grant, req.model_copy(update={field: changed}))


def test_grant_content_hash_is_checked(bundle):
    req, grant, _ = bundle
    assert publication.validate_cash_authorization(grant, req) == grant
    with pytest.raises(publication.HistoricalNavStorageError):
        publication.validate_cash_authorization(grant.model_copy(update={"authorization_id": uuid4()}), req)


@pytest.mark.parametrize("codes", [BLOCKERS, ()])
def test_real_resolver_never_turns_research_or_empty_blockers_into_a_grant(bundle, report, monkeypatch, codes):
    req, _, _ = bundle
    row = SimpleNamespace(
        run_id=req.research_run_id,
        request_key=uuid4(),
        created_at=NOW,
        report=report.model_dump(mode="json"),
        dataset_hash=report.preparation.dataset_hash,
        publication_status="MODEL_NOT_RELEASED",
    )
    check = CashPredictionCheck(
        checked_at=NOW,
        fund_code=req.fund_code,
        research_run_id=req.research_run_id,
        report_hash=req.expected_report_hash,
        blocking_codes=codes,
        comparisons=(),
        incomplete_window_ids=(),
    )
    monkeypatch.setattr(publication, "check_cash_prediction_in_session", lambda *a, **k: check)
    monkeypatch.setattr(publication, "find_research", lambda *a, **k: row)
    with pytest.raises(publication.CashPublicationUnavailable) as error:
        publication.resolve_cash_authorization(None, req, now=NOW)
    assert error.value.codes == (codes or ("FORMAL_RELEASE_NOT_ISSUED",))


def test_current_data_is_usable_and_old_window_never_rolls_forward(bundle):
    _, _, saved = bundle
    assert not cash_forecast_stale_reasons(saved, SOURCE, date(2026, 9, 7), closed_day=CUTOFF)
    late = cash_forecast_stale_reasons(saved, SOURCE, date(2026, 9, 7), closed_day=date(2026, 11, 1))
    assert "FORECAST_WINDOW_ENDED" in late and "NAV_TOO_OLD" in late
    assert saved.value.target_end_date == load_prediction_calendar(CUTOFF).future_sessions(CUTOFF)[-1]


@pytest.mark.parametrize("status", ["MODEL_NOT_RELEASED", "DATA_INSUFFICIENT", "STALE"])
def test_unavailable_projection_rejects_any_number(status):
    with pytest.raises(ValueError):
        CashForecastView(
            status=status,
            fund_code="006730",
            cutoff_date=CUTOFF,
            reason_codes=("BLOCKED",),
            up_probability=Decimal("0.5"),
        )


@pytest.mark.parametrize("headers", [{}, {"X-Service-Token": "bad"}, {**HEADERS, "Origin": "http://localhost"}])
def test_http_auth_precedes_generation_and_read(client, bundle, monkeypatch, headers):
    req, _, _ = bundle
    monkeypatch.setattr(api, "generate_cash_forecast", lambda _: pytest.fail("must not generate"))
    monkeypatch.setattr(api, "get_cash_forecast", lambda _: pytest.fail("must not read"))
    assert client.post(PATH, json=req.model_dump(mode="json", by_alias=True), headers=headers).status_code == 403
    assert client.get(f"{PATH}/{uuid4()}", headers=headers).status_code == 403


@pytest.mark.parametrize(
    "extra", [{"grant": {}}, {"force": True}, {"x": [0] * 7}, {"model": {}}, {"cutoffDate": "2025-08-07"}]
)
def test_http_never_accepts_external_grant_or_values(client, bundle, monkeypatch, extra):
    req, _, _ = bundle
    monkeypatch.setattr(api, "generate_cash_forecast", lambda _: pytest.fail("must not generate"))
    assert (
        client.post(PATH, json={**req.model_dump(mode="json", by_alias=True), **extra}, headers=HEADERS).status_code
        == 422
    )


def test_production_model_hash_is_explicit_even_while_blocked(bundle, report, monkeypatch):
    req, _, _ = bundle
    row = SimpleNamespace(
        run_id=req.research_run_id,
        request_key=uuid4(),
        created_at=NOW,
        report=report.model_dump(mode="json"),
        dataset_hash=report.preparation.dataset_hash,
        publication_status="MODEL_NOT_RELEASED",
    )
    monkeypatch.setattr(publication, "check_cash_prediction_in_session", lambda *a, **k: None)
    monkeypatch.setattr(publication, "find_research", lambda *a, **k: row)
    with pytest.raises(publication.HistoricalNavStorageError) as error:
        publication.resolve_cash_authorization(None, req.model_copy(update={"expected_model_hash": "f" * 64}), now=NOW)
    assert error.value.code == "MODEL_HASH_MISMATCH"
