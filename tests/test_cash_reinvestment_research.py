"""人工数据检验新版研究时点、指纹、门禁和模型复现；不冒充真实基金成绩。"""

import math
from dataclasses import replace
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from time import perf_counter
from types import SimpleNamespace
from uuid import UUID, uuid4

import pytest
from app.schemas.cash_reinvestment_research import CashPrepareRequest, CashResearchRequest
from app.services import cash_reinvestment_research as research
from app.services.cash_reinvestment_storage import cash_hash, restore_cash_batch
from app.services.historical_nav_evaluation import PreparedRow
from app.services.historical_nav_training import predict_artifact_scores
from app.services.trading_calendar import load_calendar
from tests.test_cash_reinvestment_storage import stored_parts
from tests.test_historical_nav_http import HEADERS
from tests.test_historical_nav_http import client as client


def small_dataset():
    return research.prepare_cash_batches((restore_cash_batch(*stored_parts()),), deadline=perf_counter() + 20)


@pytest.fixture(scope="module")
def data():
    calendar = load_calendar()
    rows = []
    for fund in research.FUNDS:
        for i, day in enumerate(calendar.sessions):
            if not date(2022, 1, 1) <= day <= date(2024, 11, 1):
                continue
            x = tuple(Decimal(str(math.sin(i / (17 + j)) / 20)) for j in range(6)) + (Decimal(i % 5),)
            y = int(math.sin((i + 20) / 17) > math.sin(i / 17))
            rows.append(
                PreparedRow(
                    batch_id=UUID(int=1),
                    fund_code=fund,
                    as_of_date=day,
                    available_at=day,
                    label_available_at=calendar.sessions[i + 20] + timedelta(days=1),
                    x=x,
                    y=y,
                    content_hash=cash_hash({"fund": fund, "date": str(day), "x": [str(v) for v in x], "y": y}),
                )
            )
    report = small_dataset().report.model_copy(update={"status": "RESEARCH_READY", "versions": research.VERSIONS})
    return research.CashDataset(tuple(rows), report)


def test_prepare_counts_and_formal_admission_remain_separate():
    data = small_dataset()
    assert data.report.input_count == data.report.unique_count == data.report.usable_count == 1
    assert data.report.status == "INSUFFICIENT_DATA" and data.report.missing_counts["001632"]["TRAIN"] == 252
    assert data.rows[0].available_at == data.rows[0].as_of_date == date(2023, 6, 20)
    assert len(data.rows[0].x) == 7 and data.rows[0].label_available_at > data.rows[0].available_at
    assert not data.report.training_eligible and not data.report.test_scored


def test_duplicate_merge_and_order_invariance():
    first = restore_cash_batch(*stored_parts())
    second = first.model_copy(update={"batch_id": uuid4()})
    a = research.prepare_cash_batches((first, second), deadline=perf_counter() + 20)
    b = research.prepare_cash_batches((second, first), deadline=perf_counter() + 20)
    assert a.report == b.report and a.report.duplicate_count == 1 and a.report.unique_count == 1


def test_conflicting_cutoff_rejected():
    first = restore_cash_batch(*stored_parts())
    item = first.preview.items[0].model_copy(update={"feature_hash": "f" * 64})
    second = first.model_copy(
        update={"batch_id": uuid4(), "preview": first.preview.model_copy(update={"items": (item,)})}
    )
    with pytest.raises(research.HistoricalNavStorageError) as error:
        research.prepare_cash_batches((first, second), deadline=perf_counter() + 20)
    assert error.value.code == "CASH_SAMPLE_CONFLICT"


def test_all_windows_purge_answers_and_split_inputs(data):
    for window in research.WINDOWS:
        rows, counts = research.cash_window_rows(data, window)
        assert all(not any(c.missing.values()) for c in counts)
        for r in rows["FIT"]:
            assert r.available_at <= r.label_available_at <= window.fit_end_date
        for r in rows["CALIBRATION"]:
            assert window.fit_end_date < r.available_at <= r.label_available_at <= window.calibration_end_date
        for r in rows["EXAM"]:
            assert window.calibration_end_date < r.available_at <= r.label_available_at <= window.evaluation_end_date


def test_real_numerical_fit_replay_and_no_publication(data):
    report = research.evaluate_cash_dataset(data)
    assert report.status == "EVALUATED" and report.model_fitted
    assert report.release_gate == "BLOCKED" and report.publication_status == "MODEL_NOT_RELEASED"
    assert not report.test_scored and set(research.BLOCKERS).issubset(report.release_blockers)
    for w in report.windows:
        assert w.model.base_model.versions == research.VERSIONS
        restored = research.restore_calibrated_artifact(w.model.model_dump_json())
        assert restored == w.model
        assert len(predict_artifact_scores(restored.base_model, (data.rows[0].x,))) == 1


def test_2024_answers_cannot_change_any_model(data):
    original = research.evaluate_cash_dataset(data)
    changed = replace(data, rows=tuple(replace(r, y=1 - r.y) if r.available_at.year == 2024 else r for r in data.rows))
    after = research.evaluate_cash_dataset(changed)
    assert [w.model for w in original.windows] == [w.model for w in after.windows]
    assert original.windows[:2] == after.windows[:2]
    assert original.windows[-1].after != after.windows[-1].after


def test_insufficient_data_does_not_fit(monkeypatch):
    monkeypatch.setattr(research, "evaluate_calibration_window_rows", lambda *a, **k: pytest.fail("must not fit"))
    report = research.evaluate_cash_dataset(small_dataset())
    assert not report.model_fitted and report.status == "INSUFFICIENT_DATA"


def test_corrupt_report_rejected():
    report = research.evaluate_cash_dataset(small_dataset())
    row = SimpleNamespace(
        run_id=uuid4(),
        request_key=uuid4(),
        created_at=datetime.now(UTC),
        dataset_hash=report.preparation.dataset_hash,
        report=report.model_dump(mode="json"),
        publication_status="MODEL_NOT_RELEASED",
    )
    assert research.restore_research(row).report == report
    row.report["release_blockers"] = []
    with pytest.raises(research.HistoricalNavStorageError):
        research.restore_research(row)


@pytest.mark.parametrize(
    "updates",
    [{"includeTest": True}, {"trainingEligible": True}, {"batchIds": []}, {"batchIds": [str(UUID(int=1))] * 2}],
)
def test_no_test_or_publication_switch(updates):
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        CashPrepareRequest.model_validate({"batchIds": [str(uuid4())], **updates})


def test_internal_http_preparation_and_busy(client, monkeypatch):
    from app.api.routes import cash_reinvestment_storage as api
    from app.services.historical_nav_training import HistoricalNavTrainingError

    data = small_dataset()
    monkeypatch.setattr(api, "load_cash_dataset", lambda _: data)
    response = client.post(
        "/internal/v1/features/cash-reinvestment/preparation", headers=HEADERS, json={"batchIds": [str(uuid4())]}
    )
    assert response.status_code == 200 and response.json()["training_eligible"] is False

    def busy(_):
        raise HistoricalNavTrainingError("TRAINING_BUSY", "当前正在训练。", 429)

    monkeypatch.setattr(api, "save_cash_research", busy)
    request = CashResearchRequest(batchIds=[uuid4()], requestKey=uuid4(), expectedDatasetHash="0" * 64)
    response = client.post(
        "/internal/v1/features/cash-reinvestment/research-runs",
        headers=HEADERS,
        json=request.model_dump(mode="json", by_alias=True),
    )
    assert response.status_code == 429 and response.json()["detail"]["code"] == "TRAINING_BUSY"
