"""数据准备与基线的离线测试；人工序列只用于验证工程规则，不代表真实基金成绩。"""

from collections import Counter
from dataclasses import replace
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from uuid import UUID, uuid4

import pytest
from app.api.routes import historical_nav_evaluation as api
from app.schemas.historical_nav_evaluation import HistoricalNavEvaluationRequest
from app.schemas.historical_nav_storage import HistoricalNavStoredBatch
from app.services.historical_nav_evaluation import (
    FEATURE_NAMES,
    HistoricalNavEvaluationError,
    calculate_baseline_metrics,
    prepare_historical_nav_dataset,
)
from app.services.historical_nav_samples import (
    HISTORICAL_NAV_FEATURE_VERSION,
    HISTORICAL_NAV_LABEL_VERSION,
    HISTORICAL_NAV_SAMPLE_RULE_VERSION,
    HistoricalNavPoint,
    HistoricalNavSampleInput,
    build_historical_nav_samples,
)
from app.services.historical_nav_storage import HistoricalNavStorageError
from app.services.momentum_baseline import fixed_momentum_score
from sqlalchemy.exc import SQLAlchemyError
from tests.test_historical_nav_http import HEADERS  # noqa: F401
from tests.test_historical_nav_http import client as client

PATH = "/internal/v1/features/historical-nav-samples/baseline-evaluation"
START = date(2022, 1, 1)


def day(offset):
    return START + timedelta(days=offset)


def make_batches(samples, *, seed=100):
    """按31自然日分为真实存储契约允许的小批次，全部使用同一人工来源水位。"""
    batches = []
    for offset in range(0, len(samples), 31):
        items = tuple(samples[offset : offset + 31])
        counts = Counter(s.eligibility_status for s in items)
        batches.append(
            HistoricalNavStoredBatch(
                batch_id=UUID(int=seed + offset),
                request_key=uuid4(),
                fund_code=items[0].fund_code,
                fund_type="STOCK",
                start_date=items[0].as_of_date,
                end_date=items[-1].as_of_date,
                source_code="EVALUATION_TEST",
                source_sync_run_id=UUID(int=2),
                feature_version=HISTORICAL_NAV_FEATURE_VERSION,
                sample_rule_version=HISTORICAL_NAV_SAMPLE_RULE_VERSION,
                label_version=HISTORICAL_NAV_LABEL_VERSION,
                purpose="LEARNING_ONLY",
                sample_count=len(items),
                scorable_count=counts["SCORABLE"],
                data_insufficient_count=counts["DATA_INSUFFICIENT"],
                label_not_matured_count=counts["LABEL_NOT_MATURED"],
                unavailable_reasons=dict(Counter(s.unavailable_reason for s in items if s.unavailable_reason)),
                created_at=datetime(2026, 9, 7, tzinfo=UTC),
                items=items,
            )
        )
    return tuple(batches)


@pytest.fixture(scope="module")
def batches():
    points = []
    for i in range(700):
        triangle = i % 80 if i % 80 < 40 else 80 - i % 80
        value = Decimal(2) + Decimal(triangle) / 100 + Decimal(i) / 10000
        points.append(HistoricalNavPoint(day(i), day(i + 1), value, value))
    samples = build_historical_nav_samples(
        HistoricalNavSampleInput(
            fund_code="008888",
            fund_type="STOCK",
            source_code="EVALUATION_TEST",
            source_sync_run_id=UUID(int=2),
            nav_points=tuple(points),
        )
    )
    return make_batches(samples)


def request_for(batches, **updates):
    values = {
        "batch_ids": tuple(b.batch_id for b in batches),
        "train_start_date": day(61),
        "train_end_date": day(332),
        "validation_end_date": day(472),
        "test_end_date": day(612),
    }
    return HistoricalNavEvaluationRequest(**(values | updates))


def test_prepare_exact_minimum_and_four_validation_baselines(batches):
    prepared = prepare_historical_nav_dataset(request_for(batches), batches)
    report = prepared.report
    assert report.status == "BASELINE_EVALUATED" and len(report.baselines) == 4
    assert (len(prepared.train), len(prepared.validation), len(prepared.test)) == (252, 120, 120)
    assert report.funds[0].missing_samples == {"TRAIN": 0, "VALIDATION": 0, "TEST": 0}
    assert all(b.validation.sample_count == 120 for b in report.baselines)
    assert report.feature_names == FEATURE_NAMES and all(len(r.x) == 7 for r in prepared.train)
    assert report.unique_sample_count == report.included_sample_count + sum(report.excluded_reasons.values())
    assert report.training_eligible is False and report.publication_status == "MODEL_NOT_RELEASED"
    assert report.persisted is False and report.purpose == "LEARNING_ONLY"
    # 来源、基金、日期、答案均没有混入固定的7列X。
    for row in prepared.train:
        original = next(s for b in batches for s in b.items if s.as_of_date == row.as_of_date)
        assert row.x == tuple(Decimal(original.feature_payload["metrics"][name]) for name in FEATURE_NAMES)
        assert row.y == original.offline_label.label_up_20d


def test_split_by_available_date_and_purge_actual_answer_date(batches):
    request = request_for(batches)
    prepared = prepare_historical_nav_dataset(request, batches)
    for rows, lower, upper in (
        (prepared.train, request.train_start_date - timedelta(days=1), request.train_end_date),
        (prepared.validation, request.train_end_date, request.validation_end_date),
        (prepared.test, request.validation_end_date, request.test_end_date),
    ):
        assert all(lower < r.available_at <= upper and r.label_available_at <= upper for r in rows)
    assert max(r.label_available_at for r in prepared.train) < min(r.available_at for r in prepared.validation)
    assert max(r.label_available_at for r in prepared.validation) < min(r.available_at for r in prepared.test)
    assert prepared.validation[0].as_of_date == request.train_end_date  # 净值日相同，但公告在下一天。
    assert all(
        prepared.report.excluded_reasons[f"{split}_LABEL_AFTER_CUTOFF"] == 20
        for split in ("TRAIN", "VALIDATION", "TEST")
    )


def test_input_order_preview_size_and_reader_pages_do_not_change_dataset(batches):
    first = prepare_historical_nav_dataset(request_for(batches), batches).report
    reversed_batches = tuple(reversed(batches))
    second = prepare_historical_nav_dataset(request_for(reversed_batches, preview_size=0), reversed_batches).report
    assert first.model_dump(exclude={"sample_preview"}) == second.model_dump(exclude={"sample_preview"})
    assert second.sample_preview == ()


def test_exact_duplicate_is_collapsed_but_original_batches_are_not_modified(batches):
    original = batches[3]
    duplicated = original.model_copy(update={"batch_id": UUID(int=99999), "request_key": uuid4()})
    with_duplicate = (*batches, duplicated)
    first = prepare_historical_nav_dataset(request_for(batches), batches)
    repeated = prepare_historical_nav_dataset(request_for(with_duplicate), with_duplicate)
    assert repeated.report.duplicate_sample_count == len(original.items)
    assert repeated.report.unique_sample_count == first.report.unique_sample_count
    assert repeated.report.train_hash == first.report.train_hash
    assert repeated.report.baselines == first.report.baselines
    assert repeated.train == first.train and len(original.items) == 31


def change_label(sample):
    label = sample.offline_label
    if label is None:
        return sample
    return replace(
        sample,
        offline_label=replace(
            label,
            future_return_20d=Decimal("-0.01") if label.label_up_20d else Decimal("0.01"),
            label_up_20d=1 - label.label_up_20d,
        ),
    )


def test_same_day_different_answer_is_conflict_not_latest_selection(batches):
    original = batches[3]
    changed = original.model_copy(
        update={"batch_id": UUID(int=99999), "items": (change_label(original.items[0]), *original.items[1:])}
    )
    selection = (*batches, changed)
    with pytest.raises(HistoricalNavEvaluationError, match="内容不一致") as error:
        prepare_historical_nav_dataset(request_for(selection), selection)
    assert error.value.code == "CONFLICTING_SAMPLES"


def test_held_out_answers_never_change_training_or_validation_and_are_not_exposed(batches):
    request = request_for(batches)
    first = prepare_historical_nav_dataset(request, batches).report
    changed = tuple(
        b.model_copy(
            update={
                "items": tuple(
                    change_label(s) if s.available_at and s.available_at > request.validation_end_date else s
                    for s in b.items
                )
            }
        )
        for b in batches
    )
    second = prepare_historical_nav_dataset(request, changed).report
    assert first.dataset_hash != second.dataset_hash
    assert first.train_hash == second.train_hash and first.baselines == second.baselines
    assert first.funds == second.funds and first.sample_preview == second.sample_preview
    assert all(s.split != "TEST" for s in second.sample_preview)
    assert "test_metrics" not in second.model_dump() and "test_up_rate" not in second.model_dump_json()


def test_validation_answers_never_fit_training_frequency(batches):
    request = request_for(batches)
    first = prepare_historical_nav_dataset(request, batches).report
    changed = tuple(
        b.model_copy(
            update={
                "items": tuple(
                    change_label(s)
                    if s.available_at and request.train_end_date < s.available_at <= request.validation_end_date
                    else s
                    for s in b.items
                )
            }
        )
        for b in batches
    )
    second = prepare_historical_nav_dataset(request, changed).report
    assert second.train_hash == first.train_hash
    assert second.funds[0].train_up_rate == first.funds[0].train_up_rate
    assert second.baselines != first.baselines


def test_minimum_is_per_fund_and_is_not_silently_pooled(batches):
    small = tuple(replace(s, fund_code="001632") for s in batches[3].items)
    selected = (*batches, *make_batches(small, seed=10000))
    report = prepare_historical_nav_dataset(request_for(selected), selected).report
    assert report.status == "INSUFFICIENT_DATA" and report.baselines == ()
    assert len(report.funds) == 2 and report.funds[0].missing_samples["TRAIN"] == 221


@pytest.mark.parametrize("field", ["source_code", "feature_version", "sample_rule_version", "label_version"])
def test_mixed_or_old_contracts_are_rejected(batches, field):
    selected = (batches[0].model_copy(update={field: "OTHER"}), *batches[1:])
    with pytest.raises(HistoricalNavEvaluationError) as error:
        prepare_historical_nav_dataset(request_for(selected), selected)
    assert error.value.code == "INCOMPATIBLE_BATCHES"


def test_missing_source_watermark_is_rejected(batches):
    selected = (batches[0].model_copy(update={"source_sync_run_id": None}), *batches[1:])
    with pytest.raises(HistoricalNavEvaluationError) as error:
        prepare_historical_nav_dataset(request_for(selected), selected)
    assert error.value.code == "SOURCE_WATERMARK_MISSING"


def test_corrupt_stored_hash_is_not_silently_filtered(batches):
    first = batches[3]
    changed = first.model_copy(update={"items": (replace(first.items[0], feature_hash="0" * 64), *first.items[1:])})
    selected = (*batches[:3], changed, *batches[4:])
    with pytest.raises(HistoricalNavStorageError) as error:
        prepare_historical_nav_dataset(request_for(selected), selected)
    assert error.value.code == "STORED_BATCH_INCONSISTENT"


def test_future_answer_announcement_is_purged_even_when_label_end_is_inside(batches):
    from app.services.historical_nav_evaluation import _sample_split

    request = request_for(batches)
    protocol = prepare_historical_nav_dataset(request, batches).report.protocol
    sample = next(s for b in batches for s in b.items if s.as_of_date == day(300))
    sample = replace(sample, offline_label=replace(sample.offline_label, label_available_at=day(333)))
    assert sample.offline_label.label_end_date < request.train_end_date
    assert _sample_split(sample, protocol) == (None, "TRAIN_LABEL_AFTER_CUTOFF")


def test_expired_deadline_fails_without_partial_report(batches):
    with pytest.raises(HistoricalNavEvaluationError) as error:
        prepare_historical_nav_dataset(request_for(batches), batches, deadline=0)
    assert error.value.code == "EVALUATION_TIMEOUT" and error.value.status_code == 503


def test_unit_nav_fallback_is_reported_but_not_used(batches):
    import hashlib
    import json
    from copy import deepcopy

    batch = batches[3]
    changed = []
    for sample in batch.items:
        payload = deepcopy(sample.feature_payload)
        payload["source"]["nav_value_basis"] = "UNIT_NAV"
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
        ).hexdigest()
        changed.append(replace(sample, nav_value_basis="UNIT_NAV", feature_payload=payload, feature_hash=digest))
    selected = (*batches[:3], batch.model_copy(update={"items": tuple(changed)}), *batches[4:])
    report = prepare_historical_nav_dataset(request_for(selected), selected).report
    assert report.excluded_reasons["UNSUPPORTED_NAV_BASIS"] == 31
    assert report.status == "INSUFFICIENT_DATA" and not report.baselines


def test_empty_batch_returns_diagnostic_without_fake_zero_scores(batches):
    empty = batches[0].model_copy(
        update={
            "items": (),
            "sample_count": 0,
            "scorable_count": 0,
            "data_insufficient_count": 0,
            "label_not_matured_count": 0,
            "unavailable_reasons": {},
        }
    )
    report = prepare_historical_nav_dataset(request_for((empty,)), (empty,)).report
    assert report.status == "INSUFFICIENT_DATA" and report.baselines == () and report.sample_preview == ()
    assert report.funds[0].missing_samples == {"TRAIN": 252, "VALIDATION": 120, "TEST": 120}
    assert report.funds[0].train_up_rate is None


def test_all_up_training_retains_warning_without_division_by_zero(batches):
    selected = tuple(
        batch.model_copy(
            update={
                "items": tuple(
                    change_label(s) if s.offline_label and not s.offline_label.label_up_20d else s for s in batch.items
                )
            }
        )
        for batch in batches
    )
    report = prepare_historical_nav_dataset(request_for(selected), selected).report
    assert report.funds[0].warnings == ("TRAIN_SINGLE_CLASS",)
    assert report.funds[0].train_up_rate == 1
    assert all(b.validation.balanced_accuracy is None for b in report.baselines)


def test_multi_fund_evaluation_reports_each_group(batches):
    copied = make_batches(tuple(replace(s, fund_code="001632") for b in batches for s in b.items), seed=10000)
    selected = (*batches, *copied)
    report = prepare_historical_nav_dataset(request_for(selected), selected).report
    assert report.status == "BASELINE_EVALUATED"
    assert all(b.validation.sample_count == 240 and len(b.per_fund) == 2 for b in report.baselines)


def test_pilot_plan_is_fixed_monthly_and_retry_keys_are_stable():
    from scripts.historical_nav_baseline_pilot import pilot_requests

    key = UUID(int=123)
    first, second = pilot_requests(key), pilot_requests(key)
    assert first == second and len(first) == len({r.request_key for r in first}) == 144
    assert {r.fund_code for r in first} == {"001632", "006730", "008888"}
    assert min(r.start_date for r in first) == date(2022, 1, 1)
    assert max(r.end_date for r in first) == date(2025, 12, 31)
    assert all(0 <= (r.end_date - r.start_date).days <= 30 for r in first)
    assert set(r.request_key for r in first).isdisjoint(r.request_key for r in pilot_requests(UUID(int=124)))


@pytest.mark.parametrize("execute,sufficient,expected", [(False, True, 0), (True, False, 2)])
def test_pilot_dry_run_or_shortage_never_writes(monkeypatch, capsys, execute, sufficient, expected):
    from types import SimpleNamespace

    from scripts import historical_nav_baseline_pilot as pilot

    argv = ["pilot", "--run-key", str(UUID(int=1))] + (["--execute"] if execute else [])
    monkeypatch.setattr("sys.argv", argv)
    monkeypatch.setattr(
        pilot, "get_settings", lambda: SimpleNamespace(ai_database_url="postgresql+psycopg://localhost/fund_ai")
    )
    monkeypatch.setattr(pilot, "preview_pilot", lambda _: {"sufficient": sufficient})
    monkeypatch.setattr(pilot, "execute_pilot", lambda _: pytest.fail("must not save"))
    assert pilot.main() == expected
    assert "PILOT_RESULT=" in capsys.readouterr().out


def test_pilot_cannot_write_to_remote_database(monkeypatch):
    from types import SimpleNamespace

    from scripts import historical_nav_baseline_pilot as pilot

    monkeypatch.setattr("sys.argv", ["pilot", "--run-key", str(UUID(int=1)), "--execute"])
    monkeypatch.setattr(
        pilot, "get_settings", lambda: SimpleNamespace(ai_database_url="postgresql+psycopg://remote.invalid/fund_ai")
    )
    monkeypatch.setattr(pilot, "preview_pilot", lambda _: pytest.fail("must not connect"))
    assert pilot.main() == 1


def test_known_metrics_and_half_score_tie():
    result = calculate_baseline_metrics((1, 0, 1, 0), tuple(map(Decimal, ("1", "0.5", "0", "0.2"))))
    assert result.correct_count == 3 and result.predicted_up_count == 1
    assert result.accuracy == result.balanced_accuracy == Decimal("0.75")
    assert result.brier_score == Decimal("0.3225")


def test_one_class_balanced_accuracy_is_undefined():
    result = calculate_baseline_metrics((1, 1), (Decimal(1), Decimal(1)))
    assert result.accuracy == 1 and result.brier_score == 0 and result.balanced_accuracy is None


@pytest.mark.parametrize("score", ["NaN", "Infinity", "-0.1", "1.1"])
def test_invalid_scores_are_rejected(score):
    with pytest.raises(ValueError):
        calculate_baseline_metrics((1,), (Decimal(score),))


@pytest.mark.parametrize("value,expected", [("-1", "0.0500"), ("0", "0.5000"), ("0.025025", "0.5501"), ("1", "0.9500")])
def test_existing_fixed_formula_is_preserved(value, expected):
    from app.services.baseline_analysis import _baseline_up_probability

    assert fixed_momentum_score(Decimal(value)) == Decimal(expected)
    assert _baseline_up_probability(Decimal(value)) == Decimal(expected)


def test_http_success_and_trace_id(batches, client, monkeypatch):
    report = prepare_historical_nav_dataset(request_for(batches), batches).report
    monkeypatch.setattr(api, "evaluate_stored_historical_nav_batches", lambda _: report)
    response = client.post(
        PATH,
        json=request_for(batches).model_dump(mode="json", by_alias=True),
        headers=HEADERS | {"X-Trace-Id": "evaluation-test"},
    )
    assert response.status_code == 200 and response.json() == report.model_dump(mode="json")
    assert response.headers["X-Trace-Id"] == "evaluation-test"


@pytest.mark.parametrize("headers", [{}, {"X-Service-Token": "wrong"}, HEADERS | {"Origin": "http://localhost"}])
def test_auth_rejected_before_io(batches, client, monkeypatch, headers):
    monkeypatch.setattr(api, "evaluate_stored_historical_nav_batches", lambda _: pytest.fail("must not read data"))
    response = client.post(PATH, json=request_for(batches).model_dump(mode="json", by_alias=True), headers=headers)
    assert response.status_code == 403


@pytest.mark.parametrize(
    "update",
    [
        {"batchIds": []},
        {"batchIds": ["not-a-uuid"]},
        {"batchIds": [str(UUID(int=1))] * 2},
        {"batchIds": [str(UUID(int=i)) for i in range(513)]},
        {"trainStartDate": "2030-01-01"},
        {"validationEndDate": "2022-01-01"},
        {"testEndDate": "2022-01-01"},
        {"previewSize": True},
        {"previewSize": 1.5},
        {"previewSize": 21},
        {"previewSize": -1},
        {"userId": "1"},
        {"minimumSamples": 1},
        {"includeTest": True},
        {"items": []},
    ],
)
def test_invalid_parameters_do_not_read_database(batches, client, monkeypatch, update):
    monkeypatch.setattr(api, "evaluate_stored_historical_nav_batches", lambda _: pytest.fail("must not read data"))
    body = request_for(batches).model_dump(mode="json", by_alias=True) | update
    assert client.post(PATH, json=body, headers=HEADERS).status_code == 422


@pytest.mark.parametrize(
    "error,status,code",
    [
        (HistoricalNavEvaluationError("CONFLICTING_SAMPLES", "conflict"), 409, "CONFLICTING_SAMPLES"),
        (HistoricalNavEvaluationError("BATCH_NOT_FOUND", "missing", 404), 404, "BATCH_NOT_FOUND"),
        (HistoricalNavStorageError("STORED_BATCH_INCONSISTENT", "invalid", 503), 503, "STORED_BATCH_INCONSISTENT"),
        (SQLAlchemyError("private database detail"), 503, "EVALUATION_UNAVAILABLE"),
    ],
)
def test_safe_http_errors(batches, client, monkeypatch, error, status, code):
    def fail(_):
        raise error

    monkeypatch.setattr(api, "evaluate_stored_historical_nav_batches", fail)
    response = client.post(PATH, json=request_for(batches).model_dump(mode="json", by_alias=True), headers=HEADERS)
    assert response.status_code == status and response.json()["detail"]["code"] == code
    assert "private database detail" not in response.text
