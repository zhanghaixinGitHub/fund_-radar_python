"""手算日期、泄漏、失败与校准边界；不联网、不加载真实权重。"""

from dataclasses import replace
from datetime import date, timedelta
from decimal import Decimal
from types import SimpleNamespace
from uuid import UUID

import pytest
from app.schemas.model_comparison import ComparisonInput
from app.services.model_comparison_adapters import (
    ChronosAdapter,
    calibrated_chronos_score,
    chronos_score,
    fit_chronos_calibrator,
)
from app.services.model_comparison_artifacts import fingerprint, verify_files, write_json
from app.services.model_comparison_dataset import from_sample, validate_input
from app.services.model_comparison_protocol import settings
from pydantic import ValidationError
from tests.test_cash_reinvestment_samples import preview, request, rows_for


def make_input(lag=1, day=date(2023, 6, 20)):
    req = request(day)
    rows = rows_for(req)
    if lag == 0:
        rows = tuple(replace(r, ann_date=r.nav_date) for r in rows)
    sample = preview(rows=rows, req=req)
    return from_sample(sample, UUID(int=1))


@pytest.mark.parametrize("lag", (0, 1))
@pytest.mark.parametrize("day", (date(2023, 6, 20), date(2023, 9, 28), date(2023, 12, 15)))
def test_calendar_and_forecast_base_are_correct_across_holidays_and_year(lag, day):
    item = make_input(lag, day)
    last = float(item.history_values[-1])
    quantiles = [[last - 2, last, last + 2] for _ in range(20 + lag)]
    quantiles[-1] = [last, last + 2, last + 4]
    if lag:
        quantiles[0] = [last - 1, last + 1, last + 3]
    result = chronos_score(item, quantiles)
    assert result["raw_score"] == pytest.approx(0.25 if lag else 0.5)
    assert result["horizon"] == 20 + lag
    assert result["base_reference"] == pytest.approx(last + lag)
    with pytest.raises(ValueError, match="SHAPE"):
        chronos_score(item, quantiles[:-1])


def test_changing_future_nav_changes_answer_but_never_input_or_raw_score():
    req = request()
    original = preview(req=req)
    changed = tuple(replace(r, unit_nav=r.unit_nav * 2) if r.nav_date > req.cutoff_date else r for r in rows_for(req))
    future = preview(rows=changed, req=req)
    a, b = from_sample(original, UUID(int=1)), from_sample(future, UUID(int=1))
    assert original.label_hash != future.label_hash
    assert a == b
    quantiles = [[99.0, 100.0, 101.0]] * 21
    assert chronos_score(a, quantiles) == chronos_score(b, quantiles)


def test_input_rejects_answers_future_values_corruption_and_protected_period():
    item = make_input()
    with pytest.raises(ValidationError):
        ComparisonInput.model_validate({**item.model_dump(), "y": 1})
    with pytest.raises(ValidationError, match="not yet available"):
        ComparisonInput.model_validate({**item.model_dump(), "history_available_at": [date(2024, 1, 1)] * 61})
    with pytest.raises(ValidationError):
        ComparisonInput.model_validate({**item.model_dump(), "cutoff_date": date(2025, 1, 1)})
    with pytest.raises(ValueError, match="input hash"):
        validate_input(item.model_copy(update={"x": (Decimal(0),) * 7}))
    # Rehashing cannot make a shifted horizon valid.
    raw = item.model_dump(mode="json")
    raw["label_end_date"] = str(item.label_end_date + timedelta(days=1))
    raw["input_hash"] = fingerprint({k: v for k, v in raw.items() if k != "input_hash"})
    with pytest.raises(ValueError, match="alignment"):
        validate_input(ComparisonInput.model_validate(raw))


@pytest.mark.parametrize(
    "row,reason",
    (
        ([100.0, 99.0, 101.0], "CROSSING"),
        ([99.0, float("nan"), 101.0], "NONFINITE"),
        ([-2.0, -1.0, 1.0], "INVALID_FORECAST"),
    ),
)
def test_invalid_quantiles_fail_instead_of_fifty_percent(row, reason):
    with pytest.raises(ValueError, match=reason):
        chronos_score(make_input(), [row] * 21)


def calibration_rows():
    from app.services.historical_nav_evaluation import PreparedRow

    rows = []
    for fund in settings()["funds"]:
        for n in range(60):
            rows.append(
                PreparedRow(
                    UUID(int=1),
                    fund,
                    date(2023, 7, 3),
                    date(2023, 7, 3),
                    date(2023, 8, 1),
                    (Decimal(0),) * 7,
                    n % 2,
                    f"{fund}-{n}",
                )
            )
    return tuple(rows)


def test_scalar_calibration_replays_and_is_bound_to_time_and_checkpoint():
    rows = calibration_rows()
    values = [2.0 * r.y - 1 for r in rows]
    model = fit_chronos_calibrator(rows, values, lower=date(2023, 6, 30), upper=date(2023, 12, 31), base_hash="a" * 64)
    assert calibrated_chronos_score(model, 1.0, "a" * 64) > 0.5
    assert calibrated_chronos_score(model, -1.0, "a" * 64) < 0.5
    with pytest.raises(ValueError, match="ARTIFACT_MISMATCH"):
        calibrated_chronos_score(model, 1.0, "b" * 64)
    wrong = (replace(rows[0], label_available_at=date(2024, 1, 1)), *rows[1:])
    with pytest.raises(ValueError, match="TIME_BOUNDARY"):
        fit_chronos_calibrator(wrong, values, lower=date(2023, 6, 30), upper=date(2023, 12, 31), base_hash="a" * 64)
    with pytest.raises(ValueError, match="SINGLE_CLASS"):
        fit_chronos_calibrator(
            tuple(replace(r, y=1) for r in rows),
            values,
            lower=date(2023, 6, 30),
            upper=date(2023, 12, 31),
            base_hash="a" * 64,
        )
    negative = fit_chronos_calibrator(
        rows, [-v for v in values], lower=date(2023, 6, 30), upper=date(2023, 12, 31), base_hash="a" * 64
    )
    with pytest.raises(ValueError, match="NONPOSITIVE"):
        calibrated_chronos_score(negative, 1.0, "a" * 64)


def test_worker_timeout_closes_only_its_owned_process():
    adapter = ChronosAdapter.__new__(ChronosAdapter)
    closed = []
    adapter.connection = SimpleNamespace(poll=lambda seconds: False)
    adapter.close = lambda: closed.append(True)
    with pytest.raises(TimeoutError, match="CHRONOS_TIMEOUT"):
        adapter._receive(0.001)
    assert closed == [True]


def test_artifacts_cannot_be_overwritten_or_escape_the_run(tmp_path):
    path = tmp_path / "result.json"
    write_json(path, {"a": 1})
    with pytest.raises(FileExistsError):
        write_json(path, {"a": 2})
    with pytest.raises(ValueError, match="integrity"):
        verify_files(tmp_path, {"../result.json": "0" * 64})
