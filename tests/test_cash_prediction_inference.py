"""纯数值分支使用人工训练、人工净值；不冒充真实模型获准发布，也不接触数据库。"""

import math
from datetime import date, timedelta
from decimal import ROUND_HALF_UP, Decimal

import pytest
from app.services import cash_prediction_inference as service
from app.services.cash_prediction_features import build_cash_prediction_feature
from app.services.cash_reinvestment_research import evaluate_cash_dataset
from app.services.cash_reinvestment_samples import _hash
from app.services.historical_nav_calibration import calibrated_model_hash
from app.services.historical_nav_evaluation import FEATURE_NAMES
from app.services.trading_calendar import CalendarCoverageError, load_prediction_calendar
from tests.test_cash_prediction_features import SOURCE, history_rows, request
from tests.test_cash_reinvestment_research import data as data


@pytest.fixture(scope="module")
def model(data):
    # 预先固定使用2024研究窗，不按成绩或基金挑模型。
    artifact = evaluate_cash_dataset(data).windows[2].model
    assert artifact is not None and artifact.calibrator.slope > 0
    return artifact


def feature(day=date(2026, 9, 4)):
    req = request(day)
    return build_cash_prediction_feature(req, SOURCE, history_rows(req), (), load_prediction_calendar(day))


def test_matches_independent_math_and_uses_cutoff_not_anchor_as_target_base(model, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("pure inference must not read the database or fit")

    from app.services import cash_prediction_features, historical_nav_training

    monkeypatch.setattr(cash_prediction_features, "read_cash_prediction_feature", forbidden)
    monkeypatch.setattr(historical_nav_training, "fit_logistic_artifact", forbidden)
    known = feature()
    result = service.calculate_cash_inference(known, model, expected_model_hash=model.model_hash)
    base = model.base_model
    x = [float(known.feature_payload.metrics[key]) for key in FEATURE_NAMES]
    z = base.intercept + sum(
        c * (n - mu) / sd for c, n, mu, sd in zip(base.coefficients, x, base.mean, base.scale, strict=True)
    )
    expected = Decimal(str(1 / (1 + math.exp(-(model.calibrator.slope * z + model.calibrator.intercept)))))
    assert result.up_score == expected.quantize(Decimal("0.00000001"), rounding=ROUND_HALF_UP)
    assert result.anchor_nav_date == date(2026, 9, 3)
    assert result.target_base_date == date(2026, 9, 4)
    assert result.target_end_date == load_prediction_calendar(known.cutoff_date).future_sessions(known.cutoff_date)[-1]
    assert result.predicted_up == (result.up_score > Decimal("0.5"))
    assert not {"up_probability", "release_id", "publication_status", "offline_label"} & result.model_dump().keys()


def test_weekend_cutoff_keeps_exact_twenty_sessions(model):
    known = feature(date(2026, 9, 5))
    result = service.calculate_cash_inference(known, model, expected_model_hash=model.model_hash)
    assert result.anchor_nav_date == result.target_base_date == date(2026, 9, 4)
    calendar = load_prediction_calendar(known.cutoff_date)
    assert (
        calendar.at_or_before_index(result.target_end_date) - calendar.at_or_before_index(result.target_base_date) == 20
    )


@pytest.mark.parametrize(
    "case",
    ["hash", "status", "issue", "stale", "fund", "cutoff", "anchor", "calendar", "dates", "late", "columns", "nan"],
)
def test_invalid_input_rejected_before_numerics(model, monkeypatch, case):
    from app.schemas.cash_reinvestment_samples import CashSampleIssue

    known = feature()
    payload = known.feature_payload
    changes = {}
    if case == "hash":
        changes["feature_hash"] = "f" * 64
    elif case == "status":
        changes["status"] = "DATA_INSUFFICIENT"
    elif case == "issue":
        changes["input_issues"] = (CashSampleIssue(day=None, code="MISSING"),)
    elif case == "stale":
        changes["anchor_lag_sessions"] = 2
    elif case == "fund":
        payload = payload.model_copy(update={"fund_code": "000001"})
    elif case == "cutoff":
        payload = payload.model_copy(update={"cutoff_date": known.cutoff_date + timedelta(days=1)})
    elif case == "anchor":
        changes["anchor_nav_date"] = known.cutoff_date
    elif case == "calendar":
        payload = payload.model_copy(update={"calendar_hash": "e" * 64})
    elif case == "dates":
        changes["history_dates"] = known.history_dates[1:]
    elif case == "late":
        payload = payload.model_copy(update={"available_at": known.cutoff_date + timedelta(days=1)})
    else:
        metrics = dict(payload.metrics)
        metrics["future_answer" if case == "columns" else FEATURE_NAMES[0]] = "NaN" if case == "nan" else "0.1"
        payload = payload.model_copy(update={"metrics": metrics})
    if payload != known.feature_payload:
        changes.update(feature_payload=payload, feature_hash=_hash(payload))
    monkeypatch.setattr(
        service, "predict_calibrated_scores", lambda *a, **k: pytest.fail("invalid input must not infer")
    )
    with pytest.raises(ValueError):
        service.calculate_cash_inference(known.model_copy(update=changes), model, expected_model_hash=model.model_hash)


def test_wrong_model_hash_rejected(model):
    with pytest.raises(ValueError, match="hash"):
        service.calculate_cash_inference(feature(), model, expected_model_hash="0" * 64)


@pytest.mark.parametrize("slope", [0, -1])
def test_nonpositive_calibrator_computes_with_explicit_diagnostic(model, slope):
    changed = model.model_copy(update={"calibrator": model.calibrator.model_copy(update={"slope": slope})})
    changed = changed.model_copy(update={"model_hash": calibrated_model_hash(changed)})
    result = service.calculate_cash_inference(feature(), changed, expected_model_hash=changed.model_hash)
    assert result.calibration.status == ("CONSTANT_PENDING_VALIDATION" if slope == 0 else "REVERSED_PENDING_VALIDATION")
    assert result.calibration.slope == slope
    assert 0 <= result.up_score <= 1
    assert not {"release_id", "publication_allowed", "up_probability"} & result.model_dump().keys()
    if slope == 0:
        expected = Decimal(str(1 / (1 + math.exp(-changed.calibrator.intercept))))
        assert result.up_score == expected.quantize(Decimal("0.00000001"), rounding=ROUND_HALF_UP)


@pytest.mark.parametrize("field", ["slope", "intercept"])
def test_nonfinite_calibration_still_rejected_before_numerics(model, monkeypatch, field):
    changed = model.model_copy(update={"calibrator": model.calibrator.model_copy(update={field: float("nan")})})
    monkeypatch.setattr(service, "predict_calibrated_scores", lambda *a, **k: pytest.fail("must not infer"))
    with pytest.raises(ValueError):
        service.calculate_cash_inference(feature(), changed, expected_model_hash=changed.model_hash)


def test_no_guessing_future_calendar(model):
    with pytest.raises(CalendarCoverageError):
        service.calculate_cash_inference(feature(date(2026, 12, 30)), model, expected_model_hash=model.model_hash)


def test_model_cannot_predict_a_day_before_its_calibration_finished(model):
    with pytest.raises(ValueError, match="time boundary"):
        service.calculate_cash_inference(feature(date(2023, 6, 20)), model, expected_model_hash=model.model_hash)


def test_next_year_calendar_not_yet_public_at_cutoff_is_rejected(model):
    with pytest.raises(ValueError, match="calendar was not available"):
        service.calculate_cash_inference(feature(date(2024, 12, 10)), model, expected_model_hash=model.model_hash)


def test_old_nav_basis_cannot_enter_even_with_internally_consistent_hashes(model):
    from app.services.historical_nav_training import artifact_hash

    base = model.base_model.model_copy(update={"versions": {"feature": "OLD_ACCUMULATED_NAV"}})
    base = base.model_copy(update={"model_hash": artifact_hash(base)})
    changed = model.model_copy(
        update={
            "base_model": base,
            "calibrator": model.calibrator.model_copy(update={"base_model_hash": base.model_hash}),
        }
    )
    changed = changed.model_copy(update={"model_hash": calibrated_model_hash(changed)})
    with pytest.raises(ValueError, match="data versions"):
        service.calculate_cash_inference(feature(), changed, expected_model_hash=changed.model_hash)


@pytest.mark.parametrize("value,up", [(0.5, False), (0.500000001, False), (0.50000001, True), (0, False), (1, True)])
def test_displayed_score_and_direction_share_same_rounding(model, monkeypatch, value, up):
    monkeypatch.setattr(service, "predict_calibrated_scores", lambda *a, **k: (value,))
    assert service.calculate_cash_inference(feature(), model, expected_model_hash=model.model_hash).predicted_up is up
