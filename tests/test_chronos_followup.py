"""新年份时点边界、不可提前读答案和固定负斜率诊断；不读取真实数据库。"""

from dataclasses import replace
from datetime import date, timedelta
from decimal import Decimal
from uuid import UUID

import pytest
from app.repositories.cash_reinvestment_samples import CashNavPoint
from app.schemas.cash_prediction_features import CashPredictionFeatureRequest
from app.schemas.model_comparison import ComparisonInput
from app.services.cash_prediction_features import build_cash_prediction_feature
from app.services.chronos_followup import signed_diagnostic_probability
from app.services.chronos_followup_data import (
    FollowupInput,
    answer_from_series,
    export_followup_answers,
    feature_to_input,
    load_followup_inputs,
)
from app.services.chronos_followup_diagnosis import score_relationship
from app.services.model_comparison_adapters import calibrated_chronos_score, chronos_score
from app.services.model_comparison_artifacts import fingerprint, write_jsonl
from app.services.trading_calendar import load_current_calendar
from pydantic import ValidationError
from tests.test_trading_nav_window import SOURCE


def sample(lag=0):
    calendar = load_current_calendar()
    cutoff = date(2026, 5, 12)
    index = calendar.at_or_before_index(cutoff)
    days = calendar.sessions[index - 61 : index + 21]
    rows = tuple(CashNavPoint(d, d + timedelta(days=lag), Decimal(10) + Decimal(n) / 100) for n, d in enumerate(days))
    feature = build_cash_prediction_feature(
        CashPredictionFeatureRequest(fundCode="006730", cutoffDate=cutoff),
        replace(SOURCE, source_code="TUSHARE_PRO_FUND"),
        tuple(r for r in rows if r.nav_date <= cutoff),
        (),
        calendar,
    )
    return feature_to_input(feature, UUID(int=1)), rows


@pytest.mark.parametrize("lag", (0, 1))
def test_new_period_has_exact_same_horizon_and_cash_label(lag):
    item, rows = sample(lag)
    assert item.anchor_lag_sessions == lag
    result = chronos_score(item, [[100.0, 101.0, 102.0]] * (20 + lag))
    assert result["horizon"] == 20 + lag
    answer = answer_from_series(item, rows, (), available_by=date(2026, 9, 8))
    nav = {r.nav_date: r.unit_nav for r in rows}
    expected = (nav[item.label_end_date] / nav[item.label_base_date] - 1).quantize(Decimal(".000000000001"))
    assert Decimal(answer["future_return_20d"]) == expected
    assert answer["y"] == 1
    assert len(answer["label_series"]) == 21


def test_old_scope_stays_protected_new_input_never_accepts_answers_or_bad_dates():
    item, _ = sample()
    with pytest.raises(ValidationError):
        ComparisonInput.model_validate(item.model_dump())
    with pytest.raises(ValidationError):
        FollowupInput.model_validate({**item.model_dump(), "y": 1})
    with pytest.raises(ValidationError):
        FollowupInput.model_validate({**item.model_dump(), "cutoff_date": date(2025, 12, 31)})
    with pytest.raises(ValidationError, match="unavailable"):
        FollowupInput.model_validate({**item.model_dump(), "history_available_at": [date(2026, 5, 13)] * 61})
    bad = item.model_dump(mode="json")
    bad["label_end_date"] = "2026-06-11"
    bad["input_hash"] = fingerprint({k: v for k, v in bad.items() if k != "input_hash"})
    with pytest.raises(ValidationError, match="alignment"):
        FollowupInput.model_validate(bad)


def test_answer_changes_do_not_modify_sealed_input_and_late_answers_are_missing():
    from dataclasses import replace

    item, rows = sample()
    before = item.model_dump_json()
    changed = tuple(replace(r, unit_nav=r.unit_nav / 2) if r.nav_date == item.label_end_date else r for r in rows)
    assert answer_from_series(item, rows, (), available_by=date(2026, 9, 8))["y"] == 1
    assert answer_from_series(item, changed, (), available_by=date(2026, 9, 8))["y"] == 0
    assert item.model_dump_json() == before
    assert answer_from_series(item, rows, (), available_by=item.cutoff_date)["reason"] == "ANSWER_NOT_MATURE"


def test_sigmoid_diagnostic_accepts_signed_map_without_changing_old_gate():
    assert signed_diagnostic_probability(-2.0, 0.0, 1.0) < 0.5
    assert signed_diagnostic_probability(-2.0, 0.0, -1.0) > 0.5
    model = {"version": "CHRONOS2_LOCAL_SIGMOID_V1", "base_model_hash": "a" * 64, "slope": -2.0, "intercept": 0.0}
    model["model_hash"] = fingerprint(model)
    with pytest.raises(ValueError, match="NONPOSITIVE"):
        calibrated_chronos_score(model, 1.0, "a" * 64)
    with pytest.raises(ValueError, match="nonfinite"):
        signed_diagnostic_probability(1.0, 0.0, float("nan"))


def test_low_up_rate_does_not_itself_explain_negative_slope():
    labels = [0] * 8 + [1] * 2
    groups = ["006730"] * 10
    increasing = score_relationship(labels, list(range(10)), groups)
    decreasing = score_relationship(labels, list(range(9, -1, -1)), groups)
    assert increasing["up_rate"] == decreasing["up_rate"] == 0.2
    assert increasing["weighted_covariance"] > 0 > decreasing["weighted_covariance"]
    assert increasing["auc"] == 1.0 and decreasing["auc"] == 0.0
    assert score_relationship([0, 0], [1.0, 2.0], ["006730"] * 2)["auc"] is None


def test_read_answers_requires_prior_receipt_before_opening_database(tmp_path, monkeypatch):
    from app.services import chronos_followup_data as data

    opened = []
    monkeypatch.setattr(data, "local_engine", lambda: opened.append(True))
    with pytest.raises(FileNotFoundError):
        export_followup_answers(tmp_path, {}, {})
    assert not opened


def test_zero_hash_input_cannot_be_reloaded(tmp_path):
    item, _ = sample()
    raw = item.model_dump(mode="json")
    raw["input_hash"] = "0" * 64
    write_jsonl(tmp_path / "inputs.jsonl", [raw])
    with pytest.raises(ValueError, match="unsealed"):
        load_followup_inputs(tmp_path)
