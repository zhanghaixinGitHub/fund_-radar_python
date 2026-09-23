"""P0-01 日期与总回报契约；测试日历是夹具，不作为真实日历覆盖证明。"""

from datetime import date, datetime
from decimal import Decimal

import pytest
from app.services.prediction_contract import (
    PredictionFailure,
    ValuationCalendar,
    add_months,
    reinvested_series,
    target_dates,
)
from app.services.trading_calendar import load_current_calendar


def calendar():
    source = load_current_calendar()
    return ValuationCalendar(
        "CN_TEST",
        source.sessions,
        source.definition.coverage_start,
        source.definition.coverage_end,
        source.content_hash,
    )


def test_current_cutoff_and_holiday():
    before = target_dates(calendar(), datetime.fromisoformat("2026-09-22T14:59:59+08:00"), "T5_V1")
    after = target_dates(calendar(), datetime.fromisoformat("2026-09-22T15:00:00+08:00"), "T5_V1")
    assert before["startDate"] == "2026-09-22"
    assert after["startDate"] == "2026-09-23"
    assert before["endDate"] == "2026-09-30"
    holiday = target_dates(calendar(), datetime.fromisoformat("2026-10-01T10:00:00+08:00"), "T5_V1")
    assert holiday["startDate"] == "2026-10-08"


def test_months_keep_nominal_when_official_year_missing():
    value = target_dates(calendar(), datetime.fromisoformat("2026-09-22T10:00:00+08:00"), "M6_V1")
    assert value["nominalEndDate"] == "2027-03-22"
    assert value["endDate"] is None
    assert value["endDateStatus"] == "PENDING_OFFICIAL_CALENDAR"
    assert add_months(date(2024, 8, 31), 6) == date(2025, 2, 28)


def test_missing_current_date_is_failure_not_pending():
    with pytest.raises(PredictionFailure) as error:
        target_dates(calendar(), datetime.fromisoformat("2027-01-04T10:00:00+08:00"), "M6_V1")
    assert error.value.payload["code"] == "CALENDAR_RANGE_MISSING"


def test_independent_targets_and_timezone():
    now = datetime.fromisoformat("2026-09-22T02:00:00+00:00")
    assert target_dates(calendar(), now, "T5_V1")["endDate"] != target_dates(calendar(), now, "T20_V1")["endDate"]
    with pytest.raises(ValueError, match="TIMEZONE"):
        target_dates(calendar(), datetime(2026, 9, 22), "T5_V1")


def test_cash_reinvestment_not_false_drop():
    dates = [date(2026, 9, 21), date(2026, 9, 22), date(2026, 9, 23)]
    nav = dict(zip(dates, map(Decimal, ["1", "0.9", "0.99"]), strict=True))
    values = reinvested_series(dates, nav, {dates[1]: Decimal("0.1")})
    assert values[-1] == Decimal("1.1")
    with pytest.raises(PredictionFailure) as error:
        reinvested_series(dates, {dates[0]: Decimal(1)}, {})
    assert error.value.payload["details"]["missingDates"] == ["2026-09-22", "2026-09-23"]
