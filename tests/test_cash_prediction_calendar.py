"""2026官方交易日、版本隔离、无未来答案的当前输入；只使用人工数据。"""

import json
from datetime import date, timedelta
from decimal import Decimal

import pytest
from app.repositories.cash_reinvestment_samples import CashNavPoint
from app.services import cash_prediction_features as features
from app.services import trading_calendar as calendars
from tests.test_cash_prediction_features import SOURCE, request


def test_2026_fixed_version_and_sources_do_not_replace_research_calendar():
    current = calendars.load_prediction_calendar(date(2026, 9, 8))
    old = calendars.load_prediction_calendar(date(2024, 12, 31))
    assert current.definition.version == "CN_A_SHARE_2026_V1"
    assert current.content_hash == calendars.EXPECTED_CURRENT_CALENDAR_HASH
    assert len(current.sessions) == 242 and {d.year for d in current.sessions} == {2026}
    assert current.sessions[0] == date(2026, 1, 5) and current.sessions[-1] == date(2026, 12, 31)
    assert len(current.definition.years[0].sources) == 2
    assert current.definition.years[0].announced_on == date(2025, 12, 22)
    assert old is calendars.load_calendar() and len(old.sessions) == 1212
    assert old.content_hash == "a7258f7368a071b9fd1df2b2ac039a60bb2be0b4e5c9bde51e836c76ca3c75fa"


@pytest.mark.parametrize(
    "day",
    [
        date(2026, 1, 2),
        date(2026, 1, 4),
        date(2026, 2, 14),
        date(2026, 2, 16),
        date(2026, 2, 23),
        date(2026, 2, 28),
        date(2026, 4, 6),
        date(2026, 5, 5),
        date(2026, 5, 9),
        date(2026, 6, 19),
        date(2026, 9, 20),
        date(2026, 9, 25),
        date(2026, 10, 7),
        date(2026, 10, 10),
    ],
)
def test_holidays_and_makeup_weekends_remain_closed(day):
    assert day not in calendars.load_current_calendar().sessions


def test_count_after_cutoff_skips_mid_autumn_and_national_holidays():
    sessions = calendars.load_current_calendar().future_sessions(date(2026, 9, 8))
    assert len(sessions) == 20 and sessions[0] == date(2026, 9, 9) and sessions[-1] == date(2026, 10, 14)
    assert date(2026, 9, 25) not in sessions and date(2026, 10, 7) not in sessions


@pytest.mark.parametrize("day", [date(2020, 12, 31), date(2027, 1, 1)])
def test_no_unverified_year_fallback(day):
    with pytest.raises(calendars.CalendarCoverageError):
        calendars.load_prediction_calendar(day)


def test_old_calendar_still_rejects_new_year_and_current_one_rejects_2025():
    with pytest.raises(calendars.CalendarCoverageError):
        calendars.load_calendar().at_or_before_index(date(2026, 1, 5))
    with pytest.raises(calendars.CalendarCoverageError):
        calendars.load_current_calendar().at_or_before_index(date(2025, 12, 31))


@pytest.mark.parametrize("day", [date(2026, 1, 5), date(2026, 4, 8)])
def test_early_2026_does_not_borrow_heldout_2025_history(day):
    with pytest.raises(calendars.CalendarCoverageError):
        features.cash_history_bounds(calendars.load_current_calendar(), request(day))


@pytest.mark.parametrize("day", [date(2026, 4, 9), date(2026, 9, 8), date(2026, 12, 31)])
def test_2026_inputs_work_without_future_values_or_calendar(monkeypatch, day):
    calendar = calendars.load_current_calendar()
    req = request(day)
    start, end = features.cash_history_bounds(calendar, req)
    nav = tuple(
        CashNavPoint(d, d + timedelta(days=1), Decimal(10) + Decimal(i) / 100)
        for i, d in enumerate(d for d in calendar.sessions if start <= d <= end)
    )
    monkeypatch.setattr(
        calendars.TradingCalendar, "future_sessions", lambda *a, **k: pytest.fail("must not depend on future")
    )
    actual = features.build_cash_prediction_feature(req, SOURCE, nav, (), calendar)
    assert actual.status == "INPUT_READY" and len(actual.history_dates) == 61
    assert max(actual.history_dates) < day and min(actual.history_dates).year == 2026
    assert actual.feature_payload.calendar_hash == calendars.EXPECTED_CURRENT_CALENDAR_HASH
    assert actual.feature_payload.calendar_version == "CN_A_SHARE_2026_V1"
    assert not actual.database_written and actual.publication_status == "MODEL_NOT_RELEASED"


@pytest.mark.parametrize("problem", ["holiday", "version", "coverage", "notice_year", "oversized"])
def test_changed_calendar_fact_or_version_rejected(tmp_path, monkeypatch, problem):
    definition = json.loads(calendars.CURRENT_CALENDAR_FILE.read_bytes())
    if problem == "holiday":
        definition["years"][0]["closed_ranges"].pop()
    elif problem == "version":
        definition["version"] = "CN_A_SHARE_2021_2025_V1"
    elif problem == "coverage":
        definition["coverage_start"] = "2025-12-01"
    elif problem == "notice_year":
        definition["years"][0]["year"] = 2025
    path = tmp_path / "calendar.json"
    path.write_text(json.dumps(definition) if problem != "oversized" else "x" * 65537, encoding="utf-8")
    monkeypatch.setattr(calendars, "CURRENT_CALENDAR_FILE", path)
    calendars.load_current_calendar.cache_clear()
    try:
        with pytest.raises(ValueError):
            calendars.load_current_calendar()
    finally:
        calendars.load_current_calendar.cache_clear()
