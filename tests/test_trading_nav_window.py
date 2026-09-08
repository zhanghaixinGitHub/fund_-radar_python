"""官方日历边界、精确窗口、不看未来决定历史、HTTP和只读仓储验收。"""

import json
from collections import Counter
from dataclasses import replace
from datetime import UTC, date, datetime, timedelta
from types import SimpleNamespace
from uuid import UUID

import pytest
from app.api.routes import trading_nav_window as api
from app.repositories.feature_snapshot import FeatureSourceReadiness
from app.repositories.historical_nav import HistoricalNavPreviewReadError
from app.repositories.trading_nav_window import NavDatePoint, read_nav_dates
from app.schemas.trading_nav_window import TradingNavWindowRequest
from app.services import trading_calendar as calendars
from app.services import trading_nav_window as service
from sqlalchemy.exc import SQLAlchemyError
from tests.test_historical_nav_http import HEADERS
from tests.test_historical_nav_http import client as client

PATH = "/internal/v1/features/historical-nav-samples/trading-window-preview"
SOURCE = FeatureSourceReadiness(UUID(int=1), "TEST", UUID(int=2), datetime(2026, 1, 1, tzinfo=UTC))


@pytest.fixture
def calendar():
    return calendars.load_calendar()


def request(day=date(2024, 3, 7)):
    return TradingNavWindowRequest(fund_code="008888", cutoff_date=day)


def points_for(calendar, req):
    start, end = service.window_read_bounds(calendar, req.cutoff_date)
    return tuple(NavDatePoint(d, d + timedelta(days=1)) for d in calendar.sessions if start <= d <= end)


def preview(calendar, req=None, points=None):
    req = req or request()
    return service.build_trading_nav_window(
        req, SOURCE, points if points is not None else points_for(calendar, req), calendar
    )


def test_calendar_official_years_counts_and_pinned_hash(calendar):
    assert dict(Counter(d.year for d in calendar.sessions)) == {2021: 243, 2022: 242, 2023: 242, 2024: 242, 2025: 243}
    assert calendar.content_hash == calendars.EXPECTED_CALENDAR_HASH
    assert all(d.weekday() < 5 for d in calendar.sessions)
    assert all(len(y.sources) == 2 for y in calendar.definition.years)


@pytest.mark.parametrize(
    "day", [date(2024, 2, 9), date(2024, 2, 18), date(2024, 4, 7), date(2023, 10, 7), date(2025, 2, 8)]
)
def test_holiday_and_makeup_workday_are_not_trading_days(calendar, day):
    assert day not in calendar.sessions


def test_spring_festival_and_cutoff_not_counted(calendar):
    assert calendar.future_sessions(date(2024, 2, 8), 1) == (date(2024, 2, 19),)
    assert calendar.future_sessions(date(2024, 2, 10), 1) == (date(2024, 2, 19),)
    assert calendar.future_sessions(date(2025, 8, 8))[-1] == date(2025, 9, 5)
    assert calendar.future_sessions(date(2025, 8, 8))[0] == date(2025, 8, 11)


@pytest.mark.parametrize("day", [date(2020, 12, 31), date(2021, 1, 4), date(2025, 12, 31), date(2026, 1, 1)])
def test_calendar_scope_does_not_fallback_to_weekdays(calendar, day):
    with pytest.raises(calendars.CalendarCoverageError):
        service.window_read_bounds(calendar, day)


@pytest.mark.parametrize("count", [0, -1, 62, True, 1.5])
def test_calendar_count_bounded(calendar, count):
    with pytest.raises(ValueError):
        calendar.future_sessions(date(2024, 3, 7), count)


def test_coverage_rejected_before_opening_database(monkeypatch):
    monkeypatch.setattr(service, "get_nav_preview_engine", lambda: pytest.fail("out-of-coverage DB access"))
    with pytest.raises(calendars.CalendarCoverageError):
        service.preview_trading_nav_window(request(date(2026, 1, 1)))


def test_calendar_file_changed_or_incomplete_rejected(tmp_path, monkeypatch):
    original = calendars.CALENDAR_FILE.read_bytes()
    path = tmp_path / "calendar.json"
    monkeypatch.setattr(calendars, "CALENDAR_FILE", path)
    try:
        definition = json.loads(original)
        definition["years"][3]["closed_ranges"].pop()
        path.write_text(json.dumps(definition), encoding="utf-8")
        calendars.load_calendar.cache_clear()
        with pytest.raises(ValueError, match="new verified version"):
            calendars.load_calendar()
        definition["years"].pop()
        path.write_text(json.dumps(definition), encoding="utf-8")
        with pytest.raises(ValueError):
            calendars.load_calendar()
        path.write_bytes(b"x" * 65537)
        with pytest.raises(ValueError, match="64KiB"):
            calendars.load_calendar()
    finally:
        calendars.load_calendar.cache_clear()


def test_complete_window_separates_nav_day_and_cutoff(calendar):
    result = preview(calendar, request(date(2025, 8, 8)))
    assert result.status == "WINDOW_DATES_COMPLETE"
    assert result.anchor_nav_date == date(2025, 8, 7) and result.anchor_ann_date == date(2025, 8, 8)
    assert result.anchor_lag_sessions == 1 and len(result.history_dates) == 61
    assert len(result.future_dates) == 20 and result.future_end_date == date(2025, 9, 5)
    assert result.anchor_based_20th_trading_date == result.legacy_20th_nav_date_from_anchor == date(2025, 9, 4)
    assert not any(
        (
            result.database_written,
            result.feature_generated,
            result.label_generated,
            result.training_eligible,
            result.nav_values_verified,
        )
    )


def test_non_trading_nav_is_retained_as_diagnostic_not_counted(calendar):
    clean = preview(calendar)
    rows = tuple(
        sorted(
            (*points_for(calendar, request()), NavDatePoint(date(2024, 3, 31), date(2024, 4, 1))),
            key=lambda p: p.nav_date,
        )
    )
    result = preview(calendar, points=rows)
    assert result.future_dates == clean.future_dates and result.history_dates == clean.history_dates
    assert result.ignored_non_trading_nav_dates == (date(2024, 3, 31),)
    assert result.legacy_20th_nav_date_from_anchor < result.anchor_based_20th_trading_date


def test_missing_future_does_not_change_endpoint_or_history(calendar):
    clean = preview(calendar)
    missing_day = clean.future_dates[3]
    rows = tuple(p for p in points_for(calendar, request()) if p.nav_date != missing_day)
    result = preview(calendar, points=rows)
    assert result.status == "FUTURE_DATES_INCOMPLETE"
    assert result.future_end_date == clean.future_end_date and result.history_dates == clean.history_dates
    assert result.anchor_nav_date == clean.anchor_nav_date
    assert result.future_issues[0].nav_date == missing_day and result.future_issues[0].reason == "MISSING_NAV"
    assert result.future_navs_available_at is None


@pytest.mark.parametrize("kind", ["missing", "late", "invalid", "no_ann"])
def test_history_hole_not_filled_by_an_older_row(calendar, kind):
    clean = preview(calendar)
    day = clean.history_dates[4]
    rows = points_for(calendar, request())
    if kind == "missing":
        rows = tuple(p for p in rows if p.nav_date != day)
    else:
        ann = (
            request().cutoff_date + timedelta(days=1)
            if kind == "late"
            else day - timedelta(days=1)
            if kind == "invalid"
            else None
        )
        rows = tuple(replace(p, ann_date=ann) if p.nav_date == day else p for p in rows)
    result = preview(calendar, points=rows)
    assert result.status == "INPUT_DATES_INCOMPLETE"
    assert result.history_dates == clean.history_dates and len(result.history_issues) == 1
    assert result.history_issues[0].nav_date == day


def test_missing_latest_known_anchor_does_not_claim_fresh(calendar):
    clean = preview(calendar)
    rows = tuple(p for p in points_for(calendar, request()) if p.nav_date != clean.anchor_nav_date)
    result = preview(calendar, points=rows)
    assert result.status == "INPUT_DATES_INCOMPLETE" and result.anchor_lag_sessions == 2
    assert result.anchor_issue == "STALE_ANCHOR_OVER_ONE_SESSION"
    assert not result.history_dates and not result.history_issues
    empty = preview(calendar, points=())
    assert empty.anchor_issue == "NO_KNOWN_TRADING_NAV" and len(empty.future_dates) == 20


def test_later_announcement_does_not_become_known_history(calendar):
    clean = preview(calendar)
    req = request()
    rows = tuple(
        replace(p, ann_date=req.cutoff_date + timedelta(days=100)) if p.nav_date > req.cutoff_date else p
        for p in points_for(calendar, req)
    )
    result = preview(calendar, points=rows)
    assert result.history_dates == clean.history_dates and result.anchor_nav_date == clean.anchor_nav_date
    assert result.future_navs_available_at > clean.future_navs_available_at


def test_same_day_announcement_and_weekend_cutoff(calendar):
    req = request(date(2024, 3, 8))
    rows = tuple(replace(p, ann_date=p.nav_date) for p in points_for(calendar, req))
    assert preview(calendar, req, rows).anchor_lag_sessions == 0
    weekend = preview(calendar, request(date(2024, 3, 10)))
    assert weekend.anchor_nav_date == date(2024, 3, 8) and weekend.anchor_lag_sessions == 0
    assert weekend.future_dates[0] == date(2024, 3, 11)


def test_next_year_calendar_not_silently_called_known_at_cutoff(calendar):
    result = preview(calendar, request(date(2023, 12, 8)))
    assert result.future_end_date.year == 2024
    assert result.calendar.future_schedule_known_at_cutoff is False


@pytest.mark.parametrize("change", ["duplicate", "unordered", "out_of_range", "oversized"])
def test_invalid_raw_date_sequence_rejected(calendar, change):
    rows = points_for(calendar, request())
    rows = (
        rows + (rows[-1],)
        if change == "duplicate"
        else tuple(reversed(rows))
        if change == "unordered"
        else (NavDatePoint(date(2020, 1, 1), None), *rows)
        if change == "out_of_range"
        else rows * 4
    )
    with pytest.raises(ValueError):
        preview(calendar, points=rows)


def test_repository_selects_only_dates_with_source_bounds():
    statements = []

    def execute(statement):
        statements.append(statement)
        return SimpleNamespace(all=lambda: [(date(2024, 3, 1), date(2024, 3, 4))])

    rows = read_nav_dates(
        SimpleNamespace(execute=execute),
        fund_code="008888",
        source_id=UUID(int=1),
        start=date(2024, 1, 1),
        end=date(2024, 5, 1),
    )
    assert len(rows) == 1
    query = statements[0]
    assert tuple(c.name for c in query.selected_columns) == ("nav_date", "ann_date")
    assert UUID(int=1) in query.compile().params.values() and 193 in query.compile().params.values()
    with pytest.raises(ValueError):
        read_nav_dates(None, fund_code="008888", source_id=UUID(int=1), start=date(2023, 1, 1), end=date(2024, 5, 1))


def test_http_complete_and_no_extra_parameters(calendar, client, monkeypatch):
    monkeypatch.setattr(api, "preview_trading_nav_window", lambda req: preview(calendar, req))
    response = client.get(
        PATH,
        params={"fundCode": "008888", "cutoffDate": "2025-08-08"},
        headers={**HEADERS, "X-Trace-Id": "calendar-preview-test"},
    )
    assert response.status_code == 200 and response.json()["future_end_date"] == "2025-09-05"
    assert response.headers["X-Trace-Id"] == "calendar-preview-test"


@pytest.mark.parametrize(
    "extra",
    [
        {"fundCode": "000001"},
        {"cutoffDate": "bad"},
        {"asOfDate": "2025-08-08"},
        {"horizon": 30},
        {"includeTest": True},
        {"calendar": "custom"},
    ],
)
def test_http_invalid_request_no_work(client, monkeypatch, extra):
    monkeypatch.setattr(api, "preview_trading_nav_window", lambda req: pytest.fail("invalid request must not query"))
    assert (
        client.get(
            PATH, params={"fundCode": "008888", "cutoffDate": "2025-08-08", **extra}, headers=HEADERS
        ).status_code
        == 422
    )


@pytest.mark.parametrize("headers", [{}, {"X-Service-Token": "wrong"}, {**HEADERS, "Origin": "http://localhost"}])
def test_http_auth_before_work(client, monkeypatch, headers):
    monkeypatch.setattr(api, "preview_trading_nav_window", lambda req: pytest.fail("unauthorized work"))
    assert (
        client.get(PATH, params={"fundCode": "008888", "cutoffDate": "2025-08-08"}, headers=headers).status_code == 403
    )


@pytest.mark.parametrize(
    "error,status",
    [
        (calendars.CalendarCoverageError("insufficient"), 409),
        (HistoricalNavPreviewReadError("FUND_NOT_FOUND", "missing"), 404),
        (HistoricalNavPreviewReadError("SOURCE_NOT_READY", "not ready"), 409),
        (SQLAlchemyError("private SQL connection"), 503),
        (ValueError("private calendar internals"), 503),
        (TimeoutError("private timeout"), 503),
    ],
)
def test_http_error_mapping_and_redaction(client, monkeypatch, error, status):
    def fail(req):
        raise error

    monkeypatch.setattr(api, "preview_trading_nav_window", fail)
    response = client.get(PATH, params={"fundCode": "008888", "cutoffDate": "2025-08-08"}, headers=HEADERS)
    assert response.status_code == status and "private" not in response.text
