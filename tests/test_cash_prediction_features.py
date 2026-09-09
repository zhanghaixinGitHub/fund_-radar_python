"""训练/预测输入一致、无未来答案依赖、只读窗口与2025保护；全部使用人工数据。"""

from dataclasses import replace
from datetime import date, datetime, timedelta
from decimal import Decimal
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pytest
from app.repositories import cash_prediction_features as repository
from app.repositories.cash_reinvestment_samples import CashNavPoint
from app.schemas.cash_prediction_features import CashPredictionFeatureRequest
from app.schemas.cash_reinvestment_samples import CashSampleRequest
from app.services import cash_prediction_features as service
from app.services.cash_reinvestment_samples import build_cash_reinvestment_sample
from app.services.trading_calendar import (
    CalendarCoverageError,
    TradingCalendar,
    load_calendar,
    load_prediction_calendar,
)
from sqlalchemy.dialects import postgresql
from tests.test_cash_reinvestment_samples import dividend, rows_for
from tests.test_trading_nav_window import SOURCE as DATE_SOURCE

SOURCE = replace(DATE_SOURCE, source_code="TUSHARE_PRO_FUND")


def request(day=date(2023, 6, 20)):
    return CashPredictionFeatureRequest(fundCode="006730", cutoffDate=day)


def history_rows(req=None):
    req = req or request()
    calendar = load_prediction_calendar(req.cutoff_date)
    start, end = service.cash_history_bounds(calendar, req)
    return tuple(
        CashNavPoint(day, day + timedelta(days=1), Decimal(10) + Decimal(index) / 100)
        for index, day in enumerate(d for d in calendar.sessions if start <= d <= end)
    )


def build(nav=None, events=(), req=None):
    req = req or request()
    return service.build_cash_prediction_feature(
        req, SOURCE, nav if nav is not None else history_rows(req), events, load_calendar()
    )


@pytest.mark.parametrize("day", [date(2022, 6, 20), date(2023, 6, 20), date(2023, 12, 5), date(2024, 11, 1)])
@pytest.mark.parametrize("with_dividend", [False, True])
def test_input_payload_and_hash_match_historical_sample(day, with_dividend):
    sample_request = CashSampleRequest(fundCode="006730", cutoffDate=day)
    all_nav = rows_for(sample_request)
    events = (dividend(all_nav[20].nav_date),) if with_dividend else ()
    old = build_cash_reinvestment_sample(sample_request, SOURCE, all_nav, events, load_calendar())
    actual = build(tuple(p for p in all_nav if p.nav_date <= day), events, request(day))
    assert actual.feature_payload == old.feature_payload and actual.feature_hash == old.feature_hash
    assert actual.history_dates == old.history_dates and actual.input_issues == old.input_issues
    assert "offline_label" not in actual.model_dump() and "future_dates" not in actual.model_dump()


def test_last_2024_day_needs_no_future_calendar_or_answers(monkeypatch):
    req = request(date(2024, 12, 31))
    nav = history_rows(req)
    monkeypatch.setattr(TradingCalendar, "future_sessions", lambda *a, **k: pytest.fail("must not plan future answers"))
    actual = build(nav, req=req)
    assert actual.status == "INPUT_READY" and actual.anchor_nav_date == date(2024, 12, 30)
    assert len(actual.history_dates) == 61 and max(actual.history_dates) < req.cutoff_date
    assert not actual.database_written and not actual.training_eligible


@pytest.mark.parametrize("mutation", ["missing", "price", "nan", "late_ann"])
def test_unannounced_cutoff_price_not_used(mutation):
    rows = history_rows()
    old = build(rows)
    day = request().cutoff_date
    if mutation == "missing":
        rows = tuple(p for p in rows if p.nav_date != day)
    else:
        updates = {
            "price": {"unit_nav": Decimal(500)},
            "nan": {"unit_nav": Decimal("NaN")},
            "late_ann": {"ann_date": day + timedelta(days=90)},
        }[mutation]
        rows = tuple(replace(p, **updates) if p.nav_date == day else p for p in rows)
    actual = build(rows)
    assert actual.feature_payload == old.feature_payload and actual.feature_hash == old.feature_hash


def test_future_announced_dividend_does_not_rewrite_history():
    old = build()
    event = dividend(
        old.history_dates[20], ann_date=request().cutoff_date + timedelta(days=1), cash_dividend=Decimal(500)
    )
    assert build(events=(event,)).feature_hash == old.feature_hash


@pytest.mark.parametrize("problem", ["missing", "missing_ann", "before_ann", "late_ann", "zero", "flat", "stale"])
def test_bad_history_has_no_input_and_matches_sample(problem):
    sample_request = CashSampleRequest(fundCode="006730", cutoffDate=request().cutoff_date)
    rows = rows_for(sample_request)
    target = rows[20].nav_date
    if problem == "missing":
        rows = tuple(p for p in rows if p.nav_date != target)
    elif problem == "flat":
        rows = tuple(replace(p, unit_nav=Decimal(1)) for p in rows)
    elif problem == "stale":
        rows = tuple(replace(p, ann_date=date(2024, 1, 1)) if p.nav_date >= date(2023, 6, 19) else p for p in rows)
    else:
        changes = {
            "missing_ann": {"ann_date": None},
            "before_ann": {"ann_date": target - timedelta(days=1)},
            "late_ann": {"ann_date": date(2024, 1, 1)},
            "zero": {"unit_nav": Decimal(0)},
        }[problem]
        rows = tuple(replace(p, **changes) if p.nav_date == target else p for p in rows)
    old = build_cash_reinvestment_sample(sample_request, SOURCE, rows, (), load_calendar())
    actual = build(tuple(p for p in rows if p.nav_date <= request().cutoff_date))
    assert actual.status == "DATA_INSUFFICIENT" and actual.feature_payload is None and actual.feature_hash is None
    assert actual.input_issues == old.input_issues and actual.history_dates == old.history_dates


@pytest.mark.parametrize("problem", ["duplicate", "reversed", "future", "oversized"])
def test_input_shape_not_silently_trimmed(problem):
    nav = history_rows()
    if problem == "duplicate":
        nav = (nav[0], *nav)
    elif problem == "reversed":
        nav = tuple(reversed(nav))
    elif problem == "future":
        nav = (
            *nav,
            CashNavPoint(
                request().cutoff_date + timedelta(days=1), request().cutoff_date + timedelta(days=2), Decimal(12)
            ),
        )
    else:
        nav = nav * 4
    with pytest.raises(ValueError):
        build(nav)


class FakeQuerySession:
    def __init__(self, nav=(), events=()):
        self.results = iter((nav, events))
        self.statements = []

    def execute(self, statement):
        self.statements.append(statement.compile(dialect=postgresql.dialect()))
        return SimpleNamespace(all=lambda: next(self.results))


def read(session, *, start=date(2023, 3, 1), cutoff=date(2023, 6, 20)):
    return repository.read_cash_history_inputs(
        session, fund_code="006730", source_id=SOURCE.source_id, start=start, cutoff=cutoff
    )


def test_sql_masks_unannounced_prices_and_limits_known_events():
    session = FakeQuerySession(nav=((date(2023, 6, 20), date(2023, 6, 21), None),))
    nav, events = read(session)
    assert nav[0].unit_nav is None and not events and len(session.statements) == 2
    sql = [str(s) for s in session.statements]
    assert "CASE WHEN" in sql[0] and "ann_date <=" in sql[0]
    assert session.statements[0].statement.selected_columns[2].else_ is None  # SQL省略ELSE等价于ELSE NULL。
    assert "nav_date <=" in sql[0] and 193 in session.statements[0].params.values()
    assert "implementation_ann_date <=" in sql[1] and 101 in session.statements[1].params.values()
    assert all(
        not any(name in s for name in ("accumulated_nav", "adj_nav", "cash_sample", "forecast_result")) for s in sql
    )


@pytest.mark.parametrize(
    "start,cutoff",
    [
        (date(2024, 12, 1), date(2025, 1, 1)),
        (date(2025, 11, 1), date(2026, 1, 2)),
        (date(2023, 1, 1), date(2023, 12, 1)),
        (date(2023, 6, 21), date(2023, 6, 20)),
    ],
)
def test_read_bounds_reject_before_sql(start, cutoff):
    session = FakeQuerySession()
    with pytest.raises(ValueError):
        read(session, start=start, cutoff=cutoff)
    assert not session.statements


@pytest.mark.parametrize(
    "problem",
    ["hidden_price", "missing_ann_price", "duplicate_nav", "late_event", "duplicate_event", "too_many_events"],
)
def test_repository_rechecks_returned_rows(problem):
    nav, events = (), ()
    if problem == "hidden_price":
        nav = ((date(2023, 6, 20), date(2023, 6, 21), Decimal(1)),)
    elif problem == "missing_ann_price":
        nav = ((date(2023, 6, 20), None, Decimal(1)),)
    elif problem == "duplicate_nav":
        nav = ((date(2023, 6, 19), date(2023, 6, 20), Decimal(1)),) * 2
    else:
        event = ("event", date(2023, 6, 1), None, date(2023, 6, 19), date(2023, 6, 19), Decimal(1), "实施")
        events = (
            ((event[0], date(2023, 6, 21), *event[2:]),)
            if problem == "late_event"
            else (event,) * (101 if problem == "too_many_events" else 2)
        )
    with pytest.raises(ValueError):
        read(FakeQuerySession(nav, events))


@pytest.mark.parametrize("day", [date(2025, 6, 20), date(2026, 1, 8)])
def test_incomplete_history_or_heldout_year_never_opens_database(monkeypatch, day):
    monkeypatch.setattr(service, "get_nav_preview_engine", lambda: pytest.fail("must not connect"))
    with pytest.raises((ValueError, CalendarCoverageError)):
        service.read_cash_prediction_feature(request(day))


def test_not_finished_cutoff_day_rejected(monkeypatch):
    monkeypatch.setattr(service, "get_nav_preview_engine", lambda: pytest.fail("must not connect"))
    with pytest.raises(service.HistoricalNavPreviewReadError) as error:
        service.read_cash_prediction_feature(request(datetime.now(ZoneInfo("Asia/Shanghai")).date()))
    assert error.value.code == "CUTOFF_DAY_NOT_CLOSED"


@pytest.mark.parametrize("day", [date(2023, 6, 20), date(2026, 9, 8)])
def test_read_adapter_uses_consistent_readonly_session(monkeypatch, day):
    sql = []
    req = request(day)

    class Session:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def begin(self):
            return self

        def execute(self, statement):
            sql.append(str(statement))

    monkeypatch.setattr(service, "get_nav_preview_engine", lambda: None)
    monkeypatch.setattr(service, "Session", lambda _: Session())
    monkeypatch.setattr(service, "read_historical_nav_source", lambda s, **kw: SOURCE)

    def inputs(s, **kw):
        assert kw["cutoff"] == day and kw["start"] < kw["cutoff"]
        assert kw["start"].year != 2025
        return history_rows(req), ()

    monkeypatch.setattr(service, "read_cash_history_inputs", inputs)
    result = service.read_cash_prediction_feature(req)
    assert result.status == "INPUT_READY" and sql == ["SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY"]
