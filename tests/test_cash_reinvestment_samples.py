"""现金再投公式、时间隔离、固定20段、只读查询及HTTP边界；不写真实库。"""

from dataclasses import replace
from datetime import date, timedelta
from decimal import ROUND_DOWN, Decimal, localcontext
from types import SimpleNamespace
from uuid import UUID

import pytest
from app.api.routes import cash_reinvestment_samples as api
from app.repositories.cash_reinvestment_samples import CashDividend, CashNavPoint, read_cash_sample_inputs
from app.repositories.historical_nav import HistoricalNavPreviewReadError
from app.schemas.cash_reinvestment_samples import CashSampleRequest
from app.services import cash_reinvestment_samples as service
from app.services.trading_calendar import CalendarCoverageError, load_calendar
from sqlalchemy.exc import SQLAlchemyError
from tests.test_historical_nav_http import HEADERS
from tests.test_historical_nav_http import client as client
from tests.test_trading_nav_window import SOURCE

D = Decimal
PATH = "/internal/v1/features/historical-nav-samples/cash-reinvestment-preview"
PARAMS = {"fundCode": "006730", "cutoffDate": "2023-06-20"}


def request(day=date(2023, 6, 20)):
    return CashSampleRequest(fundCode="006730", cutoffDate=day)


def rows_for(req=None):
    req = req or request()
    calendar = load_calendar()
    start, end = service.cash_sample_bounds(calendar, req)
    return tuple(
        CashNavPoint(day, day + timedelta(days=1), D(10) + D(index) / 100)
        for index, day in enumerate(d for d in calendar.sessions if start <= d <= end)
    )


def dividend(day, **changes):
    event = CashDividend("event-1", day - timedelta(days=1), None, day, day, D("0.5"), "实施")
    return replace(event, **changes)


def preview(rows=None, events=(), req=None):
    req = req or request()
    return service.build_cash_reinvestment_sample(
        req, SOURCE, rows if rows is not None else rows_for(req), events, load_calendar()
    )


def test_complete_input_and_answer_are_separate_versions_and_exact_twenty_intervals():
    output = preview()
    assert output.status == "RESEARCH_SAMPLE_READY"
    assert output.anchor_nav_date == date(2023, 6, 19)
    assert output.label_base_date == date(2023, 6, 20) and output.anchor_lag_sessions == 1
    assert len(output.feature_payload.history_series) == 61 and len(output.offline_label.label_series) == 21
    assert output.future_dates[0] == date(2023, 6, 21)
    assert output.future_dates[-1] == output.label_end_date
    assert output.feature_payload.available_at <= output.cutoff_date < output.offline_label.label_available_at
    assert output.feature_payload.feature_version == "CASH_REINVESTMENT_FEATURE_V1"
    assert output.offline_label.label_version == "CASH_REINVESTMENT_FORWARD_20TD_V1"
    assert output.feature_hash != output.label_hash
    assert not any((output.training_eligible, output.database_written, output.model_fitted, output.test_scored))
    nav = {p.nav_date: p.unit_nav for p in rows_for()}
    expected = (nav[output.label_end_date] / nav[output.label_base_date] - 1).quantize(D("0.000000000001"))
    assert D(output.offline_label.future_return_20d) == expected
    feature_json = output.feature_payload.model_dump_json()
    assert "label" not in feature_json and str(output.label_end_date) not in feature_json


def test_cash_reinvests_shares_and_excludes_base_day_cash():
    days = (date(2023, 6, 19), date(2023, 6, 20), date(2023, 6, 21))
    nav = {day: CashNavPoint(day, day, value) for day, value in zip(days, (D(10), D(9), D(10)), strict=True)}
    events = (
        dividend(days[0], cash_dividend=D(100), event_key="base"),
        dividend(days[1], cash_dividend=D(1), event_key="one"),
        dividend(days[2], cash_dividend=D(2), event_key="two"),
    )
    points, issues = service.build_cash_return_series(days, nav, events, latest_available=days[-1])
    assert not issues and points[0].cash_per_share == "0.000000000000"
    assert points[0].daily_return is None and not points[0].dividend_event_keys
    assert points[1].daily_return == "0.000000000000"
    # 10元买1份，第一天分红再投后有10/9份；第二天得到12元/份，总值13.3333而非13。
    assert points[-1].growth_index == "133.333333333333"
    assert points[-1].growth_index != "130.000000000000"


def test_actual_dividend_formula_fixture():
    days = (date(2023, 6, 20), date(2023, 6, 21))
    nav = {day: CashNavPoint(day, day, value) for day, value in zip(days, (D("1.4188"), D("1.2235")), strict=True)}
    points, issues = service.build_cash_return_series(
        days, nav, (dividend(days[-1], cash_dividend=D("0.1676")),), latest_available=days[-1]
    )
    assert not issues and points[-1].daily_return == "-0.019523541021"


@pytest.mark.parametrize("bad", [None, D(0), D(-1), D("NaN"), D("Infinity")])
def test_history_invalid_unit_nav_not_replaced_by_another_basis(bad):
    day = preview().history_dates[12]
    rows = tuple(replace(p, unit_nav=bad) if p.nav_date == day else p for p in rows_for())
    output = preview(rows)
    assert output.status == "INPUT_UNAVAILABLE" and output.feature_payload is None
    assert any(i.code == "UNIT_NAV_INVALID" for i in output.input_issues)


@pytest.mark.parametrize("kind", ["missing", "missing_ann", "ann_before_day", "ann_after_cutoff"])
def test_history_missing_and_late_records_not_filled_or_skipped(kind):
    day = preview().history_dates[12]
    rows = rows_for()
    if kind == "missing":
        rows = tuple(p for p in rows if p.nav_date != day)
    else:
        ann = (
            None
            if kind == "missing_ann"
            else day - timedelta(days=1)
            if kind == "ann_before_day"
            else date(2023, 6, 21)
        )
        rows = tuple(replace(p, ann_date=ann) if p.nav_date == day else p for p in rows)
    output = preview(rows)
    assert output.status == "INPUT_UNAVAILABLE" and output.history_dates == preview().history_dates


@pytest.mark.parametrize("which", ["future", "base"])
@pytest.mark.parametrize("change", ["missing", "price", "bad_price", "missing_ann", "late_ann"])
def test_future_and_unannounced_base_mutations_do_not_change_input(which, change):
    clean = preview()
    day = clean.future_dates[5] if which == "future" else clean.label_base_date
    rows = rows_for()
    if change == "missing":
        rows = tuple(p for p in rows if p.nav_date != day)
    else:
        updates = {
            "price": {"unit_nav": D(15)},
            "bad_price": {"unit_nav": D(0)},
            "missing_ann": {"ann_date": None},
            "late_ann": {"ann_date": date(2025, 1, 1)},
        }[change]
        rows = tuple(replace(p, **updates) if p.nav_date == day else p for p in rows)
    output = preview(rows)
    assert output.feature_payload == clean.feature_payload and output.feature_hash == clean.feature_hash
    assert output.label_end_date == clean.label_end_date
    if change != "price":
        assert output.status == "LABEL_UNAVAILABLE" and output.offline_label is None
    else:
        assert output.status == "RESEARCH_SAMPLE_READY"


@pytest.mark.parametrize("change", [{}, {"cash_dividend": D(3)}, {"process_status": "预案"}, {"ann_date": None}])
def test_future_dividend_data_never_enters_features(change):
    clean = preview()
    output = preview(events=(dividend(clean.future_dates[0], **change),))
    assert output.feature_payload == clean.feature_payload and output.feature_hash == clean.feature_hash
    if change.get("ann_date", 1) is None or change.get("process_status") == "预案":
        assert output.offline_label is None
    else:
        assert output.label_hash != clean.label_hash


@pytest.mark.parametrize(
    "change",
    [{"ann_date": None}, {"ann_date": date(2023, 6, 21)}, {"implementation_ann_date": date(2023, 6, 21)}],
)
def test_later_or_unknown_announcements_do_not_rewrite_historical_input(change):
    clean = preview()
    output = preview(events=(dividend(clean.history_dates[-3], **change),))
    assert output.feature_hash == clean.feature_hash and output.feature_payload == clean.feature_payload
    assert not output.dividend_history_complete_verified and not output.training_eligible


def test_known_event_and_implementation_announcement_at_cutoff_are_included():
    clean = preview()
    day = clean.history_dates[-3]
    output = preview(events=(dividend(day, implementation_ann_date=clean.cutoff_date),))
    assert output.status == "RESEARCH_SAMPLE_READY" and output.feature_hash != clean.feature_hash
    point = next(p for p in output.feature_payload.history_series if p.nav_date == day)
    assert point.available_at == clean.cutoff_date and point.cash_per_share == "0.500000000000"
    assert output.label_hash == clean.label_hash


@pytest.mark.parametrize(
    "change,code",
    [
        ({"ex_date": None, "nav_ex_date": None}, "DIVIDEND_EFFECTIVE_DATE_MISSING"),
        ({"nav_ex_date": date(2023, 6, 20)}, "DIVIDEND_DATE_CONFLICT"),
        ({"ex_date": date(2023, 6, 22), "nav_ex_date": date(2023, 6, 22)}, "DIVIDEND_NOT_ON_TRADING_DAY"),
        ({"process_status": "预案"}, "DIVIDEND_NOT_IMPLEMENTED"),
        ({"ann_date": None}, "DIVIDEND_ANN_DATE_MISSING"),
        ({"implementation_ann_date": date(2025, 1, 1)}, "DIVIDEND_NOT_AVAILABLE_BY_LIMIT"),
        ({"cash_dividend": None}, "DIVIDEND_CASH_INVALID"),
        ({"cash_dividend": D(-1)}, "DIVIDEND_CASH_INVALID"),
        ({"cash_dividend": D("NaN")}, "DIVIDEND_CASH_INVALID"),
    ],
)
def test_label_dividend_anomalies_refuse_whole_label(change, code):
    clean = preview()
    output = preview(events=(dividend(date(2023, 6, 21), **change),))
    assert output.offline_label is None
    # 日期未知且当时已经公告时，事件也可能影响历史，必须连历史一起拒收。
    if code == "DIVIDEND_EFFECTIVE_DATE_MISSING":
        assert output.feature_payload is None and any(i.code == code for i in output.input_issues)
        assert output.label_issues[0].code == "INPUT_UNAVAILABLE"
    else:
        assert output.feature_hash == clean.feature_hash
        assert any(i.code == code for i in output.label_issues)


def test_same_day_multiple_event_keys_are_not_summed():
    clean = preview()
    day = clean.future_dates[0]
    output = preview(events=(dividend(day), dividend(day, event_key="second")))
    assert output.feature_hash == clean.feature_hash and output.offline_label is None
    assert any(i.code == "MULTIPLE_DIVIDENDS_REQUIRE_REVIEW" for i in output.label_issues)


def test_known_historical_bad_event_refuses_input():
    day = preview().history_dates[-3]
    output = preview(events=(dividend(day, process_status="预案"),))
    assert output.status == "INPUT_UNAVAILABLE" and output.feature_hash is None


def test_late_implementation_sets_label_availability_and_date_fallback():
    clean = preview()
    announced = date(2023, 8, 1)
    output = preview(events=(dividend(clean.future_dates[0], nav_ex_date=None, implementation_ann_date=announced),))
    assert output.feature_hash == clean.feature_hash
    assert output.offline_label.label_available_at == announced


def test_end_price_changes_only_answer_and_future_invalid_announcement_is_label_issue():
    clean = preview()
    rows = tuple(replace(p, unit_nav=p.unit_nav + 1) if p.nav_date == clean.label_end_date else p for p in rows_for())
    output = preview(rows)
    assert output.feature_hash == clean.feature_hash and output.label_hash != clean.label_hash
    rows = tuple(
        replace(p, ann_date=p.nav_date - timedelta(days=1)) if p.nav_date == clean.label_end_date else p for p in rows
    )
    output = preview(rows)
    assert output.feature_hash == clean.feature_hash and output.offline_label is None
    assert output.label_issues[0].code == "NAV_ANN_BEFORE_NAV_DATE"


def test_weekend_cutoff_and_same_day_announced_anchor():
    weekend = preview(req=request(date(2023, 6, 18)))
    assert weekend.anchor_nav_date == weekend.label_base_date == date(2023, 6, 16)
    assert weekend.anchor_lag_sessions == 0 and weekend.future_dates[0] == date(2023, 6, 19)
    rows = tuple(replace(p, ann_date=p.nav_date) for p in rows_for())
    same = preview(rows)
    assert same.anchor_nav_date == same.label_base_date == date(2023, 6, 20)


def test_unpublished_next_year_calendar_retains_input_but_refuses_label():
    output = preview(req=request(date(2023, 12, 8)))
    assert output.feature_payload is not None and output.offline_label is None
    assert output.label_issues[0].code == "FUTURE_CALENDAR_NOT_KNOWN_AT_CUTOFF"


def test_future_non_trading_row_does_not_affect_either_hash():
    clean = preview()
    extra = CashNavPoint(date(2023, 6, 25), None, D("NaN"))
    output = preview(tuple(sorted((*rows_for(), extra), key=lambda p: p.nav_date)))
    assert output.feature_hash == clean.feature_hash and output.label_hash == clean.label_hash
    assert output.ignored_non_trading_nav_dates == (extra.nav_date,)


def test_flat_history_is_not_fabricated_into_midpoint_position():
    rows = tuple(replace(p, unit_nav=D(10)) for p in rows_for())
    output = preview(rows)
    assert output.status == "INPUT_UNAVAILABLE" and output.input_issues[0].code == "FLAT_HISTORY_POSITION_UNDEFINED"


def test_displayed_zero_return_has_zero_direction_and_decimal_context_is_fixed():
    clean = preview()
    rows = rows_for()
    base_value = next(p.unit_nav for p in rows if p.nav_date == clean.label_base_date)
    rows = tuple(
        replace(p, unit_nav=base_value + D("0.00000000000001")) if p.nav_date == clean.label_end_date else p
        for p in rows
    )
    normal = preview(rows)
    with localcontext() as context:
        context.prec = 8
        context.rounding = ROUND_DOWN
        low_precision = preview(rows)
    assert low_precision == normal
    assert normal.offline_label.future_return_20d == "0.000000000000" and normal.offline_label.label_up_20d == 0


@pytest.mark.parametrize("change", ["duplicate", "unordered", "out_of_scope", "oversized", "no_anchor"])
def test_bad_source_structure_or_no_anchor_is_not_silently_repaired(change):
    rows = rows_for()
    if change == "no_anchor":
        output = preview(())
        assert output.status == "INPUT_UNAVAILABLE" and output.input_issues[0].code == "NO_KNOWN_TRADING_NAV"
        return
    rows = (
        rows + (rows[-1],)
        if change == "duplicate"
        else tuple(reversed(rows))
        if change == "unordered"
        else (CashNavPoint(date(2020, 1, 1), None, None), *rows)
        if change == "out_of_scope"
        else rows * 4
    )
    with pytest.raises(ValueError):
        preview(rows)


@pytest.mark.parametrize("count", [2, 101])
def test_duplicate_event_identifiers_and_oversized_packet_refused(count):
    with pytest.raises(ValueError):
        preview(events=(dividend(date(2023, 6, 21)),) * count)


def test_repository_is_narrow_bounded_same_fund_source_and_keeps_conflicting_dates():
    statements = []

    def execute(statement):
        statements.append(statement)
        return SimpleNamespace(all=lambda: [])

    read_cash_sample_inputs(
        SimpleNamespace(execute=execute),
        fund_code="006730",
        source_id=UUID(int=1),
        start=date(2023, 3, 1),
        end=date(2023, 7, 20),
    )
    assert tuple(c.name for c in statements[0].selected_columns) == ("nav_date", "ann_date", "unit_nav")
    assert tuple(c.name for c in statements[1].selected_columns) == (
        "source_event_key",
        "ann_date",
        "implementation_ann_date",
        "ex_date",
        "nav_ex_date",
        "cash_dividend",
        "process_status",
    )
    for statement, limit in zip(statements, (193, 101), strict=True):
        params = statement.compile().params.values()
        assert UUID(int=1) in params and "006730" in params and limit in params
        assert "2025" not in str(params)
    assert "IS NULL" in str(statements[1]) and " OR " in str(statements[1])


@pytest.mark.parametrize("sizes", [(193, 0), (0, 101)])
def test_repository_sentinels_refuse_partial_results(sizes):
    counts = iter(sizes)
    session = SimpleNamespace(execute=lambda statement: SimpleNamespace(all=lambda: [None] * next(counts)))
    with pytest.raises(ValueError):
        read_cash_sample_inputs(
            session, fund_code="006730", source_id=UUID(int=1), start=date(2023, 3, 1), end=date(2023, 7, 20)
        )


@pytest.mark.parametrize("start,end", [(date(2023, 1, 1), date(2023, 12, 31)), (date(2024, 12, 1), date(2025, 1, 1))])
def test_repository_bounds_refused_before_query(start, end):
    with pytest.raises(ValueError):
        read_cash_sample_inputs(None, fund_code="006730", source_id=UUID(int=1), start=start, end=end)


def test_read_phase_is_repeatable_read_only_and_preserves_source(monkeypatch):
    calls = []

    class FakeSession:
        def __init__(self, engine):
            calls.append("session")

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def begin(self):
            return self

        def execute(self, sql):
            calls.append(str(sql))

    source = replace(SOURCE, source_code="TUSHARE_PRO_FUND")
    monkeypatch.setattr(service, "get_nav_preview_engine", lambda: None)
    monkeypatch.setattr(service, "Session", FakeSession)
    monkeypatch.setattr(service, "read_historical_nav_source", lambda session, **kw: source)

    def read(session, **kw):
        assert kw["source_id"] == source.source_id and kw["fund_code"] == "006730"
        assert kw["end"] <= date(2024, 12, 31)
        return rows_for(), ()

    monkeypatch.setattr(service, "read_cash_sample_inputs", read)
    output = service.preview_cash_reinvestment_sample(request())
    assert output.status == "RESEARCH_SAMPLE_READY"
    assert calls == ["session", "SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY"]


def test_cutoff_whose_answer_enters_2025_is_blocked_before_database(monkeypatch):
    monkeypatch.setattr(service, "get_nav_preview_engine", lambda: pytest.fail("protected test values queried"))
    with pytest.raises(HistoricalNavPreviewReadError) as caught:
        service.preview_cash_reinvestment_sample(request(date(2024, 12, 20)))
    assert caught.value.code == "TEST_PERIOD_PROTECTED"


def test_http_response_with_chinese_comments_and_trace(client, monkeypatch):
    monkeypatch.setattr(api, "preview_cash_reinvestment_sample", lambda req: preview(req=req))
    response = client.get(PATH, params=PARAMS, headers={**HEADERS, "X-Trace-Id": "cash-test"})
    assert response.status_code == 200 and response.json()["status"] == "RESEARCH_SAMPLE_READY"
    assert response.headers["X-Trace-Id"] == "cash-test"


@pytest.mark.parametrize(
    "extra",
    [
        {"fundCode": "000001"},
        {"cutoffDate": "bad"},
        {"cutoffDate": "2021-12-31"},
        {"cutoffDate": "2025-01-01"},
        {"asOfDate": "2023-06-20"},
        {"includeTest": True},
        {"navBasis": "ADJUSTED_NAV"},
        {"horizon": 30},
    ],
)
def test_invalid_http_request_cannot_query(client, monkeypatch, extra):
    monkeypatch.setattr(api, "preview_cash_reinvestment_sample", lambda req: pytest.fail("invalid work"))
    assert client.get(PATH, params={**PARAMS, **extra}, headers=HEADERS).status_code == 422


@pytest.mark.parametrize("headers", [{}, {"X-Service-Token": "wrong"}, {**HEADERS, "Origin": "http://localhost"}])
def test_auth_before_query(client, monkeypatch, headers):
    monkeypatch.setattr(api, "preview_cash_reinvestment_sample", lambda req: pytest.fail("unauthorized work"))
    assert client.get(PATH, params=PARAMS, headers=headers).status_code == 403


@pytest.mark.parametrize(
    "error,status",
    [
        (CalendarCoverageError("not covered"), 409),
        (HistoricalNavPreviewReadError("FUND_NOT_FOUND", "missing"), 404),
        (HistoricalNavPreviewReadError("TEST_PERIOD_PROTECTED", "protected"), 409),
        (SQLAlchemyError("private SQL"), 503),
        (ValueError("private path"), 503),
        (TimeoutError("private timeout"), 503),
    ],
)
def test_http_errors_redact_internal_details(client, monkeypatch, error, status):
    def fail(req):
        raise error

    monkeypatch.setattr(api, "preview_cash_reinvestment_sample", fail)
    response = client.get(PATH, params=PARAMS, headers=HEADERS)
    assert response.status_code == status and "private" not in response.text
