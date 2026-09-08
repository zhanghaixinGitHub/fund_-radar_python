"""可手算口径差异、坏数据不回退、同来源有界只读与HTTP鉴权验收。"""

from dataclasses import replace
from datetime import date, timedelta
from decimal import Decimal, localcontext
from types import SimpleNamespace
from uuid import UUID

import pytest
from app.api.routes import nav_basis_audit as api
from app.repositories.historical_nav import HistoricalNavPreviewReadError
from app.repositories.nav_basis_audit import BasisDividend, BasisNavPoint, read_basis_inputs
from app.schemas.nav_basis_audit import NavBasisAuditRequest
from app.services import nav_basis_audit as service
from app.services.trading_calendar import CalendarCoverageError, load_calendar
from sqlalchemy.exc import SQLAlchemyError
from tests.test_historical_nav_http import HEADERS
from tests.test_historical_nav_http import client as client
from tests.test_trading_nav_window import SOURCE

PATH = "/internal/v1/features/historical-nav-samples/nav-basis-audit"
PARAMS = {"fundCode": "006730", "startDate": "2023-06-20", "endDate": "2023-06-21"}
D = Decimal


def request(**kwargs):
    return NavBasisAuditRequest(**{**PARAMS, **kwargs})


def fixture_rows():
    # 第一天净值10→10.2；第二天每份分红1元，净值10.2→9.3。
    # 现金算式为10.3/10.2-1，复权故意给出另一种算式9.3/(10.2-1)-1。
    adjusted = D("10.2") * D("9.3") / D("9.2")
    nav = (
        BasisNavPoint(date(2023, 6, 19), date(2023, 6, 20), D("10"), D("10"), D("10"), None),
        BasisNavPoint(date(2023, 6, 20), date(2023, 6, 21), D("10.2"), D("10.2"), D("10.2"), None),
        BasisNavPoint(date(2023, 6, 21), date(2023, 6, 22), D("9.3"), D("10.3"), adjusted, None),
    )
    dividend = BasisDividend(date(2023, 6, 20), date(2023, 6, 21), date(2023, 6, 21), D("1"), "实施")
    return nav, (dividend,)


def result(nav=None, events=None, req=None):
    default_nav, default_events = fixture_rows()
    return service.build_nav_basis_audit(
        req or request(),
        SOURCE,
        nav if nav is not None else default_nav,
        events if events is not None else default_events,
        load_calendar(),
    )


def test_exact_cash_formula_differs_from_source_adjusted_and_keeps_gates():
    output = result()
    assert output.status == "DIFFERENCES_FOUND" and output.daily_gap_over_threshold_count == 1
    assert output.trading_day_count == output.checked_daily_pairs == 2
    item = output.dividend_comparisons[0]
    assert D(item.cash_inclusive_day_return) == D("0.009803921569")
    assert D(item.source_adjusted_day_return) == D("0.010869565217")
    assert item.source_adjusted_day_return == item.ex_cash_denominator_hypothesis_return
    assert item.unit_nav_ratio_return == "-0.088235294118"
    assert output.period_comparison.cash_reinvestment_candidate_return == "0.030000000000"
    assert output.admission_status == "NOT_APPROVED" and output.publication_status == "MODEL_NOT_RELEASED"
    for name in (
        "automatic_basis_fallback_allowed",
        "database_written",
        "feature_generated",
        "label_generated",
        "model_fitted",
        "test_scored",
    ):
        assert getattr(output, name) is False


def test_same_inputs_and_event_order_produce_identical_hash_and_response():
    assert result().model_dump() == result().model_dump()
    nav, events = fixture_rows()
    assert (
        result(nav=nav[:-1] + (replace(nav[-1], adjusted_nav=D("11")),), events=events).snapshot_hash
        != result().snapshot_hash
    )


def test_different_decimal_context_does_not_change_result():
    nav, events = fixture_rows()
    expected = result(nav, events)
    with localcontext() as ctx:
        ctx.prec = 12
        assert result(nav, events) == expected


def test_cash_candidate_compounds_daily_reinvestment_not_simple_cash_sum():
    nav, events = fixture_rows()
    nav = (
        nav[0],
        replace(nav[1], unit_nav=D("9.5"), adjusted_nav=D("10.5")),
        replace(nav[2], unit_nav=D("9"), adjusted_nav=D("10.5") * D("10") / D("9.5")),
    )
    events = (
        replace(events[0], ann_date=date(2023, 6, 19), ex_date=date(2023, 6, 20), nav_ex_date=date(2023, 6, 20)),
        events[0],
    )
    output = result(nav, events)
    assert output.period_comparison.cash_reinvestment_candidate_return == "0.105263157895"
    assert result(nav, tuple(reversed(events))) == output
    assert output.period_comparison.cash_reinvestment_candidate_return != "0.100000000000"


@pytest.mark.parametrize("field", ["unit_nav", "accumulated_nav", "adjusted_nav"])
@pytest.mark.parametrize("bad", [None, D(0), D(-1), D("NaN"), D("Infinity")])
def test_invalid_values_never_fallback_to_another_basis(field, bad):
    nav, events = fixture_rows()
    output = result((nav[0], replace(nav[1], **{field: bad}), nav[2]), events)
    assert output.status == "AUDIT_INCOMPLETE"
    assert output.invalid_nav_counts[field] == 1
    assert output.period_comparison.cash_reinvestment_candidate_return is None


def test_missing_required_date_not_skipped():
    nav, events = fixture_rows()
    output = result((nav[0], nav[2]), events)
    assert output.trading_day_count == 2 and output.checked_daily_pairs == 0
    assert any(i.code == "TRADING_NAV_MISSING" and i.day == nav[1].nav_date for i in output.issues)
    assert all(n == 1 for n in output.invalid_nav_counts.values())
    assert output.period_comparison.unit_nav_ratio_return == "-0.070000000000"
    assert output.period_comparison.cash_reinvestment_candidate_return is None


def test_weekend_and_holiday_points_are_not_counted():
    req = request(startDate="2023-06-16", endDate="2023-06-21")
    days = service.audit_dates(load_calendar(), req)
    nav = tuple(BasisNavPoint(day, day + timedelta(days=1), D("10"), D("10"), D("10"), None) for day in days)
    clean = result(nav, (), req)
    weekend = BasisNavPoint(date(2023, 6, 18), None, D("999"), None, D("1"), None)
    output = result(tuple(sorted((*nav, weekend), key=lambda p: p.nav_date)), (), req)
    assert output.period_comparison == clean.period_comparison
    assert output.ignored_non_trading_dates == (date(2023, 6, 18),)
    assert output.checked_daily_pairs == clean.checked_daily_pairs


def test_no_dividend_rows_is_not_proof_of_no_dividends():
    nav, _ = fixture_rows()
    output = result(nav, ())
    assert output.status == "AUDIT_INCOMPLETE"
    assert any(i.code == "UNEXPLAINED_ADJUSTMENT_WITHOUT_DIVIDEND" for i in output.issues)
    assert output.period_comparison.cash_reinvestment_candidate_return is None


def test_equal_daily_ratios_does_not_approve_basis_and_accumulated_can_differ():
    nav, _ = fixture_rows()
    nav = tuple(replace(p, adjusted_nav=p.unit_nav, accumulated_nav=p.unit_nav + D("3")) for p in nav)
    output = result(nav, ())
    assert output.status == "NO_LARGE_DAILY_DIFFERENCE_FOUND"
    assert output.admission_status == "NOT_APPROVED" and output.accumulated_dividend_missing_count == 3
    assert (
        output.period_comparison.accumulated_nav_ratio_return
        != output.period_comparison.source_adjusted_nav_ratio_return
    )


@pytest.mark.parametrize(
    "change,code",
    [
        ({"nav_ex_date": None, "ex_date": None}, "DIVIDEND_EFFECTIVE_DATE_MISSING"),
        ({"nav_ex_date": date(2023, 6, 20)}, "DIVIDEND_DATE_CONFLICT"),
        ({"nav_ex_date": date(2023, 6, 18), "ex_date": date(2023, 6, 18)}, "DIVIDEND_NOT_ON_REQUIRED_TRADING_DAY"),
        ({"process_status": "预案"}, "DIVIDEND_NOT_IMPLEMENTED"),
        ({"ann_date": None}, "DIVIDEND_ANN_DATE_INVALID"),
        ({"ann_date": date(2023, 6, 22)}, "DIVIDEND_ANN_DATE_INVALID"),
        ({"cash_dividend": None}, "DIVIDEND_CASH_INVALID"),
        ({"cash_dividend": D(-1)}, "DIVIDEND_CASH_INVALID"),
        ({"cash_dividend": D("NaN")}, "DIVIDEND_CASH_INVALID"),
    ],
)
def test_bad_dividend_not_summed_or_silently_treated_as_zero(change, code):
    nav, events = fixture_rows()
    output = result(nav, (replace(events[0], **change),))
    assert output.status == "AUDIT_INCOMPLETE" and any(i.code == code for i in output.issues)
    assert not output.dividend_comparisons and output.period_comparison.cash_reinvestment_candidate_return is None


def test_multiple_events_not_summed():
    nav, events = fixture_rows()
    output = result(nav, events * 2)
    assert any(i.code == "MULTIPLE_DIVIDENDS_REQUIRE_REVIEW" for i in output.issues)
    assert not output.dividend_comparisons


def test_date_conflict_excludes_both_dates_from_cash_comparison():
    nav, events = fixture_rows()
    output = result(nav, (replace(events[0], nav_ex_date=date(2023, 6, 20)),))
    assert output.checked_daily_pairs == 0


def test_ex_date_fallback_is_explicit_and_valid():
    nav, events = fixture_rows()
    assert result(nav, (replace(events[0], nav_ex_date=None),)).dividend_comparisons == result().dividend_comparisons


@pytest.mark.parametrize("change", ["duplicate", "unordered", "oversized", "future", "future_event", "too_many_events"])
def test_invalid_raw_scope_rejected(change):
    nav, events = fixture_rows()
    if change == "duplicate":
        nav += (nav[-1],)
    elif change == "unordered":
        nav = tuple(reversed(nav))
    elif change == "oversized":
        nav *= 130
    elif change == "future":
        nav += (replace(nav[-1], nav_date=date(2025, 1, 1)),)
    elif change == "future_event":
        events = (replace(events[0], ann_date=date(2025, 1, 1)),)
    else:
        events *= 101
    with pytest.raises(ValueError):
        result(nav, events)


def test_query_has_source_bounds_field_whitelist_and_sentinels():
    statements = []

    def execute(statement):
        statements.append(statement)
        return SimpleNamespace(all=lambda: [])

    read_basis_inputs(
        SimpleNamespace(execute=execute),
        fund_code="006730",
        source_id=UUID(int=1),
        base_date=date(2022, 12, 30),
        start=date(2023, 1, 1),
        end=date(2023, 12, 31),
    )
    assert tuple(c.name for c in statements[0].selected_columns) == (
        "nav_date",
        "ann_date",
        "unit_nav",
        "accumulated_nav",
        "adjusted_nav",
        "accumulated_dividend",
    )
    assert tuple(c.name for c in statements[1].selected_columns) == (
        "ann_date",
        "ex_date",
        "nav_ex_date",
        "cash_dividend",
        "process_status",
    )
    for statement, limit in zip(statements, (387, 101), strict=True):
        assert UUID(int=1) in statement.compile().params.values() and limit in statement.compile().params.values()
    with pytest.raises(ValueError):
        read_basis_inputs(
            None,
            fund_code="006730",
            source_id=UUID(int=1),
            base_date=date(2024, 12, 1),
            start=date(2024, 12, 2),
            end=date(2025, 1, 1),
        )


@pytest.mark.parametrize("sizes", [(387, 0), (0, 101)])
def test_repository_rejects_sentinel_rows(sizes):
    counts = iter(sizes)

    def execute(statement):
        count = next(counts)
        return SimpleNamespace(all=lambda: [None] * count)

    with pytest.raises(ValueError):
        read_basis_inputs(
            SimpleNamespace(execute=execute),
            fund_code="006730",
            source_id=UUID(int=1),
            base_date=date(2022, 12, 30),
            start=date(2023, 1, 1),
            end=date(2023, 12, 31),
        )


def test_empty_trading_range_rejected_before_database(monkeypatch):
    monkeypatch.setattr(service, "get_nav_preview_engine", lambda: pytest.fail("must reject before DB"))
    with pytest.raises(CalendarCoverageError):
        service.audit_nav_basis(request(startDate="2023-06-22", endDate="2023-06-25"))


def test_http_response_and_trace(client, monkeypatch):
    monkeypatch.setattr(api, "audit_nav_basis", lambda req: result(req=req))
    response = client.get(PATH, params=PARAMS, headers={**HEADERS, "X-Trace-Id": "basis-test"})
    assert response.status_code == 200 and response.json()["admission_status"] == "NOT_APPROVED"
    assert response.headers["X-Trace-Id"] == "basis-test"


@pytest.mark.parametrize(
    "extra",
    [
        {"fundCode": "000001"},
        {"startDate": "bad"},
        {"startDate": "2021-12-31"},
        {"endDate": "2025-01-01"},
        {"endDate": "2023-06-19"},
        {"endDate": "2024-12-31"},
        {"cutoffDate": "2023-06-20"},
        {"includeTest": True},
        {"navBasis": "UNIT_NAV"},
    ],
)
def test_invalid_http_request_does_not_query(client, monkeypatch, extra):
    monkeypatch.setattr(api, "audit_nav_basis", lambda req: pytest.fail("invalid request must not query"))
    assert client.get(PATH, params={**PARAMS, **extra}, headers=HEADERS).status_code == 422


@pytest.mark.parametrize("headers", [{}, {"X-Service-Token": "wrong"}, {**HEADERS, "Origin": "http://localhost"}])
def test_http_authorization(client, monkeypatch, headers):
    monkeypatch.setattr(api, "audit_nav_basis", lambda req: pytest.fail("unauthorized work"))
    assert client.get(PATH, params=PARAMS, headers=headers).status_code == 403


@pytest.mark.parametrize(
    "error,code",
    [
        (CalendarCoverageError("no trading days"), 409),
        (HistoricalNavPreviewReadError("FUND_NOT_FOUND", "missing"), 404),
        (HistoricalNavPreviewReadError("SOURCE_NOT_READY", "not ready"), 409),
        (SQLAlchemyError("private SQL"), 503),
        (ValueError("private file"), 503),
        (TimeoutError("private"), 503),
    ],
)
def test_http_errors_do_not_leak_details(client, monkeypatch, error, code):
    def fail(req):
        raise error

    monkeypatch.setattr(api, "audit_nav_basis", fail)
    response = client.get(PATH, params=PARAMS, headers=HEADERS)
    assert response.status_code == code and "private" not in response.text
