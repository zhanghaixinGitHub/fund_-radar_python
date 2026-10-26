"""任期回报的现金再投、净值覆盖、来源隔离及独立降级边界。"""

from datetime import date
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from app.repositories.fund_manager_performance import ManagerDividend, ManagerNavPoint
from app.services import fund_manager_performance as service
from sqlalchemy.exc import SQLAlchemyError

DAYS = (date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4))


def point(day, unit, accumulated=None, adjusted=None):
    return ManagerNavPoint(
        day, day, Decimal(unit), Decimal(accumulated or unit), Decimal(adjusted or unit),
    )


def event(day=DAYS[1], cash="0.1", **changes):
    values = dict(
        ex_date=day, nav_ex_date=day, cash_dividend=Decimal(cash), process_status="实施",
        ann_date=DAYS[0], implementation_ann_date=DAYS[0],
    )
    return ManagerDividend(**{**values, **changes})


def calculate(navs=None, dividends=(), **changes):
    return service.calculate_tenure_return(**{
        "begin": DAYS[0], "end": None, "latest_nav_date": DAYS[-1], "today": date(2024, 1, 6),
        "navs": navs or tuple(point(day, value) for day, value in zip(DAYS, ("1", "1.1", "1.2"), strict=True)),
        "dividends": dividends, "sessions": DAYS, **changes,
    })


@pytest.mark.parametrize("last, expected", [("1.2", "0.2"), ("0.8", "-0.2"), ("1", "0")])
def test_positive_negative_and_zero_are_real_returns_with_actual_nav_dates(last, expected):
    result = calculate(navs=(point(DAYS[0], "1"), point(DAYS[1], "1.1"), point(DAYS[2], last)))
    assert result.status == "AVAILABLE" and result.return_rate == Decimal(expected)
    assert result.start_date == DAYS[0] and result.end_date == DAYS[-1]
    assert result.end_date != date(2024, 1, 6)


def test_cash_is_reinvested_at_ex_date_nav_and_not_accumulated_nav_ratio():
    navs = (point(DAYS[0], "1"), point(DAYS[1], "0.9", "1", "1"),
            point(DAYS[2], "0.99", "1.09", "1.1"))
    result = calculate(navs=navs, dividends=(event(),), co_managed=True)
    assert result.status == "AVAILABLE" and result.return_rate == Decimal("0.1")
    assert result.return_rate != Decimal("0.09")
    assert result.includes_co_management
    # 首日收盘已除权，首日派息属于起点前的收益段，不能再领取一次。
    assert calculate(dividends=(event(DAYS[0]),)).return_rate == Decimal("0.2")


def test_missing_nav_duplicate_invalid_values_and_future_disclosure_refuse_full_period():
    cases = [
        (point(DAYS[0], "1"), point(DAYS[2], "1.2")),
        (point(DAYS[0], "1"), point(DAYS[1], "1.1"), point(DAYS[1], "1.1"), point(DAYS[2], "1.2")),
        (point(DAYS[0], "1"), point(DAYS[1], "0"), point(DAYS[2], "1.2")),
        (point(DAYS[0], "1"), point(DAYS[1], "NaN"), point(DAYS[2], "1.2")),
        (point(DAYS[0], "1"), point(DAYS[1], "1.1"),
         ManagerNavPoint(DAYS[2], date(2024, 1, 7), Decimal("1.2"), Decimal("1.2"), Decimal("1.2"))),
    ]
    for navs in cases:
        result = calculate(navs=navs)
        assert result.status == "UNAVAILABLE" and result.return_rate is None


@pytest.mark.parametrize("changes", [
    {"nav_ex_date": DAYS[2]}, {"process_status": "预案"}, {"ann_date": None},
    {"cash_dividend": Decimal("-1")}, {"cash_dividend": Decimal("NaN")},
    {"implementation_ann_date": date(2024, 1, 7)}, {"ex_date": None, "nav_ex_date": None},
])
def test_unverified_dividend_never_becomes_zero_cash(changes):
    assert calculate(dividends=(event(**changes),)).status == "UNAVAILABLE"


def test_missing_dividend_duplicate_dividend_and_non_cash_adjustment_are_rejected():
    cash_nav = (point(DAYS[0], "1"), point(DAYS[1], "0.9", "1", "1"),
                point(DAYS[2], "0.99", "1.09", "1.1"))
    assert "不一致" in calculate(navs=cash_nav).reason
    assert "同日" in calculate(navs=cash_nav, dividends=(event(), event())).reason
    split_nav = (point(DAYS[0], "1"), point(DAYS[1], "0.5", "0.5", "1"),
                 point(DAYS[2], "0.6", "0.6", "1.2"))
    assert "净值调整" in calculate(navs=split_nav).reason


@pytest.mark.parametrize("changes", [
    {"begin": None}, {"begin": date(2024, 1, 7)}, {"end": date(2024, 1, 1)},
    {"end": date(2024, 1, 5)}, {"begin": date(2023, 12, 31)}, {"begin": DAYS[-1]},
])
def test_invalid_short_uncovered_and_incomplete_departed_periods_remain_unknown(changes):
    assert calculate(**changes).return_rate is None


def test_holidays_select_calendar_endpoints_instead_of_filling_or_skipping_missing_nav():
    # 周末开始只使用首个应有净值日；并不将随机缺数视为休市。
    result = calculate(begin=date(2024, 1, 3), end=date(2024, 1, 6))
    assert result.status == "UNAVAILABLE"  # 离任日已到但最新净值未覆盖完整任期。
    result = calculate(begin=DAYS[1])
    assert result.start_date == DAYS[1] and result.return_rate == Decimal("0.090909090909")


def test_quarter_end_weekend_nav_is_retained_without_weakening_trading_day_gap_checks():
    friday, sunday, monday = date(2024, 3, 29), date(2024, 3, 31), date(2024, 4, 1)
    navs = (point(friday, "1"), point(sunday, "1.01"), point(monday, "1.02"))
    result = calculate(navs=navs, begin=friday, latest_nav_date=monday, today=monday, sessions=(friday, monday))
    assert result.return_rate == Decimal("0.02")
    assert calculate(navs=navs[:-1], begin=friday, latest_nav_date=monday,
                     today=monday, sessions=(friday, monday)).status == "UNAVAILABLE"
    # 周日离任可正常以已披露的周日净值收盘结束，不强求等到周一。
    result = calculate(navs=navs, begin=friday, end=sunday, latest_nav_date=monday,
                       today=monday, sessions=(friday, monday))
    assert result.end_date == sunday and result.return_rate == Decimal("0.01")


def test_weekend_departure_does_not_require_a_nonexistent_weekend_nav():
    friday, saturday, monday = date(2024, 1, 5), date(2024, 1, 6), date(2024, 1, 8)
    navs = tuple(point(day, "1") for day in (*DAYS, friday))
    result = calculate(navs=navs, end=saturday, latest_nav_date=friday, today=monday,
                       sessions=(*DAYS, friday, monday))
    assert result.status == "AVAILABLE" and result.end_date == friday


def snapshot(fund_type="MIXED", source="TUSHARE_PRO_FUND"):
    def manager(name):
        return SimpleNamespace(source_code=source, assignment=SimpleNamespace(
            manager_name=name, begin_date=DAYS[0], end_date=None, source_id="same-source-id",
        ))
    return SimpleNamespace(
        managers=(manager("甲"), manager("乙")), profile=None,
        detail=SimpleNamespace(fund=SimpleNamespace(fund_type=fund_type, fund_code="002112"),
                               source_code=source, nav_date=DAYS[-1]),
        succeeded_sync_types=frozenset({"MARKET_DETAIL_DIVIDEND"}),
    )


def test_one_fund_read_is_shared_by_all_managers_and_has_explicit_source_and_bounds(monkeypatch):
    calls = []
    monkeypatch.setattr(service, "manager_calendar", lambda: DAYS)

    def read(_session, **query):
        calls.append(query)
        return tuple(point(day, "1") for day in DAYS), ()

    monkeypatch.setattr(service, "read_manager_performance_inputs", read)
    results = service.build_manager_performances(MagicMock(), snapshot(), today=date(2024, 1, 6))
    assert len(calls) == 1 and calls[0] == {
        "fund_code": "002112", "source_id": "same-source-id", "start": DAYS[0], "end": DAYS[-1],
    }
    assert all(result.return_rate == 0 and result.includes_co_management for result in results)


@pytest.mark.parametrize("fund_type,source", [("QDII", "TUSHARE_PRO_FUND"), ("MIXED", "OTHER")])
def test_unsupported_market_or_source_is_not_silently_calculated(monkeypatch, fund_type, source):
    monkeypatch.setattr(service, "read_manager_performance_inputs", lambda *args, **kwargs: pytest.fail("应先拒绝"))
    results = service.build_manager_performances(MagicMock(), snapshot(fund_type, source))
    assert all(result.status == "UNAVAILABLE" and result.return_rate is None for result in results)


def test_reviewed_calendar_covers_old_and_current_years_without_changing_existing_versions():
    sessions = service.manager_calendar()
    assert sessions[0] == date(2015, 11, 16) and date(2026, 10, 23) in sessions
    assert date(2020, 1, 31) not in sessions
    assert len(sessions) == len(set(sessions)) and tuple(sorted(sessions)) == sessions


def test_oversized_inputs_and_read_errors_degrade_only_returns(monkeypatch):
    monkeypatch.setattr(service, "manager_calendar", lambda: DAYS)
    monkeypatch.setattr(service, "read_manager_performance_inputs", lambda *args, **kwargs: (
        (point(DAYS[0], "1"),) * 5001, (),
    ))
    result = service.build_manager_performances(MagicMock(), snapshot(), today=date(2024, 1, 6))
    assert all(item.status == "UNAVAILABLE" and "超出" in item.reason for item in result)

    def failed_read(*args, **kwargs):
        raise SQLAlchemyError("test timeout")

    monkeypatch.setattr(service, "read_manager_performance_inputs", failed_read)
    result = service.build_manager_performances(MagicMock(), snapshot(), today=date(2024, 1, 6))
    assert all(item.status == "UNAVAILABLE" and item.return_rate is None for item in result)


def test_a_missing_entire_dividend_sync_is_not_inferred_as_no_dividends():
    value = snapshot()
    value.succeeded_sync_types = frozenset()
    assert all(item.return_rate is None for item in service.build_manager_performances(MagicMock(), value))


def test_stage_and_full_tenure_use_same_inputs_but_keep_independent_dates_and_returns(monkeypatch):
    days = (*DAYS, date(2024, 1, 5), date(2024, 1, 8), date(2024, 1, 9))
    value = snapshot()
    value.detail.nav_date = days[-1]
    value.managers[1].assignment.end_date = DAYS[-1]
    calls = []
    monkeypatch.setattr(service, "manager_calendar", lambda: days)

    def read(_session, **query):
        calls.append(query)
        return tuple(point(day, str(Decimal(1) + Decimal(index) / 10)) for index, day in enumerate(days)), ()

    monkeypatch.setattr(service, "read_manager_performance_inputs", read)
    result = service.build_manager_return_bundle(MagicMock(), value, today=date(2024, 1, 10))
    assert len(calls) == 1
    assert result.managers[0].return_rate == Decimal("0.5")
    assert result.managers[0].includes_co_management
    current, earlier = result.periods
    assert current.begin_date == date(2024, 1, 5) and current.end_date is None
    assert current.manager_names == ("甲",)
    assert current.performance.start_date == date(2024, 1, 5)
    assert current.performance.end_date == date(2024, 1, 9)
    assert current.performance.return_rate == Decimal("0.153846153846")
    assert not current.performance.includes_co_management
    assert earlier.performance.return_rate == Decimal("0.2")
    assert set(earlier.manager_names) == {"甲", "乙"}
    assert earlier.performance.includes_co_management
    # 两个阶段各按自身首个净值收盘计算，不相加百分比来冒充完整任期收益。
    assert current.performance.return_rate + earlier.performance.return_rate != result.managers[0].return_rate


def test_single_nav_day_stage_is_unknown_instead_of_borrowing_another_period_return(monkeypatch):
    value = snapshot()
    value.managers[1].assignment.begin_date = DAYS[1]
    value.managers[1].assignment.end_date = DAYS[1]
    monkeypatch.setattr(service, "manager_calendar", lambda: DAYS)
    monkeypatch.setattr(service, "read_manager_performance_inputs", lambda *args, **kwargs: (
        tuple(point(day, "1") for day in DAYS), (),
    ))
    result = service.build_manager_return_bundle(MagicMock(), value, today=date(2024, 1, 6))
    one_day = next(period for period in result.periods if period.begin_date == period.end_date)
    assert one_day.performance.status == "UNAVAILABLE" and one_day.performance.return_rate is None
    assert "阶段" in one_day.performance.reason
    assert result.managers[0].return_rate == 0  # 完整任期可计算的0%仍有效，不能移用于单日阶段。


def test_team_periods_merge_same_name_renewals_and_keep_gaps_and_incomplete_records_separate():
    def manager(name, begin, end=None):
        return SimpleNamespace(assignment=SimpleNamespace(manager_name=name, begin_date=begin, end_date=end))

    rows = (manager("甲", DAYS[0], DAYS[1]), manager("甲", DAYS[1], DAYS[2]),
            manager("甲", date(2024, 1, 5)), manager("乙", None))
    assert service.manager_periods(rows, date(2024, 1, 6)) == ((DAYS[0], None, ("甲",)),)
    gap_rows = (manager("甲", DAYS[0], DAYS[0]), manager("乙", DAYS[2]))
    assert service.manager_periods(gap_rows, date(2024, 1, 6))[1] == (DAYS[1], DAYS[1], ())
