"""新版批量日历选题、分页一致、单日等价、只读事务、异常原子返回与HTTP验收。"""

import hashlib
import json
from dataclasses import replace
from datetime import date, timedelta
from decimal import Decimal
from math import ceil
from types import SimpleNamespace
from uuid import UUID

import pytest
from app.api.routes import cash_reinvestment_batch as api
from app.repositories.cash_reinvestment_samples import read_cash_nav_window
from app.repositories.historical_nav import HistoricalNavPreviewReadError
from app.schemas.cash_reinvestment_batch import CashBatchRequest
from app.schemas.cash_reinvestment_samples import CashSampleRequest
from app.services import cash_reinvestment_batch as service
from app.services.cash_reinvestment_samples import build_cash_reinvestment_sample
from app.services.trading_calendar import CalendarCoverageError, load_calendar
from sqlalchemy.exc import SQLAlchemyError
from tests.test_cash_reinvestment_samples import CashNavPoint, dividend
from tests.test_historical_nav_http import HEADERS
from tests.test_historical_nav_http import client as client
from tests.test_trading_nav_window import SOURCE

PATH = "/internal/v1/features/historical-nav-samples/cash-reinvestment-dry-run"
PARAMS = {"fundCode": "006730", "startDate": "2023-06-01", "endDate": "2023-06-30"}


def request(**updates):
    return CashBatchRequest(**{**PARAMS, **updates})


@pytest.fixture
def reader(monkeypatch):
    """内存资料替代数据库，不替代日期计划和单日公式；记录每页实际有界读取参数。"""
    calendar = load_calendar()
    state = SimpleNamespace(
        source=replace(SOURCE, source_code="TUSHARE_PRO_FUND"),
        nav=tuple(
            CashNavPoint(day, day + timedelta(days=1), Decimal(1) + Decimal(i) / 1000)
            for i, day in enumerate(calendar.sessions)
        ),
        events=(dividend(date(2023, 6, 21)),),
        calls=[],
        fail_nav_page=None,
        ro=False,
        active=0,
    )

    class FakeSession:
        def __init__(self, engine):
            state.calls.append(("session",))

        def __enter__(self):
            state.active += 1
            return self

        def __exit__(self, *args):
            state.active -= 1

        def begin(self):
            return self

        def execute(self, sql):
            assert str(sql) == "SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY"
            state.calls.append(("readonly",))
            state.ro = True

    def source(session, *, fund_code):
        assert state.active and state.ro and fund_code == "006730"
        state.calls.append(("source",))
        return state.source

    def nav(session, **kw):
        assert state.active and state.ro
        assert kw["source_id"] == state.source.source_id and kw["fund_code"] == "006730"
        state.calls.append(("nav", kw["start"], kw["end"]))
        if sum(c[0] == "nav" for c in state.calls) == state.fail_nav_page:
            raise SQLAlchemyError("private page SQL details")
        return tuple(p for p in state.nav if kw["start"] <= p.nav_date <= kw["end"])

    def events(session, **kw):
        assert state.active and state.ro
        assert kw["source_id"] == state.source.source_id and kw["fund_code"] == "006730"
        state.calls.append(("events", kw["start"], kw["end"]))
        return select_events(state.events, kw["start"], kw["end"])

    monkeypatch.setattr(service, "get_nav_preview_engine", lambda: None)
    monkeypatch.setattr(service, "Session", FakeSession)
    monkeypatch.setattr(service, "read_historical_nav_source", source)
    monkeypatch.setattr(service, "read_cash_nav_window", nav)
    monkeypatch.setattr(service, "read_cash_dividends", events)
    return state


def select_events(events, start, end):
    return tuple(
        e
        for e in events
        if (e.ann_date is None or e.ann_date <= date(2024, 12, 31))
        and (
            (e.ex_date is None and e.nav_ex_date is None)
            or (e.ex_date is not None and start <= e.ex_date <= end)
            or (e.nav_ex_date is not None and start <= e.nav_ex_date <= end)
        )
    )


def independently_build_each_single(req, reader):
    calendar = load_calendar()
    return tuple(
        build_cash_reinvestment_sample(
            CashSampleRequest(fundCode=req.fund_code, cutoffDate=p.cutoff_date),
            reader.source,
            tuple(row for row in reader.nav if p.read_start <= row.nav_date <= p.read_end),
            select_events(reader.events, p.read_start, p.read_end),
            calendar,
        )
        for p in service.plan_cash_batch(req, calendar)
    )


@pytest.mark.parametrize("size", [1, 2, 7, 10, 30])
def test_each_item_matches_single_and_all_pages_complete_in_one_snapshot(reader, size):
    req = request(pageSize=size)
    output = service.preview_cash_reinvestment_batch(req)
    assert output.items == independently_build_each_single(req, reader)
    assert (
        output.sample_count == output.ready_count == output.input_available_count == output.label_available_count == 20
    )
    assert output.page_count == ceil(20 / size)
    assert not output.input_unavailable_count and not output.label_unavailable_count
    assert len(output.skipped_cutoff_dates) == 10
    assert date(2023, 6, 22) in output.skipped_cutoff_dates and date(2023, 6, 23) in output.skipped_cutoff_dates
    assert sum(c[0] == "source" for c in reader.calls) == sum(c[0] == "events" for c in reader.calls) == 1
    assert sum(c[0] == "nav" for c in reader.calls) == output.page_count
    assert reader.calls[0:3] == [("session",), ("readonly",), ("source",)]
    assert reader.active == 0
    assert not any((output.database_written, output.training_eligible, output.model_fitted, output.test_scored))


def test_page_size_only_changes_two_transport_fields_and_hash_is_verifiable(reader):
    outputs = [service.preview_cash_reinvestment_batch(request(pageSize=n)) for n in (1, 7, 30)]
    assert (
        outputs[0].model_dump(exclude={"page_size", "page_count"})
        == outputs[1].model_dump(exclude={"page_size", "page_count"})
        == outputs[2].model_dump(exclude={"page_size", "page_count"})
    )
    body = outputs[0].model_dump(mode="json", exclude={"page_size", "page_count", "batch_hash"})
    assert (
        hashlib.sha256(json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        == outputs[0].batch_hash
    )


@pytest.mark.parametrize(
    "kind",
    [
        "missing_nav",
        "missing_announcement",
        "zero_nav",
        "future_nav",
        "conflict",
        "undated",
        "unknown_announcement",
        "late_implementation",
        "non_trading_nav",
    ],
)
def test_dirty_data_same_result_for_single_and_all_page_sizes(reader, kind):
    if kind in ("missing_nav", "missing_announcement", "zero_nav", "future_nav"):
        day = date(2023, 7, 25) if kind == "future_nav" else date(2023, 6, 20)
        if kind in ("missing_nav", "future_nav"):
            reader.nav = tuple(p for p in reader.nav if p.nav_date != day)
        else:
            changes = {"ann_date": None} if kind == "missing_announcement" else {"unit_nav": Decimal(0)}
            reader.nav = tuple(replace(p, **changes) if p.nav_date == day else p for p in reader.nav)
    elif kind == "non_trading_nav":
        reader.nav = tuple(
            sorted((*reader.nav, CashNavPoint(date(2023, 6, 18), None, Decimal("NaN"))), key=lambda p: p.nav_date)
        )
    else:
        changes = {
            "conflict": {"nav_ex_date": date(2023, 6, 20)},
            "undated": {"ex_date": None, "nav_ex_date": None},
            "unknown_announcement": {"ann_date": None},
            "late_implementation": {"implementation_ann_date": date(2023, 7, 30)},
        }[kind]
        reader.events = (replace(reader.events[0], **changes),)
    outputs = [service.preview_cash_reinvestment_batch(request(pageSize=n)) for n in (1, 7, 30)]
    assert all(o.items == independently_build_each_single(request(), reader) for o in outputs)
    assert len({o.batch_hash for o in outputs}) == 1 and outputs[0].sample_count == 20
    if kind == "future_nav":
        assert outputs[0].ready_count > 0 and outputs[0].label_unavailable_count > 0
    if kind == "missing_nav":
        assert outputs[0].input_unavailable_count > 0 and outputs[0].label_unavailable_count > 0


def test_bigger_page_does_not_add_neighbor_events_to_single_issue_list(reader):
    plans = service.plan_cash_batch(request(), load_calendar())
    # 第一题根本不读这个未来日期，但大页会读到；不能把整页分红一股脑传给单题。
    day = plans[-1].read_end
    reader.events = (dividend(day, cash_dividend=None),)
    output = service.preview_cash_reinvestment_batch(request(pageSize=30))
    assert output.items == independently_build_each_single(request(), reader)
    assert output.items[0].status == "RESEARCH_SAMPLE_READY" and output.items[-1].status == "LABEL_UNAVAILABLE"


def test_missing_trading_cutoff_is_not_silently_dropped(reader):
    reader.nav = ()
    output = service.preview_cash_reinvestment_batch(request())
    assert output.sample_count == output.input_unavailable_count == 20
    assert output.input_issue_counts == {"NO_KNOWN_TRADING_NAV": 20}
    assert output.label_issue_counts == {"INPUT_UNAVAILABLE": 20}


def test_reason_counts_count_samples_once_not_days(reader):
    reader.events = (replace(reader.events[0], ex_date=None, nav_ex_date=None),)
    reader.nav = tuple(
        replace(p, unit_nav=Decimal(0)) if p.nav_date in (date(2023, 4, 10), date(2023, 4, 11)) else p
        for p in reader.nav
    )
    output = service.preview_cash_reinvestment_batch(request())
    for kind in ("input", "label"):
        expected = {}
        for item in output.items:
            for code in {p.code for p in getattr(item, kind + "_issues")}:
                expected[code] = expected.get(code, 0) + 1
        assert getattr(output, kind + "_issue_counts") == expected
    assert output.input_issue_counts["UNIT_NAV_INVALID"] <= output.sample_count
    assert any(len(item.input_issues) > len({p.code for p in item.input_issues}) for item in output.items)


@pytest.mark.parametrize("start,end", [("2023-06-17", "2023-06-18"), ("2023-06-22", "2023-06-25")])
def test_empty_calendar_range_checks_source_but_does_not_read_values(reader, start, end):
    req = request(startDate=start, endDate=end)
    output = service.preview_cash_reinvestment_batch(req)
    assert output.status == "NO_TRADING_CUTOFFS" and output.items == ()
    assert output.sample_count == output.page_count == output.ready_count == 0
    assert len(output.skipped_cutoff_dates) == (req.end_date - req.start_date).days + 1
    assert reader.calls == [("session",), ("readonly",), ("source",)]


@pytest.mark.parametrize("size", [1, 30])
def test_later_test_period_rejected_before_any_database_work(monkeypatch, size):
    monkeypatch.setattr(service, "get_nav_preview_engine", lambda: pytest.fail("protected request queried DB"))
    with pytest.raises(HistoricalNavPreviewReadError) as caught:
        service.preview_cash_reinvestment_batch(request(startDate="2024-11-20", endDate="2024-12-20", pageSize=size))
    assert caught.value.code == "TEST_PERIOD_PROTECTED"


def test_unpublished_calendar_only_makes_affected_labels_unavailable(reader):
    req = request(startDate="2023-12-01", endDate="2023-12-08")
    output = service.preview_cash_reinvestment_batch(req)
    assert output.input_available_count == output.sample_count
    assert output.label_unavailable_count > 0
    assert output.items == independently_build_each_single(req, reader)


@pytest.mark.parametrize("size", [1, 30])
def test_dividend_limit_is_for_whole_batch_not_each_page(reader, size):
    reader.events = tuple(dividend(date(2023, 6, 21), event_key=str(i)) for i in range(101))
    with pytest.raises(ValueError):
        service.preview_cash_reinvestment_batch(request(pageSize=size))
    assert not any(c[0] == "nav" for c in reader.calls)


def test_duplicate_events_rejected_before_pages(reader):
    reader.events *= 2
    with pytest.raises(ValueError):
        service.preview_cash_reinvestment_batch(request())
    assert reader.active == 0


def test_unsupported_source_does_not_read_values(reader):
    reader.source = replace(reader.source, source_code="OTHER")
    with pytest.raises(HistoricalNavPreviewReadError) as caught:
        service.preview_cash_reinvestment_batch(request())
    assert caught.value.code == "CASH_SOURCE_UNSUPPORTED"
    assert reader.calls == [("session",), ("readonly",), ("source",)]


def test_timeout_and_page_failure_close_snapshot_and_never_return_partial_items(reader, monkeypatch):
    original = service.build_cash_reinvestment_sample
    expired = False

    def compute(*args):
        nonlocal expired
        result = original(*args)
        expired = True
        return result

    monkeypatch.setattr(service, "perf_counter", lambda: 20 if expired else 0)
    monkeypatch.setattr(service, "build_cash_reinvestment_sample", compute)
    with pytest.raises(TimeoutError):
        service.preview_cash_reinvestment_batch(request(pageSize=1))
    assert reader.active == 0 and sum(c[0] == "nav" for c in reader.calls) == 1


def test_middle_page_failure_http_has_no_partial_items(reader, client):
    reader.fail_nav_page = 2
    response = client.get(PATH, params={**PARAMS, "pageSize": 1}, headers=HEADERS)
    assert response.status_code == 503 and "items" not in response.json()
    assert "private" not in response.text and reader.active == 0
    assert sum(c[0] == "nav" for c in reader.calls) == 2


@pytest.mark.parametrize("kind", ["duplicate", "reverse", "outside", "oversized"])
def test_repository_rejects_malformed_window_before_cropping(kind):
    rows = [(date(2023, 6, 1), date(2023, 6, 2), Decimal(1)), (date(2023, 6, 2), date(2023, 6, 3), Decimal(2))]
    rows = (
        rows * 100
        if kind == "oversized"
        else list(reversed(rows))
        if kind == "reverse"
        else rows + [rows[-1]]
        if kind == "duplicate"
        else [(date(2022, 1, 1), None, None)]
    )
    session = SimpleNamespace(execute=lambda statement: SimpleNamespace(all=lambda: rows))
    with pytest.raises(ValueError):
        read_cash_nav_window(
            session, fund_code="006730", source_id=UUID(int=1), start=date(2023, 6, 1), end=date(2023, 6, 30)
        )


def test_http_no_body_returns_complete_result_with_trace(reader, client):
    response = client.get(PATH, params=PARAMS, headers={**HEADERS, "X-Trace-Id": "cash-batch-test"})
    assert response.status_code == 200 and response.headers["X-Trace-Id"] == "cash-batch-test"
    assert response.json()["mode"] == "READ_ONLY_CASH_REINVESTMENT_BATCH_DRY_RUN"
    assert len(response.json()["items"]) == 20


@pytest.mark.parametrize(
    "extra",
    [
        {"fundCode": "000001"},
        {"startDate": "bad"},
        {"startDate": "2021-12-31"},
        {"endDate": "2025-01-01"},
        {"endDate": "2023-05-31"},
        {"endDate": "2023-07-02"},
        {"pageSize": 0},
        {"pageSize": 31},
        {"pageSize": 1.5},
        {"cutoffDate": "2023-06-20"},
        {"includeTest": True},
        {"navBasis": "UNIT_NAV"},
        {"horizon": 30},
        {"requestKey": "not-a-save-endpoint"},
    ],
)
def test_bad_query_rejected_before_work(client, monkeypatch, extra):
    monkeypatch.setattr(api, "preview_cash_reinvestment_batch", lambda req: pytest.fail("invalid work"))
    assert client.get(PATH, params={**PARAMS, **extra}, headers=HEADERS).status_code == 422


@pytest.mark.parametrize("headers", [{}, {"X-Service-Token": "wrong"}, {**HEADERS, "Origin": "http://localhost"}])
def test_auth_before_work(client, monkeypatch, headers):
    monkeypatch.setattr(api, "preview_cash_reinvestment_batch", lambda req: pytest.fail("unauthorized work"))
    assert client.get(PATH, params=PARAMS, headers=headers).status_code == 403


@pytest.mark.parametrize(
    "error,status",
    [
        (HistoricalNavPreviewReadError("FUND_NOT_FOUND", "missing"), 404),
        (HistoricalNavPreviewReadError("SOURCE_NOT_READY", "not ready"), 409),
        (CalendarCoverageError("not covered"), 409),
        (SQLAlchemyError("private SQL"), 503),
        (ValueError("private file"), 503),
        (TimeoutError("private timeout"), 503),
    ],
)
def test_http_error_mapping_does_not_leak_or_return_partial_items(client, monkeypatch, error, status):
    def fail(req):
        raise error

    monkeypatch.setattr(api, "preview_cash_reinvestment_batch", fail)
    response = client.get(PATH, params=PARAMS, headers=HEADERS)
    assert response.status_code == status and "private" not in response.text
    assert "items" not in response.json()
