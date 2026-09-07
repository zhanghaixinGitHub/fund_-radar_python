"""批量出题的独立验收：改变分页大小，不能改变单日答案。

复用临时 SQLite 测试数据，不改真实 PostgreSQL 数据。本文件既跑真实查询组装，
也跑服务循环；仅将 PostgreSQL 的事务声明换成记录，真实只读事务另做本机验收。
"""

from contextlib import contextmanager
from datetime import date, timedelta
from decimal import Decimal
from math import ceil
from uuid import UUID

import pytest
from app.models.fund import NavDaily
from app.schemas.historical_nav import HistoricalNavBatchPreviewRequest
from app.services import historical_nav_preview as preview
from sqlalchemy import update
from sqlalchemy.exc import SQLAlchemyError
from tests.test_historical_nav_repository import session as session  # noqa: F401

START = date(2025, 1, 1)


@pytest.fixture
def reader(session, monkeypatch):
    """让真实仓储和构建器在临时库工作，记录事务声明与 SQL 次数供断言。"""
    statements = []
    exits = []

    class RecordingSession:
        def execute(self, statement):
            statements.append(str(statement))
            if str(statement).startswith("SET TRANSACTION"):
                return None
            return session.execute(statement)

    @contextmanager
    def test_session(_engine):
        try:
            yield RecordingSession()
        finally:
            exits.append(True)

    monkeypatch.setattr(preview, "Session", test_session)
    monkeypatch.setattr(preview, "get_nav_preview_engine", lambda: None)
    return statements, exits


def batch(first: int, last: int, page_size: int):
    """first/last 是测试数据里从0开始的日期偏移，不是数据库 ID。"""
    return preview.preview_stored_historical_nav_batch(HistoricalNavBatchPreviewRequest(
        fund_code="008888", start_date=START + timedelta(days=first),
        end_date=START + timedelta(days=last), page_size=page_size,
    ))


@pytest.mark.parametrize("first,last", [(0, 20), (55, 85), (75, 105), (129, 149)])
def test_batch_pages_and_single_days_have_identical_samples(reader, first, last):
    """覆盖历史不足、完整样本、延迟公告和答案未成熟；逐字段比较，而非只比较数量。"""
    expected = tuple(preview.preview_stored_historical_nav_sample(
        fund_code="008888", as_of_date=START + timedelta(days=index),
    ) for index in range(first, last + 1))
    for size in (1, 7, 10, 30):
        result = batch(first, last, size)
        assert result.items == expected
        assert result.sample_count == len(expected)
        assert result.page_count == ceil(len(expected) / size)
        assert result.scorable_count + result.data_insufficient_count + result.label_not_matured_count == len(expected)
        assert sum(result.unavailable_reasons.values()) == len(expected) - result.scorable_count
        assert len({item.as_of_date for item in result.items}) == len(expected)


@pytest.mark.parametrize("offset,field,value", [
    (80, "ann_date", None),
    (80, "ann_date", START),
    (79, "ann_date", START + timedelta(days=160)),
    (95, "accumulated_nav", None),
    (95, "accumulated_nav", Decimal("0")),
    (95, "ann_date", None),
])
def test_bad_or_delayed_records_do_not_change_page_semantics(reader, session, offset, field, value):
    """只修改内存测试库：坏记录不能被分页跳过，迟到公告也不能提前变成已知信息。"""
    session.execute(update(NavDaily).where(NavDaily.nav_date == START + timedelta(days=offset)).values({field: value}))
    expected = tuple(preview.preview_stored_historical_nav_sample(
        fund_code="008888", as_of_date=START + timedelta(days=index),
    ) for index in range(75, 86))
    assert batch(75, 85, 1).items == batch(75, 85, 10).items == batch(75, 85, 30).items == expected


def test_empty_range_is_not_replaced_with_another_date(reader):
    """范围没有净值时明确返回0，不自动向前或向后寻找别的日期。"""
    result = batch(200, 210, 10)
    assert result.items == ()
    assert result.sample_count == result.page_count == 0
    assert result.unavailable_reasons == {}


def test_pages_issue_bounded_queries_not_one_roundtrip_per_sample(reader):
    """21份题目分3页：1次基金检查、每页2条净值查询，加1条事务声明。"""
    statements, exits = reader
    result = batch(80, 100, 10)
    assert result.sample_count == 21
    assert statements[0] == "SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY"
    assert len(statements) == 2 + 2 * result.page_count
    assert sum("UNION ALL" in sql for sql in statements) == result.page_count
    assert exits == [True]


def test_stale_anchor_is_kept_in_batch_and_reason_count(reader):
    """第79条是夹具内的迟公告起点；它必须保留在原日期，并计入不可用数量。"""
    result = batch(78, 80, 1)
    sample = result.items[1]
    assert sample.as_of_date == START + timedelta(days=79)
    assert sample.unavailable_reason == "STALE_NAV_AT_CUTOFF"
    assert sample.feature_payload["metrics"] is None
    assert sample.offline_label is None
    assert result.sample_count == 3
    assert result.data_insufficient_count == 1
    assert result.unavailable_reasons == {"STALE_NAV_AT_CUTOFF": 1}


def test_out_of_window_announcement_has_identical_single_and_batch_results(reader, session):
    """更新公告在当前页、请求日期段和后20条之外，仍应跨页一致地拒收旧起点。"""
    target = START + timedelta(days=80)
    session.execute(update(NavDaily).where(
        NavDaily.source_id == UUID(int=1), NavDaily.nav_date == target,
    ).values(ann_date=START + timedelta(days=102)))
    session.execute(update(NavDaily).where(
        NavDaily.source_id == UUID(int=1), NavDaily.nav_date > target,
        NavDaily.nav_date <= START + timedelta(days=100),
    ).values(ann_date=START + timedelta(days=130)))
    expected = preview.preview_stored_historical_nav_sample(fund_code="008888", as_of_date=target)
    assert expected.unavailable_reason == "STALE_NAV_AT_CUTOFF"
    results = [batch(80, 82, size) for size in (1, 10, 30)]
    assert all(result.items[0] == expected for result in results)
    assert results[0].items == results[1].items == results[2].items


def test_timeout_releases_session_and_does_not_return_partial_success(reader, monkeypatch):
    """超过预算后抛出异常并离开事务，不伪造一个正常批量响应。"""
    ticks = iter((0, 16))
    monkeypatch.setattr(preview, "perf_counter", lambda: next(ticks))
    with pytest.raises(preview.HistoricalNavBatchTimeoutError):
        batch(80, 100, 10)
    assert reader[1] == [True]


def test_failure_on_second_page_discards_first_page_result(reader, monkeypatch):
    """第一页面算成功、第二页数据库异常时，整次失败，并确保释放连接。"""
    original = preview.read_historical_nav_input_page

    def fail_second_page(*args, **kwargs):
        if kwargs["after_date"] is not None:
            raise SQLAlchemyError("simulated query failure")
        return original(*args, **kwargs)

    monkeypatch.setattr(preview, "read_historical_nav_input_page", fail_second_page)
    with pytest.raises(SQLAlchemyError):
        batch(80, 100, 10)
    assert reader[1] == [True]
