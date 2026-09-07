"""用临时 SQLite 内存库验证净值查询的来源、时间边界及条数上限。

这里创建的表和数据只存在于测试中，不会修改项目的 PostgreSQL 数据库。
这些测试不证明 PostgreSQL 只读事务配置或真实数据源授权已经生效。
"""

from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from uuid import UUID

import pytest
from app.repositories.feature_snapshot import FeatureSourceReadiness
from app.repositories.historical_nav import HistoricalNavPreviewReadError, read_historical_nav_sample_input
from sqlalchemy import Column, Date, DateTime, MetaData, Numeric, String, Table, Uuid, create_engine, insert
from sqlalchemy.orm import Session


@pytest.fixture
def session(monkeypatch):
    """准备最少的表和人工净值，测试结束后释放内存数据库。"""
    engine = create_engine("sqlite://")
    metadata = MetaData()
    funds = Table("fund_share_class", metadata, Column("fund_code", String), Column("fund_type", String),
                  Column("status", String), Column("source_code", String))
    nav = Table("nav_daily", metadata, Column("fund_code", String), Column("source_id", Uuid),
                Column("nav_date", Date), Column("ann_date", Date), Column("unit_nav", Numeric),
                Column("accumulated_nav", Numeric), Column("updated_at", DateTime))
    # updated_at 供异常场景测试修改净值时使用；ORM 会自动更新它，实际项目表本来就有此列。
    metadata.create_all(engine)
    # 暂时把来源检查替换为“来源已就绪”，让这些用例专注检验净值读取规则。
    source = FeatureSourceReadiness(UUID(int=1), "TEST_SOURCE", UUID(int=3), datetime(2026, 1, 1, tzinfo=UTC))
    monkeypatch.setattr("app.repositories.historical_nav.get_enabled_feature_source", lambda *args: source)
    start = date(2025, 1, 1)
    with engine.begin() as conn:
        conn.execute(insert(funds), {"fund_code": "008888", "fund_type": "STOCK", "status": "ACTIVE",
                                     "source_code": "TEST_SOURCE"})
        rows = []
        for index in range(150):
            day = start + timedelta(days=index)
            value = Decimal("1") + Decimal(index) / 100
            rows.append({"fund_code": "008888", "source_id": source.source_id, "nav_date": day,
                         "ann_date": day + timedelta(days=10 if index == 79 else 1),
                         "unit_nav": value, "accumulated_nav": value})
            # 为同一基金、同一天再造一份其他来源的值，检查查询是否混入错误来源。
            rows.append({**rows[-1], "source_id": UUID(int=2), "accumulated_nav": Decimal("999")})
        conn.execute(insert(nav), rows)
    try:
        with Session(engine) as db_session:
            yield db_session
    finally:
        engine.dispose()


def test_reader_bounds_history_and_future_and_never_mixes_sources(session) -> None:
    """只取 60 条已知历史 + 1 条起点 + 20 条未来，并且全部来自指定来源。"""
    start = date(2025, 1, 1)
    result = read_historical_nav_sample_input(session, fund_code="008888", as_of_date=start + timedelta(days=80))
    assert len(result.nav_points) == 81
    # 紧邻起点的一条净值延迟公告，应从更早日期补足60条已知历史。
    assert result.nav_points[0].nav_date == start + timedelta(days=19)
    assert all(p.nav_date != start + timedelta(days=79) for p in result.nav_points)
    assert result.nav_points[60].nav_date == start + timedelta(days=80)
    assert result.nav_points[-1].nav_date == start + timedelta(days=100)
    assert all(p.accumulated_nav != Decimal("999") for p in result.nav_points)


def test_reader_does_not_fall_back_when_target_nav_date_is_missing(session) -> None:
    """指定日期没有净值时明确报错，不偷偷换成附近某天的净值。"""
    with pytest.raises(HistoricalNavPreviewReadError) as error:
        read_historical_nav_sample_input(session, fund_code="008888", as_of_date=date(2024, 1, 1))
    assert error.value.code == "NAV_NOT_FOUND"


def test_reader_rejects_missing_fund(session) -> None:
    """基金不存在时返回明确的业务错误，而不是继续查询或构造空基金。"""
    with pytest.raises(HistoricalNavPreviewReadError) as error:
        read_historical_nav_sample_input(session, fund_code="999999", as_of_date=date(2025, 1, 1))
    assert error.value.code == "FUND_NOT_FOUND"


def test_reader_rejects_disabled_source(session, monkeypatch) -> None:
    """来源检查没有返回可用来源时停止处理，不绕过来源限制读取净值。"""
    monkeypatch.setattr("app.repositories.historical_nav.get_enabled_feature_source", lambda *args: None)
    with pytest.raises(HistoricalNavPreviewReadError) as error:
        read_historical_nav_sample_input(session, fund_code="008888", as_of_date=date(2025, 1, 1))
    assert error.value.code == "SOURCE_NOT_READY"
