"""基准定义和日序列的数据访问；不在仓储层决定授权或发布状态。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal

from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from app.models.benchmark import BenchmarkNavDaily, BenchmarkSeries
from app.models.fund import SourceRegistry


@dataclass(frozen=True)
class BenchmarkNavPoint:
    """一条已验证的基准日收盘输入。"""

    nav_date: date
    closing_value: Decimal
    source_published_at: datetime | None
    row_hash: str


@dataclass(frozen=True)
class BenchmarkSeriesCoverage:
    """列表和预检所需的基准状态与覆盖范围。"""

    benchmark_code: str
    display_name: str
    fund_type: str
    source_code: str
    source_enabled: bool
    status: str
    license_reference: str
    point_count: int
    first_nav_date: date | None
    last_nav_date: date | None


def find_source_registry(session: Session, source_code: str) -> SourceRegistry | None:
    """按稳定来源编码读取来源登记，不改变其启用状态。"""
    return session.scalar(select(SourceRegistry).where(SourceRegistry.source_code == source_code))


def find_benchmark_series(session: Session, benchmark_code: str, *, lock: bool = False) -> BenchmarkSeries | None:
    """读取一条基准定义；状态转换调用方可选择行锁。"""
    statement = select(BenchmarkSeries).where(BenchmarkSeries.benchmark_code == benchmark_code)
    if lock:
        statement = statement.with_for_update()
    return session.scalar(statement)


def list_benchmark_coverages(session: Session, *, fund_type: str | None = None) -> tuple[BenchmarkSeriesCoverage, ...]:
    """按基准查询覆盖范围，避免逐基准加载全部日序列。"""
    statement = (
        select(
            BenchmarkSeries.benchmark_code,
            BenchmarkSeries.display_name,
            BenchmarkSeries.fund_type,
            SourceRegistry.source_code,
            SourceRegistry.enabled,
            BenchmarkSeries.status,
            BenchmarkSeries.license_reference,
            func.count(BenchmarkNavDaily.nav_date),
            func.min(BenchmarkNavDaily.nav_date),
            func.max(BenchmarkNavDaily.nav_date),
        )
        .join(SourceRegistry, SourceRegistry.source_id == BenchmarkSeries.source_id)
        .outerjoin(BenchmarkNavDaily, BenchmarkNavDaily.benchmark_code == BenchmarkSeries.benchmark_code)
        .group_by(
            BenchmarkSeries.benchmark_code,
            BenchmarkSeries.display_name,
            BenchmarkSeries.fund_type,
            SourceRegistry.source_code,
            SourceRegistry.enabled,
            BenchmarkSeries.status,
            BenchmarkSeries.license_reference,
        )
        .order_by(BenchmarkSeries.fund_type.asc(), BenchmarkSeries.benchmark_code.asc())
    )
    if fund_type is not None:
        statement = statement.where(BenchmarkSeries.fund_type == fund_type)
    return tuple(
        BenchmarkSeriesCoverage(
            benchmark_code=benchmark_code,
            display_name=display_name,
            fund_type=row_fund_type,
            source_code=source_code,
            source_enabled=source_enabled,
            status=status,
            license_reference=license_reference,
            point_count=int(point_count),
            first_nav_date=first_nav_date,
            last_nav_date=last_nav_date,
        )
        for (
            benchmark_code,
            display_name,
            row_fund_type,
            source_code,
            source_enabled,
            status,
            license_reference,
            point_count,
            first_nav_date,
            last_nav_date,
        ) in session.execute(statement).all()
    )


def find_benchmark_coverage(session: Session, benchmark_code: str) -> BenchmarkSeriesCoverage | None:
    """读取指定基准的来源和覆盖摘要。"""
    return next(
        (coverage for coverage in list_benchmark_coverages(session) if coverage.benchmark_code == benchmark_code),
        None,
    )


def list_benchmark_nav_points(session: Session, benchmark_code: str) -> tuple[BenchmarkNavPoint, ...]:
    """按日期读取基准序列；调用方负责确认是否可用于回测。"""
    return tuple(
        BenchmarkNavPoint(
            nav_date=nav_date,
            closing_value=closing_value,
            source_published_at=source_published_at,
            row_hash=row_hash,
        )
        for nav_date, closing_value, source_published_at, row_hash in session.execute(
            select(
                BenchmarkNavDaily.nav_date,
                BenchmarkNavDaily.closing_value,
                BenchmarkNavDaily.source_published_at,
                BenchmarkNavDaily.row_hash,
            )
            .where(BenchmarkNavDaily.benchmark_code == benchmark_code)
            .order_by(BenchmarkNavDaily.nav_date.asc())
        ).all()
    )


def upsert_benchmark_nav_points(
    session: Session,
    *,
    benchmark_code: str,
    points: tuple[BenchmarkNavPoint, ...],
) -> int:
    """使用 PostgreSQL UPSERT 批量幂等保存日序列，不覆盖未变化记录。"""
    if not points:
        return 0
    rows = [
        {
            "benchmark_code": benchmark_code,
            "nav_date": point.nav_date,
            "closing_value": point.closing_value,
            "source_published_at": point.source_published_at,
            "row_hash": point.row_hash,
        }
        for point in points
    ]
    statement = insert(BenchmarkNavDaily).values(rows)
    excluded = statement.excluded
    result = session.execute(
        statement.on_conflict_do_update(
            index_elements=(BenchmarkNavDaily.benchmark_code, BenchmarkNavDaily.nav_date),
            set_={
                "closing_value": excluded.closing_value,
                "source_published_at": excluded.source_published_at,
                "row_hash": excluded.row_hash,
                "updated_at": func.now(),
            },
            where=BenchmarkNavDaily.row_hash != excluded.row_hash,
        )
    )
    return result.rowcount or 0
