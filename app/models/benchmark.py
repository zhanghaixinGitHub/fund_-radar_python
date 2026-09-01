"""M3 候选模型回测所需的已授权业绩基准序列。"""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from uuid import UUID

from sqlalchemy import CheckConstraint, Date, DateTime, ForeignKey, Index, Numeric, String, func
from sqlalchemy.dialects.postgresql import UUID as PostgreSQLUUID
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class BenchmarkSeries(Base):
    """一条可追溯的业绩基准定义；启用前必须通过来源与覆盖校验。"""

    __tablename__ = "benchmark_series"
    __table_args__ = (
        CheckConstraint(
            "status IN ('DRAFT', 'ACTIVE', 'SUSPENDED')", name="ck_benchmark_series_status"
        ),
        Index("ix_benchmark_series_fund_type_status", "fund_type", "status"),
        Index("ix_benchmark_series_source_status", "source_id", "status"),
    )

    benchmark_code: Mapped[str] = mapped_column(String(64), primary_key=True)
    display_name: Mapped[str] = mapped_column(String(128), nullable=False)
    fund_type: Mapped[str] = mapped_column(String(32), nullable=False)
    source_id: Mapped[UUID] = mapped_column(
        PostgreSQLUUID(as_uuid=True), ForeignKey("source_registry.source_id"), nullable=False
    )
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="DRAFT", server_default="DRAFT")
    license_reference: Mapped[str] = mapped_column(String(512), nullable=False)
    row_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()
    )


class BenchmarkNavDaily(Base):
    """一条基准在一个交易日的收盘点；粒度固定为基准代码加交易日。"""

    __tablename__ = "benchmark_nav_daily"
    __table_args__ = (
        CheckConstraint("closing_value > 0", name="ck_benchmark_nav_daily_closing_value_positive"),
        Index("ix_benchmark_nav_daily_benchmark_date", "benchmark_code", "nav_date"),
    )

    benchmark_code: Mapped[str] = mapped_column(
        String(64), ForeignKey("benchmark_series.benchmark_code"), primary_key=True
    )
    nav_date: Mapped[date] = mapped_column(Date, primary_key=True)
    closing_value: Mapped[Decimal] = mapped_column(Numeric(20, 8), nullable=False)
    source_published_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    row_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()
    )
