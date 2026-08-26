"""由 AI 服务维护的 M1 基金目录与净值持久化模型。"""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from uuid import UUID, uuid4

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    Date,
    DateTime,
    ForeignKey,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import UUID as PostgreSQLUUID
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class SourceRegistry(Base):
    """已授权数据源的治理元数据；集成层必须拒绝使用未启用的数据源。"""

    __tablename__ = "source_registry"
    __table_args__ = (
        CheckConstraint("rate_limit_per_minute > 0", name="ck_source_registry_rate_limit_positive"),
        CheckConstraint("retention_days >= 0", name="ck_source_registry_retention_days_nonnegative"),
    )

    source_id: Mapped[UUID] = mapped_column(PostgreSQLUUID(as_uuid=True), primary_key=True, default=uuid4)
    source_code: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    display_name: Mapped[str] = mapped_column(String(128), nullable=False)
    source_kind: Mapped[str] = mapped_column(String(32), nullable=False)
    license_scope: Mapped[str] = mapped_column(Text, nullable=False)
    rate_limit_per_minute: Mapped[int] = mapped_column(Integer, nullable=False)
    retention_days: Mapped[int] = mapped_column(Integer, nullable=False)
    enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False, server_default="false")
    last_success_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    last_error_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    last_error_summary: Mapped[str | None] = mapped_column(String(512))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()
    )


class FundMaster(Base):
    """基金产品主数据；具体份额类别由 FundShareClass 单独承载。"""

    __tablename__ = "fund_master"
    __table_args__ = (UniqueConstraint("manager_name", "fund_name", name="uq_fund_master_manager_name"),)

    fund_master_id: Mapped[UUID] = mapped_column(PostgreSQLUUID(as_uuid=True), primary_key=True, default=uuid4)
    fund_name: Mapped[str] = mapped_column(String(256), nullable=False)
    manager_name: Mapped[str] = mapped_column(String(256), nullable=False)
    fund_type: Mapped[str] = mapped_column(String(32), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    established_date: Mapped[date | None] = mapped_column(Date)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()
    )


class FundShareClass(Base):
    """可交易基金份额类别，是净值数据唯一允许的存储粒度。"""

    __tablename__ = "fund_share_class"

    fund_code: Mapped[str] = mapped_column(String(32), primary_key=True)
    fund_master_id: Mapped[UUID] = mapped_column(
        PostgreSQLUUID(as_uuid=True), ForeignKey("fund_master.fund_master_id"), nullable=False
    )
    share_class: Mapped[str] = mapped_column(String(64), nullable=False)
    fund_name: Mapped[str] = mapped_column(String(256), nullable=False)
    fund_type: Mapped[str] = mapped_column(String(32), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    source_code: Mapped[str] = mapped_column(String(64), nullable=False)
    benchmark_code: Mapped[str | None] = mapped_column(String(64))
    risk_level: Mapped[str | None] = mapped_column(String(32))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()
    )


class NavDaily(Base):
    """按月分区的日净值表，粒度为基金份额类别、净值日期和数据源。"""

    __tablename__ = "nav_daily"
    __table_args__ = (
        CheckConstraint("unit_nav >= 0", name="ck_nav_daily_unit_nav_nonnegative"),
        CheckConstraint(
            "accumulated_nav IS NULL OR accumulated_nav >= 0", name="ck_nav_daily_accumulated_nav_nonnegative"
        ),
        {"postgresql_partition_by": "RANGE (nav_date)"},
    )

    fund_code: Mapped[str] = mapped_column(String(32), ForeignKey("fund_share_class.fund_code"), primary_key=True)
    nav_date: Mapped[date] = mapped_column(Date, primary_key=True)
    source_id: Mapped[UUID] = mapped_column(
        PostgreSQLUUID(as_uuid=True), ForeignKey("source_registry.source_id"), primary_key=True
    )
    unit_nav: Mapped[Decimal] = mapped_column(Numeric(20, 8), nullable=False)
    accumulated_nav: Mapped[Decimal | None] = mapped_column(Numeric(20, 8))
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    source_published_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()
    )


class SourceSyncRun(Base):
    """外部数据源的一次受控同步运行记录，不保存凭据或原始响应。"""

    __tablename__ = "source_sync_run"
    __table_args__ = (
        CheckConstraint("fetched_count >= 0", name="ck_source_sync_run_fetched_nonnegative"),
        CheckConstraint("created_count >= 0", name="ck_source_sync_run_created_nonnegative"),
        CheckConstraint("updated_count >= 0", name="ck_source_sync_run_updated_nonnegative"),
        CheckConstraint("skipped_count >= 0", name="ck_source_sync_run_skipped_nonnegative"),
    )

    sync_run_id: Mapped[UUID] = mapped_column(PostgreSQLUUID(as_uuid=True), primary_key=True, default=uuid4)
    source_id: Mapped[UUID] = mapped_column(
        PostgreSQLUUID(as_uuid=True), ForeignKey("source_registry.source_id"), nullable=False
    )
    sync_type: Mapped[str] = mapped_column(String(32), nullable=False)
    requested_nav_date: Mapped[date | None] = mapped_column(Date)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    fetched_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0, server_default="0")
    created_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0, server_default="0")
    updated_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0, server_default="0")
    skipped_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0, server_default="0")
    error_summary: Mapped[str | None] = mapped_column(String(512))
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
