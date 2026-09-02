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
    Index,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
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
    authorized_api_names: Mapped[list[str]] = mapped_column(
        JSONB,
        nullable=False,
        default=list,
        server_default=text("'[]'::jsonb"),
    )
    authorization_verified_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
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
    source_fund_code: Mapped[str | None] = mapped_column(String(16), unique=True)
    benchmark_code: Mapped[str | None] = mapped_column(String(64))
    risk_level: Mapped[str | None] = mapped_column(String(32))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()
    )


class FundProfile(Base):
    """一只基金份额的当前基础资料快照，资料来源与净值来源独立可追溯。"""

    __tablename__ = "fund_profile"
    __table_args__ = (
        UniqueConstraint("fund_code", "source_id", name="uq_fund_profile_source_fund"),
        Index("ix_fund_profile_source_fund", "source_id", "fund_code"),
    )

    fund_profile_id: Mapped[UUID] = mapped_column(PostgreSQLUUID(as_uuid=True), primary_key=True, default=uuid4)
    fund_code: Mapped[str] = mapped_column(String(32), ForeignKey("fund_share_class.fund_code"), nullable=False)
    source_id: Mapped[UUID] = mapped_column(
        PostgreSQLUUID(as_uuid=True), ForeignKey("source_registry.source_id"), nullable=False
    )
    management_company_name: Mapped[str | None] = mapped_column(String(256))
    custodian_name: Mapped[str | None] = mapped_column(String(256))
    found_date: Mapped[date | None] = mapped_column(Date)
    due_date: Mapped[date | None] = mapped_column(Date)
    list_date: Mapped[date | None] = mapped_column(Date)
    issue_date: Mapped[date | None] = mapped_column(Date)
    delist_date: Mapped[date | None] = mapped_column(Date)
    issue_amount: Mapped[Decimal | None] = mapped_column(Numeric(20, 8))
    management_fee: Mapped[Decimal | None] = mapped_column(Numeric(12, 8))
    custodian_fee: Mapped[Decimal | None] = mapped_column(Numeric(12, 8))
    duration_year: Mapped[Decimal | None] = mapped_column(Numeric(12, 4))
    par_value: Mapped[Decimal | None] = mapped_column(Numeric(20, 8))
    min_purchase_amount: Mapped[Decimal | None] = mapped_column(Numeric(20, 8))
    expected_return: Mapped[Decimal | None] = mapped_column(Numeric(20, 8))
    benchmark: Mapped[str | None] = mapped_column(String(512))
    invest_type: Mapped[str | None] = mapped_column(String(128))
    source_fund_type: Mapped[str | None] = mapped_column(String(128))
    trustee_name: Mapped[str | None] = mapped_column(String(256))
    purchase_start_date: Mapped[date | None] = mapped_column(Date)
    redemption_start_date: Mapped[date | None] = mapped_column(Date)
    market: Mapped[str | None] = mapped_column(String(8))
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)
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
    ann_date: Mapped[date | None] = mapped_column(Date)
    accumulated_dividend: Mapped[Decimal | None] = mapped_column(Numeric(20, 8))
    net_asset: Mapped[Decimal | None] = mapped_column(Numeric(24, 4))
    total_net_asset: Mapped[Decimal | None] = mapped_column(Numeric(24, 4))
    adjusted_nav: Mapped[Decimal | None] = mapped_column(Numeric(20, 8))
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    source_published_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()
    )


class FundManagerAssignment(Base):
    """基金份额的经理任职历史，只保留产品说明所需的最小公开资料。"""

    __tablename__ = "fund_manager_assignment"
    __table_args__ = (
        UniqueConstraint("fund_code", "source_id", "source_record_key", name="uq_fund_manager_assignment_source"),
        Index("ix_fund_manager_assignment_fund_date", "fund_code", "ann_date", "begin_date"),
    )

    fund_manager_assignment_id: Mapped[UUID] = mapped_column(
        PostgreSQLUUID(as_uuid=True), primary_key=True, default=uuid4
    )
    fund_code: Mapped[str] = mapped_column(String(32), ForeignKey("fund_share_class.fund_code"), nullable=False)
    source_id: Mapped[UUID] = mapped_column(
        PostgreSQLUUID(as_uuid=True), ForeignKey("source_registry.source_id"), nullable=False
    )
    source_record_key: Mapped[str] = mapped_column(String(64), nullable=False)
    manager_name: Mapped[str] = mapped_column(String(128), nullable=False)
    ann_date: Mapped[date | None] = mapped_column(Date)
    begin_date: Mapped[date | None] = mapped_column(Date)
    end_date: Mapped[date | None] = mapped_column(Date)
    education: Mapped[str | None] = mapped_column(String(128))
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()
    )


class FundShareSnapshot(Base):
    """基金份额规模历史，粒度为基金份额、变动日期和来源。"""

    __tablename__ = "fund_share_snapshot"
    __table_args__ = (
        CheckConstraint("fund_share >= 0", name="ck_fund_share_snapshot_nonnegative"),
        Index("ix_fund_share_snapshot_fund_date", "fund_code", "trade_date"),
    )

    fund_code: Mapped[str] = mapped_column(String(32), ForeignKey("fund_share_class.fund_code"), primary_key=True)
    trade_date: Mapped[date] = mapped_column(Date, primary_key=True)
    source_id: Mapped[UUID] = mapped_column(
        PostgreSQLUUID(as_uuid=True), ForeignKey("source_registry.source_id"), primary_key=True
    )
    fund_share: Mapped[Decimal] = mapped_column(Numeric(24, 4), nullable=False)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()
    )


class FundDividend(Base):
    """基金分红事件历史，事件键稳定后允许后续实施状态更新。"""

    __tablename__ = "fund_dividend"
    __table_args__ = (
        Index("ix_fund_dividend_fund_ann_date", "fund_code", "ann_date"),
    )

    fund_code: Mapped[str] = mapped_column(String(32), ForeignKey("fund_share_class.fund_code"), primary_key=True)
    source_id: Mapped[UUID] = mapped_column(
        PostgreSQLUUID(as_uuid=True), ForeignKey("source_registry.source_id"), primary_key=True
    )
    source_event_key: Mapped[str] = mapped_column(String(64), primary_key=True)
    ann_date: Mapped[date | None] = mapped_column(Date)
    implementation_ann_date: Mapped[date | None] = mapped_column(Date)
    base_date: Mapped[date | None] = mapped_column(Date)
    process_status: Mapped[str | None] = mapped_column(String(64))
    record_date: Mapped[date | None] = mapped_column(Date)
    ex_date: Mapped[date | None] = mapped_column(Date)
    pay_date: Mapped[date | None] = mapped_column(Date)
    earnings_pay_date: Mapped[date | None] = mapped_column(Date)
    nav_ex_date: Mapped[date | None] = mapped_column(Date)
    cash_dividend: Mapped[Decimal | None] = mapped_column(Numeric(20, 8))
    base_unit: Mapped[Decimal | None] = mapped_column(Numeric(24, 4))
    distributable_earnings: Mapped[Decimal | None] = mapped_column(Numeric(24, 4))
    earnings_amount: Mapped[Decimal | None] = mapped_column(Numeric(24, 4))
    reinvestment_arrival_date: Mapped[date | None] = mapped_column(Date)
    base_year: Mapped[str | None] = mapped_column(String(16))
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)
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
    parent_sync_run_id: Mapped[UUID | None] = mapped_column(
        PostgreSQLUUID(as_uuid=True), ForeignKey("source_sync_run.sync_run_id")
    )
    sync_type: Mapped[str] = mapped_column(String(32), nullable=False)
    requested_nav_date: Mapped[date | None] = mapped_column(Date)
    requested_window_start: Mapped[date | None] = mapped_column(Date)
    requested_window_end: Mapped[date | None] = mapped_column(Date)
    data_as_of_date: Mapped[date | None] = mapped_column(Date)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    fetched_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0, server_default="0")
    created_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0, server_default="0")
    updated_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0, server_default="0")
    skipped_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0, server_default="0")
    error_summary: Mapped[str | None] = mapped_column(String(512))
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class SourceSyncCursor(Base):
    """一个来源、数据域和实体的最后成功水位与失败摘要。"""

    __tablename__ = "source_sync_cursor"
    __table_args__ = (
        CheckConstraint("consecutive_failure_count >= 0", name="ck_source_sync_cursor_failure_count_nonnegative"),
        Index("ix_source_sync_cursor_dataset_updated", "dataset_code", "updated_at"),
    )

    source_id: Mapped[UUID] = mapped_column(
        PostgreSQLUUID(as_uuid=True), ForeignKey("source_registry.source_id"), primary_key=True
    )
    dataset_code: Mapped[str] = mapped_column(String(64), primary_key=True)
    entity_key: Mapped[str] = mapped_column(String(128), primary_key=True)
    last_successful_data_date: Mapped[date | None] = mapped_column(Date)
    last_successful_published_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    last_sync_run_id: Mapped[UUID | None] = mapped_column(
        PostgreSQLUUID(as_uuid=True), ForeignKey("source_sync_run.sync_run_id")
    )
    consecutive_failure_count: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, server_default="0"
    )
    last_error_summary: Mapped[str | None] = mapped_column(String(512))
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()
    )
