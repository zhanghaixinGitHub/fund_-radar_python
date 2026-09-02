"""免费已授权场内基金与市场参考指数的持久化模型。"""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from uuid import UUID

from sqlalchemy import CheckConstraint, Date, DateTime, ForeignKey, Index, Integer, Numeric, String, func
from sqlalchemy.dialects.postgresql import UUID as PostgreSQLUUID
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class FundExchangeDaily(Base):
    """场内基金交易日行情，与场外基金净值严格分表保存。"""

    __tablename__ = "fund_exchange_daily"
    __table_args__ = (
        CheckConstraint("close_price > 0", name="ck_fund_exchange_daily_close_positive"),
        Index("ix_fund_exchange_daily_fund_date", "fund_code", "trade_date"),
    )

    fund_code: Mapped[str] = mapped_column(String(32), ForeignKey("fund_share_class.fund_code"), primary_key=True)
    trade_date: Mapped[date] = mapped_column(Date, primary_key=True)
    source_id: Mapped[UUID] = mapped_column(
        PostgreSQLUUID(as_uuid=True), ForeignKey("source_registry.source_id"), primary_key=True
    )
    open_price: Mapped[Decimal | None] = mapped_column(Numeric(20, 8))
    high_price: Mapped[Decimal | None] = mapped_column(Numeric(20, 8))
    low_price: Mapped[Decimal | None] = mapped_column(Numeric(20, 8))
    close_price: Mapped[Decimal] = mapped_column(Numeric(20, 8), nullable=False)
    previous_close_price: Mapped[Decimal | None] = mapped_column(Numeric(20, 8))
    change_value: Mapped[Decimal | None] = mapped_column(Numeric(20, 8))
    change_percent: Mapped[Decimal | None] = mapped_column(Numeric(16, 8))
    volume: Mapped[Decimal | None] = mapped_column(Numeric(24, 4))
    amount: Mapped[Decimal | None] = mapped_column(Numeric(24, 4))
    source_published_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()
    )


class MarketIndexCatalog(Base):
    """Tushare 指数目录；目录入库不代表模型基准已启用。"""

    __tablename__ = "market_index_catalog"
    __table_args__ = (Index("ix_market_index_catalog_category", "category", "display_name"),)

    source_id: Mapped[UUID] = mapped_column(
        PostgreSQLUUID(as_uuid=True), ForeignKey("source_registry.source_id"), primary_key=True
    )
    index_code: Mapped[str] = mapped_column(String(64), primary_key=True)
    display_name: Mapped[str] = mapped_column(String(128), nullable=False)
    market: Mapped[str | None] = mapped_column(String(32))
    publisher: Mapped[str | None] = mapped_column(String(128))
    category: Mapped[str | None] = mapped_column(String(128))
    base_date: Mapped[date | None] = mapped_column(Date)
    list_date: Mapped[date | None] = mapped_column(Date)
    expiry_date: Mapped[date | None] = mapped_column(Date)
    row_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()
    )


class MarketIndexClassification(Base):
    """指数来源分类层级，不映射为任意基金的行业暴露。"""

    __tablename__ = "market_index_classification"
    __table_args__ = (Index("ix_market_index_classification_parent", "source_id", "parent_classification_code"),)

    source_id: Mapped[UUID] = mapped_column(
        PostgreSQLUUID(as_uuid=True), ForeignKey("source_registry.source_id"), primary_key=True
    )
    classification_code: Mapped[str] = mapped_column(String(64), primary_key=True)
    classification_name: Mapped[str] = mapped_column(String(128), nullable=False)
    parent_classification_code: Mapped[str | None] = mapped_column(String(64))
    hierarchy_level: Mapped[int | None] = mapped_column(Integer)
    source_name: Mapped[str | None] = mapped_column(String(128))
    row_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()
    )


class IndexWeightSnapshot(Base):
    """已批准市场参考指数的成分权重快照，不表示基金真实持仓。"""

    __tablename__ = "index_weight_snapshot"
    __table_args__ = (
        CheckConstraint("weight >= 0 AND weight <= 100", name="ck_index_weight_snapshot_range"),
        Index("ix_index_weight_snapshot_index_date", "index_code", "weight_date"),
    )

    source_id: Mapped[UUID] = mapped_column(
        PostgreSQLUUID(as_uuid=True), ForeignKey("source_registry.source_id"), primary_key=True
    )
    index_code: Mapped[str] = mapped_column(String(64), primary_key=True)
    weight_date: Mapped[date] = mapped_column(Date, primary_key=True)
    constituent_code: Mapped[str] = mapped_column(String(32), primary_key=True)
    weight: Mapped[Decimal] = mapped_column(Numeric(12, 8), nullable=False)
    row_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()
    )
