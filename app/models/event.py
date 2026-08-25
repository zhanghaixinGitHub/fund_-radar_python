"""由 AI 服务维护的 M2 事件持久化模型。

这些模型只保存可追溯所需的已授权元数据和摘要；除非数据源许可范围明确允许，集成层不得持久化原始正文。
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from uuid import UUID, uuid4

from sqlalchemy import Boolean, CheckConstraint, DateTime, ForeignKey, Numeric, String, Text, UniqueConstraint, func
from sqlalchemy.dialects.postgresql import UUID as PostgreSQLUUID
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class NewsItem(Base):
    """去重后的资讯主记录，只保存许可范围内的标准化元数据。"""

    __tablename__ = "news_item"
    __table_args__ = (UniqueConstraint("content_hash", name="uq_news_item_content_hash"),)

    news_id: Mapped[UUID] = mapped_column(PostgreSQLUUID(as_uuid=True), primary_key=True, default=uuid4)
    source_id: Mapped[UUID] = mapped_column(
        PostgreSQLUUID(as_uuid=True), ForeignKey("source_registry.source_id"), nullable=False
    )
    title: Mapped[str] = mapped_column(String(512), nullable=False)
    url: Mapped[str] = mapped_column(String(2048), nullable=False)
    summary: Mapped[str | None] = mapped_column(Text)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    published_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    fetched_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
    retention_until: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True, server_default="true")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()
    )


class NewsSourceReference(Base):
    """保留每个获准来源引用，避免重复存储标准化资讯内容。"""

    __tablename__ = "news_source_reference"
    __table_args__ = (UniqueConstraint("source_id", "url", name="uq_news_source_reference_source_url"),)

    reference_id: Mapped[UUID] = mapped_column(PostgreSQLUUID(as_uuid=True), primary_key=True, default=uuid4)
    news_id: Mapped[UUID] = mapped_column(PostgreSQLUUID(as_uuid=True), ForeignKey("news_item.news_id"), nullable=False)
    source_id: Mapped[UUID] = mapped_column(
        PostgreSQLUUID(as_uuid=True), ForeignKey("source_registry.source_id"), nullable=False
    )
    url: Mapped[str] = mapped_column(String(2048), nullable=False)
    published_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())


class MarketEvent(Base):
    """从单条标准化资讯提炼出的已审核事件。"""

    __tablename__ = "market_event"
    __table_args__ = (
        CheckConstraint("confidence >= 0 AND confidence <= 1", name="ck_market_event_confidence_range"),
        CheckConstraint(
            "approval_status IN ('PENDING', 'APPROVED', 'REJECTED')", name="ck_market_event_approval_status"
        ),
        UniqueConstraint("event_hash", name="uq_market_event_event_hash"),
    )

    event_id: Mapped[UUID] = mapped_column(PostgreSQLUUID(as_uuid=True), primary_key=True, default=uuid4)
    news_id: Mapped[UUID] = mapped_column(PostgreSQLUUID(as_uuid=True), ForeignKey("news_item.news_id"), nullable=False)
    event_type: Mapped[str] = mapped_column(String(64), nullable=False)
    summary: Mapped[str] = mapped_column(Text, nullable=False)
    explanation: Mapped[str] = mapped_column(Text, nullable=False)
    confidence: Mapped[Decimal] = mapped_column(Numeric(5, 4), nullable=False)
    published_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    event_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    approval_status: Mapped[str] = mapped_column(
        String(16), nullable=False, default="PENDING", server_default="PENDING"
    )
    approved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()
    )


class EventRelation(Base):
    """已审核事件与领域实体之间可解释的关联关系。"""

    __tablename__ = "event_relation"
    __table_args__ = (
        CheckConstraint("relevance_score >= 0 AND relevance_score <= 1", name="ck_event_relation_relevance_range"),
        CheckConstraint(
            "entity_type IN ('FUND_CODE', 'FUND_MANAGER', 'INDUSTRY', 'INDEX', 'COMPANY', 'POLICY_TOPIC')",
            name="ck_event_relation_entity_type",
        ),
        UniqueConstraint("event_id", "entity_type", "entity_id", name="uq_event_relation_entity"),
    )

    relation_id: Mapped[UUID] = mapped_column(PostgreSQLUUID(as_uuid=True), primary_key=True, default=uuid4)
    event_id: Mapped[UUID] = mapped_column(
        PostgreSQLUUID(as_uuid=True), ForeignKey("market_event.event_id"), nullable=False
    )
    entity_type: Mapped[str] = mapped_column(String(32), nullable=False)
    entity_id: Mapped[str] = mapped_column(String(256), nullable=False)
    relevance_score: Mapped[Decimal] = mapped_column(Numeric(5, 4), nullable=False)
    relation_reason: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
