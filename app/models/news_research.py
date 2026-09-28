"""研究公告版本与卡片；不属于已审核业务事件，不被正式事件查询读取。"""

from datetime import date, datetime
from decimal import Decimal

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
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class NewsResearchBundleRow(Base):
    """版本身份和来源清单；完整元数据包含原文位置、摘要和解析版本。"""

    __tablename__ = "news_research_bundle"
    __table_args__ = (
        UniqueConstraint("dataset_key", "revision", name="news_research_bundle_dataset_key_revision_key"),
        CheckConstraint("card_count BETWEEN 1 AND 100"),
        CheckConstraint("bundle_hash ~ '^[0-9a-f]{64}$'"),
        CheckConstraint("fund_code ~ '^[0-9]{6}$'"),
        CheckConstraint("jsonb_typeof(metadata_payload)='object'"),
        CheckConstraint("supersedes_bundle_hash IS DISTINCT FROM bundle_hash"),
    )
    bundle_hash: Mapped[str] = mapped_column(String(64), primary_key=True)
    dataset_key: Mapped[str] = mapped_column(String(128), nullable=False)
    revision: Mapped[str] = mapped_column(String(64), nullable=False)
    fund_code: Mapped[str] = mapped_column(String(6), nullable=False)
    supersedes_bundle_hash: Mapped[str | None] = mapped_column(ForeignKey("news_research_bundle.bundle_hash"))
    card_count: Mapped[int] = mapped_column(Integer, nullable=False)
    metadata_payload: Mapped[dict] = mapped_column(JSONB, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class NewsResearchCardRow(Base):
    """可按基金/日期/公告定位的研究证据；空置信分数保留未知，三个资格分别关闭。"""

    __tablename__ = "news_research_card"
    __table_args__ = (
        Index("ix_news_research_fund_date", "fund_code", "target_date"),
        Index("ix_news_research_announcement", "announcement_id", "pdf_sha256"),
        Index("ix_news_research_topics", "event_topics", postgresql_using="gin"),
        CheckConstraint("record_hash ~ '^[0-9a-f]{64}$'"),
        CheckConstraint("fund_code ~ '^[0-9]{6}$'"),
        CheckConstraint("pdf_sha256 ~ '^[0-9a-f]{64}$'"),
        CheckConstraint("jsonb_typeof(event_topics)='array'"),
        CheckConstraint("jsonb_typeof(record_payload)='object'"),
        CheckConstraint("purpose='RESEARCH'"),
        CheckConstraint("validation_status='LOCAL_ANCHORS_VERIFIED'"),
        CheckConstraint("business_eligible=false"),
        CheckConstraint("prediction_eligible=false"),
        CheckConstraint("training_eligible=false"),
        CheckConstraint("published_date < target_date"),
        CheckConstraint("confidence IS NULL"),
    )
    bundle_hash: Mapped[str] = mapped_column(ForeignKey("news_research_bundle.bundle_hash"), primary_key=True)
    sample_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    record_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    fund_code: Mapped[str] = mapped_column(String(6), nullable=False)
    stock_code: Mapped[str] = mapped_column(String(16), nullable=False)
    target_date: Mapped[date] = mapped_column(Date, nullable=False)
    published_date: Mapped[date] = mapped_column(Date, nullable=False)
    announcement_id: Mapped[str] = mapped_column(String(128), nullable=False)
    pdf_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    event_topics: Mapped[list] = mapped_column(JSONB, nullable=False)
    event_stage: Mapped[str] = mapped_column(Text, nullable=False)
    record_payload: Mapped[dict] = mapped_column(JSONB, nullable=False)
    purpose: Mapped[str] = mapped_column(String(16), nullable=False)
    validation_status: Mapped[str] = mapped_column(String(32), nullable=False)
    confidence: Mapped[Decimal | None] = mapped_column(Numeric(5, 4))
    business_eligible: Mapped[bool] = mapped_column(Boolean, nullable=False)
    prediction_eligible: Mapped[bool] = mapped_column(Boolean, nullable=False)
    training_eligible: Mapped[bool] = mapped_column(Boolean, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
