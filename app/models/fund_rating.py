"""公共基金评级存储；不含用户身份、个人持仓及风险偏好。"""

from datetime import date, datetime
from decimal import Decimal

from sqlalchemy import Boolean, CheckConstraint, Date, DateTime, ForeignKey, Index, Integer, Numeric, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class RatingMethodology(Base):
    __tablename__ = "fund_rating_methodology"
    methodology_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    family: Mapped[str] = mapped_column(String(32), nullable=False)
    config: Mapped[dict] = mapped_column(JSONB, nullable=False)
    # 候选规则不得直接发布；验收记录保存真实输入摘要和试算结果。
    validation: Mapped[dict | None] = mapped_column(JSONB)
    active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class RatingClassification(Base):
    __tablename__ = "fund_rating_classification"
    __table_args__ = (Index("ix_rating_classification_code_date", "fund_code", "effective_from"),)
    classification_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    fund_code: Mapped[str] = mapped_column(String(6), nullable=False)
    product_id: Mapped[str] = mapped_column(String(128), nullable=False)
    category_code: Mapped[str] = mapped_column(String(64), nullable=False)
    family: Mapped[str] = mapped_column(String(32), nullable=False)
    currency: Mapped[str] = mapped_column(String(3), nullable=False)
    eligibility: Mapped[str] = mapped_column(String(64), nullable=False)
    effective_from: Mapped[date] = mapped_column(Date, nullable=False)
    effective_to: Mapped[date | None] = mapped_column(Date)
    source_refs: Mapped[list] = mapped_column(JSONB, nullable=False)
    verified_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    revoked: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)


class RatingEvidence(Base):
    """核验后的规范化原始资料包，按内容寻址；资料更正追加新包并撤回旧包。"""

    __tablename__ = "fund_rating_evidence"
    __table_args__ = (Index("ix_rating_evidence_code_date", "fund_code", "as_of_date"),)
    evidence_hash: Mapped[str] = mapped_column(String(64), primary_key=True)
    fund_code: Mapped[str] = mapped_column(String(6), nullable=False)
    classification_id: Mapped[str] = mapped_column(ForeignKey("fund_rating_classification.classification_id"))
    as_of_date: Mapped[date] = mapped_column(Date, nullable=False)
    payload: Mapped[dict] = mapped_column(JSONB, nullable=False)
    acquired_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    revoked: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)


class RatingBatch(Base):
    __tablename__ = "fund_rating_batch"
    __table_args__ = (
        CheckConstraint("status IN ('COMPLETE','PUBLISHED')", name="ck_rating_batch_status"),
        Index("ix_rating_batch_category_date", "category_code", "as_of_date", "status"),
    )
    # 内容寻址同时充当幂等标识；冻结时刻单独保存，不制造重复计算日期。
    batch_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    category_code: Mapped[str] = mapped_column(String(64), nullable=False)
    methodology_id: Mapped[str | None] = mapped_column(ForeignKey("fund_rating_methodology.methodology_id"))
    as_of_date: Mapped[date] = mapped_column(Date, nullable=False)
    frozen_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    input_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    sample_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    input_path: Mapped[str] = mapped_column(String(128), nullable=False)
    # 实际规范化数值同时保存于 DB；外部原表后续修订不破坏复算。
    input_snapshot: Mapped[dict] = mapped_column(JSONB, nullable=False)
    member_count: Mapped[int] = mapped_column(Integer, nullable=False)
    product_count: Mapped[int] = mapped_column(Integer, nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    completed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    withdrawn: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)


class RatingMember(Base):
    __tablename__ = "fund_rating_member"
    batch_id: Mapped[str] = mapped_column(ForeignKey("fund_rating_batch.batch_id"), primary_key=True)
    fund_code: Mapped[str] = mapped_column(String(6), primary_key=True)
    product_id: Mapped[str | None] = mapped_column(String(128))
    representative: Mapped[bool] = mapped_column(Boolean, nullable=False)
    admitted: Mapped[bool] = mapped_column(Boolean, nullable=False)
    reasons: Mapped[list] = mapped_column(JSONB, nullable=False)
    evidence_hash: Mapped[str | None] = mapped_column(ForeignKey("fund_rating_evidence.evidence_hash"))


class RatingResult(Base):
    __tablename__ = "fund_rating_result"
    __table_args__ = (
        CheckConstraint("score IS NULL OR (score >= 0 AND score <= 100)", name="ck_rating_score_range"),
        CheckConstraint(
            "(status = 'RATED' AND score IS NOT NULL AND grade IS NOT NULL) OR "
            "(status IN ('NOT_RATED','UNSUPPORTED') AND score IS NULL AND grade IS NULL)",
            name="ck_rating_score_state",
        ),
        CheckConstraint(
            "grade IS NULL OR grade = CASE WHEN score < 50 THEN 'WEAK' WHEN score < 70 THEN 'AVERAGE' "
            "WHEN score < 90 THEN 'GOOD' ELSE 'EXCELLENT' END",
            name="ck_rating_score_grade",
        ),
        Index("ix_rating_result_code_batch", "fund_code", "batch_id"),
    )
    batch_id: Mapped[str] = mapped_column(ForeignKey("fund_rating_batch.batch_id"), primary_key=True)
    fund_code: Mapped[str] = mapped_column(String(6), primary_key=True)
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    raw_score: Mapped[str | None] = mapped_column(String(128))
    score: Mapped[Decimal | None] = mapped_column(Numeric(20, 12))
    grade: Mapped[str | None] = mapped_column(String(16))
    dimension_values: Mapped[dict | None] = mapped_column(JSONB)
    public_evidence: Mapped[dict] = mapped_column(JSONB, nullable=False)
    valid_until: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class RatingCurrent(Base):
    __tablename__ = "fund_rating_current"
    category_code: Mapped[str] = mapped_column(String(64), primary_key=True)
    batch_id: Mapped[str] = mapped_column(ForeignKey("fund_rating_batch.batch_id"), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class RatingAudit(Base):
    """只保存任务范围和发布/撤回/回退理由，不保存个人资料。"""

    __tablename__ = "fund_rating_audit"
    audit_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    category_code: Mapped[str] = mapped_column(String(64), nullable=False)
    batch_id: Mapped[str | None] = mapped_column(String(64))
    action: Mapped[str] = mapped_column(String(32), nullable=False)
    reason: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
