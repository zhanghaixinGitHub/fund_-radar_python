"""独立现金研究存储；不覆盖累计净值版本，答案不进入特征列。"""

from datetime import date, datetime
from uuid import UUID

from sqlalchemy import CheckConstraint, Date, DateTime, ForeignKey, Index, String, UniqueConstraint, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class CashSampleBatch(Base):
    """不可变批次封面；请求凭证用于安全重试，内容哈希用于复现。"""

    __tablename__ = "cash_sample_batch"
    __table_args__ = (
        UniqueConstraint("request_key", name="uq_cash_batch_request"),
        CheckConstraint("end_date >= start_date AND end_date - start_date < 31", name="ck_cash_batch_range"),
        Index("ix_cash_batch_fund_date", "fund_code", "start_date"),
        {"comment": "现金再投V1研究批次，不代表正式训练准入"},
    )
    batch_id: Mapped[UUID] = mapped_column(PGUUID, primary_key=True, comment="本次不可变批次编号")
    request_key: Mapped[UUID] = mapped_column(PGUUID, nullable=False, comment="重试凭证；重算使用新编号")
    fund_code: Mapped[str] = mapped_column(String(32), ForeignKey("fund_share_class.fund_code"), comment="所属基金份额")
    start_date: Mapped[date] = mapped_column(Date, comment="信息截止日范围起点，含当天")
    end_date: Mapped[date] = mapped_column(Date, comment="信息截止日范围终点，含当天")
    summary: Mapped[dict] = mapped_column(JSONB, comment="原试跑汇总，不含items；保留规则和来源水位")
    batch_hash: Mapped[str] = mapped_column(String(64), comment="整份试跑内容SHA256，不含分页方式")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), comment="数据库保存时刻"
    )


class CashSampleRecord(Base):
    """每一道题的历史输入；context是诊断与日期，不作为模型矩阵。"""

    __tablename__ = "cash_sample"
    __table_args__ = (
        UniqueConstraint("batch_id", "cutoff_date", name="uq_cash_sample_cutoff"),
        {"comment": "现金再投研究样本；只允许七项历史metrics进入模型"},
    )
    sample_id: Mapped[UUID] = mapped_column(PGUUID, primary_key=True, comment="题目编号")
    batch_id: Mapped[UUID] = mapped_column(PGUUID, ForeignKey("cash_sample_batch.batch_id"), comment="所属批次")
    cutoff_date: Mapped[date] = mapped_column(Date, comment="信息截止日，不是批次创建日")
    context: Mapped[dict] = mapped_column(JSONB, comment="除特征和答案外的原样本诊断，不能作为X")
    feature_payload: Mapped[dict | None] = mapped_column(
        JSONB(none_as_null=True), comment="仅历史输入，禁止答案字段；不可用时SQL NULL"
    )


class CashSampleLabel(Base):
    """后来答案单独存放，训练读取仍必须做时点隔离。"""

    __tablename__ = "cash_sample_label"
    __table_args__ = ({"comment": "现金再投20交易日离线答案，不是预测结果"},)
    sample_id: Mapped[UUID] = mapped_column(
        PGUUID, ForeignKey("cash_sample.sample_id"), primary_key=True, comment="对应题目"
    )
    label_payload: Mapped[dict] = mapped_column(JSONB, comment="独立答案及复算序列，仅离线评估读取")


class CashResearchRun(Base):
    """冻结研究报告和数值模型，任何研究成功都不隐式发布。"""

    __tablename__ = "cash_research_run"
    __table_args__ = (
        UniqueConstraint("request_key", name="uq_cash_research_request"),
        CheckConstraint("publication_status = 'MODEL_NOT_RELEASED'", name="ck_cash_research_no_release"),
        Index("ix_cash_research_created", "created_at"),
        {"comment": "现金口径研究运行，发布准入失败关闭；不存个人数据"},
    )
    run_id: Mapped[UUID] = mapped_column(PGUUID, primary_key=True, comment="本轮研究编号")
    request_key: Mapped[UUID] = mapped_column(PGUUID, nullable=False, comment="研究保存重试凭证")
    dataset_hash: Mapped[str] = mapped_column(String(64), comment="明确批次与数据规则的冻结指纹")
    report: Mapped[dict] = mapped_column(JSONB, comment="准备、窗口成绩、数值JSON模型及未发布原因；无pickle")
    publication_status: Mapped[str] = mapped_column(
        String(32), default="MODEL_NOT_RELEASED", comment="固定未发布，不能用手改ACTIVE替代准入"
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), comment="研究报告保存时刻"
    )
