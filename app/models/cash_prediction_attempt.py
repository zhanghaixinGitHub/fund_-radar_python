"""独立的生成拒绝回执，不写旧预测表，也不覆盖研究资料。"""

from datetime import date, datetime
from uuid import UUID

from sqlalchemy import CheckConstraint, Date, DateTime, ForeignKey, Index, String, UniqueConstraint, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class CashPredictionAttemptRecord(Base):
    __tablename__ = "cash_prediction_attempt"
    __table_args__ = (
        UniqueConstraint("request_key", name="uq_cash_prediction_attempt_request"),
        CheckConstraint(
            "cutoff_date >= DATE '2022-01-01' AND EXTRACT(YEAR FROM cutoff_date) <> 2025",
            name="ck_cash_prediction_attempt_cutoff",
        ),
        CheckConstraint(
            "(check_payload->>'status' = 'GENERATION_BLOCKED' "
            "AND check_payload->'up_probability' = 'null'::jsonb "
            "AND check_payload->'direction' = 'null'::jsonb "
            "AND check_payload->'inference_executed' = 'false'::jsonb "
            "AND check_payload->'forecast_created' = 'false'::jsonb "
            "AND jsonb_array_length(check_payload->'blocking_codes') > 0) IS TRUE",
            name="ck_cash_prediction_attempt_rejected",
        ),
        Index("ix_cash_prediction_attempt_fund_created", "fund_code", "created_at", "attempt_id"),
        {"comment": "现金预测生成拒绝回执；不是正式发布或预测结果"},
    )
    attempt_id: Mapped[UUID] = mapped_column(PGUUID, primary_key=True, comment="这份不可变历史回执的编号")
    request_key: Mapped[UUID] = mapped_column(PGUUID, nullable=False, comment="重试凭证；重新检查使用新编号")
    request_hash: Mapped[str] = mapped_column(String(64), comment="基金、日期、报告及版本的请求指纹，不含重试编号")
    fund_code: Mapped[str] = mapped_column(
        String(32), ForeignKey("fund_share_class.fund_code"), comment="计划生成的基金"
    )
    cutoff_date: Mapped[date] = mapped_column(Date, comment="计划使用的信息截止日，拒绝时并未读取该日输入")
    research_run_id: Mapped[UUID] = mapped_column(
        PGUUID, ForeignKey("cash_research_run.run_id"), comment="此次明确检查的冻结研究报告编号"
    )
    report_hash: Mapped[str] = mapped_column(String(64), comment="当时核验的研究报告指纹，不选最新或成绩最优记录")
    check_payload: Mapped[dict] = mapped_column(JSONB, comment="当时的拒绝原因和比较证据；无概率、方向或模型输入")
    receipt_hash: Mapped[str] = mapped_column(String(64), comment="请求指纹与完整检查快照的联合指纹")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), comment="数据库保存回执的实际时刻"
    )
