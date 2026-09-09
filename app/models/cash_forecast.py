"""独立的现金20交易日计算结果；与旧固定评分及拒绝回执分开保存。"""

from datetime import date, datetime
from uuid import UUID

from sqlalchemy import CheckConstraint, Date, DateTime, ForeignKey, Index, String, UniqueConstraint, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class CashForecastRecord(Base):
    __tablename__ = "cash_forecast_result"
    __table_args__ = (
        UniqueConstraint("request_key", name="uq_cash_forecast_request"),
        UniqueConstraint(
            "fund_code",
            "cutoff_date",
            "authorization_hash",
            "feature_hash",
            "source_revision_id",
            name="uq_cash_forecast_business",
        ),
        CheckConstraint(
            "cutoff_date >= DATE '2022-01-01' AND EXTRACT(YEAR FROM cutoff_date) <> 2025",
            name="ck_cash_forecast_cutoff",
        ),
        CheckConstraint(
            "(payload->>'protocol_version' = 'CASH_FORECAST_STORAGE_V1' "
            "AND payload#>>'{feature,status}' = 'INPUT_READY' "
            "AND (payload#>>'{value,up_score}')::numeric BETWEEN 0 AND 1) IS TRUE",
            name="ck_cash_forecast_payload",
        ),
        Index("ix_cash_forecast_fund_date", "fund_code", "cutoff_date", "forecast_id"),
        {"comment": "现金20交易日计算结果和已知输入快照；读取时仍须核验当前发布与数据时效"},
    )
    forecast_id: Mapped[UUID] = mapped_column(PGUUID, primary_key=True, comment="不可变结果编号")
    request_key: Mapped[UUID] = mapped_column(PGUUID, nullable=False, comment="首次成功生成的重试凭证")
    request_hash: Mapped[str] = mapped_column(String(64), comment="生成请求指纹，不含requestKey")
    fund_code: Mapped[str] = mapped_column(String(32), ForeignKey("fund_share_class.fund_code"), comment="所属基金份额")
    cutoff_date: Mapped[date] = mapped_column(Date, comment="本次信息截止日，不随读取时间变化")
    research_run_id: Mapped[UUID] = mapped_column(PGUUID, ForeignKey("cash_research_run.run_id"), comment="原研究编号")
    authorization_hash: Mapped[str] = mapped_column(String(64), comment="当时正式授权指纹；原始凭证引用保存在快照中")
    model_hash: Mapped[str] = mapped_column(String(64), comment="本次真实计算的模型指纹，不是旧模型版本名")
    feature_hash: Mapped[str] = mapped_column(String(64), comment="本次仅含已知信息的输入指纹")
    source_revision_id: Mapped[UUID] = mapped_column(PGUUID, comment="该来源所有数据同步的水位，不只检查净值同步")
    payload: Mapped[dict] = mapped_column(JSONB, comment="明确请求、授权引用、历史输入和真实计算值；无未来答案")
    content_hash: Mapped[str] = mapped_column(String(64), comment="整个结果快照的SHA256，用于回读完整性核对")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), comment="数据库保存实际计算结果的时刻"
    )
