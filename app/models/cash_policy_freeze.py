"""已确认规则的不可变快照；数据库另用触发器阻止UPDATE、DELETE和TRUNCATE。"""

from datetime import datetime
from uuid import UUID

from sqlalchemy import CheckConstraint, DateTime, String, UniqueConstraint, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class CashPolicyFreezeRecord(Base):
    __tablename__ = "cash_policy_freeze"
    __table_args__ = (
        UniqueConstraint("request_key", name="uq_cash_policy_request"),
        UniqueConstraint("policy_version", name="uq_cash_policy_version"),
        CheckConstraint(
            "policy_hash ~ '^[0-9a-f]{64}$' AND binding_hash ~ '^[0-9a-f]{64}$' AND content_hash ~ '^[0-9a-f]{64}$'",
            name="ck_cash_policy_hashes",
        ),
        CheckConstraint(
            "(snapshot->>'version' = 'CASH_POLICY_FREEZE_V1' "
            "AND snapshot#>>'{policy,approval_state}' = 'APPROVED' "
            "AND snapshot#>>'{policy,version}' = policy_version "
            "AND length(btrim(snapshot#>>'{policy,approval_reference}')) > 0) IS TRUE",
            name="ck_cash_policy_approved",
        ),
        {"comment": "已确认的现金发布规则不可变快照；不代表模型获准发布或最终测试已完成"},
    )
    freeze_id: Mapped[UUID] = mapped_column(PGUUID, primary_key=True, comment="不可变规则快照编号，非模型授权")
    request_key: Mapped[UUID] = mapped_column(PGUUID, comment="首次冻结请求的重试凭证")
    policy_version: Mapped[str] = mapped_column(String(64), comment="规则版本，每版本仅允许一份内容")
    policy_hash: Mapped[str] = mapped_column(String(64), comment="规则正文SHA256，不是审批签名")
    binding_hash: Mapped[str] = mapped_column(String(64), comment="固定窗口、口径及比较约定的SHA256")
    snapshot: Mapped[dict] = mapped_column(JSONB, comment="当时已确认的完整规则和比较约定，不含模型或答案")
    content_hash: Mapped[str] = mapped_column(String(64), comment="身份、时间与完整快照的SHA256")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), comment="数据库时钟取得的实际冻结时间"
    )
