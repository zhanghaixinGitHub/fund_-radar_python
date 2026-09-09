"""三试点源值变化的本地观察日志；不是供应商首次公开版本或事件完整性证明。"""

from datetime import date, datetime
from uuid import UUID

from sqlalchemy import BigInteger, CheckConstraint, Date, DateTime, ForeignKey, Identity, Index, String
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class CashSourceObservation(Base):
    __tablename__ = "cash_source_observation"
    __table_args__ = (
        CheckConstraint("kind IN ('NAV','DIVIDEND')", name="ck_cash_observation_kind"),
        CheckConstraint("fund_code IN ('001632','006730','008888')", name="ck_cash_observation_fund"),
        CheckConstraint("operation IN ('INSERT','UPDATE','DELETE')", name="ck_cash_observation_operation"),
        CheckConstraint("expires_at > observed_at", name="ck_cash_observation_expiry"),
        CheckConstraint("content_hash ~ '^[0-9a-f]{64}$'", name="ck_cash_observation_hash"),
        CheckConstraint(
            "(operation='INSERT' AND before_payload IS NULL AND after_payload IS NOT NULL "
            "AND jsonb_typeof(after_payload)='object') OR "
            "(operation='UPDATE' AND before_payload IS NOT NULL AND after_payload IS NOT NULL "
            "AND jsonb_typeof(before_payload)='object' AND jsonb_typeof(after_payload)='object') OR "
            "(operation='DELETE' AND before_payload IS NOT NULL "
            "AND jsonb_typeof(before_payload)='object' AND after_payload IS NULL)",
            name="ck_cash_observation_payloads",
        ),
        Index("ix_cash_observation_lookup", "source_id", "fund_code", "kind", "event_date", "observation_id"),
        Index("ix_cash_observation_expiry", "expires_at", "observation_id"),
        {"comment": "现金三试点的源值变化观察；只证明本地写入时刻，不代表历史首次公开或完整事件清单"},
    )
    observation_id: Mapped[int] = mapped_column(
        BigInteger, Identity(), primary_key=True, comment="观察记录游标，按生成顺序分页，不当提交顺序证明"
    )
    source_id: Mapped[UUID] = mapped_column(
        PGUUID, ForeignKey("source_registry.source_id"), comment="实际来源主键，不跨来源拼接"
    )
    fund_code: Mapped[str] = mapped_column(
        String(32), ForeignKey("fund_share_class.fund_code"), comment="仅三只现金研究试点"
    )
    kind: Mapped[str] = mapped_column(String(16), comment="NAV净值或DIVIDEND现金分红事件")
    record_key: Mapped[str] = mapped_column(String(64), comment="净值业务日ISO字符串或稳定的来源分红事件键")
    event_date: Mapped[date | None] = mapped_column(Date, comment="净值日或分红净值除息/除息/公告日，均缺失时保留空值")
    operation: Mapped[str] = mapped_column(String(8), comment="源行新增、更新或删除；删除不是无事件证明")
    observed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), comment="触发器实际执行的数据库时间，不是首次公告时间或事务提交时间"
    )
    expires_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), comment="按观察时来源保留天数计算的失效时间；到期不能用作有效证据"
    )
    before_payload: Mapped[dict | None] = mapped_column(
        JSONB, comment="本次写入前的现金相关源值；首次INSERT为空，不将旧值观察时间倒填"
    )
    after_payload: Mapped[dict | None] = mapped_column(
        JSONB, comment="本次写入后的现金相关源值；DELETE为空，不包含来源凭据或原始响应"
    )
    content_hash: Mapped[str] = mapped_column(
        String(64), comment="种类及前后JSONB源值的SHA256，不含观察时间，不是审批签名"
    )
