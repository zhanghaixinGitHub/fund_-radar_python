"""新研究与训练前既有规则的不可变关联；不能更新为另一个模型或补绑旧报告。"""

from datetime import datetime
from uuid import UUID

from sqlalchemy import CheckConstraint, DateTime, ForeignKey, String, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class CashPlannedResearchBinding(Base):
    __tablename__ = "cash_planned_research_binding"
    __table_args__ = (
        UniqueConstraint("request_key", name="uq_cash_planned_request"),
        UniqueConstraint("research_run_id", name="uq_cash_planned_research"),
        CheckConstraint("completed_at >= evaluation_started_at", name="ck_cash_planned_times"),
        CheckConstraint(
            "policy_freeze_hash ~ '^[0-9a-f]{64}$' AND report_hash ~ '^[0-9a-f]{64}$' "
            "AND content_hash ~ '^[0-9a-f]{64}$'",
            name="ck_cash_planned_hashes",
        ),
        {"comment": "计算前选定已确认规则的新研究绑定；仅研究、不代表正式数据准入或发布"},
    )
    binding_id: Mapped[UUID] = mapped_column(
        PGUUID, primary_key=True, comment="不可变研究绑定编号，也是内部研究请求凭证"
    )
    request_key: Mapped[UUID] = mapped_column(PGUUID, comment="新研究流程的幂等请求号，不关联旧研究请求号")
    research_run_id: Mapped[UUID] = mapped_column(
        PGUUID, ForeignKey("cash_research_run.run_id"), comment="同事务新建的研究报告编号"
    )
    policy_freeze_id: Mapped[UUID] = mapped_column(
        PGUUID, ForeignKey("cash_policy_freeze.freeze_id"), comment="计算前已保存的规则快照"
    )
    policy_freeze_hash: Mapped[str] = mapped_column(String(64), comment="当时核对的规则快照完整指纹")
    report_hash: Mapped[str] = mapped_column(String(64), comment="本流程生成的研究报告指纹，防止引用被替换")
    preparation: Mapped[dict] = mapped_column(JSONB, comment="计算前资料覆盖准备，含明确批次、日历计划和各窗口可用题数")
    evaluation_started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), comment="核对计划与资料后、评估前取得的数据库时间"
    )
    completed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), comment="评估结束后保存前取得的数据库时间")
    content_hash: Mapped[str] = mapped_column(String(64), comment="身份、时刻、规则、资料及报告引用的SHA256")
