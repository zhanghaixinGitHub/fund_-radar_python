"""基金经理公共目录：人员、来源观察与已发布快照分开，避免同名误合并。"""

from datetime import date, datetime
from decimal import Decimal
from uuid import UUID, uuid4

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    Date,
    DateTime,
    ForeignKey,
    Index,
    Numeric,
    String,
    Text,
    UniqueConstraint,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class FundManager(Base):
    """稳定人物编号；锚定某只基金不等于已证明所有同名任职属于同一人。"""

    __tablename__ = "fund_manager"
    __table_args__ = {"comment": "基金经理人员主档，姓名不唯一，不以基金公司替代人员身份"}
    manager_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True), primary_key=True, default=uuid4, comment="稳定经理编号"
    )
    anchor_key: Mapped[str] = mapped_column(
        String(256), unique=True, comment="原始基金与姓名锚点；仅幂等建档，不作为跨基金归并证明"
    )
    display_name: Mapped[str] = mapped_column(String(128), index=True, comment="公开姓名，可重名")
    identity_status: Mapped[str] = mapped_column(
        String(32), default="ANCHORED", comment="ANCHORED本基金锚点；VERIFIED有跨基金身份依据"
    )
    redirect_to: Mapped[UUID | None] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("fund_manager.manager_id"), comment="有证据合并后的目标人员；空表示未合并"
    )
    current_snapshot_id: Mapped[UUID | None] = mapped_column(
        PGUUID(as_uuid=True), comment="当前已发布快照；查询另校验快照所属人员"
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), comment="本机建档时间"
    )


class FundManagerIdentityEvidence(Base):
    """跨基金任职的显式核对依据；保留原件、接受范围和决定，不能只比较姓名。"""

    __tablename__ = "fund_manager_identity_evidence"
    __table_args__ = (
        UniqueConstraint("manager_id", "content_hash", name="uq_manager_identity_evidence_hash"),
        {"comment": "经理身份归并依据及其接受或待核对决定"},
    )
    evidence_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True), primary_key=True, default=uuid4, comment="身份依据编号"
    )
    manager_id: Mapped[UUID] = mapped_column(ForeignKey("fund_manager.manager_id"), index=True, comment="对应经理")
    source_id: Mapped[UUID] = mapped_column(ForeignKey("source_registry.source_id"), comment="来源登记编号")
    decision: Mapped[str] = mapped_column(String(32), comment="ACCEPTED明确接受的关系；PENDING待核对")
    evidence_ref: Mapped[dict] = mapped_column(JSONB, comment="原件哈希、出处、取得时间、任职锚点及明确接受的基金代码")
    content_hash: Mapped[str] = mapped_column(String(64), comment="依据规范化内容SHA256")
    reviewed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), comment="事实核对时间，非原文公开日")


class FundManagerProfileVersion(Base):
    """公开职业资料按内容版本保存；不请求出生年份、性别等非业务字段。"""

    __tablename__ = "fund_manager_profile_version"
    __table_args__ = (
        UniqueConstraint("manager_id", "content_hash", name="uq_manager_profile_hash"),
        {"comment": "基金经理职业履历版本，保留来源与时间"},
    )
    profile_version_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True), primary_key=True, default=uuid4, comment="履历版本编号"
    )
    manager_id: Mapped[UUID] = mapped_column(ForeignKey("fund_manager.manager_id"), index=True, comment="对应经理")
    education: Mapped[str | None] = mapped_column(String(128), comment="来源明确披露学历，未知为空")
    professional_resume: Mapped[str | None] = mapped_column(Text, comment="与本基金任职锚点匹配的公开职业履历")
    evidence_ref: Mapped[dict] = mapped_column(JSONB, comment="履历出处、锚点及原件引用")
    content_hash: Mapped[str] = mapped_column(String(64), comment="履历内容哈希")
    published_on: Mapped[date | None] = mapped_column(Date, comment="来源公开日，未提供则为空")
    received_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), comment="本机实际取得时间")


class FundManagerRelatedFund(Base):
    """经理关联的外部基金目录；与原ACTIVE市场范围隔离，避免采集自动扩散。"""

    __tablename__ = "fund_manager_related_fund"
    __table_args__ = (
        UniqueConstraint("source_id", "source_fund_code", name="uq_manager_related_source_code"),
        {"comment": "经理关联基金基础目录，不自动加入原日常任务"},
    )
    related_fund_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True), primary_key=True, default=uuid4, comment="关联份额编号"
    )
    source_id: Mapped[UUID] = mapped_column(ForeignKey("source_registry.source_id"), comment="来源编号")
    source_fund_code: Mapped[str] = mapped_column(String(32), comment="来源完整代码，保留交易场所后缀")
    mapped_fund_code: Mapped[str | None] = mapped_column(
        ForeignKey("fund_share_class.fund_code"), comment="已有市场代码；外部份额允许为空"
    )
    fund_name: Mapped[str] = mapped_column(String(256), comment="来源基金份额名称")
    company_name: Mapped[str | None] = mapped_column(String(256), comment="管理公司，不是基金经理姓名")
    fund_type: Mapped[str] = mapped_column(String(32), comment="基金类型，未核实使用OTHER")
    product_identity: Mapped[dict | None] = mapped_column(
        JSONB, comment="经正式资料核对的产品家族、适用期和出处；未知为空"
    )


class FundManagerTenureVersion(Base):
    """规范任期版本；补写离任日期替换当前版本，不产生永久幽灵在管记录。"""

    __tablename__ = "fund_manager_tenure_version"
    __table_args__ = (
        CheckConstraint(
            "end_date IS NULL OR begin_date IS NULL OR end_date >= begin_date", name="ck_manager_tenure_dates"
        ),
        UniqueConstraint("tenure_key", "content_hash", name="uq_manager_tenure_hash"),
        Index("uq_manager_tenure_current", "tenure_key", unique=True, postgresql_where=text("is_current")),
        {"comment": "经理规范任期及来源修订，历史版本不覆盖"},
    )
    tenure_version_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True), primary_key=True, default=uuid4, comment="任期版本编号"
    )
    tenure_key: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), index=True, comment="同一次任职的稳定标识")
    manager_id: Mapped[UUID] = mapped_column(ForeignKey("fund_manager.manager_id"), index=True, comment="经理编号")
    related_fund_id: Mapped[UUID] = mapped_column(
        ForeignKey("fund_manager_related_fund.related_fund_id"), comment="基金份额编号"
    )
    begin_date: Mapped[date | None] = mapped_column(Date, comment="任职首日，未知为空")
    end_date: Mapped[date | None] = mapped_column(Date, comment="最后管理日，空值仍须结合来源覆盖判断在任")
    verification_status: Mapped[str] = mapped_column(String(32), comment="CONFIRMED已锚定关系；PENDING身份或日期待核对")
    team_status: Mapped[str] = mapped_column(
        String(32), default="UNKNOWN", comment="SOLO独立；CO_MANAGED共管；UNKNOWN团队资料不完整"
    )
    is_current: Mapped[bool] = mapped_column(Boolean, default=True, comment="是否当前有效版本，不等于当前仍在任")
    content_hash: Mapped[str] = mapped_column(String(64), comment="本版本任职内容哈希")
    evidence_refs: Mapped[list] = mapped_column(JSONB, comment="任职事实与完整扫描的原件引用")
    observed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), comment="本轮事实取得时间")


class FundManagerCoverageSnapshot(Base):
    """不可变只读投影；目录引用各经理快照，分页不拼接新旧版本。"""

    __tablename__ = "fund_manager_coverage_snapshot"
    __table_args__ = (
        CheckConstraint(
            "(scope_kind='DIRECTORY' AND manager_id IS NULL) OR (scope_kind='MANAGER' AND manager_id IS NOT NULL)",
            name="ck_manager_snapshot_scope",
        ),
        {"comment": "经理资料及目录的不可变发布快照，覆盖与完整性分别说明"},
    )
    snapshot_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True), primary_key=True, default=uuid4, comment="已发布资料版本"
    )
    manager_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("fund_manager.manager_id"), index=True, comment="经理快照所属人员；目录为空"
    )
    scope_kind: Mapped[str] = mapped_column(String(32), index=True, comment="MANAGER经理；DIRECTORY目录")
    payload: Mapped[dict] = mapped_column(JSONB, comment="无用户态的已核实公共读模型和来源覆盖")
    content_hash: Mapped[str] = mapped_column(String(64), comment="发布内容哈希，支持幂等与审计")
    published_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), index=True, comment="原子发布时间，不代替净值日或公告日"
    )


class FundManagerMetricSnapshot(Base):
    """具体份额与任期的指标缓存，绑定任期及输入版本，收益采用小数比例。"""

    __tablename__ = "fund_manager_metric_snapshot"
    __table_args__ = (
        UniqueConstraint("tenure_version_id", "input_hash", "method_version", name="uq_manager_metric_input"),
        {"comment": "经理具体基金任期表现，缺项保留为空，不生成个人总收益"},
    )
    metric_id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True, default=uuid4, comment="指标编号")
    manager_id: Mapped[UUID] = mapped_column(ForeignKey("fund_manager.manager_id"), index=True, comment="经理编号")
    related_fund_id: Mapped[UUID] = mapped_column(
        ForeignKey("fund_manager_related_fund.related_fund_id"), comment="份额编号"
    )
    tenure_version_id: Mapped[UUID] = mapped_column(
        ForeignKey("fund_manager_tenure_version.tenure_version_id"), comment="对应任期版本"
    )
    return_rate: Mapped[Decimal | None] = mapped_column(Numeric(24, 12), comment="收益小数比例，0.1为10%，不可用为空")
    max_drawdown_rate: Mapped[Decimal | None] = mapped_column(Numeric(24, 12), comment="非负最大回撤比例，未核实为空")
    annualized_rate: Mapped[Decimal | None] = mapped_column(
        Numeric(24, 12), comment="完整不少于365日的年化比例，否则为空"
    )
    input_hash: Mapped[str] = mapped_column(String(64), comment="实际计算输入哈希")
    method_version: Mapped[str] = mapped_column(String(64), comment="分红再投等确定计算方法版本")
    payload: Mapped[dict] = mapped_column(JSONB, comment="计算日期、状态、缺项、币种和出处")
    calculated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), comment="计算时间，不代替数据日")
