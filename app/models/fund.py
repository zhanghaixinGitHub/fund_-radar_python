"""由 AI 服务维护的 M1 基金目录与净值持久化模型。"""

from __future__ import annotations

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
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import UUID as PostgreSQLUUID
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class SourceRegistry(Base):
    """来源登记表：说明数据来自哪里、允许怎样使用，以及这个来源是否启用。

    这是数据库实体：Mapped[T]标注Python字段类型，mapped_column说明数据库列的类型和约束。
    nullable表示数据库可否为空，primary_key表示主键，ForeignKey表示关联另一张表。
    与历史样本的dataclass不同，修改实体并提交事务可以写库；本次GET在只读事务中只取字段。
    """

    __tablename__ = "source_registry"
    __table_args__ = (
        CheckConstraint("rate_limit_per_minute > 0", name="ck_source_registry_rate_limit_positive"),
        CheckConstraint("retention_days >= 0", name="ck_source_registry_retention_days_nonnegative"),
    )

    # 来源主键（UUID唯一标识）；nav_daily.source_id用它关联来源并隔离不同来源的净值。
    source_id: Mapped[UUID] = mapped_column(PostgreSQLUUID(as_uuid=True), primary_key=True, default=uuid4)
    # 可读的唯一来源编码，例如TUSHARE_PRO_FUND；基金目录用source_code指定自己的来源。
    source_code: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    # 来源的展示名称，给人看，不用于替代唯一编码关联数据。
    display_name: Mapped[str] = mapped_column(String(128), nullable=False)
    # 来源类别标记，用于区分不同来源的接入/治理方式。
    source_kind: Mapped[str] = mapped_column(String(32), nullable=False)
    # 登记的授权用途和使用范围说明；不存储来源账号密码或Token。
    license_scope: Mapped[str] = mapped_column(Text, nullable=False)
    # 来源允许的每分钟请求次数；约束采集频率，本次读库GET不会调用来源接口。
    rate_limit_per_minute: Mapped[int] = mapped_column(Integer, nullable=False)
    # 登记的数据保留天数；其含义是来源治理配置，不是模型预测天数。
    retention_days: Mapped[int] = mapped_column(Integer, nullable=False)
    # 是否启用该来源；False时本次预览会拒绝使用其数据。
    enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False, server_default="false")
    # 已登记授权的来源API名称列表，以JSON数组保存；不是用户可访问的HTTP接口列表。
    authorized_api_names: Mapped[list[str]] = mapped_column(
        JSONB,
        nullable=False,
        default=list,
        server_default=text("'[]'::jsonb"),
    )
    # 最近一次核验来源授权的时间；None表示没有登记该核验时间。
    authorization_verified_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    # 最近一次来源同步成功的时间，不等于某条净值的公告日。
    last_success_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    # 最近一次来源错误发生时间，供排障和状态展示。
    last_error_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    # 错误摘要，应保持简洁脱敏；不是完整来源响应。
    last_error_summary: Mapped[str | None] = mapped_column(String(512))
    # 来源登记记录的创建时间，不是来源数据本身的业务日期。
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
    # 来源登记记录的更新时间；DateTime(timezone=True)表示该时间字段携带时区语义。
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()
    )


class FundMaster(Base):
    """基金产品主数据；具体份额类别由 FundShareClass 单独承载。"""

    __tablename__ = "fund_master"
    __table_args__ = (UniqueConstraint("manager_name", "fund_name", name="uq_fund_master_manager_name"),)

    fund_master_id: Mapped[UUID] = mapped_column(PostgreSQLUUID(as_uuid=True), primary_key=True, default=uuid4)
    fund_name: Mapped[str] = mapped_column(String(256), nullable=False)
    manager_name: Mapped[str] = mapped_column(String(256), nullable=False)
    fund_type: Mapped[str] = mapped_column(String(32), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    established_date: Mapped[date | None] = mapped_column(Date)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()
    )


class FundShareClass(Base):
    """基金份额目录：同一产品的A/C等份额可以有不同代码，净值按具体份额代码关联。

    本次预览读取fund_type、status、source_code，分别决定是否适用、是否启用、选哪个来源。
    """

    __tablename__ = "fund_share_class"

    # 份额代码，如008888，是本表主键，也是GET参数fundCode的查询目标。
    fund_code: Mapped[str] = mapped_column(String(32), primary_key=True)
    # 归属的基金产品主键，关联fund_master；同一产品下可以有多个份额类别。
    fund_master_id: Mapped[UUID] = mapped_column(
        PostgreSQLUUID(as_uuid=True), ForeignKey("fund_master.fund_master_id"), nullable=False
    )
    # 份额类别，如A/C；不同类别不能只因属于同一产品就混合净值历史。
    share_class: Mapped[str] = mapped_column(String(64), nullable=False)
    # 该份额的展示名称，不参与本次历史特征计算。
    fund_name: Mapped[str] = mapped_column(String(256), nullable=False)
    # 标准化基金类型，本阶段只接受STOCK（股票型）。
    fund_type: Mapped[str] = mapped_column(String(32), nullable=False)
    # 当前目录状态，本阶段只读取ACTIVE（启用）的基金。
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    # 来源编码，用于找到来源登记表，再按来源主键筛选净值。
    source_code: Mapped[str] = mapped_column(String(64), nullable=False)
    # 来源系统中的完整基金代码，可能含.OF等后缀；与页面使用的6位fund_code区分。
    # 后缀	含义
    # .OF	场外基金
    # .SH	上海证券交易所
    # .SZ	深圳证券交易所
    source_fund_code: Mapped[str | None] = mapped_column(String(16), unique=True)
    # 已登记的参考基准编码，可为空；本阶段只用自身净值，不使用此字段作特征。
    benchmark_code: Mapped[str | None] = mapped_column(String(64))
    # 目录中的风险等级，可为空；不是本次计算产生的风险评分。
    risk_level: Mapped[str | None] = mapped_column(String(32))
    # 本地目录记录何时创建，与基金成立日不同。
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
    # 本地目录记录何时更新，不能作为历史净值的可得日期。
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()
    )


class FundProfile(Base):
    """一只基金份额的当前基础资料快照，资料来源与净值来源独立可追溯。"""

    __tablename__ = "fund_profile"
    __table_args__ = (
        UniqueConstraint("fund_code", "source_id", name="uq_fund_profile_source_fund"),
        Index("ix_fund_profile_source_fund", "source_id", "fund_code"),
    )

    fund_profile_id: Mapped[UUID] = mapped_column(PostgreSQLUUID(as_uuid=True), primary_key=True, default=uuid4)
    fund_code: Mapped[str] = mapped_column(String(32), ForeignKey("fund_share_class.fund_code"), nullable=False)
    source_id: Mapped[UUID] = mapped_column(
        PostgreSQLUUID(as_uuid=True), ForeignKey("source_registry.source_id"), nullable=False
    )
    management_company_name: Mapped[str | None] = mapped_column(String(256))
    custodian_name: Mapped[str | None] = mapped_column(String(256))
    found_date: Mapped[date | None] = mapped_column(Date)
    due_date: Mapped[date | None] = mapped_column(Date)
    list_date: Mapped[date | None] = mapped_column(Date)
    issue_date: Mapped[date | None] = mapped_column(Date)
    delist_date: Mapped[date | None] = mapped_column(Date)
    issue_amount: Mapped[Decimal | None] = mapped_column(Numeric(20, 8))
    management_fee: Mapped[Decimal | None] = mapped_column(Numeric(12, 8))
    custodian_fee: Mapped[Decimal | None] = mapped_column(Numeric(12, 8))
    duration_year: Mapped[Decimal | None] = mapped_column(Numeric(12, 4))
    par_value: Mapped[Decimal | None] = mapped_column(Numeric(20, 8))
    min_purchase_amount: Mapped[Decimal | None] = mapped_column(Numeric(20, 8))
    expected_return: Mapped[Decimal | None] = mapped_column(Numeric(20, 8))
    benchmark: Mapped[str | None] = mapped_column(String(512))
    invest_type: Mapped[str | None] = mapped_column(String(128))
    source_fund_type: Mapped[str | None] = mapped_column(String(128))
    trustee_name: Mapped[str | None] = mapped_column(String(256))
    purchase_start_date: Mapped[date | None] = mapped_column(Date)
    redemption_start_date: Mapped[date | None] = mapped_column(Date)
    market: Mapped[str | None] = mapped_column(String(8))
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()
    )


class NavDaily(Base):
    """日净值实体，一行由“基金代码 + 净值业务日 + 来源ID”共同确定。

    同一基金同一天可以有不同来源的记录，因此读取时必须同时限制来源。
    本次GET只取nav_date、ann_date、unit_nav、accumulated_nav四列计算，其余列用于关联或追溯。
    """

    __tablename__ = "nav_daily"
    # 表按净值日做范围分区。数据库约束允许存0以记录来源事实，但样本计算要求净值严格大于0。
    __table_args__ = (
        CheckConstraint("unit_nav >= 0", name="ck_nav_daily_unit_nav_nonnegative"),
        CheckConstraint(
            "accumulated_nav IS NULL OR accumulated_nav >= 0", name="ck_nav_daily_accumulated_nav_nonnegative"
        ),
        {"postgresql_partition_by": "RANGE (nav_date)"},
    )

    # 净值属于哪只基金份额；关联fund_share_class，也是联合主键的一部分。
    fund_code: Mapped[str] = mapped_column(String(32), ForeignKey("fund_share_class.fund_code"), primary_key=True)
    # 净值业务日，例如2025-08-07；作为分区键和联合主键，不表示当日已经公告。
    nav_date: Mapped[date] = mapped_column(Date, primary_key=True)
    # 净值来源ID，关联source_registry；防止把不同来源的记录重复计数或拼接。
    source_id: Mapped[UUID] = mapped_column(
        PostgreSQLUUID(as_uuid=True), ForeignKey("source_registry.source_id"), primary_key=True
    )
    # 来源公布的单位净值；Numeric(20,8)表示最多20位数字、其中小数8位，Python用Decimal承接。
    unit_nav: Mapped[Decimal] = mapped_column(Numeric(20, 8), nullable=False)
    # 来源公布的累计净值，可缺失；当时可见窗口全部有效时，阶段2优先整段使用此口径。
    accumulated_nav: Mapped[Decimal | None] = mapped_column(Numeric(20, 8))
    # 来源记录的公告日期，是本次历史可见性筛选的依据；没有小时/分钟精度。
    ann_date: Mapped[date | None] = mapped_column(Date)
    # 来源提供的累计分红字段，阶段2未参与计算，也没有据此自行还原复权收益。
    accumulated_dividend: Mapped[Decimal | None] = mapped_column(Numeric(20, 8))
    # 来源的净资产字段；此实体保存数值、不在这里换算金额单位，阶段2不读取。
    net_asset: Mapped[Decimal | None] = mapped_column(Numeric(24, 4))
    # 来源的合计净资产字段，与net_asset分开保存；阶段2不作为特征。
    total_net_asset: Mapped[Decimal | None] = mapped_column(Numeric(24, 4))
    # 调整后净值字段，可为空；当前阶段2构建器选择累计或单位净值，不使用此列。
    adjusted_nav: Mapped[Decimal | None] = mapped_column(Numeric(20, 8))
    # 原始净值业务内容的摘要，用于同步时识别是否变化；不同于输出样本的feature_hash。
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    # 来源发布时刻的时间字段，可为空；当前历史样本使用ann_date，没有用它实现盘中回放。
    source_published_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    # 这行记录进入本地数据库的时间，可能远晚于净值业务日/公告日，不能当历史可得日。
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
    # 本地记录最后更新时间；当前表不因保留此字段就自动具备历史修订版本回放能力。
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()
    )


class FundManagerAssignment(Base):
    """基金份额的经理任职历史，只保留产品说明所需的最小公开资料。"""

    __tablename__ = "fund_manager_assignment"
    __table_args__ = (
        UniqueConstraint("fund_code", "source_id", "source_record_key", name="uq_fund_manager_assignment_source"),
        Index("ix_fund_manager_assignment_fund_date", "fund_code", "ann_date", "begin_date"),
    )

    fund_manager_assignment_id: Mapped[UUID] = mapped_column(
        PostgreSQLUUID(as_uuid=True), primary_key=True, default=uuid4
    )
    fund_code: Mapped[str] = mapped_column(String(32), ForeignKey("fund_share_class.fund_code"), nullable=False)
    source_id: Mapped[UUID] = mapped_column(
        PostgreSQLUUID(as_uuid=True), ForeignKey("source_registry.source_id"), nullable=False
    )
    source_record_key: Mapped[str] = mapped_column(String(64), nullable=False)
    manager_name: Mapped[str] = mapped_column(String(128), nullable=False)
    ann_date: Mapped[date | None] = mapped_column(Date)
    begin_date: Mapped[date | None] = mapped_column(Date)
    end_date: Mapped[date | None] = mapped_column(Date)
    education: Mapped[str | None] = mapped_column(String(128))
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()
    )


class FundShareSnapshot(Base):
    """基金份额规模历史，粒度为基金份额、变动日期和来源。"""

    __tablename__ = "fund_share_snapshot"
    __table_args__ = (
        CheckConstraint("fund_share >= 0", name="ck_fund_share_snapshot_nonnegative"),
        Index("ix_fund_share_snapshot_fund_date", "fund_code", "trade_date"),
    )

    fund_code: Mapped[str] = mapped_column(String(32), ForeignKey("fund_share_class.fund_code"), primary_key=True)
    trade_date: Mapped[date] = mapped_column(Date, primary_key=True)
    source_id: Mapped[UUID] = mapped_column(
        PostgreSQLUUID(as_uuid=True), ForeignKey("source_registry.source_id"), primary_key=True
    )
    fund_share: Mapped[Decimal] = mapped_column(Numeric(24, 4), nullable=False)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()
    )


class FundDividend(Base):
    """基金分红事件历史，事件键稳定后允许后续实施状态更新。"""

    __tablename__ = "fund_dividend"
    __table_args__ = (
        Index("ix_fund_dividend_fund_ann_date", "fund_code", "ann_date"),
    )

    fund_code: Mapped[str] = mapped_column(String(32), ForeignKey("fund_share_class.fund_code"), primary_key=True)
    source_id: Mapped[UUID] = mapped_column(
        PostgreSQLUUID(as_uuid=True), ForeignKey("source_registry.source_id"), primary_key=True
    )
    source_event_key: Mapped[str] = mapped_column(String(64), primary_key=True)
    ann_date: Mapped[date | None] = mapped_column(Date)
    implementation_ann_date: Mapped[date | None] = mapped_column(Date)
    base_date: Mapped[date | None] = mapped_column(Date)
    process_status: Mapped[str | None] = mapped_column(String(64))
    record_date: Mapped[date | None] = mapped_column(Date)
    ex_date: Mapped[date | None] = mapped_column(Date)
    pay_date: Mapped[date | None] = mapped_column(Date)
    earnings_pay_date: Mapped[date | None] = mapped_column(Date)
    nav_ex_date: Mapped[date | None] = mapped_column(Date)
    cash_dividend: Mapped[Decimal | None] = mapped_column(Numeric(20, 8))
    base_unit: Mapped[Decimal | None] = mapped_column(Numeric(24, 4))
    distributable_earnings: Mapped[Decimal | None] = mapped_column(Numeric(24, 4))
    earnings_amount: Mapped[Decimal | None] = mapped_column(Numeric(24, 4))
    reinvestment_arrival_date: Mapped[date | None] = mapped_column(Date)
    base_year: Mapped[str | None] = mapped_column(String(16))
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()
    )


class SourceSyncRun(Base):
    """一次数据同步的执行记录，相当于“这次搬运数据的工作记录”。

    记录何时执行、处理多少条、是否成功；不保存凭据或外部接口原始响应。
    本次预览只读取成功的净值同步记录作为来源信息，不会启动新的同步。
    """

    __tablename__ = "source_sync_run"
    __table_args__ = (
        CheckConstraint("fetched_count >= 0", name="ck_source_sync_run_fetched_nonnegative"),
        CheckConstraint("created_count >= 0", name="ck_source_sync_run_created_nonnegative"),
        CheckConstraint("updated_count >= 0", name="ck_source_sync_run_updated_nonnegative"),
        CheckConstraint("skipped_count >= 0", name="ck_source_sync_run_skipped_nonnegative"),
    )

    # 本次同步的唯一编号；样本里的 source_sync_run_id 对应这里的编号。
    sync_run_id: Mapped[UUID] = mapped_column(PostgreSQLUUID(as_uuid=True), primary_key=True, default=uuid4)
    # 本次使用哪个数据来源，关联 source_registry；不是基金编号。
    source_id: Mapped[UUID] = mapped_column(
        PostgreSQLUUID(as_uuid=True), ForeignKey("source_registry.source_id"), nullable=False
    )
    # 如果本次同步由上级同步任务拆分而来，记录上级编号；没有上级时为空。
    parent_sync_run_id: Mapped[UUID | None] = mapped_column(
        PostgreSQLUUID(as_uuid=True), ForeignKey("source_sync_run.sync_run_id")
    )
    # 同步的数据类别；本次预览查找 NAV（净值）类别的成功记录。
    sync_type: Mapped[str] = mapped_column(String(32), nullable=False)
    # 发起同步时指定的净值日期；不按单日同步时可以为空。
    requested_nav_date: Mapped[date | None] = mapped_column(Date)
    # 发起同步时指定的日期范围起点；是否包含边界由对应同步流程决定。
    requested_window_start: Mapped[date | None] = mapped_column(Date)
    # 发起同步时指定的日期范围终点；没有使用范围查询时可以为空。
    requested_window_end: Mapped[date | None] = mapped_column(Date)
    # 本次同步记录的数据截至日期；它不是同步执行时间，也不是模型预测日期。
    data_as_of_date: Mapped[date | None] = mapped_column(Date)
    # 执行状态；预览要求 SUCCEEDED（成功），并且 finished_at 不为空。
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    # 从来源获取的记录数；计数为 0 不等于接口报错，还需结合 status 判断。
    fetched_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0, server_default="0")
    # 本次向数据库新建的记录数；这里只是记录同步结果，预览不会增加该值。
    created_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0, server_default="0")
    # 本次更新数据库已有记录的条数。
    updated_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0, server_default="0")
    # 本次跳过的记录数；具体跳过原因由对应同步流程决定。
    skipped_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0, server_default="0")
    # 失败原因的简短摘要；成功时通常为空，禁止把密码或 Token 放在这里。
    error_summary: Mapped[str | None] = mapped_column(String(512))
    # 同步任务实际开始执行的时间，带时区；与净值所属的业务日期不同。
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
    # 同步任务实际结束的时间；尚未结束时可以为空。
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class SourceSyncCursor(Base):
    """一个来源、数据域和实体的最后成功水位与失败摘要。"""

    __tablename__ = "source_sync_cursor"
    __table_args__ = (
        CheckConstraint("consecutive_failure_count >= 0", name="ck_source_sync_cursor_failure_count_nonnegative"),
        Index("ix_source_sync_cursor_dataset_updated", "dataset_code", "updated_at"),
    )

    source_id: Mapped[UUID] = mapped_column(
        PostgreSQLUUID(as_uuid=True), ForeignKey("source_registry.source_id"), primary_key=True
    )
    dataset_code: Mapped[str] = mapped_column(String(64), primary_key=True)
    entity_key: Mapped[str] = mapped_column(String(128), primary_key=True)
    last_successful_data_date: Mapped[date | None] = mapped_column(Date)
    last_successful_published_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    last_sync_run_id: Mapped[UUID | None] = mapped_column(
        PostgreSQLUUID(as_uuid=True), ForeignKey("source_sync_run.sync_run_id")
    )
    consecutive_failure_count: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, server_default="0"
    )
    last_error_summary: Mapped[str | None] = mapped_column(String(512))
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()
    )
