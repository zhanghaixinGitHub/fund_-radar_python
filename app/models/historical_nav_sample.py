"""历史净值学习样本的存储格式；本文件不查询、不保存，也不训练模型。

先读 Batch（练习册封面），再读 Record（题目），最后读 Label（答案）。
这里的 ORM 实体把 Python 属性对应到数据库列；声明类或导入类不会自动建表。
真正建表由 Alembic 迁移执行，真正保存由 services/historical_nav_storage.py 的事务代码完成。

Mapped[T] 表示属性的 Python 类型；mapped_column 声明数据库列；comment 会成为
数据库字段注释。主键是本行编号，外键是关联另一张表的编号，唯一约束用于阻止重复。
下面只声明数据库能直接检查的规则；跨表统计核对、答案与题目状态核对、禁止覆盖旧批次，
由保存服务及 historical_nav_storage_validation.py 负责。分表本身不代表已经防住训练时的未来信息泄漏。
"""

from __future__ import annotations

# 标准库类型：date/datetime分别表达业务日期/保存时刻，Decimal保留十进制数，UUID用作唯一编号。
from datetime import date, datetime
from decimal import Decimal
from uuid import UUID, uuid4

# SQLAlchemy提供“声明表结构”的工具，不是在导入基金数据或加载训练好的模型。
from sqlalchemy import (
    CheckConstraint,
    Date,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    PrimaryKeyConstraint,
    SmallInteger,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import UUID as PostgreSQLUUID
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class HistoricalNavSampleBatch(Base):
    """练习册封面：一只基金、一个小日期段、同一来源水位与同一套规则。

    当前仅保存学习快照，purpose 固定为 LEARNING_ONLY，不能作为模型发布凭证。
    保存成功才留下整批记录，所以这里没有 RUNNING 等任务状态，也没有更新旧批次的字段。
    """

    __tablename__ = "historical_nav_sample_batch"
    __table_args__ = (
        PrimaryKeyConstraint("batch_id", name="pk_historical_nav_sample_batch"),
        UniqueConstraint("request_key", name="uq_historical_nav_batch_request"),
        CheckConstraint("fund_type = 'STOCK'", name="ck_historical_nav_batch_fund_type"),
        CheckConstraint("purpose = 'LEARNING_ONLY'", name="ck_historical_nav_batch_purpose"),
        # 日期含首尾，所以相差 30 天就是最多 31 个自然日；不是 31 个交易日。
        CheckConstraint("end_date - start_date BETWEEN 0 AND 30", name="ck_historical_nav_batch_window"),
        CheckConstraint(
            "btrim(source_code) <> '' AND btrim(feature_version) <> '' "
            "AND btrim(sample_rule_version) <> '' AND btrim(label_version) <> ''",
            name="ck_historical_nav_batch_versions",
        ),
        CheckConstraint(
            "sample_count >= 0 AND sample_count <= end_date - start_date + 1 "
            "AND scorable_count >= 0 AND data_insufficient_count >= 0 AND label_not_matured_count >= 0 "
            "AND sample_count = scorable_count + data_insufficient_count + label_not_matured_count",
            name="ck_historical_nav_batch_counts",
        ),
        CheckConstraint("jsonb_typeof(unavailable_reasons) = 'object'", name="ck_historical_nav_batch_reasons"),
        # 按基金浏览历史批次时，用创建时间和批次编号稳定分页，不必扫描全部基金。
        Index("ix_historical_nav_batch_fund_created", "fund_code", "created_at", "batch_id"),
        {"comment": "历史净值学习样本批次（练习册封面，仅供学习，不是模型或预测结果）"},
    )

    batch_id: Mapped[UUID] = mapped_column(
        PostgreSQLUUID(as_uuid=True), default=uuid4, comment="批次唯一编号；一次完整保存对应一份练习册"
    )
    # 不自动生成请求凭证：调用保存服务的一方必须在重试时继续使用原凭证。
    request_key: Mapped[UUID] = mapped_column(
        PostgreSQLUUID(as_uuid=True), nullable=False, comment="保存请求的幂等凭证；重试沿用，明确重算时换新凭证"
    )
    fund_code: Mapped[str] = mapped_column(
        String(32),
        ForeignKey("fund_share_class.fund_code", name="fk_historical_nav_batch_fund"),
        nullable=False,
        comment="整批所属基金份额代码，例如008888；关联现有基金档案",
    )
    fund_type: Mapped[str] = mapped_column(
        String(32), nullable=False, server_default="STOCK", comment="保存时的基金品类快照，本版固定为STOCK股票型"
    )
    source_code: Mapped[str] = mapped_column(
        String(64), nullable=False, comment="保存时的数据来源编码快照；不是模型输入指标"
    )
    source_sync_run_id: Mapped[UUID | None] = mapped_column(
        PostgreSQLUUID(as_uuid=True),
        ForeignKey("source_sync_run.sync_run_id", name="fk_historical_nav_batch_source_run"),
        comment="来源同步水位编号；不能证明每条净值首次可得时间或恢复历史修订，未知时为空",
    )
    start_date: Mapped[date] = mapped_column(Date, nullable=False, comment="样本起点日期范围的第一天，包含当天")
    end_date: Mapped[date] = mapped_column(Date, nullable=False, comment="样本起点日期范围的最后一天，包含当天")
    feature_version: Mapped[str] = mapped_column(
        String(128), nullable=False, comment="全批统一的历史指标计算规则版本，不是训练出的模型版本"
    )
    sample_rule_version: Mapped[str] = mapped_column(
        String(128), nullable=False, comment="全批统一的样本筛选规则版本，例如是否拒收过时起点"
    )
    label_version: Mapped[str] = mapped_column(
        String(128), nullable=False, comment="全批统一的答案计算规则版本；没有成熟答案时也登记本次采用的规则"
    )
    purpose: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        server_default="LEARNING_ONLY",
        comment="用途固定为LEARNING_ONLY；保存不等于可训练或可发布",
    )
    sample_count: Mapped[int] = mapped_column(
        Integer, nullable=False, comment="本批实际样本总数，包含拒收和答案未成熟样本"
    )
    scorable_count: Mapped[int] = mapped_column(
        Integer, nullable=False, comment="特征和答案均可用的样本数，不代表模型合格"
    )
    data_insufficient_count: Mapped[int] = mapped_column(
        Integer, nullable=False, comment="数据不合格的样本数，例如历史不足或起点过时"
    )
    label_not_matured_count: Mapped[int] = mapped_column(
        Integer, nullable=False, comment="特征已有但未来答案尚未齐备的样本数"
    )
    unavailable_reasons: Mapped[dict[str, int]] = mapped_column(
        JSONB, nullable=False, comment="不可用原因到样本数的汇总对象，例如STALE_NAV_AT_CUTOFF对应1；无原因时为{}"
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        comment="批次保存时间（带时区），不是净值公告日",
    )


class HistoricalNavSampleRecord(Base):
    """练习册中的一道题；Record 后缀用于区别纯计算模块里的 HistoricalNavSample。

    同批的基金、来源水位和版本从封面读取，不在每一行复制。题目正文仍保存完整 feature_payload，
    保存服务会核对其中的来源/版本与封面一致。这里只定义格式，导入实体不会自动执行核对。
    """

    __tablename__ = "historical_nav_sample"
    __table_args__ = (
        PrimaryKeyConstraint("sample_id", name="pk_historical_nav_sample"),
        UniqueConstraint("batch_id", "as_of_date", name="uq_historical_nav_sample_batch_date"),
        CheckConstraint(
            "eligibility_status IN ('SCORABLE', 'DATA_INSUFFICIENT', 'LABEL_NOT_MATURED')",
            name="ck_historical_nav_sample_status",
        ),
        CheckConstraint(
            "nav_value_basis IN ('ACCUMULATED_NAV', 'UNIT_NAV', 'UNDETERMINED')",
            name="ck_historical_nav_sample_basis",
        ),
        CheckConstraint(
            "(eligibility_status = 'SCORABLE' AND unavailable_reason IS NULL) OR "
            "(eligibility_status <> 'SCORABLE' AND unavailable_reason IS NOT NULL AND btrim(unavailable_reason) <> '')",
            name="ck_historical_nav_sample_reason",
        ),
        # 拒收行可能正是因为公告日缺失或早于净值日；不能用一刀切约束让错误事实无法保留。
        CheckConstraint(
            "eligibility_status = 'DATA_INSUFFICIENT' OR "
            "(available_at IS NOT NULL AND available_at >= as_of_date AND nav_value_basis <> 'UNDETERMINED')",
            name="ck_historical_nav_sample_available",
        ),
        CheckConstraint("jsonb_typeof(feature_payload) = 'object'", name="ck_historical_nav_sample_payload"),
        CheckConstraint("feature_hash ~ '^[0-9a-f]{64}$'", name="ck_historical_nav_sample_hash"),
        {"comment": "历史净值学习样本题目；保留正常、拒收及答案未成熟的样本，不存未来答案"},
    )

    sample_id: Mapped[UUID] = mapped_column(
        PostgreSQLUUID(as_uuid=True), default=uuid4, comment="题目唯一编号，用于关联单独存放的答案"
    )
    # 不配置级联删除：误删封面时让外键拦住，不连带抹掉题目和答案。
    batch_id: Mapped[UUID] = mapped_column(
        PostgreSQLUUID(as_uuid=True),
        ForeignKey("historical_nav_sample_batch.batch_id", name="fk_historical_nav_sample_batch"),
        nullable=False,
        comment="所属练习册的批次编号；同批共享基金、来源水位和规则版本",
    )
    as_of_date: Mapped[date] = mapped_column(
        Date, nullable=False, comment="起点净值所属的业务日期，不代表当天已经看到净值"
    )
    available_at: Mapped[date | None] = mapped_column(
        Date, comment="起点公告日，按日终可见作为历史输入截止；缺失或错误的公告仍随拒收样本保留"
    )
    nav_value_basis: Mapped[str] = mapped_column(
        String(32), nullable=False, comment="统一净值口径：ACCUMULATED_NAV累计、UNIT_NAV单位、UNDETERMINED未确定"
    )
    eligibility_status: Mapped[str] = mapped_column(
        String(32), nullable=False, comment="样本状态：SCORABLE完整、DATA_INSUFFICIENT不合格、LABEL_NOT_MATURED答案未齐"
    )
    unavailable_reason: Mapped[str | None] = mapped_column(
        Text, comment="拒收或答案未成熟的原因代码；完整样本为空，不能用0代替原因"
    )
    feature_payload: Mapped[dict[str, object]] = mapped_column(
        JSONB, nullable=False, comment="当时已知条件的完整JSON包；包含来源、质量和指标，不得放入未来答案"
    )
    feature_hash: Mapped[str] = mapped_column(
        String(64), nullable=False, comment="仅对feature_payload计算的SHA-256指纹；不包含答案、批次编号或筛选规则版本"
    )


class HistoricalNavSampleLabel(Base):
    """题目的后来真实答案，不是模型预测值；没有答案时整行不写。

    sample_id 同时是主键和外键，一道题至多有一份答案；0 表示真实的非上涨，不能表示未知。
    只给 SCORABLE 行配答案、日期晚于题目截止日等跨表规则，由下一步保存服务核对。
    """

    __tablename__ = "historical_nav_sample_label"
    __table_args__ = (
        PrimaryKeyConstraint("sample_id", name="pk_historical_nav_sample_label"),
        CheckConstraint("horizon_trading_days = 20", name="ck_historical_nav_label_horizon"),
        CheckConstraint("label_available_at >= label_end_date", name="ck_historical_nav_label_available"),
        CheckConstraint(
            "future_return_20d > -1 AND future_return_20d::text NOT IN ('NaN', 'Infinity', '-Infinity')",
            name="ck_historical_nav_label_return",
        ),
        CheckConstraint(
            "(future_return_20d > 0 AND label_up_20d = 1) OR (future_return_20d <= 0 AND label_up_20d = 0)",
            name="ck_historical_nav_label_direction",
        ),
        {"comment": "历史净值学习样本的离线答案，与输入分表；非预测概率，未成熟或拒收时没有答案行"},
    )

    sample_id: Mapped[UUID] = mapped_column(
        PostgreSQLUUID(as_uuid=True),
        ForeignKey("historical_nav_sample.sample_id", name="fk_historical_nav_label_sample"),
        comment="所回答题目的唯一编号；同时作为本表主键，保证一道题至多一份答案",
    )
    horizon_trading_days: Mapped[int] = mapped_column(
        SmallInteger,
        nullable=False,
        server_default="20",
        comment="从起点向后数的净值区间数，本版固定20，不是20个自然日",
    )
    label_end_date: Mapped[date] = mapped_column(Date, nullable=False, comment="起点后第20条净值的业务日期，即答案终点")
    label_available_at: Mapped[date] = mapped_column(
        Date, nullable=False, comment="答案所依赖的未来记录全部公告完成的日期，不是保存时间"
    )
    # 不指定小数位数：当前构建器返回 Decimal，不应在保存时被 NUMERIC(20,8) 截成另一份答案。
    future_return_20d: Mapped[Decimal] = mapped_column(
        Numeric(), nullable=False, comment="后来实际区间收益：终点净值/起点净值-1；不固定小数位，0.14表示14%"
    )
    label_up_20d: Mapped[int] = mapped_column(
        SmallInteger, nullable=False, comment="真实收益大于0记1；持平或下跌记0；不允许以0代替缺失答案"
    )
