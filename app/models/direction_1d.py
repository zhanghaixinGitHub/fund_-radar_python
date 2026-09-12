"""1日公共证据的SQLAlchemy表定义；写入由独立仓储和Alembic管理。"""

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB, UUID

from app.db.base import Base

direction_1d_assessment_ack = sa.Table(
    "direction_1d_assessment_ack",
    Base.metadata,
    sa.Column("task_key", sa.String(256), nullable=False, primary_key=True, comment="协议及基金目标日幂等键"),
    sa.Column("label_hash", sa.CHAR(64), nullable=False, primary_key=True, comment="已核对答案摘要"),
    sa.Column(
        "assessed_at", sa.DateTime(timezone=True), nullable=False, primary_key=False, comment="Java真实核对完成时刻"
    ),
    sa.Column(
        "received_at",
        sa.DateTime(timezone=True),
        nullable=False,
        primary_key=False,
        comment="实际读取接纳时刻，非历史公布时刻",
        server_default=sa.text("clock_timestamp()"),
    ),
    comment="Java先核对后的公共学习许可回执",
)

direction_1d_job = sa.Table(
    "direction_1d_job",
    Base.metadata,
    sa.Column("job_id", UUID(as_uuid=True), nullable=False, primary_key=True, comment="异步作业编号"),
    sa.Column("task_key", sa.String(256), nullable=False, primary_key=False, comment="协议及基金目标日幂等键"),
    sa.Column("kind", sa.String(16), nullable=False, primary_key=False, comment="证据或作业类别"),
    sa.Column("state", sa.String(32), nullable=False, primary_key=False, comment="作业状态"),
    sa.Column("payload", JSONB(), nullable=False, primary_key=False, comment="不含用户身份的冻结业务原文"),
    sa.Column("result", JSONB(), nullable=True, primary_key=False, comment="稳定结果或脱敏失败原因"),
    sa.Column(
        "created_at",
        sa.DateTime(timezone=True),
        nullable=False,
        primary_key=False,
        comment="数据库创建时刻",
        server_default=sa.text("clock_timestamp()"),
    ),
    sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True, primary_key=False, comment="真实结束时刻"),
    sa.Column(
        "lease_until",
        sa.DateTime(timezone=True),
        nullable=True,
        primary_key=False,
        comment="队列或进程恢复租约，不修改已成功原文",
    ),
    sa.UniqueConstraint("task_key", name="direction_1d_job_task_key_key"),
    comment="有界异步作业和稳定结果，不含用户身份",
)

direction_1d_model = sa.Table(
    "direction_1d_model",
    Base.metadata,
    sa.Column("model_id", UUID(as_uuid=True), nullable=False, primary_key=True, comment="注册模型编号"),
    sa.Column("run_id", UUID(as_uuid=True), nullable=False, primary_key=False, comment="冻结训练运行编号"),
    sa.Column("cohort_id", sa.String(80), nullable=False, primary_key=False, comment="冻结训练集合版本"),
    sa.Column("group_id", sa.String(32), nullable=False, primary_key=False, comment="经过核验的资产组"),
    sa.Column("horizon", sa.Integer(), nullable=False, primary_key=False, comment="预测交易日数，恒为1"),
    sa.Column("file_name", sa.String(128), nullable=False, primary_key=False, comment="允许目录内的模型文件名"),
    sa.Column("content_hash", sa.CHAR(64), nullable=False, primary_key=False, comment="规范原文SHA256"),
    sa.Column("metadata", JSONB(), nullable=False, primary_key=False, comment="特征标签与训练证据身份"),
    sa.Column("trained_at", sa.DateTime(timezone=True), nullable=False, primary_key=False, comment="真实拟合完成时刻"),
    sa.Column(
        "registered_at",
        sa.DateTime(timezone=True),
        nullable=False,
        primary_key=False,
        comment="真实注册时刻",
        server_default=sa.text("clock_timestamp()"),
    ),
    sa.Column(
        "expires_at",
        sa.DateTime(timezone=True),
        nullable=False,
        primary_key=False,
        comment="来源许可范围内的证据失效时间",
    ),
    sa.ForeignKeyConstraint(["run_id"], ["direction_1d_training_run.run_id"], name="direction_1d_model_run_id_fkey"),
    sa.UniqueConstraint("cohort_id", "group_id", "run_id", name="direction_1d_model_cohort_id_group_id_run_id_key"),
    sa.CheckConstraint("horizon = 1", name="direction_1d_model_horizon_check"),
    comment="仅供一日实验的已核验模型，不是正式发布表",
)

direction_1d_snapshot = sa.Table(
    "direction_1d_snapshot",
    Base.metadata,
    sa.Column("snapshot_id", UUID(as_uuid=True), nullable=False, primary_key=True, comment="证据快照唯一编号"),
    sa.Column("kind", sa.String(16), nullable=False, primary_key=False, comment="证据或作业类别"),
    sa.Column("task_key", sa.String(256), nullable=False, primary_key=False, comment="协议及基金目标日幂等键"),
    sa.Column("as_of", sa.DateTime(timezone=True), nullable=False, primary_key=False, comment="一致性读取界限"),
    sa.Column("payload", JSONB(), nullable=False, primary_key=False, comment="不含用户身份的冻结业务原文"),
    sa.Column("content_hash", sa.CHAR(64), nullable=False, primary_key=False, comment="规范原文SHA256"),
    sa.Column(
        "expires_at",
        sa.DateTime(timezone=True),
        nullable=False,
        primary_key=False,
        comment="来源许可范围内的证据失效时间",
    ),
    sa.Column(
        "created_at",
        sa.DateTime(timezone=True),
        nullable=False,
        primary_key=False,
        comment="数据库创建时刻",
        server_default=sa.text("clock_timestamp()"),
    ),
    comment="可复算的一日输入和答案不可变快照",
)
sa.Index(
    "ix_direction_1d_snapshot_task",
    direction_1d_snapshot.c.task_key,
    direction_1d_snapshot.c.kind,
    direction_1d_snapshot.c.as_of,
)

direction_1d_source_version = sa.Table(
    "direction_1d_source_version",
    Base.metadata,
    sa.Column("version_id", UUID(as_uuid=True), nullable=False, primary_key=True, comment="来源版本唯一编号"),
    sa.Column("fund_code", sa.String(32), nullable=False, primary_key=False, comment="公共基金份额代码"),
    sa.Column("source_id", UUID(as_uuid=True), nullable=False, primary_key=False, comment="来源登记编号"),
    sa.Column("kind", sa.String(16), nullable=False, primary_key=False, comment="证据或作业类别"),
    sa.Column("business_date", sa.Date(), nullable=False, primary_key=False, comment="源数据业务日期"),
    sa.Column("content_hash", sa.CHAR(64), nullable=False, primary_key=False, comment="规范原文SHA256"),
    sa.Column("payload", JSONB(), nullable=False, primary_key=False, comment="不含用户身份的冻结业务原文"),
    sa.Column(
        "received_at",
        sa.DateTime(timezone=True),
        nullable=False,
        primary_key=False,
        comment="实际读取接纳时刻，非历史公布时刻",
        server_default=sa.text("clock_timestamp()"),
    ),
    sa.Column(
        "stored_at",
        sa.DateTime(timezone=True),
        nullable=False,
        primary_key=False,
        comment="数据库实际写入时刻",
        server_default=sa.text("clock_timestamp()"),
    ),
    sa.Column(
        "expires_at",
        sa.DateTime(timezone=True),
        nullable=False,
        primary_key=False,
        comment="来源许可范围内的证据失效时间",
    ),
    sa.UniqueConstraint(
        "fund_code",
        "source_id",
        "kind",
        "business_date",
        "content_hash",
        name="direction_1d_source_version_fund_code_source_id_kind_busine_key",
    ),
    comment="公开来源实际读取版本，不倒填历史首次公布时间",
)

direction_1d_training_run = sa.Table(
    "direction_1d_training_run",
    Base.metadata,
    sa.Column("run_id", UUID(as_uuid=True), nullable=False, primary_key=True, comment="冻结训练运行编号"),
    sa.Column("cohort_id", sa.String(80), nullable=False, primary_key=False, comment="冻结训练集合版本"),
    sa.Column(
        "train_as_of", sa.DateTime(timezone=True), nullable=False, primary_key=False, comment="成熟样本训练读取界限"
    ),
    sa.Column("spec", JSONB(), nullable=False, primary_key=False, comment="固定训练配方与名单"),
    sa.Column("spec_hash", sa.CHAR(64), nullable=False, primary_key=False, comment="冻结配方摘要"),
    sa.Column("status", sa.String(32), nullable=False, primary_key=False, comment="技术状态，不表示预测有效性"),
    sa.Column("evidence", JSONB(), nullable=False, primary_key=False, comment="训练复现与样本清单证据"),
    sa.Column(
        "created_at",
        sa.DateTime(timezone=True),
        nullable=False,
        primary_key=False,
        comment="数据库创建时刻",
        server_default=sa.text("clock_timestamp()"),
    ),
    comment="固定配方训练运行和冻结清单",
)

direction_1d_window_model = sa.Table(
    "direction_1d_window_model",
    Base.metadata,
    sa.Column("cohort_id", sa.String(80), nullable=False, primary_key=True, comment="冻结训练集合版本"),
    sa.Column("group_id", sa.String(32), nullable=False, primary_key=True, comment="经过核验的资产组"),
    sa.Column("base_nav_date", sa.Date(), nullable=False, primary_key=False, comment="基准交易日T"),
    sa.Column("target_nav_date", sa.Date(), nullable=False, primary_key=True, comment="相邻目标交易日U"),
    sa.Column("branch_id", sa.String(16), nullable=False, primary_key=True, comment="冻结或每周更新分支"),
    sa.Column(
        "activation_policy",
        sa.String(40),
        nullable=False,
        primary_key=True,
        server_default=sa.text("'BEFORE_WINDOW_V1'"),
        comment="模型选择策略：旧BEFORE_WINDOW_V1；预测时已可用AVAILABLE_AT_PREDICTION_V2",
    ),
    sa.Column("model_id", UUID(as_uuid=True), nullable=True, primary_key=False, comment="注册模型编号"),
    sa.Column(
        "locked_at",
        sa.DateTime(timezone=True),
        nullable=False,
        primary_key=False,
        comment="映射实际锁定时刻",
        server_default=sa.text("clock_timestamp()"),
    ),
    sa.ForeignKeyConstraint(
        ["model_id"], ["direction_1d_model.model_id"], name="direction_1d_window_model_model_id_fkey"
    ),
    sa.CheckConstraint(
        "branch_id::text = ANY (ARRAY['FIXED'::character varying, 'WEEKLY'::character varying]::text[])",
        name="direction_1d_window_model_branch_id_check",
    ),
    comment="按窗口不可变的FIXED和WEEKLY映射",
)
