"""新增历史净值学习样本的封面、题目和离线答案三张表。

Revision ID: 20260907_13
Revises: 20260905_12
Create Date: 2026-09-07

这是独立冻结的数据库版本定义，不导入当前 ORM 实体，避免将来改实体时改变旧迁移。
只创建空表、约束、索引和中文注释；不复制净值、不保存样本，也不触发训练。
执行前应确认目标为 fund_ai；本次交付只准备脚本，不自动执行。
"""

from collections.abc import Sequence

# sa声明列和约束，op发出迁移动作，postgresql提供本库的UUID/JSONB等专用类型。
import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "20260907_13"
down_revision: str | None = "20260905_12"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """按封面、题目、答案的依赖顺序建表；不修改既有表和业务数据。"""
    op.create_table(
        "historical_nav_sample_batch",
        sa.Column(
            "batch_id",
            postgresql.UUID(as_uuid=True),
            nullable=False,
            comment="批次唯一编号；一次完整保存对应一份练习册",
        ),
        sa.Column(
            "request_key",
            postgresql.UUID(as_uuid=True),
            nullable=False,
            comment="保存请求的幂等凭证；重试沿用，明确重算时换新凭证",
        ),
        sa.Column(
            "fund_code",
            sa.String(length=32),
            nullable=False,
            comment="整批所属基金份额代码，例如008888；关联现有基金档案",
        ),
        sa.Column(
            "fund_type",
            sa.String(length=32),
            nullable=False,
            server_default="STOCK",
            comment="保存时的基金品类快照，本版固定为STOCK股票型",
        ),
        sa.Column(
            "source_code",
            sa.String(length=64),
            nullable=False,
            comment="保存时的数据来源编码快照；不是模型输入指标",
        ),
        sa.Column(
            "source_sync_run_id",
            postgresql.UUID(as_uuid=True),
            nullable=True,
            comment="来源同步水位编号；不能证明每条净值首次可得时间或恢复历史修订，未知时为空",
        ),
        sa.Column(
            "start_date",
            sa.Date(),
            nullable=False,
            comment="样本起点日期范围的第一天，包含当天",
        ),
        sa.Column(
            "end_date",
            sa.Date(),
            nullable=False,
            comment="样本起点日期范围的最后一天，包含当天",
        ),
        sa.Column(
            "feature_version",
            sa.String(length=128),
            nullable=False,
            comment="全批统一的历史指标计算规则版本，不是训练出的模型版本",
        ),
        sa.Column(
            "sample_rule_version",
            sa.String(length=128),
            nullable=False,
            comment="全批统一的样本筛选规则版本，例如是否拒收过时起点",
        ),
        sa.Column(
            "label_version",
            sa.String(length=128),
            nullable=False,
            comment="全批统一的答案计算规则版本；没有成熟答案时也登记本次采用的规则",
        ),
        sa.Column(
            "purpose",
            sa.String(length=32),
            nullable=False,
            server_default="LEARNING_ONLY",
            comment="用途固定为LEARNING_ONLY；保存不等于可训练或可发布",
        ),
        sa.Column(
            "sample_count",
            sa.Integer(),
            nullable=False,
            comment="本批实际样本总数，包含拒收和答案未成熟样本",
        ),
        sa.Column(
            "scorable_count",
            sa.Integer(),
            nullable=False,
            comment="特征和答案均可用的样本数，不代表模型合格",
        ),
        sa.Column(
            "data_insufficient_count",
            sa.Integer(),
            nullable=False,
            comment="数据不合格的样本数，例如历史不足或起点过时",
        ),
        sa.Column(
            "label_not_matured_count",
            sa.Integer(),
            nullable=False,
            comment="特征已有但未来答案尚未齐备的样本数",
        ),
        sa.Column(
            "unavailable_reasons",
            postgresql.JSONB(),
            nullable=False,
            comment="不可用原因到样本数的汇总对象，例如STALE_NAV_AT_CUTOFF对应1；无原因时为{}",
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
            comment="批次保存时间（带时区），不是净值公告日",
        ),
        sa.CheckConstraint(
            "sample_count >= 0 AND sample_count <= end_date - start_date + 1 "
            "AND scorable_count >= 0 AND data_insufficient_count >= 0 AND label_not_matured_count >= 0 "
            "AND sample_count = scorable_count + data_insufficient_count + label_not_matured_count",
            name="ck_historical_nav_batch_counts",
        ),
        sa.CheckConstraint(
            "fund_type = 'STOCK'",
            name="ck_historical_nav_batch_fund_type",
        ),
        sa.CheckConstraint(
            "purpose = 'LEARNING_ONLY'",
            name="ck_historical_nav_batch_purpose",
        ),
        sa.CheckConstraint(
            "jsonb_typeof(unavailable_reasons) = 'object'",
            name="ck_historical_nav_batch_reasons",
        ),
        sa.CheckConstraint(
            "btrim(source_code) <> '' AND btrim(feature_version) <> '' "
            "AND btrim(sample_rule_version) <> '' AND btrim(label_version) <> ''",
            name="ck_historical_nav_batch_versions",
        ),
        sa.CheckConstraint(
            "end_date - start_date BETWEEN 0 AND 30",
            name="ck_historical_nav_batch_window",
        ),
        sa.ForeignKeyConstraint(["fund_code"], ["fund_share_class.fund_code"], name="fk_historical_nav_batch_fund"),
        sa.ForeignKeyConstraint(
            ["source_sync_run_id"], ["source_sync_run.sync_run_id"], name="fk_historical_nav_batch_source_run"
        ),
        sa.PrimaryKeyConstraint("batch_id", name="pk_historical_nav_sample_batch"),
        sa.UniqueConstraint("request_key", name="uq_historical_nav_batch_request"),
        comment="历史净值学习样本批次（练习册封面，仅供学习，不是模型或预测结果）",
    )
    op.create_index(
        "ix_historical_nav_batch_fund_created",
        "historical_nav_sample_batch",
        ["fund_code", "created_at", "batch_id"],
        unique=False,
    )

    op.create_table(
        "historical_nav_sample",
        sa.Column(
            "sample_id",
            postgresql.UUID(as_uuid=True),
            nullable=False,
            comment="题目唯一编号，用于关联单独存放的答案",
        ),
        sa.Column(
            "batch_id",
            postgresql.UUID(as_uuid=True),
            nullable=False,
            comment="所属练习册的批次编号；同批共享基金、来源水位和规则版本",
        ),
        sa.Column(
            "as_of_date",
            sa.Date(),
            nullable=False,
            comment="起点净值所属的业务日期，不代表当天已经看到净值",
        ),
        sa.Column(
            "available_at",
            sa.Date(),
            nullable=True,
            comment="起点公告日，按日终可见作为历史输入截止；缺失或错误的公告仍随拒收样本保留",
        ),
        sa.Column(
            "nav_value_basis",
            sa.String(length=32),
            nullable=False,
            comment="统一净值口径：ACCUMULATED_NAV累计、UNIT_NAV单位、UNDETERMINED未确定",
        ),
        sa.Column(
            "eligibility_status",
            sa.String(length=32),
            nullable=False,
            comment="样本状态：SCORABLE完整、DATA_INSUFFICIENT不合格、LABEL_NOT_MATURED答案未齐",
        ),
        sa.Column(
            "unavailable_reason",
            sa.Text(),
            nullable=True,
            comment="拒收或答案未成熟的原因代码；完整样本为空，不能用0代替原因",
        ),
        sa.Column(
            "feature_payload",
            postgresql.JSONB(),
            nullable=False,
            comment="当时已知条件的完整JSON包；包含来源、质量和指标，不得放入未来答案",
        ),
        sa.Column(
            "feature_hash",
            sa.String(length=64),
            nullable=False,
            comment="仅对feature_payload计算的SHA-256指纹；不包含答案、批次编号或筛选规则版本",
        ),
        sa.CheckConstraint(
            "eligibility_status = 'DATA_INSUFFICIENT' OR "
            "(available_at IS NOT NULL AND available_at >= as_of_date AND nav_value_basis <> 'UNDETERMINED')",
            name="ck_historical_nav_sample_available",
        ),
        sa.CheckConstraint(
            "nav_value_basis IN ('ACCUMULATED_NAV', 'UNIT_NAV', 'UNDETERMINED')",
            name="ck_historical_nav_sample_basis",
        ),
        sa.CheckConstraint(
            "feature_hash ~ '^[0-9a-f]{64}$'",
            name="ck_historical_nav_sample_hash",
        ),
        sa.CheckConstraint(
            "jsonb_typeof(feature_payload) = 'object'",
            name="ck_historical_nav_sample_payload",
        ),
        sa.CheckConstraint(
            "(eligibility_status = 'SCORABLE' AND unavailable_reason IS NULL) OR "
            "(eligibility_status <> 'SCORABLE' AND unavailable_reason IS NOT NULL AND btrim(unavailable_reason) <> '')",
            name="ck_historical_nav_sample_reason",
        ),
        sa.CheckConstraint(
            "eligibility_status IN ('SCORABLE', 'DATA_INSUFFICIENT', 'LABEL_NOT_MATURED')",
            name="ck_historical_nav_sample_status",
        ),
        sa.ForeignKeyConstraint(
            ["batch_id"], ["historical_nav_sample_batch.batch_id"], name="fk_historical_nav_sample_batch"
        ),
        sa.PrimaryKeyConstraint("sample_id", name="pk_historical_nav_sample"),
        sa.UniqueConstraint("batch_id", "as_of_date", name="uq_historical_nav_sample_batch_date"),
        comment="历史净值学习样本题目；保留正常、拒收及答案未成熟的样本，不存未来答案",
    )

    op.create_table(
        "historical_nav_sample_label",
        sa.Column(
            "sample_id",
            postgresql.UUID(as_uuid=True),
            nullable=False,
            comment="所回答题目的唯一编号；同时作为本表主键，保证一道题至多一份答案",
        ),
        sa.Column(
            "horizon_trading_days",
            sa.SmallInteger(),
            nullable=False,
            server_default="20",
            comment="从起点向后数的净值区间数，本版固定20，不是20个自然日",
        ),
        sa.Column(
            "label_end_date",
            sa.Date(),
            nullable=False,
            comment="起点后第20条净值的业务日期，即答案终点",
        ),
        sa.Column(
            "label_available_at",
            sa.Date(),
            nullable=False,
            comment="答案所依赖的未来记录全部公告完成的日期，不是保存时间",
        ),
        sa.Column(
            "future_return_20d",
            sa.Numeric(),
            nullable=False,
            comment="后来实际区间收益：终点净值/起点净值-1；不固定小数位，0.14表示14%",
        ),
        sa.Column(
            "label_up_20d",
            sa.SmallInteger(),
            nullable=False,
            comment="真实收益大于0记1；持平或下跌记0；不允许以0代替缺失答案",
        ),
        sa.CheckConstraint(
            "label_available_at >= label_end_date",
            name="ck_historical_nav_label_available",
        ),
        sa.CheckConstraint(
            "(future_return_20d > 0 AND label_up_20d = 1) OR (future_return_20d <= 0 AND label_up_20d = 0)",
            name="ck_historical_nav_label_direction",
        ),
        sa.CheckConstraint(
            "horizon_trading_days = 20",
            name="ck_historical_nav_label_horizon",
        ),
        sa.CheckConstraint(
            "future_return_20d > -1 AND future_return_20d::text NOT IN ('NaN', 'Infinity', '-Infinity')",
            name="ck_historical_nav_label_return",
        ),
        sa.ForeignKeyConstraint(
            ["sample_id"], ["historical_nav_sample.sample_id"], name="fk_historical_nav_label_sample"
        ),
        sa.PrimaryKeyConstraint("sample_id", name="pk_historical_nav_sample_label"),
        comment="历史净值学习样本的离线答案，与输入分表；非预测概率，未成熟或拒收时没有答案行",
    )


def downgrade() -> None:
    """仅允许回退空表；任何一张表已有数据则整次拒绝，不级联删除。

    先获取三张表的排他锁，再检查是否为空，避免检查后又有人写入。
    NOWAIT 表示取不到锁就立即报错，不长时间阻塞正在使用表的服务。
    应在 Alembic 的 PostgreSQL 事务中执行，失败后回滚，不保留部分删除。
    """
    op.execute(
        "LOCK TABLE historical_nav_sample_label, historical_nav_sample, historical_nav_sample_batch "
        "IN ACCESS EXCLUSIVE MODE NOWAIT"
    )
    op.execute(
        """
        DO $$
        BEGIN
            IF EXISTS (SELECT 1 FROM historical_nav_sample_label)
                OR EXISTS (SELECT 1 FROM historical_nav_sample)
                OR EXISTS (SELECT 1 FROM historical_nav_sample_batch) THEN
                RAISE EXCEPTION 'Historical NAV sample tables are not empty; downgrade refused';
            END IF;
        END;
        $$
        """
    )
    # 逆序删除；不使用 CASCADE，其他对象若依赖这些表也应阻止回退。
    op.drop_table("historical_nav_sample_label")
    op.drop_table("historical_nav_sample")
    op.drop_index("ix_historical_nav_batch_fund_created", table_name="historical_nav_sample_batch")
    op.drop_table("historical_nav_sample_batch")
