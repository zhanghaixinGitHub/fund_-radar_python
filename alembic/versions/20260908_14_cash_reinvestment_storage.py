"""独立现金样本及研究报告；加性迁移，旧表不动，回退仅允许空表。"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql as pg

revision = "20260908_14"
down_revision = "20260907_13"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "cash_sample_batch",
        sa.Column("batch_id", pg.UUID(), primary_key=True, comment="本次不可变批次编号"),
        sa.Column("request_key", pg.UUID(), nullable=False, comment="重试凭证；重算使用新编号"),
        sa.Column(
            "fund_code",
            sa.String(32),
            sa.ForeignKey("fund_share_class.fund_code"),
            nullable=False,
            comment="所属基金份额",
        ),
        sa.Column("start_date", sa.Date(), nullable=False, comment="信息截止日范围起点，含当天"),
        sa.Column("end_date", sa.Date(), nullable=False, comment="信息截止日范围终点，含当天"),
        sa.Column("summary", pg.JSONB(), nullable=False, comment="原试跑汇总，不含items；保留规则和来源水位"),
        sa.Column("batch_hash", sa.String(64), nullable=False, comment="整份试跑内容SHA256，不含分页方式"),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
            comment="数据库保存时刻",
        ),
        sa.UniqueConstraint("request_key", name="uq_cash_batch_request"),
        sa.CheckConstraint("end_date >= start_date AND end_date - start_date < 31", name="ck_cash_batch_range"),
        comment="现金再投V1研究批次，不代表正式训练准入",
    )
    op.create_index("ix_cash_batch_fund_date", "cash_sample_batch", ["fund_code", "start_date"])
    op.create_table(
        "cash_sample",
        sa.Column("sample_id", pg.UUID(), primary_key=True, comment="题目编号"),
        sa.Column(
            "batch_id", pg.UUID(), sa.ForeignKey("cash_sample_batch.batch_id"), nullable=False, comment="所属批次"
        ),
        sa.Column("cutoff_date", sa.Date(), nullable=False, comment="信息截止日，不是批次创建日"),
        sa.Column("context", pg.JSONB(), nullable=False, comment="除特征和答案外的原样本诊断，不能作为X"),
        sa.Column(
            "feature_payload",
            pg.JSONB(none_as_null=True),
            nullable=True,
            comment="仅历史输入，禁止答案字段；不可用时SQL NULL",
        ),
        sa.UniqueConstraint("batch_id", "cutoff_date", name="uq_cash_sample_cutoff"),
        comment="现金再投研究样本；只允许七项历史metrics进入模型",
    )
    op.create_table(
        "cash_sample_label",
        sa.Column("sample_id", pg.UUID(), sa.ForeignKey("cash_sample.sample_id"), primary_key=True, comment="对应题目"),
        sa.Column("label_payload", pg.JSONB(), nullable=False, comment="独立答案及复算序列，仅离线评估读取"),
        comment="现金再投20交易日离线答案，不是预测结果",
    )
    op.create_table(
        "cash_research_run",
        sa.Column("run_id", pg.UUID(), primary_key=True, comment="本轮研究编号"),
        sa.Column("request_key", pg.UUID(), nullable=False, comment="研究保存重试凭证"),
        sa.Column("dataset_hash", sa.String(64), nullable=False, comment="明确批次与数据规则的冻结指纹"),
        sa.Column("report", pg.JSONB(), nullable=False, comment="准备、窗口成绩、数值JSON模型及未发布原因；无pickle"),
        sa.Column("publication_status", sa.String(32), nullable=False, comment="固定未发布，不能用手改ACTIVE替代准入"),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
            comment="研究报告保存时刻",
        ),
        sa.UniqueConstraint("request_key", name="uq_cash_research_request"),
        sa.CheckConstraint("publication_status = 'MODEL_NOT_RELEASED'", name="ck_cash_research_no_release"),
        comment="现金口径研究运行，发布准入失败关闭；不存个人数据",
    )
    op.create_index("ix_cash_research_created", "cash_research_run", ["created_at"])


def downgrade():
    # 一条事务中先锁定并检查四表，任何资料存在就拒绝回退，不能误删研究证据。
    op.execute(
        "LOCK TABLE cash_sample_batch, cash_sample, cash_sample_label, cash_research_run IN ACCESS EXCLUSIVE MODE"
    )
    op.execute("""DO $$ BEGIN
        IF EXISTS (SELECT 1 FROM cash_sample_batch) OR EXISTS (SELECT 1 FROM cash_sample)
          OR EXISTS (SELECT 1 FROM cash_sample_label) OR EXISTS (SELECT 1 FROM cash_research_run)
        THEN RAISE EXCEPTION 'cash research tables are not empty; downgrade refused'; END IF;
        END $$""")
    for table in ("cash_research_run", "cash_sample_label", "cash_sample", "cash_sample_batch"):
        op.drop_table(table)
