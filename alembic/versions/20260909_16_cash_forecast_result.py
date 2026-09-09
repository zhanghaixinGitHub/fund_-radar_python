"""独立现金预测结果；不改旧结果或拒绝回执，非空回退失败关闭。"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql as pg

revision = "20260909_16"
down_revision = "20260909_15"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "cash_forecast_result",
        sa.Column("forecast_id", pg.UUID(), primary_key=True, comment="不可变结果编号"),
        sa.Column("request_key", pg.UUID(), nullable=False, comment="首次成功生成的重试凭证"),
        sa.Column("request_hash", sa.String(64), nullable=False, comment="生成请求指纹，不含requestKey"),
        sa.Column(
            "fund_code",
            sa.String(32),
            sa.ForeignKey("fund_share_class.fund_code"),
            nullable=False,
            comment="所属基金份额",
        ),
        sa.Column("cutoff_date", sa.Date(), nullable=False, comment="本次信息截止日，不随读取时间变化"),
        sa.Column(
            "research_run_id",
            pg.UUID(),
            sa.ForeignKey("cash_research_run.run_id"),
            nullable=False,
            comment="原研究编号",
        ),
        sa.Column(
            "authorization_hash", sa.String(64), nullable=False, comment="当时正式授权指纹；原始凭证引用保存在快照中"
        ),
        sa.Column("model_hash", sa.String(64), nullable=False, comment="本次真实计算的模型指纹，不是旧模型版本名"),
        sa.Column("feature_hash", sa.String(64), nullable=False, comment="本次仅含已知信息的输入指纹"),
        sa.Column(
            "source_revision_id", pg.UUID(), nullable=False, comment="该来源所有数据同步的水位，不只检查净值同步"
        ),
        sa.Column(
            "payload", pg.JSONB(), nullable=False, comment="明确请求、授权引用、历史输入和真实计算值；无未来答案"
        ),
        sa.Column("content_hash", sa.String(64), nullable=False, comment="整个结果快照的SHA256，用于回读完整性核对"),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
            comment="数据库保存实际计算结果的时刻",
        ),
        sa.UniqueConstraint("request_key", name="uq_cash_forecast_request"),
        sa.UniqueConstraint(
            "fund_code",
            "cutoff_date",
            "authorization_hash",
            "feature_hash",
            "source_revision_id",
            name="uq_cash_forecast_business",
        ),
        sa.CheckConstraint(
            "cutoff_date >= DATE '2022-01-01' AND EXTRACT(YEAR FROM cutoff_date) <> 2025",
            name="ck_cash_forecast_cutoff",
        ),
        sa.CheckConstraint(
            "(payload->>'protocol_version' = 'CASH_FORECAST_STORAGE_V1' "
            "AND payload#>>'{feature,status}' = 'INPUT_READY' "
            "AND (payload#>>'{value,up_score}')::numeric BETWEEN 0 AND 1) IS TRUE",
            name="ck_cash_forecast_payload",
        ),
        comment="现金20交易日计算结果和已知输入快照；读取时仍须核验当前发布与数据时效",
    )
    op.create_index("ix_cash_forecast_fund_date", "cash_forecast_result", ["fund_code", "cutoff_date", "forecast_id"])


def downgrade():
    op.execute("LOCK TABLE cash_forecast_result IN ACCESS EXCLUSIVE MODE")
    op.execute("""DO $$ BEGIN IF EXISTS (SELECT 1 FROM cash_forecast_result)
        THEN RAISE EXCEPTION 'cash forecasts are not empty; downgrade refused'; END IF; END $$""")
    op.drop_table("cash_forecast_result")
