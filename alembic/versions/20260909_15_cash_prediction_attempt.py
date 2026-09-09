"""生成拒绝回执的加性迁移；不动旧数据，非空时禁止回退删除证据。"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql as pg

revision = "20260909_15"
down_revision = "20260908_14"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "cash_prediction_attempt",
        sa.Column("attempt_id", pg.UUID(), primary_key=True, comment="这份不可变历史回执的编号"),
        sa.Column("request_key", pg.UUID(), nullable=False, comment="重试凭证；重新检查使用新编号"),
        sa.Column(
            "request_hash", sa.String(64), nullable=False, comment="基金、日期、报告及版本的请求指纹，不含重试编号"
        ),
        sa.Column(
            "fund_code",
            sa.String(32),
            sa.ForeignKey("fund_share_class.fund_code"),
            nullable=False,
            comment="计划生成的基金",
        ),
        sa.Column("cutoff_date", sa.Date(), nullable=False, comment="计划使用的信息截止日，拒绝时并未读取该日输入"),
        sa.Column(
            "research_run_id",
            pg.UUID(),
            sa.ForeignKey("cash_research_run.run_id"),
            nullable=False,
            comment="此次明确检查的冻结研究报告编号",
        ),
        sa.Column(
            "report_hash", sa.String(64), nullable=False, comment="当时核验的研究报告指纹，不选最新或成绩最优记录"
        ),
        sa.Column(
            "check_payload", pg.JSONB(), nullable=False, comment="当时的拒绝原因和比较证据；无概率、方向或模型输入"
        ),
        sa.Column("receipt_hash", sa.String(64), nullable=False, comment="请求指纹与完整检查快照的联合指纹"),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
            comment="数据库保存回执的实际时刻",
        ),
        sa.UniqueConstraint("request_key", name="uq_cash_prediction_attempt_request"),
        sa.CheckConstraint(
            "cutoff_date >= DATE '2022-01-01' AND EXTRACT(YEAR FROM cutoff_date) <> 2025",
            name="ck_cash_prediction_attempt_cutoff",
        ),
        sa.CheckConstraint(
            "(check_payload->>'status' = 'GENERATION_BLOCKED' "
            "AND check_payload->'up_probability' = 'null'::jsonb "
            "AND check_payload->'direction' = 'null'::jsonb "
            "AND check_payload->'inference_executed' = 'false'::jsonb "
            "AND check_payload->'forecast_created' = 'false'::jsonb "
            "AND jsonb_array_length(check_payload->'blocking_codes') > 0) IS TRUE",
            name="ck_cash_prediction_attempt_rejected",
        ),
        comment="现金预测生成拒绝回执；不是正式发布或预测结果",
    )
    op.create_index(
        "ix_cash_prediction_attempt_fund_created", "cash_prediction_attempt", ["fund_code", "created_at", "attempt_id"]
    )


def downgrade():
    op.execute("LOCK TABLE cash_prediction_attempt IN ACCESS EXCLUSIVE MODE")
    op.execute("""DO $$ BEGIN
        IF EXISTS (SELECT 1 FROM cash_prediction_attempt)
        THEN RAISE EXCEPTION 'cash prediction attempts are not empty; downgrade refused'; END IF;
        END $$""")
    op.drop_table("cash_prediction_attempt")
