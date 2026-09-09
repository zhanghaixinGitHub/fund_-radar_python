"""计划绑定的新研究；加性迁移，禁止更改关联或删除非空历史。"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql as pg

revision = "20260909_18"
down_revision = "20260909_17"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "cash_planned_research_binding",
        sa.Column("binding_id", pg.UUID(), primary_key=True, comment="不可变研究绑定编号，也是内部研究请求凭证"),
        sa.Column("request_key", pg.UUID(), nullable=False, comment="新研究流程的幂等请求号，不关联旧研究请求号"),
        sa.Column(
            "research_run_id",
            pg.UUID(),
            sa.ForeignKey("cash_research_run.run_id"),
            nullable=False,
            comment="同事务新建的研究报告编号",
        ),
        sa.Column(
            "policy_freeze_id",
            pg.UUID(),
            sa.ForeignKey("cash_policy_freeze.freeze_id"),
            nullable=False,
            comment="计算前已保存的规则快照",
        ),
        sa.Column("policy_freeze_hash", sa.String(64), nullable=False, comment="当时核对的规则快照完整指纹"),
        sa.Column("report_hash", sa.String(64), nullable=False, comment="本流程生成的研究报告指纹，防止引用被替换"),
        sa.Column(
            "preparation",
            pg.JSONB(),
            nullable=False,
            comment="计算前资料覆盖准备，含明确批次、日历计划和各窗口可用题数",
        ),
        sa.Column(
            "evaluation_started_at",
            sa.DateTime(timezone=True),
            nullable=False,
            comment="核对计划与资料后、评估前取得的数据库时间",
        ),
        sa.Column(
            "completed_at", sa.DateTime(timezone=True), nullable=False, comment="评估结束后保存前取得的数据库时间"
        ),
        sa.Column("content_hash", sa.String(64), nullable=False, comment="身份、时刻、规则、资料及报告引用的SHA256"),
        sa.UniqueConstraint("request_key", name="uq_cash_planned_request"),
        sa.UniqueConstraint("research_run_id", name="uq_cash_planned_research"),
        sa.CheckConstraint("completed_at >= evaluation_started_at", name="ck_cash_planned_times"),
        sa.CheckConstraint(
            "policy_freeze_hash ~ '^[0-9a-f]{64}$' AND report_hash ~ '^[0-9a-f]{64}$' "
            "AND content_hash ~ '^[0-9a-f]{64}$'",
            name="ck_cash_planned_hashes",
        ),
        comment="计算前选定已确认规则的新研究绑定；仅研究、不代表正式数据准入或发布",
    )
    op.execute("""CREATE FUNCTION cash_planned_research_immutable() RETURNS trigger LANGUAGE plpgsql AS $$
        BEGIN RAISE EXCEPTION 'planned research bindings are immutable' USING ERRCODE='55000'; END; $$""")
    op.execute(
        "CREATE TRIGGER cash_planned_research_no_mutation BEFORE UPDATE OR DELETE OR TRUNCATE "
        "ON cash_planned_research_binding FOR EACH STATEMENT EXECUTE FUNCTION cash_planned_research_immutable()"
    )


def downgrade():
    op.execute("LOCK TABLE cash_planned_research_binding IN ACCESS EXCLUSIVE MODE")
    op.execute("""DO $$ BEGIN IF EXISTS (SELECT 1 FROM cash_planned_research_binding)
        THEN RAISE EXCEPTION 'planned research bindings are not empty; downgrade refused'; END IF; END $$""")
    op.drop_table("cash_planned_research_binding")
    op.execute("DROP FUNCTION cash_planned_research_immutable()")
