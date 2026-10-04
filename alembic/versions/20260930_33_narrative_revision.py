"""为解释内容升级保留各版原文；原预测和已保存正文都不改写。"""

from alembic import op

revision = "20260930_33"
down_revision = "20260930_32"
branch_labels = None
depends_on = None


def upgrade():
    # 原有行原样保留。版本只由服务端发布决定，浏览器不能通过换版本绕过调用预算。
    op.execute("""
      ALTER TABLE prediction_narrative DROP CONSTRAINT prediction_narrative_pkey;
      ALTER TABLE prediction_narrative ADD PRIMARY KEY(source_kind,source_id,style_version);
      COMMENT ON COLUMN prediction_narrative.style_version IS '解释内容规范版本；每版成功后不可改写，升级时保留旧版';
      COMMENT ON COLUMN prediction_narrative.attempts IS '同一原预测同一解释版本最多三次外部尝试，失败冷却十五分钟';
    """)


def downgrade():
    raise RuntimeError("保留多版解释；回退应用不删除历史正文。")
