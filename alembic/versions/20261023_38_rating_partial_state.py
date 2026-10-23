"""允许明确标注的部分评分；总分和等级仍为空，不伪装完整评级。"""

from alembic import op

revision = "20261023_38"
down_revision = "20261020_37"
branch_labels = None
depends_on = None


def upgrade():
    op.drop_constraint("ck_rating_score_state", "fund_rating_result", type_="check")
    op.create_check_constraint(
        "ck_rating_score_state",
        "fund_rating_result",
        "(status = 'RATED' AND score IS NOT NULL AND grade IS NOT NULL) OR "
        "(status IN ('NOT_RATED','UNSUPPORTED','PARTIAL') AND score IS NULL AND grade IS NULL)",
    )


def downgrade():
    # 尚有部分结果时旧约束会拒绝降级；不得删除/篡改已发布历史来迁就回退。
    op.drop_constraint("ck_rating_score_state", "fund_rating_result", type_="check")
    op.create_check_constraint(
        "ck_rating_score_state",
        "fund_rating_result",
        "(status = 'RATED' AND score IS NOT NULL AND grade IS NOT NULL) OR "
        "(status IN ('NOT_RATED','UNSUPPORTED') AND score IS NULL AND grade IS NULL)",
    )
