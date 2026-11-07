"""完整数字分与尚未验证的等级分开，历史完整评级和部分评分保持原约束。"""

from alembic import op

revision = "20261028_40"
down_revision = "20261026_39"
branch_labels = None
depends_on = None


def upgrade():
    op.drop_constraint("ck_rating_score_state", "fund_rating_result", type_="check")
    op.create_check_constraint(
        "ck_rating_score_state",
        "fund_rating_result",
        "(status = 'RATED' AND score IS NOT NULL AND grade IS NOT NULL) OR "
        "(status = 'SCORED' AND score IS NOT NULL AND grade IS NULL) OR "
        "(status IN ('NOT_RATED','UNSUPPORTED','PARTIAL') AND score IS NULL AND grade IS NULL)",
    )


def downgrade():
    # 已有SCORED历史时拒绝降级；不能为满足旧约束删除或伪造历史等级。
    op.drop_constraint("ck_rating_score_state", "fund_rating_result", type_="check")
    op.create_check_constraint(
        "ck_rating_score_state",
        "fund_rating_result",
        "(status = 'RATED' AND score IS NOT NULL AND grade IS NOT NULL) OR "
        "(status IN ('NOT_RATED','UNSUPPORTED','PARTIAL') AND score IS NULL AND grade IS NULL)",
    )
