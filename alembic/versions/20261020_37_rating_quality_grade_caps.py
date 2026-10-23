"""保留原分数与等级，允许八维质量规则的关键短板限制等级；不更新历史结果。"""

from alembic import op

revision = "20261020_37"
down_revision = "20261014_36"
branch_labels = None
depends_on = None


def upgrade():
    # 历史直接映射继续有效。新口径由仓储逐次重算，不能任填一个较低等级。
    op.execute("ALTER TABLE fund_rating_result DROP CONSTRAINT ck_rating_score_grade")
    op.execute("""
        ALTER TABLE fund_rating_result ADD CONSTRAINT ck_rating_score_grade CHECK (
          grade IS NULL OR grade = CASE WHEN score < 50 THEN 'WEAK' WHEN score < 70 THEN 'AVERAGE'
          WHEN score < 90 THEN 'GOOD' ELSE 'EXCELLENT' END OR
          (grade='AVERAGE' AND score>=70 AND LEAST(
            (dimension_values->>'risk')::numeric,(dimension_values->>'efficiency')::numeric,
            (dimension_values->>'management')::numeric,(dimension_values->>'liquidity')::numeric)<50) OR
          (grade='GOOD' AND score>=90 AND LEAST(
            (dimension_values->>'risk')::numeric,(dimension_values->>'efficiency')::numeric,
            (dimension_values->>'management')::numeric,(dimension_values->>'liquidity')::numeric)>=50 AND LEAST(
            (dimension_values->>'risk')::numeric,(dimension_values->>'efficiency')::numeric,
            (dimension_values->>'management')::numeric,(dimension_values->>'liquidity')::numeric)<70)
        )
    """)


def downgrade():
    # 有上限结果时旧约束会拒绝降级；不为迁就回退而篡改已经发布的总分或等级。
    op.execute("ALTER TABLE fund_rating_result DROP CONSTRAINT ck_rating_score_grade")
    op.execute("""
        ALTER TABLE fund_rating_result ADD CONSTRAINT ck_rating_score_grade CHECK (
          grade IS NULL OR grade = CASE WHEN score < 50 THEN 'WEAK' WHEN score < 70 THEN 'AVERAGE'
          WHEN score < 90 THEN 'GOOD' ELSE 'EXCELLENT' END
        )
    """)
