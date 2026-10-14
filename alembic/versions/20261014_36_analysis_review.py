"""综合事后回看单独留档，不进入正式预测和准确率核验表。"""

from alembic import op

revision = "20261014_36"
down_revision = "20261010_35"
branch_labels = None
depends_on = None


def upgrade():
    op.execute("""
    CREATE TABLE fund_analysis_review (
      review_id uuid PRIMARY KEY,
      fund_code varchar(6) NOT NULL CHECK(fund_code ~ '^[0-9]{6}$'),
      target_nav_date date NOT NULL,
      base_nav_date date NOT NULL CHECK(base_nav_date < target_nav_date),
      information_cutoff timestamptz NOT NULL,
      actual_generated_at timestamptz NOT NULL CHECK(actual_generated_at >= information_cutoff),
      input_hash char(64) NOT NULL,
      content_hash char(64) NOT NULL UNIQUE,
      archive jsonb NOT NULL,
      view_payload jsonb NOT NULL,
      imported_at timestamptz NOT NULL DEFAULT clock_timestamp(),
      CHECK(actual_generated_at >= ((target_nav_date + time '15:00') AT TIME ZONE 'Asia/Shanghai')),
      CHECK(view_payload->>'recordType' = 'COMPREHENSIVE_ASOF_REVIEW'),
      CHECK(view_payload->>'excludedFromAccuracy' = 'true')
    );
    CREATE INDEX ix_analysis_review_fund_time ON fund_analysis_review
      (fund_code, actual_generated_at DESC, review_id DESC);
    COMMENT ON TABLE fund_analysis_review IS '公开资料综合事后回看；仅历史展示，不参与正式预测、当前方向和准确率';
    COMMENT ON COLUMN fund_analysis_review.review_id IS '回看独立编号，不复用正式预测编号';
    COMMENT ON COLUMN fund_analysis_review.fund_code IS '原冻结包明确指定的六位基金代码';
    COMMENT ON COLUMN fund_analysis_review.target_nav_date IS '回看对应的目标净值日期';
    COMMENT ON COLUMN fund_analysis_review.base_nav_date IS '原分析比较基准日，不按后来净值变更';
    COMMENT ON COLUMN fund_analysis_review.information_cutoff IS '重建所采用的公开资料截止时刻，不代表当时已实际取得';
    COMMENT ON COLUMN fund_analysis_review.actual_generated_at IS '原文件记录的实际生成时间，不回填为事前';
    COMMENT ON COLUMN fund_analysis_review.input_hash IS '原输入规范化JSON的SHA256摘要';
    COMMENT ON COLUMN fund_analysis_review.content_hash IS '完整原输入、结果、核验及来源清单的联合摘要，重复导入复用';
    COMMENT ON COLUMN fund_analysis_review.archive IS '完整冻结输入、首次结果、原核验和来源回执，不向日常页面透出路径';
    COMMENT ON COLUMN fund_analysis_review.view_payload IS '简洁中文回看内容及安全来源链接';
    COMMENT ON COLUMN fund_analysis_review.imported_at IS '实际入库时刻，与原分析生成时刻分别保存';
    CREATE FUNCTION fund_analysis_review_guard() RETURNS trigger LANGUAGE plpgsql AS $$
    BEGIN RAISE EXCEPTION 'immutable analysis review'; END $$;
    CREATE TRIGGER tg_analysis_review_guard BEFORE UPDATE OR DELETE ON fund_analysis_review
      FOR EACH ROW EXECUTE FUNCTION fund_analysis_review_guard();
    """)


def downgrade():
    raise RuntimeError("回退时停止读取回看，不删除已经留档的原文")
