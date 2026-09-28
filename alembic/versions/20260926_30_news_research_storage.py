"""追加公告研究版本和完整卡片；不修改现有业务事件表或预测链。"""

from alembic import op

revision = "20260926_30"
down_revision = "20260924_29"
branch_labels = None
depends_on = None


def upgrade():
    op.execute("""
      CREATE TABLE news_research_bundle (
        bundle_hash varchar(64) PRIMARY KEY CHECK(bundle_hash ~ '^[0-9a-f]{64}$'),
        dataset_key varchar(128) NOT NULL,
        revision varchar(64) NOT NULL,
        fund_code varchar(6) NOT NULL CHECK(fund_code ~ '^[0-9]{6}$'),
        supersedes_bundle_hash varchar(64) REFERENCES news_research_bundle(bundle_hash),
        card_count integer NOT NULL CHECK(card_count BETWEEN 1 AND 100),
        metadata_payload jsonb NOT NULL CHECK(jsonb_typeof(metadata_payload)='object'),
        created_at timestamptz NOT NULL DEFAULT now(),
        UNIQUE(dataset_key, revision), CHECK(supersedes_bundle_hash IS DISTINCT FROM bundle_hash)
      );
      CREATE TABLE news_research_card (
        bundle_hash varchar(64) NOT NULL REFERENCES news_research_bundle(bundle_hash),
        sample_id varchar(64) NOT NULL,
        record_hash varchar(64) NOT NULL CHECK(record_hash ~ '^[0-9a-f]{64}$'),
        fund_code varchar(6) NOT NULL CHECK(fund_code ~ '^[0-9]{6}$'),
        stock_code varchar(16) NOT NULL, target_date date NOT NULL, published_date date NOT NULL,
        announcement_id varchar(128) NOT NULL, pdf_sha256 varchar(64) NOT NULL CHECK(pdf_sha256 ~ '^[0-9a-f]{64}$'),
        event_topics jsonb NOT NULL CHECK(jsonb_typeof(event_topics)='array'), event_stage text NOT NULL,
        record_payload jsonb NOT NULL CHECK(jsonb_typeof(record_payload)='object'),
        purpose varchar(16) NOT NULL CHECK(purpose='RESEARCH'),
        validation_status varchar(32) NOT NULL CHECK(validation_status='LOCAL_ANCHORS_VERIFIED'),
        confidence numeric(5,4),
        business_eligible boolean NOT NULL CHECK(business_eligible=false),
        prediction_eligible boolean NOT NULL CHECK(prediction_eligible=false),
        training_eligible boolean NOT NULL CHECK(training_eligible=false),
        created_at timestamptz NOT NULL DEFAULT now(),
        PRIMARY KEY(bundle_hash,sample_id), CHECK(published_date < target_date), CHECK(confidence IS NULL)
      );
      CREATE INDEX ix_news_research_fund_date ON news_research_card(fund_code,target_date);
      CREATE INDEX ix_news_research_announcement ON news_research_card(announcement_id,pdf_sha256);
      CREATE INDEX ix_news_research_topics ON news_research_card USING gin(event_topics);
      CREATE FUNCTION guard_news_research_immutable() RETURNS trigger LANGUAGE plpgsql AS $$
        BEGIN RAISE EXCEPTION 'NEWS_RESEARCH_IMMUTABLE_CREATE_NEW_REVISION'; END $$;
      CREATE TRIGGER news_research_bundle_immutable BEFORE UPDATE OR DELETE ON news_research_bundle
        FOR EACH ROW EXECUTE FUNCTION guard_news_research_immutable();
      CREATE TRIGGER news_research_card_immutable BEFORE UPDATE OR DELETE ON news_research_card
        FOR EACH ROW EXECUTE FUNCTION guard_news_research_immutable();
      COMMENT ON TABLE news_research_bundle IS '不可变公告研究版本；来源摘要和文件位置可追溯，更正追加版本';
      COMMENT ON TABLE news_research_card IS '本地证据已核验的研究卡片，不代表语义自动正确、业务审核或预测有效';
      COMMENT ON COLUMN news_research_card.record_payload IS
        '原格式卡与原分析结构，保留金额单位、阶段、锚点、未知项与披露持仓版本';
      COMMENT ON COLUMN news_research_card.published_date IS '来源仅证明公开日期，不虚构首发时分秒';
      COMMENT ON COLUMN news_research_card.confidence IS '当前样例没有可核验分数，必须为空';
    """)


def downgrade():
    # 仅空研究表可直接回退；已有证据不得通过普通迁移悄悄删除。
    op.execute("""DO $$ BEGIN
      IF EXISTS(SELECT 1 FROM news_research_bundle) THEN
        RAISE EXCEPTION 'NEWS_RESEARCH_NOT_EMPTY_BACKUP_REQUIRED';
      END IF; END $$;""")
    op.drop_table("news_research_card")
    op.drop_table("news_research_bundle")
    op.execute("DROP FUNCTION guard_news_research_immutable()")
