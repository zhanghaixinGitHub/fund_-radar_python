"""公共基金八维四级评级：增量表、不可变快照和数据库约束。"""

from alembic import op

revision = "20261010_35"
down_revision = "20261009_34"
branch_labels = None
depends_on = None


def upgrade():
    # 本 revision 固化 DDL，未来模型改动不会改变历史迁移。
    op.execute("""
CREATE TABLE fund_rating_methodology (
	methodology_id VARCHAR(64) NOT NULL,
	family VARCHAR(32) NOT NULL,
	config JSONB NOT NULL,
	validation JSONB,
	active BOOLEAN NOT NULL,
	created_at TIMESTAMP WITH TIME ZONE NOT NULL,
	PRIMARY KEY (methodology_id)
);
CREATE TABLE fund_rating_classification (
	classification_id VARCHAR(64) NOT NULL,
	fund_code VARCHAR(6) NOT NULL,
	product_id VARCHAR(128) NOT NULL,
	category_code VARCHAR(64) NOT NULL,
	family VARCHAR(32) NOT NULL,
	currency VARCHAR(3) NOT NULL,
	eligibility VARCHAR(64) NOT NULL,
	effective_from DATE NOT NULL,
	effective_to DATE,
	source_refs JSONB NOT NULL,
	verified_at TIMESTAMP WITH TIME ZONE NOT NULL,
	revoked BOOLEAN NOT NULL,
	PRIMARY KEY (classification_id)
);
CREATE INDEX ix_rating_classification_code_date ON fund_rating_classification (fund_code, effective_from);
CREATE TABLE fund_rating_evidence (
	evidence_hash VARCHAR(64) NOT NULL,
	fund_code VARCHAR(6) NOT NULL,
	classification_id VARCHAR(64) NOT NULL,
	as_of_date DATE NOT NULL,
	payload JSONB NOT NULL,
	acquired_at TIMESTAMP WITH TIME ZONE NOT NULL,
	revoked BOOLEAN NOT NULL,
	PRIMARY KEY (evidence_hash),
	FOREIGN KEY(classification_id) REFERENCES fund_rating_classification (classification_id)
);
CREATE INDEX ix_rating_evidence_code_date ON fund_rating_evidence (fund_code, as_of_date);
CREATE TABLE fund_rating_batch (
	batch_id VARCHAR(64) NOT NULL,
	category_code VARCHAR(64) NOT NULL,
	methodology_id VARCHAR(64),
	as_of_date DATE NOT NULL,
	frozen_at TIMESTAMP WITH TIME ZONE NOT NULL,
	input_hash VARCHAR(64) NOT NULL,
	sample_hash VARCHAR(64) NOT NULL,
	input_path VARCHAR(128) NOT NULL,
	input_snapshot JSONB NOT NULL,
	member_count INTEGER NOT NULL,
	product_count INTEGER NOT NULL,
	status VARCHAR(16) NOT NULL,
	completed_at TIMESTAMP WITH TIME ZONE NOT NULL,
	withdrawn BOOLEAN NOT NULL,
	PRIMARY KEY (batch_id),
	CONSTRAINT ck_rating_batch_status CHECK (status IN ('COMPLETE','PUBLISHED')),
	FOREIGN KEY(methodology_id) REFERENCES fund_rating_methodology (methodology_id)
);
CREATE INDEX ix_rating_batch_category_date ON fund_rating_batch (category_code, as_of_date, status);
CREATE TABLE fund_rating_member (
	batch_id VARCHAR(64) NOT NULL,
	fund_code VARCHAR(6) NOT NULL,
	product_id VARCHAR(128),
	representative BOOLEAN NOT NULL,
	admitted BOOLEAN NOT NULL,
	reasons JSONB NOT NULL,
	evidence_hash VARCHAR(64),
	PRIMARY KEY (batch_id, fund_code),
	FOREIGN KEY(batch_id) REFERENCES fund_rating_batch (batch_id),
	FOREIGN KEY(evidence_hash) REFERENCES fund_rating_evidence (evidence_hash)
);
CREATE TABLE fund_rating_result (
	batch_id VARCHAR(64) NOT NULL,
	fund_code VARCHAR(6) NOT NULL,
	status VARCHAR(16) NOT NULL,
	raw_score VARCHAR(128),
	score NUMERIC(20, 12),
	grade VARCHAR(16),
	dimension_values JSONB,
	public_evidence JSONB NOT NULL,
	valid_until TIMESTAMP WITH TIME ZONE,
	PRIMARY KEY (batch_id, fund_code),
	CONSTRAINT ck_rating_score_range CHECK (score IS NULL OR (score >= 0 AND score <= 100)),
	CONSTRAINT ck_rating_score_state CHECK ((status = 'RATED' AND score IS NOT NULL AND grade IS NOT NULL)
        OR (status IN ('NOT_RATED','UNSUPPORTED') AND score IS NULL AND grade IS NULL)),
	CONSTRAINT ck_rating_score_grade CHECK (grade IS NULL OR grade = CASE WHEN score < 50 THEN 'WEAK'
        WHEN score < 70 THEN 'AVERAGE' WHEN score < 90 THEN 'GOOD' ELSE 'EXCELLENT' END),
	FOREIGN KEY(batch_id) REFERENCES fund_rating_batch (batch_id)
);
CREATE INDEX ix_rating_result_code_batch ON fund_rating_result (fund_code, batch_id);
CREATE TABLE fund_rating_current (
	category_code VARCHAR(64) NOT NULL,
	batch_id VARCHAR(64) NOT NULL,
	updated_at TIMESTAMP WITH TIME ZONE NOT NULL,
	PRIMARY KEY (category_code),
	FOREIGN KEY(batch_id) REFERENCES fund_rating_batch (batch_id)
);
CREATE TABLE fund_rating_audit (
	audit_id VARCHAR(64) NOT NULL,
	category_code VARCHAR(64) NOT NULL,
	batch_id VARCHAR(64),
	action VARCHAR(32) NOT NULL,
	reason TEXT NOT NULL,
	created_at TIMESTAMP WITH TIME ZONE NOT NULL,
	PRIMARY KEY (audit_id)
);
    """)
    op.execute("""
      CREATE UNIQUE INDEX uq_rating_representative ON fund_rating_member(batch_id, product_id)
        WHERE representative;
      ALTER TABLE fund_rating_result ADD CONSTRAINT ck_rating_eight_dimensions CHECK (
        status <> 'RATED' OR (jsonb_typeof(dimension_values) = 'object'
        AND dimension_values IS NOT NULL
        AND dimension_values ?& ARRAY['return','risk','efficiency','stability',
                                     'management','holdings','liquidity','cost']
        AND (dimension_values - ARRAY['return','risk','efficiency','stability',
                                     'management','holdings','liquidity','cost']) = '{}'));
      CREATE FUNCTION fund_rating_immutable() RETURNS trigger LANGUAGE plpgsql AS $$
      BEGIN RAISE EXCEPTION 'immutable rating snapshot'; END $$;
      CREATE TRIGGER tg_rating_result_immutable BEFORE UPDATE OR DELETE ON fund_rating_result
        FOR EACH ROW EXECUTE FUNCTION fund_rating_immutable();
      CREATE TRIGGER tg_rating_member_immutable BEFORE UPDATE OR DELETE ON fund_rating_member
        FOR EACH ROW EXECUTE FUNCTION fund_rating_immutable();
      CREATE TRIGGER tg_rating_audit_immutable BEFORE UPDATE OR DELETE ON fund_rating_audit
        FOR EACH ROW EXECUTE FUNCTION fund_rating_immutable();
      CREATE FUNCTION fund_rating_classification_guard() RETURNS trigger LANGUAGE plpgsql AS $$
      BEGIN
        PERFORM pg_advisory_xact_lock(hashtextextended('rating-class:' || NEW.fund_code,0));
        IF EXISTS (SELECT 1 FROM fund_rating_classification c WHERE c.fund_code=NEW.fund_code
          AND c.classification_id<>NEW.classification_id AND NOT c.revoked AND NOT NEW.revoked
          AND daterange(c.effective_from,c.effective_to,'[]') &&
              daterange(NEW.effective_from,NEW.effective_to,'[]')) THEN
          RAISE EXCEPTION 'overlapping rating classification';
        END IF;
        RETURN NEW;
      END $$;
      CREATE TRIGGER tg_rating_classification_guard BEFORE INSERT OR UPDATE ON fund_rating_classification
        FOR EACH ROW EXECUTE FUNCTION fund_rating_classification_guard();
      COMMENT ON TABLE fund_rating_result IS '基金公共评级；八维完整才有等级，非评级缺值不补零';
      COMMENT ON TABLE fund_rating_current IS '完整类别发布指针；事务内锁定切换，页面仅查询';
      CREATE FUNCTION fund_rating_snapshot_guard() RETURNS trigger LANGUAGE plpgsql AS $$
      DECLARE mutable_fields text[];
      BEGIN
        IF TG_OP='DELETE' THEN RAISE EXCEPTION 'immutable rating evidence'; END IF;
        mutable_fields := CASE TG_TABLE_NAME
          WHEN 'fund_rating_batch' THEN ARRAY['status','withdrawn']
          WHEN 'fund_rating_methodology' THEN ARRAY['active','validation']
          ELSE ARRAY['revoked'] END;
        IF (to_jsonb(NEW)-mutable_fields) IS DISTINCT FROM (to_jsonb(OLD)-mutable_fields) THEN
          RAISE EXCEPTION 'immutable rating evidence';
        END IF;
        RETURN NEW;
      END $$;
      CREATE TRIGGER tg_rating_batch_guard BEFORE UPDATE OR DELETE ON fund_rating_batch
        FOR EACH ROW EXECUTE FUNCTION fund_rating_snapshot_guard();
      CREATE TRIGGER tg_rating_evidence_guard BEFORE UPDATE OR DELETE ON fund_rating_evidence
        FOR EACH ROW EXECUTE FUNCTION fund_rating_snapshot_guard();
      CREATE TRIGGER tg_rating_method_guard BEFORE UPDATE OR DELETE ON fund_rating_methodology
        FOR EACH ROW EXECUTE FUNCTION fund_rating_snapshot_guard();
      CREATE TRIGGER tg_rating_class_immutable BEFORE UPDATE OR DELETE ON fund_rating_classification
        FOR EACH ROW EXECUTE FUNCTION fund_rating_snapshot_guard();
      CREATE FUNCTION fund_rating_pointer_guard() RETURNS trigger LANGUAGE plpgsql AS $$
      DECLARE b fund_rating_batch; actual_count integer;
      BEGIN
        SELECT * INTO b FROM fund_rating_batch WHERE batch_id=NEW.batch_id;
        SELECT count(*) INTO actual_count FROM fund_rating_result WHERE batch_id=NEW.batch_id;
        IF b.batch_id IS NULL OR b.category_code<>NEW.category_code OR b.status<>'PUBLISHED'
           OR b.withdrawn OR b.member_count<>actual_count
           OR b.member_count<>(SELECT count(*) FROM fund_rating_member WHERE batch_id=NEW.batch_id) THEN
          RAISE EXCEPTION 'invalid rating publication pointer';
        END IF;
        RETURN NEW;
      END $$;
      CREATE TRIGGER tg_rating_pointer_guard BEFORE INSERT OR UPDATE ON fund_rating_current
        FOR EACH ROW EXECUTE FUNCTION fund_rating_pointer_guard();
    """)


def downgrade():
    raise RuntimeError("评级回退应撤回展示或切换有效批次，禁止清表删除历史依据")
