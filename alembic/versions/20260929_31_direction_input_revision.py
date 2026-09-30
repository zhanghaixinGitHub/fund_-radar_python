"""追加一日输入版本；原预测、模型、研究与答案均不改写。"""

from alembic import op

revision = "20260929_31"
down_revision = "20260926_30"
branch_labels = None
depends_on = None


def upgrade():
    op.execute("""
      CREATE TABLE direction_1d_input_revision (
        revision_sequence bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
        fund_code varchar(6) NOT NULL CHECK(fund_code ~ '^[0-9]{6}$'),
        target_nav_date date NOT NULL, protocol varchar(32) NOT NULL,
        input_identity char(64) NOT NULL CHECK(input_identity ~ '^[a-f0-9]{64}$'),
        identity_payload jsonb NOT NULL, result jsonb,
        observed_at timestamptz NOT NULL DEFAULT clock_timestamp(),
        completed_at timestamptz, deadline_at timestamptz NOT NULL,
        CHECK(completed_at IS NULL OR completed_at < deadline_at),
        CHECK((result IS NULL) = (completed_at IS NULL))
      );
      CREATE INDEX ix_direction_1d_input_revision_scope
        ON direction_1d_input_revision(fund_code,target_nav_date,protocol,revision_sequence DESC);
      COMMENT ON TABLE direction_1d_input_revision IS
        '当前实际输入的有序版本；相同输入复用，变化后追加，成功原文不可更改';
      COMMENT ON COLUMN direction_1d_input_revision.revision_sequence IS '数据库分配的输入先后顺序，不代表生成完成先后';
      COMMENT ON COLUMN direction_1d_input_revision.fund_code IS '公共基金份额六位代码，不含账户身份';
      COMMENT ON COLUMN direction_1d_input_revision.target_nav_date IS '预测目标净值日期';
      COMMENT ON COLUMN direction_1d_input_revision.protocol IS '一日预测协议，用于隔离不同结果口径';
      COMMENT ON COLUMN direction_1d_input_revision.input_identity IS '实际采用资料和模型的 SHA256 摘要';
      COMMENT ON COLUMN direction_1d_input_revision.identity_payload IS '摘要所依据的公共来源版本及模型标识';
      COMMENT ON COLUMN direction_1d_input_revision.result IS '成功预测的原文和摘要；成功前为空';
      COMMENT ON COLUMN direction_1d_input_revision.observed_at IS '数据库登记实际输入版本的时刻';
      COMMENT ON COLUMN direction_1d_input_revision.completed_at IS '成功原文落库时刻，必须早于截止';
      COMMENT ON COLUMN direction_1d_input_revision.deadline_at IS '该目标日的真实截止时刻';
      CREATE FUNCTION direction_1d_revision_immutable() RETURNS trigger LANGUAGE plpgsql AS $$
      BEGIN
        IF TG_OP='DELETE' OR OLD.result IS NOT NULL THEN
          RAISE EXCEPTION 'direction revision is immutable';
        END IF;
        IF (to_jsonb(NEW)-'result'-'completed_at') IS DISTINCT FROM
           (to_jsonb(OLD)-'result'-'completed_at') THEN
          RAISE EXCEPTION 'direction input identity is immutable';
        END IF;
        RETURN NEW;
      END $$;
      CREATE TRIGGER direction_1d_revision_guard BEFORE UPDATE OR DELETE
        ON direction_1d_input_revision FOR EACH ROW EXECUTE FUNCTION direction_1d_revision_immutable();
    """)


def downgrade():
    raise RuntimeError("保留已生成的预测证据，不支持删除式回退；停用新入口并继续读取历史。")
