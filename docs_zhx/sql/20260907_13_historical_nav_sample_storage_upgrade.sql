-- 历史净值学习样本存储：PostgreSQL 等价原生 DDL（20260907_13）。
-- 前置：fund_ai 已由 Alembic 升级至 20260905_12，三张新表尚不存在。
-- 当前只供审阅；本轮未执行。只建空表、索引和注释，不包含业务 DML。
-- 常规部署只执行 Alembic upgrade 20260907_13，不再重复执行本文件。
-- 本文件不更新 alembic_version；手工执行须另行核对版本管理，禁止盲目 stamp。
-- batch_id/sample_id 由 Python 保存时生成 UUID；原生 INSERT 须显式提供，不依赖数据库扩展。
-- BEGIN GENERATED ALEMBIC SQL
BEGIN;

CREATE TABLE historical_nav_sample_batch (
    batch_id UUID NOT NULL,
    request_key UUID NOT NULL,
    fund_code VARCHAR(32) NOT NULL,
    fund_type VARCHAR(32) DEFAULT 'STOCK' NOT NULL,
    source_code VARCHAR(64) NOT NULL,
    source_sync_run_id UUID,
    start_date DATE NOT NULL,
    end_date DATE NOT NULL,
    feature_version VARCHAR(128) NOT NULL,
    sample_rule_version VARCHAR(128) NOT NULL,
    label_version VARCHAR(128) NOT NULL,
    purpose VARCHAR(32) DEFAULT 'LEARNING_ONLY' NOT NULL,
    sample_count INTEGER NOT NULL,
    scorable_count INTEGER NOT NULL,
    data_insufficient_count INTEGER NOT NULL,
    label_not_matured_count INTEGER NOT NULL,
    unavailable_reasons JSONB NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL,
    CONSTRAINT pk_historical_nav_sample_batch PRIMARY KEY (batch_id),
    CONSTRAINT ck_historical_nav_batch_counts CHECK (sample_count >= 0 AND sample_count <= end_date - start_date + 1 AND scorable_count >= 0 AND data_insufficient_count >= 0 AND label_not_matured_count >= 0 AND sample_count = scorable_count + data_insufficient_count + label_not_matured_count),
    CONSTRAINT ck_historical_nav_batch_fund_type CHECK (fund_type = 'STOCK'),
    CONSTRAINT ck_historical_nav_batch_purpose CHECK (purpose = 'LEARNING_ONLY'),
    CONSTRAINT ck_historical_nav_batch_reasons CHECK (jsonb_typeof(unavailable_reasons) = 'object'),
    CONSTRAINT ck_historical_nav_batch_versions CHECK (btrim(source_code) <> '' AND btrim(feature_version) <> '' AND btrim(sample_rule_version) <> '' AND btrim(label_version) <> ''),
    CONSTRAINT ck_historical_nav_batch_window CHECK (end_date - start_date BETWEEN 0 AND 30),
    CONSTRAINT fk_historical_nav_batch_fund FOREIGN KEY(fund_code) REFERENCES fund_share_class (fund_code),
    CONSTRAINT fk_historical_nav_batch_source_run FOREIGN KEY(source_sync_run_id) REFERENCES source_sync_run (sync_run_id),
    CONSTRAINT uq_historical_nav_batch_request UNIQUE (request_key)
);

COMMENT ON TABLE historical_nav_sample_batch IS '历史净值学习样本批次（练习册封面，仅供学习，不是模型或预测结果）';

COMMENT ON COLUMN historical_nav_sample_batch.batch_id IS '批次唯一编号；一次完整保存对应一份练习册';

COMMENT ON COLUMN historical_nav_sample_batch.request_key IS '保存请求的幂等凭证；重试沿用，明确重算时换新凭证';

COMMENT ON COLUMN historical_nav_sample_batch.fund_code IS '整批所属基金份额代码，例如008888；关联现有基金档案';

COMMENT ON COLUMN historical_nav_sample_batch.fund_type IS '保存时的基金品类快照，本版固定为STOCK股票型';

COMMENT ON COLUMN historical_nav_sample_batch.source_code IS '保存时的数据来源编码快照；不是模型输入指标';

COMMENT ON COLUMN historical_nav_sample_batch.source_sync_run_id IS '来源同步水位编号；不能证明每条净值首次可得时间或恢复历史修订，未知时为空';

COMMENT ON COLUMN historical_nav_sample_batch.start_date IS '样本起点日期范围的第一天，包含当天';

COMMENT ON COLUMN historical_nav_sample_batch.end_date IS '样本起点日期范围的最后一天，包含当天';

COMMENT ON COLUMN historical_nav_sample_batch.feature_version IS '全批统一的历史指标计算规则版本，不是训练出的模型版本';

COMMENT ON COLUMN historical_nav_sample_batch.sample_rule_version IS '全批统一的样本筛选规则版本，例如是否拒收过时起点';

COMMENT ON COLUMN historical_nav_sample_batch.label_version IS '全批统一的答案计算规则版本；没有成熟答案时也登记本次采用的规则';

COMMENT ON COLUMN historical_nav_sample_batch.purpose IS '用途固定为LEARNING_ONLY；保存不等于可训练或可发布';

COMMENT ON COLUMN historical_nav_sample_batch.sample_count IS '本批实际样本总数，包含拒收和答案未成熟样本';

COMMENT ON COLUMN historical_nav_sample_batch.scorable_count IS '特征和答案均可用的样本数，不代表模型合格';

COMMENT ON COLUMN historical_nav_sample_batch.data_insufficient_count IS '数据不合格的样本数，例如历史不足或起点过时';

COMMENT ON COLUMN historical_nav_sample_batch.label_not_matured_count IS '特征已有但未来答案尚未齐备的样本数';

COMMENT ON COLUMN historical_nav_sample_batch.unavailable_reasons IS '不可用原因到样本数的汇总对象，例如STALE_NAV_AT_CUTOFF对应1；无原因时为{}';

COMMENT ON COLUMN historical_nav_sample_batch.created_at IS '批次保存时间（带时区），不是净值公告日';

CREATE INDEX ix_historical_nav_batch_fund_created ON historical_nav_sample_batch (fund_code, created_at, batch_id);

CREATE TABLE historical_nav_sample (
    sample_id UUID NOT NULL,
    batch_id UUID NOT NULL,
    as_of_date DATE NOT NULL,
    available_at DATE,
    nav_value_basis VARCHAR(32) NOT NULL,
    eligibility_status VARCHAR(32) NOT NULL,
    unavailable_reason TEXT,
    feature_payload JSONB NOT NULL,
    feature_hash VARCHAR(64) NOT NULL,
    CONSTRAINT pk_historical_nav_sample PRIMARY KEY (sample_id),
    CONSTRAINT ck_historical_nav_sample_available CHECK (eligibility_status = 'DATA_INSUFFICIENT' OR (available_at IS NOT NULL AND available_at >= as_of_date AND nav_value_basis <> 'UNDETERMINED')),
    CONSTRAINT ck_historical_nav_sample_basis CHECK (nav_value_basis IN ('ACCUMULATED_NAV', 'UNIT_NAV', 'UNDETERMINED')),
    CONSTRAINT ck_historical_nav_sample_hash CHECK (feature_hash ~ '^[0-9a-f]{64}$'),
    CONSTRAINT ck_historical_nav_sample_payload CHECK (jsonb_typeof(feature_payload) = 'object'),
    CONSTRAINT ck_historical_nav_sample_reason CHECK ((eligibility_status = 'SCORABLE' AND unavailable_reason IS NULL) OR (eligibility_status <> 'SCORABLE' AND unavailable_reason IS NOT NULL AND btrim(unavailable_reason) <> '')),
    CONSTRAINT ck_historical_nav_sample_status CHECK (eligibility_status IN ('SCORABLE', 'DATA_INSUFFICIENT', 'LABEL_NOT_MATURED')),
    CONSTRAINT fk_historical_nav_sample_batch FOREIGN KEY(batch_id) REFERENCES historical_nav_sample_batch (batch_id),
    CONSTRAINT uq_historical_nav_sample_batch_date UNIQUE (batch_id, as_of_date)
);

COMMENT ON TABLE historical_nav_sample IS '历史净值学习样本题目；保留正常、拒收及答案未成熟的样本，不存未来答案';

COMMENT ON COLUMN historical_nav_sample.sample_id IS '题目唯一编号，用于关联单独存放的答案';

COMMENT ON COLUMN historical_nav_sample.batch_id IS '所属练习册的批次编号；同批共享基金、来源水位和规则版本';

COMMENT ON COLUMN historical_nav_sample.as_of_date IS '起点净值所属的业务日期，不代表当天已经看到净值';

COMMENT ON COLUMN historical_nav_sample.available_at IS '起点公告日，按日终可见作为历史输入截止；缺失或错误的公告仍随拒收样本保留';

COMMENT ON COLUMN historical_nav_sample.nav_value_basis IS '统一净值口径：ACCUMULATED_NAV累计、UNIT_NAV单位、UNDETERMINED未确定';

COMMENT ON COLUMN historical_nav_sample.eligibility_status IS '样本状态：SCORABLE完整、DATA_INSUFFICIENT不合格、LABEL_NOT_MATURED答案未齐';

COMMENT ON COLUMN historical_nav_sample.unavailable_reason IS '拒收或答案未成熟的原因代码；完整样本为空，不能用0代替原因';

COMMENT ON COLUMN historical_nav_sample.feature_payload IS '当时已知条件的完整JSON包；包含来源、质量和指标，不得放入未来答案';

COMMENT ON COLUMN historical_nav_sample.feature_hash IS '仅对feature_payload计算的SHA-256指纹；不包含答案、批次编号或筛选规则版本';

CREATE TABLE historical_nav_sample_label (
    sample_id UUID NOT NULL,
    horizon_trading_days SMALLINT DEFAULT '20' NOT NULL,
    label_end_date DATE NOT NULL,
    label_available_at DATE NOT NULL,
    future_return_20d NUMERIC NOT NULL,
    label_up_20d SMALLINT NOT NULL,
    CONSTRAINT pk_historical_nav_sample_label PRIMARY KEY (sample_id),
    CONSTRAINT ck_historical_nav_label_available CHECK (label_available_at >= label_end_date),
    CONSTRAINT ck_historical_nav_label_direction CHECK ((future_return_20d > 0 AND label_up_20d = 1) OR (future_return_20d <= 0 AND label_up_20d = 0)),
    CONSTRAINT ck_historical_nav_label_horizon CHECK (horizon_trading_days = 20),
    CONSTRAINT ck_historical_nav_label_return CHECK (future_return_20d > -1 AND future_return_20d::text NOT IN ('NaN', 'Infinity', '-Infinity')),
    CONSTRAINT fk_historical_nav_label_sample FOREIGN KEY(sample_id) REFERENCES historical_nav_sample (sample_id)
);

COMMENT ON TABLE historical_nav_sample_label IS '历史净值学习样本的离线答案，与输入分表；非预测概率，未成熟或拒收时没有答案行';

COMMENT ON COLUMN historical_nav_sample_label.sample_id IS '所回答题目的唯一编号；同时作为本表主键，保证一道题至多一份答案';

COMMENT ON COLUMN historical_nav_sample_label.horizon_trading_days IS '从起点向后数的净值区间数，本版固定20，不是20个自然日';

COMMENT ON COLUMN historical_nav_sample_label.label_end_date IS '起点后第20条净值的业务日期，即答案终点';

COMMENT ON COLUMN historical_nav_sample_label.label_available_at IS '答案所依赖的未来记录全部公告完成的日期，不是保存时间';

COMMENT ON COLUMN historical_nav_sample_label.future_return_20d IS '后来实际区间收益：终点净值/起点净值-1；不固定小数位，0.14表示14%';

COMMENT ON COLUMN historical_nav_sample_label.label_up_20d IS '真实收益大于0记1；持平或下跌记0；不允许以0代替缺失答案';

COMMIT;
