"""基金经理目录和任期版本；冻结DDL，不随未来ORM变化。"""

from alembic import op
from sqlalchemy import text

revision = "20261026_39"
down_revision = "20261023_38"
branch_labels = None
depends_on = None

STATEMENTS = (
    """
CREATE TABLE fund_manager (
	manager_id UUID NOT NULL,
	anchor_key VARCHAR(256) NOT NULL,
	display_name VARCHAR(128) NOT NULL,
	identity_status VARCHAR(32) NOT NULL,
	redirect_to UUID,
	current_snapshot_id UUID,
	created_at TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL,
	PRIMARY KEY (manager_id),
	UNIQUE (anchor_key),
	FOREIGN KEY(redirect_to) REFERENCES fund_manager (manager_id)
)
""",
    "CREATE INDEX ix_fund_manager_display_name ON fund_manager (display_name)",
    "COMMENT ON TABLE fund_manager IS '基金经理人员主档，姓名不唯一，不以基金公司替代人员身份'",
    "COMMENT ON COLUMN fund_manager.manager_id IS '稳定经理编号'",
    "COMMENT ON COLUMN fund_manager.anchor_key IS '原始基金与姓名锚点；仅幂等建档，不作为跨基金归并证明'",
    "COMMENT ON COLUMN fund_manager.display_name IS '公开姓名，可重名'",
    "COMMENT ON COLUMN fund_manager.identity_status IS 'ANCHORED本基金锚点；VERIFIED有跨基金身份依据'",
    "COMMENT ON COLUMN fund_manager.redirect_to IS '有证据合并后的目标人员；空表示未合并'",
    "COMMENT ON COLUMN fund_manager.current_snapshot_id IS '当前已发布快照；查询另校验快照所属人员'",
    "COMMENT ON COLUMN fund_manager.created_at IS '本机建档时间'",
    """
CREATE TABLE fund_manager_identity_evidence (
	evidence_id UUID NOT NULL,
	manager_id UUID NOT NULL,
	source_id UUID NOT NULL,
	decision VARCHAR(32) NOT NULL,
	evidence_ref JSONB NOT NULL,
	content_hash VARCHAR(64) NOT NULL,
	reviewed_at TIMESTAMP WITH TIME ZONE NOT NULL,
	PRIMARY KEY (evidence_id),
	CONSTRAINT uq_manager_identity_evidence_hash UNIQUE (manager_id, content_hash),
	FOREIGN KEY(manager_id) REFERENCES fund_manager (manager_id),
	FOREIGN KEY(source_id) REFERENCES source_registry (source_id)
)
""",
    "CREATE INDEX ix_fund_manager_identity_evidence_manager_id ON fund_manager_identity_evidence (manager_id)",
    "COMMENT ON TABLE fund_manager_identity_evidence IS '经理身份归并依据及其接受或待核对决定'",
    "COMMENT ON COLUMN fund_manager_identity_evidence.evidence_id IS '身份依据编号'",
    "COMMENT ON COLUMN fund_manager_identity_evidence.manager_id IS '对应经理'",
    "COMMENT ON COLUMN fund_manager_identity_evidence.source_id IS '来源登记编号'",
    "COMMENT ON COLUMN fund_manager_identity_evidence.decision IS 'ACCEPTED明确接受的关系；PENDING待核对'",
    "COMMENT ON COLUMN fund_manager_identity_evidence.evidence_ref "
    "IS '原件哈希、出处、取得时间、任职锚点及明确接受的基金代码'",
    "COMMENT ON COLUMN fund_manager_identity_evidence.content_hash IS '依据规范化内容SHA256'",
    "COMMENT ON COLUMN fund_manager_identity_evidence.reviewed_at IS '事实核对时间，非原文公开日'",
    """
CREATE TABLE fund_manager_profile_version (
	profile_version_id UUID NOT NULL,
	manager_id UUID NOT NULL,
	education VARCHAR(128),
	professional_resume TEXT,
	evidence_ref JSONB NOT NULL,
	content_hash VARCHAR(64) NOT NULL,
	published_on DATE,
	received_at TIMESTAMP WITH TIME ZONE NOT NULL,
	PRIMARY KEY (profile_version_id),
	CONSTRAINT uq_manager_profile_hash UNIQUE (manager_id, content_hash),
	FOREIGN KEY(manager_id) REFERENCES fund_manager (manager_id)
)
""",
    "CREATE INDEX ix_fund_manager_profile_version_manager_id ON fund_manager_profile_version (manager_id)",
    "COMMENT ON TABLE fund_manager_profile_version IS '基金经理职业履历版本，保留来源与时间'",
    "COMMENT ON COLUMN fund_manager_profile_version.profile_version_id IS '履历版本编号'",
    "COMMENT ON COLUMN fund_manager_profile_version.manager_id IS '对应经理'",
    "COMMENT ON COLUMN fund_manager_profile_version.education IS '来源明确披露学历，未知为空'",
    "COMMENT ON COLUMN fund_manager_profile_version.professional_resume IS '与本基金任职锚点匹配的公开职业履历'",
    "COMMENT ON COLUMN fund_manager_profile_version.evidence_ref IS '履历出处、锚点及原件引用'",
    "COMMENT ON COLUMN fund_manager_profile_version.content_hash IS '履历内容哈希'",
    "COMMENT ON COLUMN fund_manager_profile_version.published_on IS '来源公开日，未提供则为空'",
    "COMMENT ON COLUMN fund_manager_profile_version.received_at IS '本机实际取得时间'",
    """
CREATE TABLE fund_manager_related_fund (
	related_fund_id UUID NOT NULL,
	source_id UUID NOT NULL,
	source_fund_code VARCHAR(32) NOT NULL,
	mapped_fund_code VARCHAR(32),
	fund_name VARCHAR(256) NOT NULL,
	company_name VARCHAR(256),
	fund_type VARCHAR(32) NOT NULL,
	product_identity JSONB,
	PRIMARY KEY (related_fund_id),
	CONSTRAINT uq_manager_related_source_code UNIQUE (source_id, source_fund_code),
	FOREIGN KEY(source_id) REFERENCES source_registry (source_id),
	FOREIGN KEY(mapped_fund_code) REFERENCES fund_share_class (fund_code)
)
""",
    "COMMENT ON TABLE fund_manager_related_fund IS '经理关联基金基础目录，不自动加入原日常任务'",
    "COMMENT ON COLUMN fund_manager_related_fund.related_fund_id IS '关联份额编号'",
    "COMMENT ON COLUMN fund_manager_related_fund.source_id IS '来源编号'",
    "COMMENT ON COLUMN fund_manager_related_fund.source_fund_code IS '来源完整代码，保留交易场所后缀'",
    "COMMENT ON COLUMN fund_manager_related_fund.mapped_fund_code IS '已有市场代码；外部份额允许为空'",
    "COMMENT ON COLUMN fund_manager_related_fund.fund_name IS '来源基金份额名称'",
    "COMMENT ON COLUMN fund_manager_related_fund.company_name IS '管理公司，不是基金经理姓名'",
    "COMMENT ON COLUMN fund_manager_related_fund.fund_type IS '基金类型，未核实使用OTHER'",
    "COMMENT ON COLUMN fund_manager_related_fund.product_identity "
    "IS '经正式资料核对的产品家族、适用期和出处；未知为空'",
    """
CREATE TABLE fund_manager_tenure_version (
	tenure_version_id UUID NOT NULL,
	tenure_key UUID NOT NULL,
	manager_id UUID NOT NULL,
	related_fund_id UUID NOT NULL,
	begin_date DATE,
	end_date DATE,
	verification_status VARCHAR(32) NOT NULL,
	team_status VARCHAR(32) NOT NULL,
	is_current BOOLEAN NOT NULL,
	content_hash VARCHAR(64) NOT NULL,
	evidence_refs JSONB NOT NULL,
	observed_at TIMESTAMP WITH TIME ZONE NOT NULL,
	PRIMARY KEY (tenure_version_id),
	CONSTRAINT ck_manager_tenure_dates CHECK (end_date IS NULL OR begin_date IS NULL OR end_date >= begin_date),
	CONSTRAINT uq_manager_tenure_hash UNIQUE (tenure_key, content_hash),
	FOREIGN KEY(manager_id) REFERENCES fund_manager (manager_id),
	FOREIGN KEY(related_fund_id) REFERENCES fund_manager_related_fund (related_fund_id)
)
""",
    "CREATE INDEX ix_fund_manager_tenure_version_manager_id ON fund_manager_tenure_version (manager_id)",
    "CREATE INDEX ix_fund_manager_tenure_version_tenure_key ON fund_manager_tenure_version (tenure_key)",
    "CREATE UNIQUE INDEX uq_manager_tenure_current ON fund_manager_tenure_version (tenure_key) WHERE is_current",
    "COMMENT ON TABLE fund_manager_tenure_version IS '经理规范任期及来源修订，历史版本不覆盖'",
    "COMMENT ON COLUMN fund_manager_tenure_version.tenure_version_id IS '任期版本编号'",
    "COMMENT ON COLUMN fund_manager_tenure_version.tenure_key IS '同一次任职的稳定标识'",
    "COMMENT ON COLUMN fund_manager_tenure_version.manager_id IS '经理编号'",
    "COMMENT ON COLUMN fund_manager_tenure_version.related_fund_id IS '基金份额编号'",
    "COMMENT ON COLUMN fund_manager_tenure_version.begin_date IS '任职首日，未知为空'",
    "COMMENT ON COLUMN fund_manager_tenure_version.end_date IS '最后管理日，空值仍须结合来源覆盖判断在任'",
    "COMMENT ON COLUMN fund_manager_tenure_version.verification_status "
    "IS 'CONFIRMED已锚定关系；PENDING身份或日期待核对'",
    "COMMENT ON COLUMN fund_manager_tenure_version.team_status IS 'SOLO独立；CO_MANAGED共管；UNKNOWN团队资料不完整'",
    "COMMENT ON COLUMN fund_manager_tenure_version.is_current IS '是否当前有效版本，不等于当前仍在任'",
    "COMMENT ON COLUMN fund_manager_tenure_version.content_hash IS '本版本任职内容哈希'",
    "COMMENT ON COLUMN fund_manager_tenure_version.evidence_refs IS '任职事实与完整扫描的原件引用'",
    "COMMENT ON COLUMN fund_manager_tenure_version.observed_at IS '本轮事实取得时间'",
    """
CREATE TABLE fund_manager_coverage_snapshot (
	snapshot_id UUID NOT NULL,
	manager_id UUID,
	scope_kind VARCHAR(32) NOT NULL,
	payload JSONB NOT NULL,
	content_hash VARCHAR(64) NOT NULL,
	published_at TIMESTAMP WITH TIME ZONE NOT NULL,
	PRIMARY KEY (snapshot_id),
	CONSTRAINT ck_manager_snapshot_scope CHECK (
        (scope_kind='DIRECTORY' AND manager_id IS NULL) OR (scope_kind='MANAGER' AND manager_id IS NOT NULL)),
	FOREIGN KEY(manager_id) REFERENCES fund_manager (manager_id)
)
""",
    "CREATE INDEX ix_fund_manager_coverage_snapshot_manager_id ON fund_manager_coverage_snapshot (manager_id)",
    "CREATE INDEX ix_fund_manager_coverage_snapshot_published_at ON fund_manager_coverage_snapshot (published_at)",
    "CREATE INDEX ix_fund_manager_coverage_snapshot_scope_kind ON fund_manager_coverage_snapshot (scope_kind)",
    "COMMENT ON TABLE fund_manager_coverage_snapshot IS '经理资料及目录的不可变发布快照，覆盖与完整性分别说明'",
    "COMMENT ON COLUMN fund_manager_coverage_snapshot.snapshot_id IS '已发布资料版本'",
    "COMMENT ON COLUMN fund_manager_coverage_snapshot.manager_id IS '经理快照所属人员；目录为空'",
    "COMMENT ON COLUMN fund_manager_coverage_snapshot.scope_kind IS 'MANAGER经理；DIRECTORY目录'",
    "COMMENT ON COLUMN fund_manager_coverage_snapshot.payload IS '无用户态的已核实公共读模型和来源覆盖'",
    "COMMENT ON COLUMN fund_manager_coverage_snapshot.content_hash IS '发布内容哈希，支持幂等与审计'",
    "COMMENT ON COLUMN fund_manager_coverage_snapshot.published_at IS '原子发布时间，不代替净值日或公告日'",
    """
CREATE TABLE fund_manager_metric_snapshot (
	metric_id UUID NOT NULL,
	manager_id UUID NOT NULL,
	related_fund_id UUID NOT NULL,
	tenure_version_id UUID NOT NULL,
	return_rate NUMERIC(24, 12),
	max_drawdown_rate NUMERIC(24, 12),
	annualized_rate NUMERIC(24, 12),
	input_hash VARCHAR(64) NOT NULL,
	method_version VARCHAR(64) NOT NULL,
	payload JSONB NOT NULL,
	calculated_at TIMESTAMP WITH TIME ZONE NOT NULL,
	PRIMARY KEY (metric_id),
	CONSTRAINT uq_manager_metric_input UNIQUE (tenure_version_id, input_hash, method_version),
	FOREIGN KEY(manager_id) REFERENCES fund_manager (manager_id),
	FOREIGN KEY(related_fund_id) REFERENCES fund_manager_related_fund (related_fund_id),
	FOREIGN KEY(tenure_version_id) REFERENCES fund_manager_tenure_version (tenure_version_id)
)
""",
    "CREATE INDEX ix_fund_manager_metric_snapshot_manager_id ON fund_manager_metric_snapshot (manager_id)",
    "COMMENT ON TABLE fund_manager_metric_snapshot IS '经理具体基金任期表现，缺项保留为空，不生成个人总收益'",
    "COMMENT ON COLUMN fund_manager_metric_snapshot.metric_id IS '指标编号'",
    "COMMENT ON COLUMN fund_manager_metric_snapshot.manager_id IS '经理编号'",
    "COMMENT ON COLUMN fund_manager_metric_snapshot.related_fund_id IS '份额编号'",
    "COMMENT ON COLUMN fund_manager_metric_snapshot.tenure_version_id IS '对应任期版本'",
    "COMMENT ON COLUMN fund_manager_metric_snapshot.return_rate IS '收益小数比例，0.1为10%%，不可用为空'",
    "COMMENT ON COLUMN fund_manager_metric_snapshot.max_drawdown_rate IS '非负最大回撤比例，未核实为空'",
    "COMMENT ON COLUMN fund_manager_metric_snapshot.annualized_rate IS '完整不少于365日的年化比例，否则为空'",
    "COMMENT ON COLUMN fund_manager_metric_snapshot.input_hash IS '实际计算输入哈希'",
    "COMMENT ON COLUMN fund_manager_metric_snapshot.method_version IS '分红再投等确定计算方法版本'",
    "COMMENT ON COLUMN fund_manager_metric_snapshot.payload IS '计算日期、状态、缺项、币种和出处'",
    "COMMENT ON COLUMN fund_manager_metric_snapshot.calculated_at IS '计算时间，不代替数据日'",
    "ALTER TABLE source_sync_run ADD COLUMN scope_snapshot JSONB",
    "ALTER TABLE source_sync_run ADD COLUMN result_summary JSONB",
    "COMMENT ON COLUMN source_sync_run.scope_snapshot IS '独立资料任务的冻结目标范围，历史任务可空'",
    "COMMENT ON COLUMN source_sync_run.result_summary IS '资料任务结果与真实覆盖，历史任务可空'",
)


def upgrade():
    """只新增隔离结构和可空审计字段，不修改旧业务数据。"""
    for statement in STATEMENTS:
        op.execute(text(statement))


def downgrade():
    """已有真实资料时拒绝删表；回滚应用保留历史。"""
    raise RuntimeError("经理资料迁移保留历史，请回退应用；结构清理由明确维护任务执行。")
