"""发布规则快照，不是模型发布凭证；不允许调用者自报已完成考试或数据审核。"""

from decimal import Decimal
from typing import Annotated, Literal, Self
from uuid import UUID

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, model_validator

from app.schemas.cash_exam_plan import CashExamPlan
from app.schemas.cash_release_review import CashReleasePolicy
from app.schemas.historical_nav_calibration import CalibrationWindow
from app.schemas.historical_nav_training import Hash


class CashRuleBinding(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
    version: Literal["CASH_RELEASE_RULE_BINDING_V1"] = "CASH_RELEASE_RULE_BINDING_V1"
    evaluator_version: Literal["CASH_RELEASE_REVIEW_V1"] = Field(
        default="CASH_RELEASE_REVIEW_V1", description="本规则审查算法的版本，不是代码签名"
    )
    windows: tuple[CalibrationWindow, ...] = Field(
        min_length=3, max_length=3, description="事先固定的两轮滚动和2024验证日期/数量"
    )
    feature_versions: dict[str, str] = Field(description="现金输入、标签、样本与研究协议版本")
    bin_edges: tuple[Decimal, ...] = Field(
        min_length=6, max_length=6, description="固定五档的六个边界，不得看成绩后合并"
    )
    ece_comparison: Literal["RECOMPUTE_FROM_UNROUNDED_BINS"] = Field(
        default="RECOMPUTE_FROM_UNROUNDED_BINS", description="门槛按原分档重算，不先四舍五入"
    )
    annual_calibration_only: Literal[True] = Field(default=True, description="完整分档数量门槛用于年度，非季度")
    coverage_plan_required: Literal[True] = Field(
        default=True, description="仍须另有事前考试计划及其证据，不从现有样本反推分母"
    )
    independent_test_required: Literal[True] = Field(
        default=True, description="仍须另行完成独立考试；此快照不授权读取2025"
    )


class CashPlannedRuleBinding(CashRuleBinding):
    version: Literal["CASH_RELEASE_RULE_BINDING_V2"] = "CASH_RELEASE_RULE_BINDING_V2"
    exam_plan: CashExamPlan = Field(description="独立于批次和成绩的固定应考日期，含不读取数值的2025日期计划")


RuleBinding = Annotated[CashRuleBinding | CashPlannedRuleBinding, Field(discriminator="version")]


class CashPolicySnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    version: Literal["CASH_POLICY_FREEZE_V1"] = "CASH_POLICY_FREEZE_V1"
    policy: CashReleasePolicy = Field(description="已获服务器业务确认的规则正文")
    binding: RuleBinding = Field(description="这套规则对应的固定时间窗口、口径及比较方式；V2包含日期计划")

    @model_validator(mode="after")
    def must_be_approved(self) -> Self:
        if self.policy.approval_state != "APPROVED":
            raise ValueError("draft policy cannot be frozen")
        return self


class CashPolicyDescriptor(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    policy: CashReleasePolicy = Field(description="服务器当前规则，可为草案")
    binding: RuleBinding = Field(description="服务器当前固定比较约定")
    policy_hash: Hash = Field(description="规则正文指纹；冻结请求须原样核对")
    binding_hash: Hash = Field(description="窗口/口径/比较约定的内容指纹，不是程序签名")
    approval_ready: bool = Field(description="规则是否已获业务确认；不表示已冻结或可发布模型")
    database_written: Literal[False] = False
    publication_allowed: Literal[False] = False


class CashPolicyFreezeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)
    request_key: UUID = Field(alias="requestKey", description="同一冻结操作的重试凭证")
    expected_policy_hash: Hash = Field(alias="expectedPolicyHash", description="调用者核对过的服务器规则指纹")
    expected_binding_hash: Hash = Field(alias="expectedBindingHash", description="调用者核对过的固定比较约定指纹")


class CashFrozenPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    freeze_id: UUID = Field(description="不可变规则快照编号，不是模型授权编号")
    request_key: UUID = Field(description="首次冻结的请求凭证")
    frozen_at: AwareDatetime = Field(description="实际从数据库时钟取得的冻结时间，不接受调用者回填")
    policy_hash: Hash = Field(description="保存的规则正文指纹")
    binding_hash: Hash = Field(description="保存的比较约定指纹")
    content_hash: Hash = Field(description="包含身份、时间与完整快照的内容指纹，不是密码学审批签名")
    snapshot: CashPolicySnapshot = Field(description="当时实际保存的内容；GET不改成今天的新配置")
    created: bool = Field(default=False, description="仅首次POST成功保存为true；重试和GET为false")
    database_written: bool = Field(default=False, description="此次调用是否新增一条快照，不表示模型或预测被修改")
    publication_allowed: Literal[False] = Field(default=False, description="保存了规则不等于模型合格或发布")
    independent_test_read: Literal[False] = Field(default=False, description="冻结操作不读取2025数值或答案")

    @model_validator(mode="after")
    def writes_match_creation(self) -> Self:
        if self.created != self.database_written:
            raise ValueError("freeze write flag must match actual creation")
        return self
