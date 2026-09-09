"""先选定已冻结规则再开展新研究；不接受已有报告或客户端自报的训练时间。"""

from typing import Literal, Self
from uuid import UUID

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, model_validator

from app.schemas.cash_exam_plan import CashExamPreparation
from app.schemas.cash_policy_freeze import CashFrozenPolicy
from app.schemas.cash_reinvestment_research import CashResearchRequest, CashStoredResearch
from app.schemas.historical_nav_training import Hash


class CashPlannedResearchRequest(CashResearchRequest):
    policy_freeze_id: UUID = Field(
        alias="policyFreezeId", description="计算前已经保存的规则快照编号，必须包含V2日期计划"
    )
    expected_policy_freeze_hash: Hash = Field(
        alias="expectedPolicyFreezeHash", description="读回规则快照时核对的完整content_hash，不是规则正文policy_hash"
    )


class CashPlannedResearch(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    version: Literal["CASH_PLANNED_RESEARCH_BINDING_V1"] = "CASH_PLANNED_RESEARCH_BINDING_V1"
    binding_id: UUID = Field(description="新研究与既有计划的不可变绑定回执编号，不是模型授权编号")
    request_key: UUID = Field(description="此新流程的幂等请求号，不用于给旧研究补绑定")
    evaluation_started_at: AwareDatetime = Field(
        description="核验快照和资料后、进入评估函数前取得的数据库时间；不保证资料不足时仍拟合模型"
    )
    completed_at: AwareDatetime = Field(description="评估结束、准备原子保存新报告与绑定时取得的数据库时间")
    policy_freeze: CashFrozenPolicy = Field(description="计算前已存在的完整规则快照；本操作不创建或批准规则")
    preparation: CashExamPreparation = Field(description="计算前的资料覆盖准备；其预览标志不是事前绑定凭证")
    research: CashStoredResearch = Field(description="本流程刚执行并保存的新研究，保持研究用途和未发布状态")
    content_hash: Hash = Field(description="身份、时间、规则及资料/报告关联的完整指纹，不是审批签名")
    plan_bound_before_evaluation: Literal[True] = Field(
        default=True, description="服务器流程先验证已存在的冻结计划，再调用研究器；不接受事后关联已有报告"
    )
    created: bool = Field(default=False, description="仅本次首次成功保存为true，重试及GET为false")
    database_written: bool = Field(default=False, description="本次是否新增研究与绑定两条记录，GET不写库")
    publication_allowed: Literal[False] = False
    independent_test_read: Literal[False] = False

    @model_validator(mode="after")
    def time_and_write_flags(self) -> Self:
        if not self.policy_freeze.frozen_at <= self.evaluation_started_at <= self.completed_at:
            raise ValueError("frozen policy must exist before evaluation and completion")
        if self.created != self.database_written or self.policy_freeze.database_written:
            raise ValueError("planned research must not claim to create its policy")
        return self
