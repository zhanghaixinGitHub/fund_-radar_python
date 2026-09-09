"""新版训练资料及研究运行契约；正式准入与研究计算结果明确分开。"""

from datetime import datetime
from typing import Literal, Self
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.schemas.historical_nav_calibration import CalibrationWindowReport
from app.schemas.historical_nav_training import Hash


class CashPrepareRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)
    batch_ids: tuple[UUID, ...] = Field(
        alias="batchIds",
        min_length=1,
        max_length=128,
        description="显式选择独立现金版本批次；不接受旧累计净值批次或外部X/y",
    )

    @model_validator(mode="after")
    def unique_ids(self) -> Self:
        if len(set(self.batch_ids)) != len(self.batch_ids):
            raise ValueError("batchIds不能重复")
        return self


class CashResearchRequest(CashPrepareRequest):
    request_key: UUID = Field(alias="requestKey", description="这次研究的保存凭证；重试不重新训练")
    expected_dataset_hash: Hash = Field(alias="expectedDatasetHash", description="准备报告中的资料指纹")


class CashPreparation(BaseModel):
    protocol_version: Literal["CASH_RESEARCH_PROTOCOL_V1"] = "CASH_RESEARCH_PROTOCOL_V1"
    status: Literal["RESEARCH_READY", "INSUFFICIENT_DATA"] = Field(description="仅研究计算数量是否满足，非正式准入")
    batch_ids: tuple[UUID, ...] = Field(description="排序后的明确批次清单")
    dataset_hash: Hash = Field(description="规则版本及批次内容指纹")
    feature_names: tuple[str, ...] = Field(description="模型X的七列固定顺序，不含日期、基金代码和答案")
    versions: dict[str, str] = Field(description="现金特征、标签、样本与研究规则的独立版本")
    input_count: int = Field(description="所有批次中的题数，含重复和不可用")
    duplicate_count: int = Field(description="内容完全相同的重复题数")
    unique_count: int = Field(description="按基金及cutoff去重后的题数")
    usable_count: int = Field(description="输入及答案都能计算的研究题数；窗口还会剔除跨界答案")
    excluded_reasons: dict[str, int] = Field(description="每道去重题互斥的不可用原因")
    fund_counts: dict[str, dict[str, int]] = Field(
        description="每基金TRAIN/VALIDATION可用数量；2025未读取，不报0当作已检查"
    )
    missing_counts: dict[str, dict[str, int]] = Field(description="2022–2023至少252，2024至少120；仅研究数量检查")
    training_eligible: Literal[False] = Field(default=False, description="当前资料仍未获正式训练准入")
    test_scored: Literal[False] = Field(default=False, description="2025独立测试未读取数值或评分")
    admission_blockers: tuple[str, ...] = Field(description="阻止正式发布的证据缺口，不能由请求参数清除")


class CashResearchReport(BaseModel):
    protocol_version: Literal["CASH_RESEARCH_PROTOCOL_V1"] = "CASH_RESEARCH_PROTOCOL_V1"
    status: Literal["EVALUATED", "PARTIAL_EVALUATION", "INSUFFICIENT_DATA"] = Field(
        description="固定三窗真实研究完成情况"
    )
    preparation: CashPreparation = Field(description="本次冻结资料及数量/准入检查")
    windows: tuple[CalibrationWindowReport, ...] = Field(description="固定DEV两窗及2024验证；各自模型和成绩，不选赢家")
    model_fitted: bool = Field(description="是否至少有一窗完成基础模型和校准")
    release_gate: Literal["BLOCKED"] = Field(default="BLOCKED", description="当前协议缺正式证据，始终拒绝发布")
    publication_status: Literal["MODEL_NOT_RELEASED"] = "MODEL_NOT_RELEASED"
    release_blockers: tuple[str, ...] = Field(description="数据/最终独立测试及本轮窗口问题")
    test_scored: Literal[False] = False
    report_hash: str = Field(description="除本字段外报告SHA256，防止保存后模型/成绩错配")


class CashStoredResearch(BaseModel):
    run_id: UUID = Field(description="已保存研究运行编号")
    request_key: UUID = Field(description="幂等凭证")
    created_at: datetime = Field(description="实际保存时刻")
    database_written: Literal[True] = Field(default=True, description="报告及数值模型已保存，GET不重新训练")
    report: CashResearchReport = Field(description="不可变报告，内含研究模型，不是用户可见预测")
