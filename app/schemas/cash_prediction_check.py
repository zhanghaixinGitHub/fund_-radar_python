"""生成前检查只返回证据和拒绝原因，不接受客户端模型、标签或发布标记。"""

from datetime import datetime
from decimal import Decimal
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.historical_nav_training import Hash


class CashPredictionCheckRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)
    fund_code: str = Field(alias="fundCode", pattern=r"^[0-9]{6}$", description="拟生成预测的基金，不接收用户身份")
    research_run_id: UUID = Field(alias="researchRunId", description="明确检查哪一份已保存的现金研究报告")
    expected_report_hash: Hash = Field(alias="expectedReportHash", description="调用者核对过的报告指纹，防止检查错版本")


class CashComparisonCheck(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    window_id: str = Field(description="原先固定的考试窗口，不根据成绩改时间")
    scope: str = Field(description="ALL表示整体，否则为基金代码")
    baseline_id: str = Field(description="原报告中的简单对照规则")
    sample_count: int = Field(description="这组实际考试题数，不是训练样本数")
    accuracy_delta: Decimal = Field(description="模型准确率减对照，正数才是方向成绩更好")
    brier_delta: Decimal = Field(description="模型误差减对照，负数才是概率误差更小")
    strict_gain_observed: bool = Field(description="仅观察到本组两项均严格改善；不等于稳定增益或正式通过")


class CashPredictionCheck(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    mode: Literal["READ_ONLY_CASH_PREDICTION_PRECHECK"] = Field(
        default="READ_ONLY_CASH_PREDICTION_PRECHECK", description="只读生成前检查，不是已完成的预测"
    )
    status: Literal["GENERATION_BLOCKED"] = Field(
        default="GENERATION_BLOCKED", description="当前研究协议不允许生成产品预测"
    )
    checked_at: datetime = Field(description="实际执行检查的时刻，不是预测截至日")
    fund_code: str = Field(description="本次检查的基金")
    research_run_id: UUID = Field(description="已校验的研究编号")
    report_hash: Hash = Field(description="已校验的报告内容指纹")
    blocking_codes: tuple[str, ...] = Field(description="阻止生成的证据缺口和已观察失败，不允许客户端清除")
    comparisons: tuple[CashComparisonCheck, ...] = Field(description="已完成窗口的总体和逐基金对照；缺窗不伪造成绩")
    incomplete_window_ids: tuple[str, ...] = Field(description="尚未完成的固定窗口")
    publication_status: Literal["MODEL_NOT_RELEASED"] = Field(
        default="MODEL_NOT_RELEASED", description="未获正式发布资格"
    )
    inference_executed: Literal[False] = Field(default=False, description="未通过发布门槛，不执行数值推理")
    forecast_created: Literal[False] = Field(default=False, description="没有生成或保存一份产品预测")
    database_written: Literal[False] = Field(default=False, description="本接口只读，不写运行记录、模型或预测表")
    test_scored: Literal[False] = Field(default=False, description="不读取2025答案、不做最终测试")
    up_probability: None = Field(default=None, description="拒绝生成时必须为空，不泄露研究分数")
    direction: None = Field(default=None, description="拒绝生成不是预测下跌")
