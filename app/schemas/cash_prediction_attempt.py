"""生成尝试的历史回执：保存拒绝事实，不把写入一条记录说成生成了一份预测。"""

from datetime import date, datetime
from typing import Literal, Self
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.schemas.cash_prediction_check import CashPredictionCheck, CashPredictionCheckRequest
from app.schemas.historical_nav_training import Hash


class CashPredictionAttemptRequest(CashPredictionCheckRequest):
    request_key: UUID = Field(alias="requestKey", description="本次生成尝试的重试凭证；重新检查使用新编号")
    cutoff_date: date = Field(alias="cutoffDate", ge=date(2022, 1, 1), description="计划使用的信息截止日，必须已结束")

    @model_validator(mode="after")
    def protect_final_test(self) -> Self:
        if self.cutoff_date.year == 2025:
            raise ValueError("2025仍为受保护的最终测试期，不能用于生成尝试")
        return self

    def check_request(self) -> CashPredictionCheckRequest:
        return CashPredictionCheckRequest.model_validate(self.model_dump(exclude={"request_key", "cutoff_date"}))


class CashPredictionAttempt(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    attempt_id: UUID = Field(description="这份已保存回执的编号，可按编号读回")
    request_key: UUID = Field(description="对应的重试凭证；同key不得换基金、日期或报告")
    cutoff_date: date = Field(description="当时计划生成预测的信息截止日；拒绝时并未读取这些输入")
    created_at: datetime = Field(description="数据库实际保存回执的时刻，不是信息截止日")
    check: CashPredictionCheck = Field(description="当时实际执行的闸门检查，含已核验报告编号、指纹和拒绝原因")
    receipt_hash: Hash = Field(description="请求内容与检查快照的联合指纹，回读时校验")
    historical_receipt: Literal[True] = Field(default=True, description="这是当时的回执，不是现在的发布资格")
    database_written: Literal[True] = Field(default=True, description="回执已经持久化；GET本身仍然只读")
    forecast_created: Literal[False] = Field(default=False, description="保存拒绝回执不代表已生成产品预测")
