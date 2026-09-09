"""生成请求、不可变内部快照与最小结果投影；外部不能提交发布凭证或模型。"""

from datetime import date, datetime
from decimal import Decimal
from typing import Literal, Self
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.schemas.cash_prediction_attempt import CashPredictionAttemptRequest
from app.schemas.cash_prediction_features import CashPredictionFeature, CashPredictionFeatureRequest
from app.schemas.cash_prediction_inference import CashInferenceValue
from app.schemas.historical_nav_calibration import CalibratedModelArtifact
from app.schemas.historical_nav_training import Hash


class CashForecastRequest(CashPredictionAttemptRequest):
    fund_code: Literal["001632", "006730", "008888"] = Field(alias="fundCode", description="本版三只股票型试点")
    expected_model_hash: Hash = Field(
        alias="expectedModelHash", description="明确选择已核对模型，不按最新或最好成绩选模型"
    )

    def feature_request(self) -> CashPredictionFeatureRequest:
        return CashPredictionFeatureRequest(fundCode=self.fund_code, cutoffDate=self.cutoff_date)

    def check_request(self):
        # 基类的预检契约不接收模型指纹；模型绑定由正式授权解析器独立核对。
        from app.schemas.cash_prediction_check import CashPredictionCheckRequest

        return CashPredictionCheckRequest.model_validate(
            self.model_dump(exclude={"request_key", "cutoff_date", "expected_model_hash"})
        )


class CashAuthorizedModel(BaseModel):
    """仅限服务器内的授权解析结果；没有接受此对象的HTTP入口，也不等于已实现凭证签发。"""

    model_config = ConfigDict(extra="forbid", frozen=True)
    authorization_id: UUID = Field(description="正式发布凭证编号；不是研究编号或旧ACTIVE记录")
    authorization_hash: Hash = Field(description="凭证、研究、模型、适用范围的共同指纹")
    research_run_id: UUID = Field(description="获准模型来源的明确研究编号")
    report_hash: Hash = Field(description="获准研究的冻结指纹")
    artifact: CalibratedModelArtifact = Field(description="与授权绑定的数值JSON模型，只留在服务器")
    calendar_hashes: tuple[Hash, ...] = Field(min_length=1, max_length=2, description="经过发布核验的日历内容指纹")


class CashForecastSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    protocol_version: Literal["CASH_FORECAST_STORAGE_V1"] = "CASH_FORECAST_STORAGE_V1"
    request: CashForecastRequest = Field(description="首次生成时的明确请求，不接收外部X或答案")
    authorization_id: UUID = Field(description="当时通过的正式发布凭证编号，读取时还须再次检查有效性")
    authorization_hash: Hash = Field(description="当时正式发布凭证的指纹")
    source_revision_id: UUID = Field(description="当时该来源最新完成的同步编号，覆盖净值及分红等更新")
    feature: CashPredictionFeature = Field(description="只含历史信息的可复现输入快照，绝不保存未来答案")
    value: CashInferenceValue = Field(description="本次模型真实算出的数值结果，不通过发布检查则不得保存")


class CashForecastView(BaseModel):
    """仅有效且仍获授权的结果才可带概率；旧数据、撤销授权和不足都清空数字。"""

    model_config = ConfigDict(extra="forbid", frozen=True)
    status: Literal["AVAILABLE", "MODEL_NOT_RELEASED", "DATA_INSUFFICIENT", "STALE"] = Field(
        description="当前可显示状态"
    )
    fund_code: str = Field(description="结果所属基金")
    cutoff_date: date = Field(description="原预测的信息截止日，不会随读取时间滚动变成今天")
    forecast_id: UUID | None = Field(default=None, description="已保存结果编号；被闸门拒绝时不存在")
    generated_at: datetime | None = Field(default=None, description="实际计算并保存的时间，不是净值日期")
    target_base_date: date | None = Field(default=None, description="原预测区间基准日，不是更早的输入净值日")
    target_end_date: date | None = Field(default=None, description="原预测区间的第20个交易日，不滚动延期")
    horizon_trading_days: Literal[20] = Field(default=20, description="20个交易日的整体方向，不是20条逐日预测")
    model_hash: Hash | None = Field(default=None, description="模型内容指纹，不返回模型参数")
    up_probability: Decimal | None = Field(default=None, ge=0, le=1, description="0至1的上涨概率；仅AVAILABLE非空")
    direction: Literal["UP", "NON_UP"] | None = Field(default=None, description="非上涨包含持平，不能标成必然下跌")
    reason_codes: tuple[str, ...] = Field(default=(), description="未发布、不足或失效原因；不是预测方向")
    created: bool = Field(default=False, description="本次调用是否新建记录；GET和重试均为false")

    @model_validator(mode="after")
    def no_number_without_available_result(self) -> Self:
        if self.status == "AVAILABLE":
            if (
                self.forecast_id is None
                or self.generated_at is None
                or self.model_hash is None
                or self.target_base_date is None
                or self.target_end_date is None
                or self.target_end_date <= self.target_base_date
                or self.up_probability is None
                or self.reason_codes
                or self.direction != ("UP" if self.up_probability > Decimal("0.5") else "NON_UP")
            ):
                raise ValueError("available forecast must have intact identity, dates and consistent probability")
        elif self.up_probability is not None or self.direction is not None or not self.reason_codes:
            raise ValueError("unavailable forecast must explain why and carry no probability or direction")
        return self
