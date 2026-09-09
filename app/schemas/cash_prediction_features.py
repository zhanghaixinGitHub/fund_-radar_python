"""实际推理前的历史输入契约；没有未来答案、预测概率或发布通过开关。"""

from datetime import date
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.cash_reinvestment_samples import CashFeaturePayload, CashSampleIssue


class CashPredictionFeatureRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)
    fund_code: Literal["001632", "006730", "008888"] = Field(alias="fundCode", description="本版三只股票型试点")
    cutoff_date: date = Field(alias="cutoffDate", ge=date(2022, 1, 1), description="信息截止日，按已结束自然日解释")


class CashPredictionFeature(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    mode: Literal["CASH_HISTORY_INPUT_ONLY_V1"] = Field(
        default="CASH_HISTORY_INPUT_ONLY_V1", description="只构造历史输入"
    )
    status: Literal["INPUT_READY", "DATA_INSUFFICIENT"] = Field(description="输入能否算出，不代表模型可以发布")
    fund_code: str = Field(description="输入属于哪只基金，不作为模型数值指标")
    cutoff_date: date = Field(description="筛选公告可得日期的上限，不用今天冒充历史日期")
    anchor_nav_date: date | None = Field(description="当前截止时已公布的最新交易日净值日")
    anchor_lag_sessions: int | None = Field(description="相对截止日最近交易日落后多少日；最多1日")
    history_dates: tuple[date, ...] = Field(description="要求的61个精确历史交易日，缺数不跳过")
    input_issues: tuple[CashSampleIssue, ...] = Field(description="只根据历史信息产生的拒收原因")
    feature_payload: CashFeaturePayload | None = Field(description="与训练共用的现金序列和七项历史指标")
    feature_hash: str | None = Field(description="只对输入内容计算的指纹，可与历史样本输入逐字段核对")
    ignored_non_trading_nav_dates: tuple[date, ...] = Field(description="读取窗口中非交易日记录，仅忽略不删除")
    database_written: Literal[False] = Field(default=False, description="本构建器不保存或更改原始资料")
    training_eligible: Literal[False] = Field(default=False, description="首次版本、事件完整性等正式准入尚未获证实")
    publication_status: Literal["MODEL_NOT_RELEASED"] = Field(
        default="MODEL_NOT_RELEASED", description="输入可计算不授予发布资格"
    )
