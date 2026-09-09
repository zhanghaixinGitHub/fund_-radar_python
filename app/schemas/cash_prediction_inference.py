"""内部数值结果契约；计算成功不等于模型获准发布。"""

from datetime import date
from decimal import Decimal

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.historical_nav_training import Hash


class CashInferenceValue(BaseModel):
    """内部计算结果，不是已发布预测；外层生成器仍必须具备正式发布凭证。"""

    model_config = ConfigDict(extra="forbid", frozen=True)
    fund_code: str = Field(description="这次计算属于哪只基金，不是模型的输入列")
    cutoff_date: date = Field(description="信息截止日；所有输入在该日结束前可得")
    anchor_nav_date: date = Field(description="模型用到的最新已公告净值日，可以比预测基准日晚1个交易日")
    target_base_date: date = Field(description="cutoff及以前最近交易日；只规划日期，不读取该日未知净值")
    target_end_date: date = Field(description="cutoff之后第20个交易日，不读取其价格或答案")
    feature_hash: Hash = Field(description="本次已知历史输入的指纹")
    model_hash: Hash = Field(description="已回读并核对的基础模型与校准器共同指纹")
    up_score: Decimal = Field(ge=0, le=1, description="内部校准计算值；未经正式发布不能称为产品概率")
    predicted_up: bool = Field(description="对返回的8位计算值严格判断大于0.5；否则为非上涨而非必然下跌")
