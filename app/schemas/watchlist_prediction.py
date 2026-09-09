"""给Java个人关注页的最小读模型；当前现金研究协议不允许发布任何概率。"""

from datetime import date, datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field


class WatchlistPrediction(BaseModel):
    fund_code: str = Field(description="经过Java本人关注关系校验的基金代码")
    status: Literal["MODEL_NOT_RELEASED", "DATA_INSUFFICIENT", "NOT_APPLICABLE", "UNAVAILABLE"] = Field(
        description="业务状态；未发布不是预测下跌"
    )
    horizon_trading_days: Literal[20] = Field(default=20, description="未来20交易日整体方向，不是每天一条预测")
    up_probability: None = Field(default=None, description="发布未通过必须为空，不能返回研究分数或填50%")
    direction: None = Field(default=None, description="未发布不生成上涨/下跌方向")
    latest_nav_date: date | None = Field(default=None, description="本地当前来源最新已公告净值日期，不是预测截止日")
    research_run_id: UUID | None = Field(default=None, description="支撑本状态的研究编号，不暴露模型系数或考试答案")
    research_evaluated_at: datetime | None = Field(default=None, description="研究报告保存时刻，不是净值更新时间")
    model_version: str | None = Field(default=None, description="研究协议版本；不表示模型已发布")
    reason_codes: tuple[str, ...] = Field(description="稳定原因代码")
    reasons: tuple[str, ...] = Field(description="对应的通俗中文解释")
    message: str = Field(description="页面主状态说明")
    disclaimer: str = Field(default="仅供研究参考，不构成投资建议；未发布不表示基金会下跌。", description="边界说明")
