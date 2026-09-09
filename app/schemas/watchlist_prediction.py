"""给Java本人关注页的最小投影；数字仅来自仍获授权且有效的已保存现金结果。"""

from datetime import date, datetime
from decimal import Decimal
from typing import Literal, Self
from uuid import UUID

from pydantic import BaseModel, Field, model_validator

from app.schemas.cash_forecast import CashForecastView
from app.schemas.historical_nav_training import Hash


class WatchlistPrediction(BaseModel):
    fund_code: str = Field(description="经过Java本人关注关系校验的基金代码")
    status: Literal[
        "AVAILABLE", "STALE", "MODEL_NOT_RELEASED", "DATA_INSUFFICIENT", "NOT_APPLICABLE", "UNAVAILABLE"
    ] = Field(description="业务状态；未发布不是预测下跌")
    horizon_trading_days: Literal[20] = Field(default=20, description="未来20交易日整体方向，不是每天一条预测")
    up_probability: Decimal | None = Field(default=None, ge=0, le=1, description="仅有效已发布结果可返回0至1概率")
    direction: Literal["UP", "NON_UP"] | None = Field(default=None, description="非上涨含持平；不可用状态为空")
    latest_nav_date: date | None = Field(default=None, description="本地当前来源最新已公告净值日期，不是预测截止日")
    research_run_id: UUID | None = Field(default=None, description="支撑本状态的研究编号，不暴露模型系数或考试答案")
    research_evaluated_at: datetime | None = Field(default=None, description="研究报告保存时刻，不是净值更新时间")
    model_version: str | None = Field(default=None, description="研究或结果契约版本；版本名自身不代表模型获准")
    forecast_id: UUID | None = Field(default=None, description="已存现金结果编号，不是研究报告或拒绝回执编号")
    cutoff_date: date | None = Field(default=None, description="原结果的信息截止日，不随刷新变成今天")
    target_base_date: date | None = Field(default=None, description="原20交易日回报的基准日期")
    target_end_date: date | None = Field(default=None, description="原20交易日期间的终点，不随查询滚动延期")
    generated_at: datetime | None = Field(default=None, description="这份结果实际生成保存的时间")
    model_hash: Hash | None = Field(default=None, description="实际使用的数值模型指纹，不含模型参数")
    reason_codes: tuple[str, ...] = Field(description="稳定原因代码")
    reasons: tuple[str, ...] = Field(description="对应的通俗中文解释")
    message: str = Field(description="页面主状态说明")
    disclaimer: str = Field(default="仅供研究参考，不构成投资建议；未发布不表示基金会下跌。", description="边界说明")

    @model_validator(mode="after")
    def validate_probability_projection(self) -> Self:
        if len(self.reason_codes) != len(self.reasons):
            raise ValueError("reason codes and explanations must correspond")
        if self.status != "AVAILABLE":
            if self.up_probability is not None or self.direction is not None or not self.reason_codes:
                raise ValueError("unavailable state cannot contain numbers")
            return self
        if self.model_version != "CASH_FORECAST_STORAGE_V1" or self.research_run_id is None or self.cutoff_date is None:
            raise ValueError("available projection requires stored cash result identity")
        CashForecastView(
            status=self.status,
            fund_code=self.fund_code,
            cutoff_date=self.cutoff_date,
            forecast_id=self.forecast_id,
            generated_at=self.generated_at,
            target_base_date=self.target_base_date,
            target_end_date=self.target_end_date,
            model_hash=self.model_hash,
            up_probability=self.up_probability,
            direction=self.direction,
            reason_codes=self.reason_codes,
        )
        return self
