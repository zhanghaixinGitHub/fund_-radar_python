"""可复现 M3 评分读模型的 Pydantic 内部接口契约。"""

from datetime import date, datetime
from decimal import Decimal
from uuid import UUID

from pydantic import BaseModel, ConfigDict


class InternalSignalSummary(BaseModel):
    """一条评分结果；未评分结果必须显式省略方向与方向概率。"""

    model_config = ConfigDict(frozen=True)

    forecast_id: UUID
    fund_code: str
    as_of_date: date
    score_status: str
    direction: str | None
    directional_probability: Decimal | None
    confidence: Decimal | None
    risk_level: str | None
    max_drawdown_estimate: Decimal | None
    explanation: str
    model_version: str
    feature_version: str
    feature_completeness: Decimal
    scored_at: datetime


class InternalSignalPage(BaseModel):
    """返回给 Java 核心服务的评分结果兼容游标分页响应。"""

    model_config = ConfigDict(frozen=True)

    items: tuple[InternalSignalSummary, ...]
    next_cursor: str | None = None
