"""M0 Mock 基金读模型的 Pydantic 内部接口契约。"""

from datetime import date

from pydantic import BaseModel, ConfigDict, Field


class InternalFundSummary(BaseModel):
    """传递给 Java 核心服务的最小且可公开展示的基金摘要。"""

    model_config = ConfigDict(frozen=True)

    fund_code: str
    fund_name: str
    fund_type: str
    status: str
    as_of_date: date


class InternalFundDetail(InternalFundSummary):
    """带数据状态与来源信息的 M0 Mock 基金详情读模型。"""

    nav_status: str
    data_source: str


class InternalFundPage(BaseModel):
    """返回给 Java 核心服务的兼容游标分页响应。"""

    model_config = ConfigDict(frozen=True)

    items: tuple[InternalFundSummary, ...]
    next_cursor: str | None = Field(default=None, serialization_alias="next_cursor")
