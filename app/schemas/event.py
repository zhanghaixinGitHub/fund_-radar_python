"""已审核、受许可范围约束的事件读模型 Pydantic 契约。"""

from datetime import datetime
from decimal import Decimal
from uuid import UUID

from pydantic import BaseModel, ConfigDict


class InternalEventSummary(BaseModel):
    """与请求基金关联的一条已审核事件，不作因果关系承诺。"""

    model_config = ConfigDict(frozen=True)

    event_id: UUID
    fund_code: str
    event_type: str
    summary: str
    source_name: str
    source_url: str
    published_at: datetime
    confidence: Decimal
    relevance_score: Decimal
    relation_reason: str


class InternalEventPage(BaseModel):
    """返回给 Java 核心服务的已审核事件兼容游标分页响应。"""

    model_config = ConfigDict(frozen=True)

    items: tuple[InternalEventSummary, ...]
    next_cursor: str | None = None
