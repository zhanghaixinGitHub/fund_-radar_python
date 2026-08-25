"""AI 服务拥有的已审核事件只读查询。"""

from datetime import UTC, datetime
from uuid import UUID

from sqlalchemy import Select, and_, or_, select
from sqlalchemy.orm import Session

from app.models.event import EventRelation, MarketEvent, NewsItem
from app.models.fund import SourceRegistry


def list_approved_events_for_fund(
    session: Session, fund_code: str, page_size: int, cursor: UUID | None
) -> tuple[tuple[MarketEvent, NewsItem, EventRelation, SourceRegistry], ...]:
    """查询指定基金关联的已审核且未过期事件，并按稳定顺序返回一页加一条探测记录。

    Args:
        session: 当前数据库会话，仅用于读取。
        fund_code: 六位基金代码。
        page_size: 调用方期望返回的最大记录数。
        cursor: 上一页最后一条事件标识；不存在时抛出 ValueError。
    """
    statement: Select[tuple[MarketEvent, NewsItem, EventRelation, SourceRegistry]] = (
        select(MarketEvent, NewsItem, EventRelation, SourceRegistry)
        .join(NewsItem, MarketEvent.news_id == NewsItem.news_id)
        .join(EventRelation, EventRelation.event_id == MarketEvent.event_id)
        .join(SourceRegistry, SourceRegistry.source_id == NewsItem.source_id)
        .where(
            MarketEvent.approval_status == "APPROVED",
            NewsItem.active.is_(True),
            NewsItem.retention_until > datetime.now(UTC),
            EventRelation.entity_type == "FUND_CODE",
            EventRelation.entity_id == fund_code,
        )
        .order_by(MarketEvent.published_at.desc(), MarketEvent.event_id.desc())
    )
    if cursor is not None:
        cursor_event = session.get(MarketEvent, cursor)
        if cursor_event is None:
            raise ValueError("Event cursor does not exist.")
        statement = statement.where(
            or_(
                MarketEvent.published_at < cursor_event.published_at,
                and_(
                    MarketEvent.published_at == cursor_event.published_at, MarketEvent.event_id < cursor_event.event_id
                ),
            )
        )
    return tuple(session.execute(statement.limit(page_size + 1)).all())
