"""将已审核事件记录映射为 Java 到 Python 的内部读取契约。"""

from uuid import UUID

from sqlalchemy.orm import Session

from app.db.session import get_engine
from app.repositories.event_read import list_approved_events_for_fund
from app.schemas.event import InternalEventPage, InternalEventSummary


def list_reviewed_events(fund_code: str, page_size: int, cursor: UUID | None) -> InternalEventPage:
    """返回已审核且未过期的事件摘要，不调用任何外部数据源。

    仓储层多取一条记录用于计算下一页游标，实际响应仅返回 page_size 条。
    """
    with Session(get_engine()) as session:
        rows = list_approved_events_for_fund(session, fund_code, page_size, cursor)
    page_rows = rows[:page_size]
    return InternalEventPage(
        items=tuple(
            InternalEventSummary(
                event_id=event.event_id,
                fund_code=fund_code,
                event_type=event.event_type,
                summary=event.summary,
                source_name=source.display_name,
                source_url=news.url,
                published_at=event.published_at,
                confidence=event.confidence,
                relevance_score=relation.relevance_score,
                relation_reason=relation.relation_reason,
            )
            for event, news, relation, source in page_rows
        ),
        next_cursor=str(page_rows[-1][0].event_id) if len(rows) > page_size else None,
    )
