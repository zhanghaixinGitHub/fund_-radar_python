"""供 Java 核心服务认证读取的 M2 已审核事件接口。"""

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status

from app.api.dependencies import require_service_token
from app.core.logging import get_logger
from app.core.middleware import get_trace_id
from app.schemas.event import InternalEventPage
from app.services.event_read import list_reviewed_events

router = APIRouter()
logger = get_logger(__name__)


@router.get("", response_model=InternalEventPage, dependencies=[Depends(require_service_token)])
def list_internal_events(
    fund_code: Annotated[str, Query(alias="fundCode", min_length=6, max_length=6, pattern=r"^\d{6}$")],
    page_size: Annotated[int, Query(alias="pageSize", ge=1, le=100)] = 20,
    cursor: UUID | None = None,
) -> InternalEventPage:
    """只返回已审核、未过期且与指定基金相关的事件摘要。

    游标参数格式不合法时转换为 400；原始资讯正文和未审核事件不会穿透此接口。
    """
    try:
        payload = list_reviewed_events(fund_code, page_size, cursor)
    except ValueError as error:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(error)) from error
    logger.info(
        "events.list_internal_events >>> returned reviewed event page, trace_id=%s, fund_code=%s, count=%s",
        get_trace_id(),
        fund_code,
        len(payload.items),
    )
    return payload
