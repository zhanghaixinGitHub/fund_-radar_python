"""供 Java 核心服务认证读取的 M3 评分结果接口。"""

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status

from app.api.dependencies import require_service_token
from app.core.logging import get_logger
from app.core.middleware import get_trace_id
from app.schemas.signal import InternalSignalPage
from app.services.signal_read import list_completed_signals

router = APIRouter()
logger = get_logger(__name__)


@router.get("", response_model=InternalSignalPage, dependencies=[Depends(require_service_token)])
def list_internal_signals(
    fund_code: Annotated[str, Query(alias="fundCode", min_length=6, max_length=6, pattern=r"^\d{6}$")],
    page_size: Annotated[int, Query(alias="pageSize", ge=1, le=100)] = 20,
    cursor: UUID | None = None,
) -> InternalSignalPage:
    """返回已落库的评分结果，不执行模型，也不会补造缺失的预测方向。"""
    try:
        payload = list_completed_signals(fund_code, page_size, cursor)
    except ValueError as error:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(error)) from error
    logger.info(
        "signals.list_internal_signals >>> returned score result page, trace_id=%s, fund_code=%s, count=%s",
        get_trace_id(),
        fund_code,
        len(payload.items),
    )
    return payload
