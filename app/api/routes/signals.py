"""供 Java 核心服务认证读取的 M3 评分结果接口。"""

from datetime import datetime
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status

from app.api.dependencies import require_service_token
from app.core.logging import get_logger
from app.core.middleware import get_trace_id
from app.schemas.signal import InternalSignalChangePage, InternalSignalPage
from app.services.signal_read import list_active_scored_changes, list_completed_signals

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


@router.get("/changes", response_model=InternalSignalChangePage, dependencies=[Depends(require_service_token)])
def list_internal_signal_changes(
    page_size: Annotated[int, Query(alias="pageSize", ge=1, le=100)] = 100,
    after_scored_at: Annotated[datetime | None, Query(alias="afterScoredAt")] = None,
    after_forecast_id: Annotated[UUID | None, Query(alias="afterForecastId")] = None,
) -> InternalSignalChangePage:
    """返回 ACTIVE 发布的已评分增量，读取不会启动评分、回测或模型发布。"""
    if (after_scored_at is None) != (after_forecast_id is None):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="afterScoredAt and afterForecastId must be supplied together.",
        )
    try:
        payload = list_active_scored_changes(page_size, after_scored_at, after_forecast_id)
    except ValueError as error:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(error)) from error
    logger.info(
        "signals.list_internal_signal_changes >>> returned signal changes, trace_id=%s, count=%s, has_more=%s",
        get_trace_id(),
        len(payload.items),
        payload.has_more,
    )
    return payload
