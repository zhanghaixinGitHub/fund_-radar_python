"""Authenticated M0 fund read-model endpoints for the Java core service."""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Path, Query, status

from app.api.dependencies import require_service_token
from app.core.logging import get_logger
from app.core.middleware import get_trace_id
from app.schemas.fund import InternalFundDetail, InternalFundPage
from app.services.mock_fund_catalog import get_mock_fund, list_mock_funds

router = APIRouter()
logger = get_logger(__name__)


@router.get("", response_model=InternalFundPage, dependencies=[Depends(require_service_token)])
async def list_internal_funds(
    keyword: Annotated[str | None, Query(max_length=50)] = None,
    page_size: Annotated[int, Query(alias="pageSize", ge=1, le=100)] = 20,
    cursor: Annotated[str | None, Query(pattern=r"^\d+$")] = None,
) -> InternalFundPage:
    """Return the M0 mock list using a versioned, cursor-compatible contract."""
    logger.info(
        "funds.list_internal_funds >>> M0 mock fund page requested, trace_id=%s, page_size=%s",
        get_trace_id(),
        page_size,
    )
    return list_mock_funds(keyword, page_size, cursor)


@router.get("/{fund_code}", response_model=InternalFundDetail, dependencies=[Depends(require_service_token)])
async def get_internal_fund(
    fund_code: Annotated[str, Path(min_length=6, max_length=6, pattern=r"^\d{6}$")],
) -> InternalFundDetail:
    """Return a single M0 mock fund detail or a stable internal error code."""
    logger.info(
        "funds.get_internal_fund >>> M0 mock fund detail requested, trace_id=%s, fund_code=%s",
        get_trace_id(),
        fund_code,
    )
    fund = get_mock_fund(fund_code)
    if fund is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "FUND_NOT_FOUND", "message": "Fund is not available."},
        )
    return fund
