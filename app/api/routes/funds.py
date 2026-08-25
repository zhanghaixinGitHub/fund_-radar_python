"""供 Java 核心服务认证访问的 M0 基金读模型接口。"""

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
    """按兼容游标的内部契约返回 M0 Mock 基金列表。

    该接口只向通过服务令牌校验的 Java 核心服务开放；M0 数据仅用于链路验证，不代表真实行情。
    """
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
    """返回单只 M0 Mock 基金详情；基金不存在时给出稳定的内部 404 错误码。"""
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
