"""供 Java 核心服务认证访问的 M0 基金读模型接口。"""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Path, Query, status

from app.api.dependencies import require_service_token
from app.core.logging import get_logger
from app.core.middleware import get_trace_id
from app.schemas.fund import InternalFundDetail, InternalFundPage
from app.services.fund_catalog_read import get_fund, list_funds

router = APIRouter()
logger = get_logger(__name__)


@router.get("", response_model=InternalFundPage, dependencies=[Depends(require_service_token)])
async def list_internal_funds(
    keyword: Annotated[str | None, Query(max_length=50)] = None,
    page_size: Annotated[int, Query(alias="pageSize", ge=1, le=100)] = 20,
    cursor: Annotated[str | None, Query(pattern=r"^\d+$")] = None,
) -> InternalFundPage:
    """按兼容游标返回已落库的真实基金目录样本。

    该接口只向通过服务令牌校验的 Java 核心服务开放；目录为一次性手工核验样本，
    ``as_of_date`` 为空时表示尚未获得合规净值同步，不能视为实时行情。
    """
    logger.info(
        "funds.list_internal_funds >>> persisted fund catalog page requested, trace_id=%s, page_size=%s",
        get_trace_id(),
        page_size,
    )
    return list_funds(keyword, page_size, cursor)


@router.get("/{fund_code}", response_model=InternalFundDetail, dependencies=[Depends(require_service_token)])
async def get_internal_fund(
    fund_code: Annotated[str, Path(min_length=6, max_length=6, pattern=r"^\d{6}$")],
) -> InternalFundDetail:
    """返回单只已落库基金详情；基金不存在时给出稳定的内部 404 错误码。"""
    logger.info(
        "funds.get_internal_fund >>> persisted fund detail requested, trace_id=%s, fund_code=%s",
        get_trace_id(),
        fund_code,
    )
    fund = get_fund(fund_code)
    if fund is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "FUND_NOT_FOUND", "message": "Fund is not available."},
        )
    return fund
