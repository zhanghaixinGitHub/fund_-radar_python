"""供 Java 核心服务认证访问的 M0 基金读模型接口。"""

from datetime import date
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Path, Query, status

from app.api.dependencies import require_service_token
from app.core.config import get_settings
from app.core.logging import get_logger
from app.core.middleware import get_trace_id
from app.integrations.tushare import TushareIntegrationError
from app.schemas.fund import InternalFocusedNavSyncResult, InternalFundDetail, InternalFundNavHistory, InternalFundPage
from app.services.fund_catalog_read import get_fund, get_fund_nav_history, list_funds
from app.services.tushare_fund_sync import (
    FocusedNavIncrementalInProgressError,
    FocusedNavIncrementalPreconditionError,
    TushareFundSyncService,
)

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


@router.post(
    "/sync/focused-nav-incremental",
    response_model=InternalFocusedNavSyncResult,
    dependencies=[Depends(require_service_token)],
)
def sync_internal_focused_nav_incremental() -> InternalFocusedNavSyncResult:
    """手动补齐六只重点基金净值；同步执行，不依赖本机 Celery Beat 或 Worker。"""
    service: TushareFundSyncService | None = None
    try:
        settings = get_settings()
        ts_codes = settings.focused_fund_ts_codes
        logger.info(
            "funds.sync_internal_focused_nav_incremental >>> manual focused NAV sync requested, "
            "trace_id=%s, fund_count=%s",
            get_trace_id(),
            len(ts_codes),
        )
        service = TushareFundSyncService()
        outcome = service.sync_focused_nav_incremental(ts_codes)
    except FocusedNavIncrementalInProgressError as error:
        logger.warning(
            "funds.sync_internal_focused_nav_incremental >>> duplicate sync rejected, trace_id=%s",
            get_trace_id(),
        )
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={"code": "FOCUSED_SYNC_IN_PROGRESS", "message": "已有重点基金同步正在执行，请稍后重试。"},
        ) from error
    except FocusedNavIncrementalPreconditionError as error:
        logger.warning(
            "funds.sync_internal_focused_nav_incremental >>> missing baseline rejected, trace_id=%s",
            get_trace_id(),
        )
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail={"code": "FOCUSED_SYNC_BASELINE_MISSING", "message": "请先完成重点基金历史净值回填。"},
        ) from error
    except TushareIntegrationError as error:
        logger.warning(
            "funds.sync_internal_focused_nav_incremental >>> Tushare sync failed, trace_id=%s, api=%s",
            get_trace_id(),
            error.api_name,
        )
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail={"code": "FOCUSED_SYNC_FAILED", "message": "净值同步未完成，请稍后重试。"},
        ) from error
    except ValueError as error:
        logger.error(
            "funds.sync_internal_focused_nav_incremental >>> manual sync configuration is invalid, trace_id=%s",
            get_trace_id(),
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"code": "FOCUSED_SYNC_UNAVAILABLE", "message": "净值同步服务尚未完成配置。"},
        ) from error
    finally:
        if service is not None:
            service.close()
    return InternalFocusedNavSyncResult(
        sync_run_id=outcome.sync_run_id,
        requested_nav_date=outcome.requested_nav_date,
        fund_codes=ts_codes,
        fetched_count=outcome.fetched_count,
        created_count=outcome.created_count,
        updated_count=outcome.updated_count,
        skipped_count=outcome.skipped_count,
    )


@router.get(
    "/{fund_code}/nav-history",
    response_model=InternalFundNavHistory,
    dependencies=[Depends(require_service_token)],
)
async def get_internal_fund_nav_history(
    fund_code: Annotated[str, Path(min_length=6, max_length=6, pattern=r"^\d{6}$")],
    start_date: Annotated[date, Query(alias="startDate")],
    end_date: Annotated[date, Query(alias="endDate")],
) -> InternalFundNavHistory:
    """返回已落库的历史净值；读取过程不调用 Tushare，最长窗口限制为约 13 年。"""
    if start_date > end_date:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="startDate must not be after endDate",
        )
    if (end_date - start_date).days > 5_000:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="requested NAV history window is too large",
        )
    logger.info(
        "funds.get_internal_fund_nav_history >>> persisted NAV history requested, "
        "trace_id=%s, fund_code=%s, start=%s, end=%s",
        get_trace_id(),
        fund_code,
        start_date,
        end_date,
    )
    history = get_fund_nav_history(fund_code, start_date, end_date)
    if history is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "FUND_NOT_FOUND", "message": "Fund is not available."},
        )
    return history


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
