"""供 Java 核心服务认证访问的 M0 基金读模型接口。"""

import re
from datetime import date
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Path, Query, status

from app.api.dependencies import require_service_token
from app.core.logging import get_logger
from app.core.middleware import get_trace_id
from app.schemas.fund import (
    InternalFundDetail,
    InternalFundNavHistory,
    InternalFundPage,
    InternalFundSameTypeComparison,
    InternalFundShareHistory,
    InternalFundSummary,
    InternalFundWatchlistDetail,
    InternalSyncJobLastSuccess,
    InternalSyncJobStatus,
)
from app.services.fund_catalog_read import (
    get_fund,
    get_fund_nav_history,
    get_fund_same_type_comparison,
    get_fund_share_history,
    get_fund_watchlist_detail,
    get_funds_by_codes,
    list_funds,
)
from app.services.sync_jobs import (
    MARKET_DETAIL_JOB_TYPE,
    MARKET_NAV_INCREMENTAL_JOB_TYPE,
    STOCK_FEATURE_SNAPSHOT_JOB_TYPE,
    SyncJobInProgressError,
    SyncJobSnapshot,
    get_sync_job_manager,
)

router = APIRouter()
logger = get_logger(__name__)
_FUND_TYPES = "BOND|STOCK|MIXED|INDEX|MONEY|QDII|FOF|OTHER"
_FUND_CODE_PATTERN = re.compile(r"^\d{6}$")
_MAX_BATCH_FUND_CODES = 50


@router.get("", response_model=InternalFundPage, dependencies=[Depends(require_service_token)])
async def list_internal_funds(
    keyword: Annotated[str | None, Query(max_length=50)] = None,
    fund_type: Annotated[str | None, Query(alias="fundType", pattern=rf"^({_FUND_TYPES})$")] = None,
    page_size: Annotated[int, Query(alias="pageSize", ge=1, le=100)] = 10,
    cursor: Annotated[str | None, Query(pattern=r"^\d+$")] = None,
    page: Annotated[int | None, Query(ge=1, le=10_000)] = None,
) -> InternalFundPage:
    """按兼容游标或页码返回已落库的真实基金目录样本。

    该接口只向通过服务令牌校验的 Java 核心服务开放；目录为一次性手工核验样本，
    ``as_of_date`` 为空时表示尚未获得净值同步，不能视为实时行情。
    """
    if page is not None and cursor is not None:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail={
                "code": "PAGINATION_MODE_CONFLICT",
                "message": "page 与 cursor 不能同时使用。",
            },
        )
    logger.info(
        "funds.list_internal_funds >>> persisted fund catalog page requested, trace_id=%s, page_size=%s, page=%s",
        get_trace_id(),
        page_size,
        page,
    )
    return list_funds(keyword, fund_type, page_size, cursor, page)


@router.get("/batch", response_model=tuple[InternalFundSummary, ...], dependencies=[Depends(require_service_token)])
async def list_internal_fund_summaries_by_codes(
    fund_codes: Annotated[list[str], Query(alias="fundCode")],
) -> tuple[InternalFundSummary, ...]:
    """批量返回指定基金的摘要，供 Java 丰富当前用户关注页而不形成逐行请求。"""
    if (
        not fund_codes
        or len(fund_codes) > _MAX_BATCH_FUND_CODES
        or len(set(fund_codes)) != len(fund_codes)
        or any(_FUND_CODE_PATTERN.fullmatch(fund_code) is None for fund_code in fund_codes)
    ):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail={"code": "INVALID_FUND_CODE_BATCH", "message": "基金代码批量参数无效。"},
        )
    logger.info(
        "funds.list_internal_fund_summaries_by_codes >>> persisted fund summaries requested, "
        "trace_id=%s, fund_count=%s",
        get_trace_id(),
        len(fund_codes),
    )
    return get_funds_by_codes(tuple(fund_codes))


@router.post(
    "/sync-jobs/market-nav-incremental",
    response_model=InternalSyncJobStatus,
    status_code=status.HTTP_202_ACCEPTED,
    dependencies=[Depends(require_service_token)],
)
def start_internal_market_nav_incremental_job() -> InternalSyncJobStatus:
    """创建基金市场手动同步任务；执行由本机后台线程承担，不依赖 Celery Worker。"""
    try:
        logger.info(
            "funds.start_internal_market_nav_incremental_job >>> manual market NAV sync requested, trace_id=%s",
            get_trace_id(),
        )
        snapshot = get_sync_job_manager().start_market_nav_incremental()
    except SyncJobInProgressError as error:
        logger.warning(
            "funds.start_internal_market_nav_incremental_job >>> duplicate sync rejected, trace_id=%s",
            get_trace_id(),
        )
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={"code": "MARKET_SYNC_IN_PROGRESS", "message": "已有基金市场同步正在执行，请稍后重试。"},
        ) from error
    except ValueError as error:
        logger.error(
            "funds.start_internal_market_nav_incremental_job >>> invalid sync configuration, trace_id=%s",
            get_trace_id(),
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"code": "MARKET_SYNC_UNAVAILABLE", "message": "基金市场同步服务尚未完成配置。"},
        ) from error
    return _to_internal_sync_job_status(snapshot)


@router.post(
    "/sync-jobs/market-details",
    response_model=InternalSyncJobStatus,
    status_code=status.HTTP_202_ACCEPTED,
    dependencies=[Depends(require_service_token)],
)
def start_internal_market_detail_job() -> InternalSyncJobStatus:
    """创建基金完整资料同步任务；实际执行在本机后台线程中完成。"""
    try:
        logger.info(
            "funds.start_internal_market_detail_job >>> manual market detail sync requested, trace_id=%s",
            get_trace_id(),
        )
        snapshot = get_sync_job_manager().start_market_details()
    except SyncJobInProgressError as error:
        logger.warning(
            "funds.start_internal_market_detail_job >>> duplicate sync rejected, trace_id=%s", get_trace_id()
        )
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={"code": "MARKET_SYNC_IN_PROGRESS", "message": "已有基金市场同步正在执行，请稍后重试。"},
        ) from error
    except ValueError as error:
        logger.error(
            "funds.start_internal_market_detail_job >>> invalid sync configuration, trace_id=%s", get_trace_id()
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"code": "MARKET_DETAIL_SYNC_UNAVAILABLE", "message": "基金完整资料同步服务尚未完成配置。"},
        ) from error
    return _to_internal_sync_job_status(snapshot)


@router.post(
    "/sync-jobs/stock-feature-snapshots",
    response_model=InternalSyncJobStatus,
    status_code=status.HTTP_202_ACCEPTED,
    dependencies=[Depends(require_service_token)],
)
def start_internal_stock_feature_snapshot_job() -> InternalSyncJobStatus:
    """创建手动特征快照任务；只读取已落库净值，不访问外部来源。"""
    try:
        logger.info(
            "funds.start_internal_stock_feature_snapshot_job >>> manual feature snapshot sync requested, trace_id=%s",
            get_trace_id(),
        )
        snapshot = get_sync_job_manager().start_stock_feature_snapshots()
    except SyncJobInProgressError as error:
        logger.warning(
            "funds.start_internal_stock_feature_snapshot_job >>> duplicate sync rejected, trace_id=%s",
            get_trace_id(),
        )
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={"code": "FEATURE_SYNC_IN_PROGRESS", "message": "已有同步任务正在执行，请稍后重试。"},
        ) from error
    except ValueError as error:
        logger.error(
            "funds.start_internal_stock_feature_snapshot_job >>> invalid feature configuration, trace_id=%s",
            get_trace_id(),
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"code": "FEATURE_SYNC_UNAVAILABLE", "message": "特征快照同步服务尚未完成配置。"},
        ) from error
    return _to_internal_sync_job_status(snapshot)


@router.get(
    "/sync-jobs/market-nav-incremental/latest",
    response_model=InternalSyncJobStatus | None,
    dependencies=[Depends(require_service_token)],
)
def get_latest_internal_market_nav_incremental_job() -> InternalSyncJobStatus | None:
    """读取当前 FastAPI 进程最近一次基金市场同步任务，页面刷新后可继续观察进度。"""
    snapshot = get_sync_job_manager().get_latest_job()
    return _to_internal_sync_job_status(snapshot) if snapshot else None


@router.get(
    "/sync-jobs/market-details/latest",
    response_model=InternalSyncJobStatus | None,
    dependencies=[Depends(require_service_token)],
)
def get_latest_internal_market_detail_job() -> InternalSyncJobStatus | None:
    """读取当前 Python 进程最近一次完整资料同步任务。"""
    snapshot = get_sync_job_manager().get_latest_job(MARKET_DETAIL_JOB_TYPE)
    return _to_internal_sync_job_status(snapshot) if snapshot else None


@router.get(
    "/sync-jobs/stock-feature-snapshots/latest",
    response_model=InternalSyncJobStatus | None,
    dependencies=[Depends(require_service_token)],
)
def get_latest_internal_stock_feature_snapshot_job() -> InternalSyncJobStatus | None:
    """读取当前 Python 进程最近一次特征快照任务，供同步中心轮询。"""
    snapshot = get_sync_job_manager().get_latest_job(STOCK_FEATURE_SNAPSHOT_JOB_TYPE)
    return _to_internal_sync_job_status(snapshot) if snapshot else None


@router.get(
    "/sync-jobs/last-success",
    response_model=tuple[InternalSyncJobLastSuccess, ...],
    dependencies=[Depends(require_service_token)],
)
def get_internal_sync_job_last_success_times() -> tuple[InternalSyncJobLastSuccess, ...]:
    """读取两类任务最近一次完整成功时间；不触发外部数据请求。"""
    manager = get_sync_job_manager()
    return (
        InternalSyncJobLastSuccess(
            job_type=MARKET_NAV_INCREMENTAL_JOB_TYPE,
            last_successful_at=manager.get_last_successful_time(MARKET_NAV_INCREMENTAL_JOB_TYPE),
        ),
        InternalSyncJobLastSuccess(
            job_type=MARKET_DETAIL_JOB_TYPE,
            last_successful_at=manager.get_last_successful_time(MARKET_DETAIL_JOB_TYPE),
        ),
    )


@router.get(
    "/sync-jobs/{job_id}",
    response_model=InternalSyncJobStatus,
    dependencies=[Depends(require_service_token)],
)
def get_internal_sync_job(job_id: UUID) -> InternalSyncJobStatus:
    """按任务标识返回安全进度摘要；本接口不触发同步。"""
    snapshot = get_sync_job_manager().get_job(job_id)
    if snapshot is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "SYNC_JOB_NOT_FOUND", "message": "未找到同步任务。"},
        )
    return _to_internal_sync_job_status(snapshot)


def _to_internal_sync_job_status(snapshot: SyncJobSnapshot) -> InternalSyncJobStatus:
    """将进程内任务快照转换为明确的 Python 内部 HTTP 契约。"""
    return InternalSyncJobStatus(
        job_id=snapshot.job_id,
        job_type=snapshot.job_type,
        status=snapshot.status,
        requested_nav_date=snapshot.requested_nav_date,
        fund_codes=snapshot.fund_codes,
        progress_current=snapshot.progress_current,
        progress_total=snapshot.progress_total,
        current_fund_code=snapshot.current_fund_code,
        progress_message=snapshot.progress_message,
        sync_run_id=snapshot.sync_run_id,
        fetched_count=snapshot.fetched_count,
        created_count=snapshot.created_count,
        updated_count=snapshot.updated_count,
        skipped_count=snapshot.skipped_count,
        error_code=snapshot.error_code,
        error_message=snapshot.error_message,
        started_at=snapshot.started_at,
        finished_at=snapshot.finished_at,
    )


@router.get(
    "/{fund_code}/watchlist-detail",
    response_model=InternalFundWatchlistDetail,
    dependencies=[Depends(require_service_token)],
)
async def get_internal_watchlist_fund_detail(
    fund_code: Annotated[str, Path(min_length=6, max_length=6, pattern=r"^\d{6}$")],
) -> InternalFundWatchlistDetail:
    """返回本地完整详情；用户关注关系必须由 Java 在调用前完成校验。"""
    logger.info(
        "funds.get_internal_watchlist_fund_detail >>> persisted full detail requested, "
        "trace_id=%s, fund_code=%s",
        get_trace_id(),
        fund_code,
    )
    detail = get_fund_watchlist_detail(fund_code)
    if detail is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "FUND_NOT_FOUND", "message": "Fund is not available."},
        )
    return detail


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


@router.get(
    "/{fund_code}/share-history",
    response_model=InternalFundShareHistory,
    dependencies=[Depends(require_service_token)],
)
async def get_internal_fund_share_history(
    fund_code: Annotated[str, Path(min_length=6, max_length=6, pattern=r"^\d{6}$")],
    start_date: Annotated[date, Query(alias="startDate")],
    end_date: Annotated[date, Query(alias="endDate")],
) -> InternalFundShareHistory:
    """返回已落库的份额规模历史；调用方必须先完成当前用户关注关系校验。"""
    if start_date > end_date:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="startDate must not be after endDate",
        )
    if (end_date - start_date).days > 5_000:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="requested share history window is too large",
        )
    logger.info(
        "funds.get_internal_fund_share_history >>> persisted share history requested, "
        "trace_id=%s, fund_code=%s, start=%s, end=%s",
        get_trace_id(),
        fund_code,
        start_date,
        end_date,
    )
    history = get_fund_share_history(fund_code, start_date, end_date)
    if history is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "FUND_NOT_FOUND", "message": "Fund is not available."},
        )
    return history


@router.get(
    "/{fund_code}/same-type-comparison",
    response_model=InternalFundSameTypeComparison,
    dependencies=[Depends(require_service_token)],
)
async def get_internal_fund_same_type_comparison(
    fund_code: Annotated[str, Path(min_length=6, max_length=6, pattern=r"^\d{6}$")],
) -> InternalFundSameTypeComparison:
    """返回受控当前基金市场样本内的同类比较，不触发同步或全市场检索。"""
    logger.info(
        "funds.get_internal_fund_same_type_comparison >>> current-market comparison requested, "
        "trace_id=%s, fund_code=%s",
        get_trace_id(),
        fund_code,
    )
    comparison = get_fund_same_type_comparison(fund_code)
    if comparison is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "FUND_NOT_FOUND", "message": "Fund is not available."},
        )
    return comparison


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
