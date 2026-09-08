"""历史样本保存与只读回查；只供携带服务Token的内部调用，不开放浏览器入口。"""

from time import perf_counter
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Response
from sqlalchemy.exc import SQLAlchemyError

from app.api.dependencies import require_service_token
from app.core.logging import get_logger
from app.core.middleware import get_trace_id
from app.repositories.historical_nav import HistoricalNavPreviewReadError
from app.schemas.historical_nav_storage import HistoricalNavBatchSaveRequest, HistoricalNavStoredBatch
from app.services.historical_nav_preview import HistoricalNavBatchTimeoutError
from app.services.historical_nav_storage import (
    HistoricalNavStorageError,
    get_historical_nav_batch,
    save_historical_nav_batch,
)

router = APIRouter(dependencies=[Depends(require_service_token)])
logger = get_logger(__name__)


def _http_error(error: Exception, *, operation: str, identifier: object) -> HTTPException:
    """只向客户端返回稳定错误码；数据库异常保留脱敏堆栈，日志不含完整特征/标签或Token。"""
    if isinstance(error, HistoricalNavStorageError):
        status_code, code, message = error.status_code, error.code, str(error)
    elif isinstance(error, HistoricalNavPreviewReadError):
        status_code = 404 if error.code == "FUND_NOT_FOUND" else 409
        code, message = error.code, str(error)
    elif isinstance(error, (ValueError, ArithmeticError)):
        status_code, code, message = 422, "INVALID_NAV_SAMPLES", "样本未通过保存校验，请检查数据质量。"
    else:
        status_code, code, message = (
            503,
            "BATCH_STORAGE_UNAVAILABLE",
            "样本存储暂时不可用或超时，请沿用原requestKey重试。",
        )
    logger.log(
        40 if status_code >= 500 else 30,
        "historical_nav_storage.%s >>> request rejected, trace_id=%s, identifier=%s, code=%s",
        operation,
        get_trace_id(),
        identifier,
        code,
        exc_info=not isinstance(error, (HistoricalNavPreviewReadError, HistoricalNavStorageError))
        or status_code >= 500,
    )
    return HTTPException(status_code=status_code, detail={"code": code, "message": message})


@router.post("/historical-nav-samples/batches", response_model=HistoricalNavStoredBatch, status_code=201)
def save_batch(request: HistoricalNavBatchSaveRequest, response: Response) -> HistoricalNavStoredBatch:
    """首次完整保存201；相同凭证/身份重试200，同凭证异参数409。不接受外部样本JSON。"""
    started = perf_counter()
    try:
        result, created = save_historical_nav_batch(request)
    except (
        HistoricalNavStorageError,
        HistoricalNavPreviewReadError,
        SQLAlchemyError,
        HistoricalNavBatchTimeoutError,
        ValueError,
        ArithmeticError,
    ) as error:
        raise _http_error(error, operation="save_batch", identifier=request.request_key) from error
    response.status_code = 201 if created else 200
    logger.info(
        "historical_nav_storage.save_batch >>> completed, trace_id=%s, batch_id=%s, request_key=%s, "
        "fund_code=%s, created=%s, samples=%s, elapsed_ms=%.2f",
        get_trace_id(),
        result.batch_id,
        result.request_key,
        result.fund_code,
        created,
        result.sample_count,
        (perf_counter() - started) * 1000,
    )
    return result


@router.get("/historical-nav-samples/batches/{batch_id}", response_model=HistoricalNavStoredBatch)
def read_batch(batch_id: UUID) -> HistoricalNavStoredBatch:
    """按批次编号返回已存内容，不重新计算；批次最多31份样本，读取有界。"""
    started = perf_counter()
    try:
        result = get_historical_nav_batch(batch_id)
    except (HistoricalNavStorageError, SQLAlchemyError) as error:
        raise _http_error(error, operation="read_batch", identifier=batch_id) from error
    logger.info(
        "historical_nav_storage.read_batch >>> completed, trace_id=%s, batch_id=%s, fund_code=%s, "
        "samples=%s, elapsed_ms=%.2f",
        get_trace_id(),
        batch_id,
        result.fund_code,
        result.sample_count,
        (perf_counter() - started) * 1000,
    )
    return result
