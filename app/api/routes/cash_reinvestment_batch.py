"""新版批量GET：使用现有Token、范围校验和脱敏错误，不暴露部分成功结果。"""

from time import perf_counter
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.exc import SQLAlchemyError

from app.api.dependencies import require_service_token
from app.core.logging import get_logger
from app.core.middleware import get_trace_id
from app.repositories.historical_nav import HistoricalNavPreviewReadError
from app.schemas.cash_reinvestment_batch import CashBatchRequest, CashBatchResponse
from app.services.cash_reinvestment_batch import preview_cash_reinvestment_batch
from app.services.trading_calendar import CalendarCoverageError

router = APIRouter(dependencies=[Depends(require_service_token)])
logger = get_logger(__name__)


@router.get("/historical-nav-samples/cash-reinvestment-dry-run", response_model=CashBatchResponse)
def preview_batch(request: Annotated[CashBatchRequest, Query()]) -> CashBatchResponse:
    started = perf_counter()
    try:
        result = preview_cash_reinvestment_batch(request)
    except (HistoricalNavPreviewReadError, CalendarCoverageError) as error:
        code = error.code if isinstance(error, HistoricalNavPreviewReadError) else "CALENDAR_COVERAGE_INSUFFICIENT"
        logger.warning(
            "cash_reinvestment_batch.preview_batch >>> rejected, trace_id=%s, fund=%s, code=%s",
            get_trace_id(),
            request.fund_code,
            code,
        )
        raise HTTPException(
            status_code=404 if code == "FUND_NOT_FOUND" else 409, detail={"code": code, "message": str(error)}
        ) from error
    except (SQLAlchemyError, OSError, ValueError, ArithmeticError, TypeError) as error:
        logger.exception(
            "cash_reinvestment_batch.preview_batch >>> failed, trace_id=%s, fund=%s", get_trace_id(), request.fund_code
        )
        raise HTTPException(
            status_code=503,
            detail={
                "code": "CASH_BATCH_UNAVAILABLE",
                "message": "现金再投批量预览暂不可用，未修改数据，未返回部分样本。",
            },
        ) from error
    logger.info(
        "cash_reinvestment_batch.preview_batch >>> completed, trace_id=%s, fund=%s, start=%s, end=%s, "
        "pages=%s, samples=%s, ready=%s, input_unavailable=%s, label_unavailable=%s, elapsed_ms=%.2f",
        get_trace_id(),
        request.fund_code,
        request.start_date,
        request.end_date,
        result.page_count,
        result.sample_count,
        result.ready_count,
        result.input_unavailable_count,
        result.label_unavailable_count,
        (perf_counter() - started) * 1000,
    )
    return result
