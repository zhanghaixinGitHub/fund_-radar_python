"""现金再投研究样本GET，保留现有身份认证与错误脱敏方式。"""

from time import perf_counter
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.exc import SQLAlchemyError

from app.api.dependencies import require_service_token
from app.core.logging import get_logger
from app.core.middleware import get_trace_id
from app.repositories.historical_nav import HistoricalNavPreviewReadError
from app.schemas.cash_reinvestment_samples import CashSampleRequest, CashSampleResponse
from app.services.cash_reinvestment_samples import preview_cash_reinvestment_sample
from app.services.trading_calendar import CalendarCoverageError

router = APIRouter(dependencies=[Depends(require_service_token)])
logger = get_logger(__name__)


@router.get("/historical-nav-samples/cash-reinvestment-preview", response_model=CashSampleResponse)
def preview_sample(request: Annotated[CashSampleRequest, Query()]) -> CashSampleResponse:
    started = perf_counter()
    try:
        result = preview_cash_reinvestment_sample(request)
    except (HistoricalNavPreviewReadError, CalendarCoverageError) as error:
        code = error.code if isinstance(error, HistoricalNavPreviewReadError) else "CALENDAR_COVERAGE_INSUFFICIENT"
        logger.warning(
            "cash_reinvestment_samples.preview_sample >>> rejected, trace_id=%s, fund=%s, code=%s",
            get_trace_id(),
            request.fund_code,
            code,
        )
        raise HTTPException(
            status_code=404 if code == "FUND_NOT_FOUND" else 409, detail={"code": code, "message": str(error)}
        ) from error
    except (SQLAlchemyError, OSError, ValueError, ArithmeticError, TypeError) as error:
        logger.exception(
            "cash_reinvestment_samples.preview_sample >>> failed, trace_id=%s, fund=%s",
            get_trace_id(),
            request.fund_code,
        )
        raise HTTPException(
            status_code=503,
            detail={
                "code": "CASH_SAMPLE_UNAVAILABLE",
                "message": "现金再投样本预览暂不可用，未修改数据，请联系维护人员。",
            },
        ) from error
    logger.info(
        "cash_reinvestment_samples.preview_sample >>> completed, trace_id=%s, fund=%s, cutoff=%s, "
        "status=%s, elapsed_ms=%.2f",
        get_trace_id(),
        request.fund_code,
        request.cutoff_date,
        result.status,
        (perf_counter() - started) * 1000,
    )
    return result
