"""维护人员GET看严格交易日窗口；同原接口鉴权，不生成特征或预测。"""

from time import perf_counter
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.exc import SQLAlchemyError

from app.api.dependencies import require_service_token
from app.core.logging import get_logger
from app.core.middleware import get_trace_id
from app.repositories.historical_nav import HistoricalNavPreviewReadError
from app.schemas.trading_nav_window import TradingNavWindowRequest, TradingNavWindowResponse
from app.services.trading_calendar import CalendarCoverageError
from app.services.trading_nav_window import preview_trading_nav_window

router = APIRouter(dependencies=[Depends(require_service_token)])
logger = get_logger(__name__)


@router.get("/historical-nav-samples/trading-window-preview", response_model=TradingNavWindowResponse)
def preview_window(request: Annotated[TradingNavWindowRequest, Query()]) -> TradingNavWindowResponse:
    started = perf_counter()
    try:
        result = preview_trading_nav_window(request)
    except (HistoricalNavPreviewReadError, CalendarCoverageError) as error:
        code = error.code if isinstance(error, HistoricalNavPreviewReadError) else "CALENDAR_COVERAGE_INSUFFICIENT"
        logger.warning(
            "trading_nav_window.preview_window >>> rejected, trace_id=%s, fund=%s, cutoff=%s, code=%s",
            get_trace_id(),
            request.fund_code,
            request.cutoff_date,
            code,
        )
        raise HTTPException(
            status_code=404 if code == "FUND_NOT_FOUND" else 409, detail={"code": code, "message": str(error)}
        ) from error
    except (SQLAlchemyError, OSError, ValueError, ArithmeticError, TypeError) as error:
        logger.exception(
            "trading_nav_window.preview_window >>> failed, trace_id=%s, fund=%s, cutoff=%s",
            get_trace_id(),
            request.fund_code,
            request.cutoff_date,
        )
        raise HTTPException(
            status_code=503,
            detail={
                "code": "TRADING_WINDOW_UNAVAILABLE",
                "message": "日历或净值日期检查暂不可用，未修改数据，请联系维护人员。",
            },
        ) from error
    logger.info(
        "trading_nav_window.preview_window >>> completed, trace_id=%s, fund=%s, cutoff=%s, status=%s, elapsed_ms=%.2f",
        get_trace_id(),
        request.fund_code,
        request.cutoff_date,
        result.status,
        (perf_counter() - started) * 1000,
    )
    return result
