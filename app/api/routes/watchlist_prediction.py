"""仅供已授权Java服务读取；本人关注关系由Java会话层强制校验。"""

from time import perf_counter
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, Path, Response

from app.api.dependencies import require_service_token
from app.api.routes.cash_reinvestment_storage import ERRORS, cash_http_error
from app.core.logging import get_logger
from app.core.middleware import get_trace_id
from app.schemas.cash_forecast import CashForecastRequest, CashForecastView
from app.schemas.cash_prediction_attempt import CashPredictionAttempt, CashPredictionAttemptRequest
from app.schemas.cash_prediction_check import CashPredictionCheck, CashPredictionCheckRequest
from app.schemas.watchlist_prediction import WatchlistPrediction
from app.services.cash_forecast import generate_cash_forecast, get_cash_forecast
from app.services.cash_prediction_attempt import get_cash_prediction_attempt, save_cash_prediction_attempt
from app.services.cash_prediction_check import check_cash_prediction
from app.services.watchlist_prediction import get_watchlist_prediction

router = APIRouter(dependencies=[Depends(require_service_token)])
logger = get_logger(__name__)


@router.post("/generation-check", response_model=CashPredictionCheck)
def check_prediction_generation(request: CashPredictionCheckRequest, response: Response):
    """维护人员检查指定研究是否允许生成；不是强制发布、训练或产品预测入口。"""
    response.headers["Cache-Control"] = "no-store"
    try:
        result = check_cash_prediction(request)
    except ERRORS as error:
        raise cash_http_error(error, "check_prediction_generation") from error
    logger.info(
        "watchlist_prediction.check_prediction_generation >>> complete, trace_id=%s, fund=%s, run=%s, status=%s",
        get_trace_id(),
        request.fund_code,
        request.research_run_id,
        result.status,
    )
    return result


@router.post("/generation-attempts", response_model=CashPredictionAttempt, status_code=201)
def create_generation_attempt(request: CashPredictionAttemptRequest, response: Response):
    """保存一次真实的拒绝回执；不是强制执行模型或修改正式发布状态的入口。"""
    started = perf_counter()
    response.headers["Cache-Control"] = "no-store"
    try:
        result, created = save_cash_prediction_attempt(request)
    except ERRORS as error:
        raise cash_http_error(error, "create_generation_attempt") from error
    response.status_code = 201 if created else 200
    logger.info(
        "watchlist_prediction.create_generation_attempt >>> complete, trace_id=%s, fund=%s, "
        "attempt_id=%s, cutoff=%s, status=%s, created=%s, elapsed_ms=%.2f",
        get_trace_id(),
        request.fund_code,
        result.attempt_id,
        request.cutoff_date,
        result.check.status,
        created,
        (perf_counter() - started) * 1000,
    )
    return result


@router.get("/generation-attempts/{attempt_id}", response_model=CashPredictionAttempt)
def read_generation_attempt(attempt_id: UUID, response: Response):
    """只读历史回执；读回不重新检查发布，也不触发计算。"""
    response.headers["Cache-Control"] = "no-store"
    try:
        return get_cash_prediction_attempt(attempt_id)
    except ERRORS as error:
        raise cash_http_error(error, "read_generation_attempt") from error


@router.post("/cash-forecasts", response_model=CashForecastView)
def generate_forecast(request: CashForecastRequest, response: Response):
    """先检查当前发布资格；未通过200返回未发布，只有真实新建结果才返回201。"""
    started = perf_counter()
    response.headers["Cache-Control"] = "no-store"
    try:
        result = generate_cash_forecast(request)
    except ERRORS as error:
        raise cash_http_error(error, "generate_forecast") from error
    response.status_code = 201 if result.created else 200
    logger.info(
        "watchlist_prediction.generate_forecast >>> complete, trace_id=%s, fund=%s, cutoff=%s, status=%s, "
        "forecast_id=%s, created=%s, elapsed_ms=%.2f",
        get_trace_id(),
        request.fund_code,
        request.cutoff_date,
        result.status,
        result.forecast_id,
        result.created,
        (perf_counter() - started) * 1000,
    )
    return result


@router.get("/cash-forecasts/{forecast_id}", response_model=CashForecastView)
def read_forecast(forecast_id: UUID, response: Response):
    response.headers["Cache-Control"] = "no-store"
    try:
        return get_cash_forecast(forecast_id)
    except ERRORS as error:
        raise cash_http_error(error, "read_forecast") from error


@router.get("/{fund_code}", response_model=WatchlistPrediction)
def read_prediction(fund_code: Annotated[str, Path(pattern=r"^\d{6}$")], response: Response):
    response.headers["Cache-Control"] = "no-store"
    try:
        result = get_watchlist_prediction(fund_code)
    except ERRORS as error:
        raise cash_http_error(error, "read_prediction") from error
    logger.info(
        "watchlist_prediction.read_prediction >>> complete, trace_id=%s, fund=%s, status=%s",
        get_trace_id(),
        fund_code,
        result.status,
    )
    return result
