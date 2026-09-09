"""仅供已授权Java服务读取；本人关注关系由Java会话层强制校验。"""

from typing import Annotated

from fastapi import APIRouter, Depends, Path, Response

from app.api.dependencies import require_service_token
from app.api.routes.cash_reinvestment_storage import ERRORS, cash_http_error
from app.core.logging import get_logger
from app.core.middleware import get_trace_id
from app.schemas.cash_prediction_check import CashPredictionCheck, CashPredictionCheckRequest
from app.schemas.watchlist_prediction import WatchlistPrediction
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
