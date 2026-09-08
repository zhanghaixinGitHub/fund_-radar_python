"""时间隔离校准与滚动研究的内部HTTP入口；Token保护、无数据库或文件写入。"""

from time import perf_counter

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.exc import SQLAlchemyError

from app.api.dependencies import require_service_token
from app.core.logging import get_logger
from app.core.middleware import get_trace_id
from app.schemas.historical_nav_calibration import HistoricalNavCalibrationRequest, HistoricalNavCalibrationResponse
from app.services.historical_nav_calibration import evaluate_stored_calibration
from app.services.historical_nav_evaluation import HistoricalNavEvaluationError
from app.services.historical_nav_storage import HistoricalNavStorageError
from app.services.historical_nav_training import HistoricalNavTrainingError

router = APIRouter(dependencies=[Depends(require_service_token)])
logger = get_logger(__name__)


@router.post("/historical-nav-samples/calibration-evaluation", response_model=HistoricalNavCalibrationResponse)
def evaluate_calibration(request: HistoricalNavCalibrationRequest) -> HistoricalNavCalibrationResponse:
    """只做冻结的三窗研究；不接受阈值、折叠窗口或测试开关，不激活模型。"""
    started = perf_counter()
    try:
        result = evaluate_stored_calibration(request)
    except (HistoricalNavTrainingError, HistoricalNavEvaluationError, HistoricalNavStorageError) as error:
        logger.log(
            40 if error.status_code >= 500 else 30,
            "historical_nav_calibration.evaluate_calibration >>> rejected, trace_id=%s, batches=%s, code=%s",
            get_trace_id(),
            len(request.batch_ids),
            error.code,
            exc_info=error.status_code >= 500,
        )
        raise HTTPException(
            status_code=error.status_code, detail={"code": error.code, "message": str(error)}
        ) from error
    except (SQLAlchemyError, ValueError, ArithmeticError, TypeError, KeyError, ImportError, RuntimeWarning) as error:
        logger.exception(
            "historical_nav_calibration.evaluate_calibration >>> failed, trace_id=%s, batches=%s",
            get_trace_id(),
            len(request.batch_ids),
        )
        raise HTTPException(
            status_code=503,
            detail={
                "code": "CALIBRATION_UNAVAILABLE",
                "message": "校准验证暂不可用，本次未写库或发布，请联系维护人员。",
            },
        ) from error
    logger.info(
        "historical_nav_calibration.evaluate_calibration >>> completed, trace_id=%s, batches=%s, "
        "status=%s, evaluated_windows=%s, dataset_hash=%s, elapsed_ms=%.2f",
        get_trace_id(),
        len(request.batch_ids),
        result.status,
        result.evaluated_window_count,
        result.preparation.dataset_hash,
        (perf_counter() - started) * 1000,
    )
    return result
