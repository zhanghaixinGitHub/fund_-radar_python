"""服务Token保护的候选训练；只读数据库，不发布、不替客户端保存服务器文件。"""

from time import perf_counter

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.exc import SQLAlchemyError

from app.api.dependencies import require_service_token
from app.core.logging import get_logger
from app.core.middleware import get_trace_id
from app.schemas.historical_nav_training import HistoricalNavTrainingRequest, HistoricalNavTrainingResponse
from app.services.historical_nav_evaluation import HistoricalNavEvaluationError
from app.services.historical_nav_storage import HistoricalNavStorageError
from app.services.historical_nav_training import HistoricalNavTrainingError, train_stored_historical_nav_candidate

router = APIRouter(dependencies=[Depends(require_service_token)])
logger = get_logger(__name__)


@router.post("/historical-nav-samples/candidate-training", response_model=HistoricalNavTrainingResponse)
def train_candidate(request: HistoricalNavTrainingRequest) -> HistoricalNavTrainingResponse:
    """POST执行一次固定实验；足量才训练，忙碌429，任何异常都不返回模型成功。"""
    started = perf_counter()
    try:
        result = train_stored_historical_nav_candidate(request)
    except (HistoricalNavTrainingError, HistoricalNavEvaluationError, HistoricalNavStorageError) as error:
        logger.log(
            40 if error.status_code >= 500 else 30,
            "historical_nav_training.train_candidate >>> rejected, trace_id=%s, batches=%s, code=%s",
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
            "historical_nav_training.train_candidate >>> failed, trace_id=%s, batches=%s",
            get_trace_id(),
            len(request.batch_ids),
        )
        raise HTTPException(
            status_code=503,
            detail={
                "code": "TRAINING_UNAVAILABLE",
                "message": "候选训练暂不可用，本次没有写库或发布，请联系维护人员。",
            },
        ) from error
    logger.info(
        "historical_nav_training.train_candidate >>> completed, trace_id=%s, batches=%s, status=%s, "
        "dataset_hash=%s, model_hash=%s, elapsed_ms=%.2f",
        get_trace_id(),
        len(request.batch_ids),
        result.status,
        result.preparation.dataset_hash,
        result.model.model_hash if result.model else None,
        (perf_counter() - started) * 1000,
    )
    return result
