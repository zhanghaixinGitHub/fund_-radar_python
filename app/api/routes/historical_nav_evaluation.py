"""历史样本候选集与基线验证的内部只读HTTP入口。"""

from time import perf_counter

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.exc import SQLAlchemyError

from app.api.dependencies import require_service_token
from app.core.logging import get_logger
from app.core.middleware import get_trace_id
from app.schemas.historical_nav_evaluation import HistoricalNavEvaluationRequest, HistoricalNavEvaluationResponse
from app.services.historical_nav_evaluation import HistoricalNavEvaluationError, evaluate_stored_historical_nav_batches
from app.services.historical_nav_storage import HistoricalNavStorageError

router = APIRouter(dependencies=[Depends(require_service_token)])
logger = get_logger(__name__)


@router.post("/historical-nav-samples/baseline-evaluation", response_model=HistoricalNavEvaluationResponse)
def evaluate_batches(request: HistoricalNavEvaluationRequest) -> HistoricalNavEvaluationResponse:
    """POST仅为容纳批次清单，整个流程只读；数量不足是200诊断，不伪造成绩。"""
    started = perf_counter()
    try:
        result = evaluate_stored_historical_nav_batches(request)
    except (HistoricalNavEvaluationError, HistoricalNavStorageError) as error:
        logger.log(
            40 if error.status_code >= 500 else 30,
            "historical_nav_evaluation.evaluate_batches >>> rejected, trace_id=%s, batches=%s, code=%s",
            get_trace_id(),
            len(request.batch_ids),
            error.code,
            exc_info=error.status_code >= 500,
        )
        raise HTTPException(
            status_code=error.status_code, detail={"code": error.code, "message": str(error)}
        ) from error
    except (SQLAlchemyError, ValueError, ArithmeticError, TypeError, KeyError) as error:
        logger.exception(
            "historical_nav_evaluation.evaluate_batches >>> failed, trace_id=%s, batches=%s",
            get_trace_id(),
            len(request.batch_ids),
        )
        raise HTTPException(
            status_code=503,
            detail={
                "code": "EVALUATION_UNAVAILABLE",
                "message": "数据读取或评估校验失败，本次未写入数据，请稍后重试或联系维护人员。",
            },
        ) from error
    logger.info(
        "historical_nav_evaluation.evaluate_batches >>> completed, trace_id=%s, batches=%s, unique_samples=%s, "
        "included_samples=%s, status=%s, dataset_hash=%s, elapsed_ms=%.2f",
        get_trace_id(),
        len(request.batch_ids),
        result.unique_sample_count,
        result.included_sample_count,
        result.status,
        result.dataset_hash,
        (perf_counter() - started) * 1000,
    )
    return result
