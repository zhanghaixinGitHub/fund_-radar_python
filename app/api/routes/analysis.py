"""M3-05 仅供 Java 管理端调用的受控分析与模型发布内部接口。"""

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status

from app.api.dependencies import require_service_token
from app.core.logging import get_logger
from app.core.middleware import get_trace_id
from app.schemas.analysis_run import (
    InternalAnalysisRunStatus,
    InternalModelReleaseStatus,
    InternalModelReleaseTransitionRequest,
    InternalRollingBacktestRequest,
)
from app.schemas.analysis_summary import InternalFundAnalysisSummary
from app.services.analysis_runs import (
    AnalysisRunInProgressError,
    AnalysisRunNotFoundError,
    activate_model_release,
    get_analysis_run,
    start_stock_rolling_backtest,
    suspend_model_release,
)
from app.services.analysis_summary import get_fund_analysis_summary
from app.services.baseline_analysis import ModelReleaseTransitionError

router = APIRouter()
logger = get_logger(__name__)


@router.get(
    "/fund-summary",
    response_model=InternalFundAnalysisSummary,
    dependencies=[Depends(require_service_token)],
)
def get_internal_fund_analysis_summary(
    fund_code: Annotated[str, Query(alias="fundCode", min_length=6, max_length=6, pattern=r"^\d{6}$")],
) -> InternalFundAnalysisSummary:
    """返回已发布模型及回测摘要；读取不会泄露候选版本或启动任何分析任务。"""
    payload = get_fund_analysis_summary(fund_code)
    logger.info(
        "analysis.get_internal_fund_analysis_summary >>> returned published summary, "
        "trace_id=%s, fund_code=%s, status=%s",
        get_trace_id(),
        fund_code,
        payload.availability_status,
    )
    return payload


@router.post(
    "/runs/rolling-backtest",
    response_model=InternalAnalysisRunStatus,
    status_code=status.HTTP_202_ACCEPTED,
    dependencies=[Depends(require_service_token)],
)
def start_internal_rolling_backtest(request: InternalRollingBacktestRequest) -> InternalAnalysisRunStatus:
    """创建受控股票型回测；不接受基准数据或任意模型参数的浏览器输入。"""
    try:
        payload = start_stock_rolling_backtest(fee_rate=request.fee_rate, trace_id=get_trace_id())
    except AnalysisRunInProgressError as error:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(error)) from error
    except RuntimeError as error:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(error)) from error
    logger.info(
        "analysis.start_internal_rolling_backtest >>> analysis run queued, trace_id=%s, analysis_run_id=%s",
        get_trace_id(),
        payload.analysis_run_id,
    )
    return payload


@router.get(
    "/runs/{analysis_run_id}",
    response_model=InternalAnalysisRunStatus,
    dependencies=[Depends(require_service_token)],
)
def get_internal_analysis_run(analysis_run_id: UUID) -> InternalAnalysisRunStatus:
    """读取持久化分析运行状态；不会重试、重跑或自动发布模型。"""
    try:
        return get_analysis_run(analysis_run_id)
    except AnalysisRunNotFoundError as error:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(error)) from error


@router.post(
    "/model-releases/{model_release_id}/activate",
    response_model=InternalModelReleaseStatus,
    dependencies=[Depends(require_service_token)],
)
def activate_internal_model_release(
    model_release_id: UUID,
    request: InternalModelReleaseTransitionRequest,
) -> InternalModelReleaseStatus:
    """显式激活已通过 M3-04 发布闸门的模型，拒绝隐式或自动上线。"""
    try:
        payload = activate_model_release(model_release_id, reason=request.reason)
    except (ModelReleaseTransitionError, ValueError) as error:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(error)) from error
    logger.info(
        "analysis.activate_internal_model_release >>> model release activated, trace_id=%s, release_id=%s",
        get_trace_id(),
        model_release_id,
    )
    return payload


@router.post(
    "/model-releases/{model_release_id}/suspend",
    response_model=InternalModelReleaseStatus,
    dependencies=[Depends(require_service_token)],
)
def suspend_internal_model_release(
    model_release_id: UUID,
    request: InternalModelReleaseTransitionRequest,
) -> InternalModelReleaseStatus:
    """显式暂停模型发布，保留历史评分和审计链路。"""
    try:
        payload = suspend_model_release(model_release_id, reason=request.reason)
    except (ModelReleaseTransitionError, ValueError) as error:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(error)) from error
    logger.info(
        "analysis.suspend_internal_model_release >>> model release suspended, trace_id=%s, release_id=%s",
        get_trace_id(),
        model_release_id,
    )
    return payload
