"""内部现金研究写入/读取路由；浏览器Origin与服务Token边界沿用项目配置。"""

from time import perf_counter
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Response
from sqlalchemy.exc import SQLAlchemyError

from app.api.dependencies import require_service_token
from app.core.logging import get_logger
from app.core.middleware import get_trace_id
from app.repositories.historical_nav import HistoricalNavPreviewReadError
from app.schemas.cash_exam_plan import CashExamPreparation, CashExamPreparationRequest
from app.schemas.cash_planned_research import CashPlannedResearch, CashPlannedResearchRequest
from app.schemas.cash_reinvestment_research import (
    CashPreparation,
    CashPrepareRequest,
    CashResearchRequest,
    CashStoredResearch,
)
from app.schemas.cash_reinvestment_storage import CashBatchSaveRequest, CashStoredBatch
from app.schemas.cash_source_observation import CashObservationCheck, CashObservationCheckRequest
from app.services.cash_exam_preparation import prepare_cash_exam_data
from app.services.cash_planned_research import get_planned_research, save_planned_research
from app.services.cash_reinvestment_research import get_cash_research, load_cash_dataset, save_cash_research
from app.services.cash_reinvestment_storage import get_cash_batch, save_cash_batch
from app.services.cash_source_observation import check_cash_source_observations
from app.services.historical_nav_storage import HistoricalNavStorageError
from app.services.historical_nav_training import HistoricalNavTrainingError
from app.services.trading_calendar import CalendarCoverageError

router = APIRouter(dependencies=[Depends(require_service_token)])
logger = get_logger(__name__)


@router.post("/cash-reinvestment/source-observation-check", response_model=CashObservationCheck)
def check_source_observations(request: CashObservationCheckRequest, response: Response):
    """仅诊断本地变更留档；不是供应商历史首次版本或分红完整性的审批入口。"""
    response.headers["Cache-Control"] = "no-store"
    started = perf_counter()
    try:
        result = check_cash_source_observations(request)
    except ERRORS as error:
        raise cash_http_error(error, "source_observation_check") from error
    logger.info(
        "cash_reinvestment.source_observation_check >>> complete, trace_id=%s, fund=%s, status=%s, elapsed_ms=%.2f",
        get_trace_id(),
        request.fund_code,
        result.status,
        (perf_counter() - started) * 1000,
    )
    return result


def cash_http_error(error: Exception, operation: str) -> HTTPException:
    """不向浏览器返回SQL、连接配置、模型文件或内部堆栈。"""
    if isinstance(error, (HistoricalNavStorageError, HistoricalNavTrainingError)):
        status, code, message = error.status_code, error.code, str(error)
    elif isinstance(error, HistoricalNavPreviewReadError):
        status, code, message = 404 if error.code == "FUND_NOT_FOUND" else 409, error.code, str(error)
    elif isinstance(error, CalendarCoverageError):
        status, code, message = 409, "CALENDAR_COVERAGE_INSUFFICIENT", str(error)
    else:
        status, code, message = 503, "CASH_RESEARCH_UNAVAILABLE", "现金研究服务暂不可用，请沿用原requestKey重试。"
    logger.log(
        40 if status >= 500 else 30,
        "cash_reinvestment.%s >>> rejected, trace_id=%s, code=%s",
        operation,
        get_trace_id(),
        code,
        exc_info=status >= 500,
    )
    return HTTPException(status_code=status, detail={"code": code, "message": message})


ERRORS = (
    HistoricalNavTrainingError,
    HistoricalNavStorageError,
    HistoricalNavPreviewReadError,
    CalendarCoverageError,
    SQLAlchemyError,
    ValueError,
    TypeError,
    ArithmeticError,
    OSError,
)


@router.post("/cash-reinvestment/batches", response_model=CashStoredBatch, status_code=201)
def save_batch(request: CashBatchSaveRequest, response: Response):
    started = perf_counter()
    try:
        result, created = save_cash_batch(request)
    except ERRORS as error:
        raise cash_http_error(error, "save_batch") from error
    response.status_code = 201 if created else 200
    logger.info(
        "cash_reinvestment.save_batch >>> complete, trace_id=%s, batch_id=%s, fund=%s, "
        "created=%s, samples=%s, elapsed_ms=%.2f",
        get_trace_id(),
        result.batch_id,
        request.fund_code,
        created,
        result.preview.sample_count,
        (perf_counter() - started) * 1000,
    )
    return result


@router.get("/cash-reinvestment/batches/{batch_id}", response_model=CashStoredBatch)
def read_batch(batch_id: UUID):
    try:
        return get_cash_batch(batch_id)
    except ERRORS as error:
        raise cash_http_error(error, "read_batch") from error


@router.post("/cash-reinvestment/preparation", response_model=CashPreparation)
def prepare(request: CashPrepareRequest):
    """只读准备；不自动选择最新批次，也不自动扩大年份。"""
    try:
        return load_cash_dataset(request).report
    except ERRORS as error:
        raise cash_http_error(error, "prepare") from error


@router.post("/cash-reinvestment/research-runs", response_model=CashStoredResearch, status_code=201)
def research(request: CashResearchRequest, response: Response):
    """固定三窗研究并保存报告；不能由请求打开测试或发布开关。"""
    started = perf_counter()
    try:
        result, created = save_cash_research(request)
    except ERRORS as error:
        raise cash_http_error(error, "research") from error
    response.status_code = 201 if created else 200
    logger.info(
        "cash_reinvestment.research >>> complete, trace_id=%s, run_id=%s, status=%s, created=%s, elapsed_ms=%.2f",
        get_trace_id(),
        result.run_id,
        result.report.status,
        created,
        (perf_counter() - started) * 1000,
    )
    return result


@router.post("/cash-reinvestment/exam-preparation", response_model=CashExamPreparation)
def prepare_exam(request: CashExamPreparationRequest, response: Response):
    """核对计划和批次指纹后只读准备覆盖；不执行考试，不把旧报告补盖成事前证据。"""
    started = perf_counter()
    response.headers["Cache-Control"] = "no-store"
    try:
        result = prepare_cash_exam_data(request)
    except ERRORS as error:
        raise cash_http_error(error, "prepare_exam") from error
    logger.info(
        "cash_reinvestment.prepare_exam >>> complete, trace_id=%s, plan_hash=%s, "
        "dataset_hash=%s, groups=%s, elapsed_ms=%.2f",
        get_trace_id(),
        result.plan.plan_hash,
        result.preparation.dataset_hash,
        len(result.coverage),
        (perf_counter() - started) * 1000,
    )
    return result


@router.get("/cash-reinvestment/research-runs/{run_id}", response_model=CashStoredResearch)
def read_research(run_id: UUID):
    try:
        return get_cash_research(run_id)
    except ERRORS as error:
        raise cash_http_error(error, "read_research") from error


@router.post("/cash-reinvestment/planned-research-runs", response_model=CashPlannedResearch)
def create_planned_research(request: CashPlannedResearchRequest, response: Response):
    """已确认规则先行的新研究，不接收旧报告编号、模型、时间或强制发布开关。"""
    started = perf_counter()
    response.headers["Cache-Control"] = "no-store"
    try:
        result = save_planned_research(request)
    except ERRORS as error:
        raise cash_http_error(error, "create_planned_research") from error
    response.status_code = 201 if result.created else 200
    logger.info(
        "cash_reinvestment.create_planned_research >>> complete, trace_id=%s, binding_id=%s, "
        "run_id=%s, created=%s, elapsed_ms=%.2f",
        get_trace_id(),
        result.binding_id,
        result.research.run_id,
        result.created,
        (perf_counter() - started) * 1000,
    )
    return result


@router.get("/cash-reinvestment/planned-research-runs/{binding_id}", response_model=CashPlannedResearch)
def read_planned_research(binding_id: UUID, response: Response):
    response.headers["Cache-Control"] = "no-store"
    try:
        return get_planned_research(binding_id)
    except ERRORS as error:
        raise cash_http_error(error, "read_planned_research") from error
