"""M3-05 仅供 Java 管理端调用的受控分析与模型发布内部接口。"""

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status

from app.api.dependencies import require_service_token
from app.core.logging import get_logger
from app.core.middleware import get_trace_id
from app.schemas.analysis_run import (
    InternalAnalysisRunStatus,
    InternalBenchmarkPointImportRequest,
    InternalBenchmarkRegistrationRequest,
    InternalBenchmarkSeriesStatus,
    InternalFundExplanationRunRequest,
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
    start_fund_explanation,
    start_stock_rolling_backtest,
    suspend_model_release,
)
from app.services.analysis_summary import get_fund_analysis_summary
from app.services.baseline_analysis import ModelReleaseTransitionError
from app.services.benchmark_registry import (
    BenchmarkPointInput,
    BenchmarkRegistryError,
    activate_stock_benchmark,
    import_stock_benchmark_points,
    list_stock_benchmarks,
    register_stock_benchmark,
    suspend_stock_benchmark,
)

router = APIRouter()
logger = get_logger(__name__)


def _benchmark_status_payload(coverage: object) -> InternalBenchmarkSeriesStatus:
    """将服务层覆盖摘要转换为严格白名单的内部响应。"""
    return InternalBenchmarkSeriesStatus(
        benchmark_code=coverage.benchmark_code,
        display_name=coverage.display_name,
        fund_type=coverage.fund_type,
        source_code=coverage.source_code,
        source_enabled=coverage.source_enabled,
        status=coverage.status,
        license_reference=coverage.license_reference,
        point_count=coverage.point_count,
        first_nav_date=coverage.first_nav_date,
        last_nav_date=coverage.last_nav_date,
    )


@router.get(
    "/benchmarks",
    response_model=list[InternalBenchmarkSeriesStatus],
    dependencies=[Depends(require_service_token)],
)
def list_internal_stock_benchmarks() -> list[InternalBenchmarkSeriesStatus]:
    """列出股票候选回测可见的本地基准状态；不返回原始日序列。"""
    return [_benchmark_status_payload(coverage) for coverage in list_stock_benchmarks()]


@router.put(
    "/benchmarks/{benchmark_code}",
    response_model=InternalBenchmarkSeriesStatus,
    dependencies=[Depends(require_service_token)],
)
def register_internal_stock_benchmark(
    benchmark_code: str,
    request: InternalBenchmarkRegistrationRequest,
) -> InternalBenchmarkSeriesStatus:
    """登记 DRAFT 基准；来源状态和模型状态都不会由本接口自动改变。"""
    try:
        coverage = register_stock_benchmark(
            benchmark_code=benchmark_code,
            display_name=request.display_name,
            source_code=request.source_code,
            license_reference=request.license_reference,
        )
    except BenchmarkRegistryError as error:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(error)) from error
    logger.info(
        "analysis.register_internal_stock_benchmark >>> registered, trace_id=%s, benchmark_code=%s",
        get_trace_id(),
        coverage.benchmark_code,
    )
    return _benchmark_status_payload(coverage)


@router.put(
    "/benchmarks/{benchmark_code}/points",
    response_model=InternalBenchmarkSeriesStatus,
    dependencies=[Depends(require_service_token)],
)
def import_internal_stock_benchmark_points(
    benchmark_code: str,
    request: InternalBenchmarkPointImportRequest,
) -> InternalBenchmarkSeriesStatus:
    """导入人工核验的已授权基准点；ACTIVE 基准必须先显式暂停。"""
    try:
        coverage = import_stock_benchmark_points(
            benchmark_code=benchmark_code,
            points=tuple(
                BenchmarkPointInput(
                    nav_date=point.nav_date,
                    closing_value=point.closing_value,
                    source_published_at=point.source_published_at,
                )
                for point in request.points
            ),
        )
    except BenchmarkRegistryError as error:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(error)) from error
    logger.info(
        "analysis.import_internal_stock_benchmark_points >>> imported, trace_id=%s, benchmark_code=%s, input=%s",
        get_trace_id(),
        coverage.benchmark_code,
        len(request.points),
    )
    return _benchmark_status_payload(coverage)


@router.post(
    "/benchmarks/{benchmark_code}/activate",
    response_model=InternalBenchmarkSeriesStatus,
    dependencies=[Depends(require_service_token)],
)
def activate_internal_stock_benchmark(benchmark_code: str) -> InternalBenchmarkSeriesStatus:
    """显式启用具备来源授权和历史覆盖的基准，不影响任何模型发布状态。"""
    try:
        coverage = activate_stock_benchmark(benchmark_code)
    except BenchmarkRegistryError as error:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(error)) from error
    logger.info(
        "analysis.activate_internal_stock_benchmark >>> activated, trace_id=%s, benchmark_code=%s",
        get_trace_id(),
        coverage.benchmark_code,
    )
    return _benchmark_status_payload(coverage)


@router.post(
    "/benchmarks/{benchmark_code}/suspend",
    response_model=InternalBenchmarkSeriesStatus,
    dependencies=[Depends(require_service_token)],
)
def suspend_internal_stock_benchmark(benchmark_code: str) -> InternalBenchmarkSeriesStatus:
    """显式暂停基准，阻止新的回测使用该序列，保留历史运行。"""
    try:
        coverage = suspend_stock_benchmark(benchmark_code)
    except BenchmarkRegistryError as error:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(error)) from error
    logger.info(
        "analysis.suspend_internal_stock_benchmark >>> suspended, trace_id=%s, benchmark_code=%s",
        get_trace_id(),
        coverage.benchmark_code,
    )
    return _benchmark_status_payload(coverage)


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
        payload = start_stock_rolling_backtest(
            fee_rate=request.fee_rate,
            benchmark_code=request.benchmark_code,
            trace_id=get_trace_id(),
        )
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


@router.post(
    "/runs/fund-explanations",
    response_model=InternalAnalysisRunStatus,
    status_code=status.HTTP_202_ACCEPTED,
    dependencies=[Depends(require_service_token)],
)
def start_internal_fund_explanation(request: InternalFundExplanationRunRequest) -> InternalAnalysisRunStatus:
    """排队已发布评分的 DeepSeek 解释；不能通过本接口影响回测、评分或模型发布。"""
    try:
        payload = start_fund_explanation(fund_code=request.fund_code, trace_id=get_trace_id())
    except AnalysisRunInProgressError as error:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(error)) from error
    except (RuntimeError, ValueError) as error:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(error)) from error
    logger.info(
        "analysis.start_internal_fund_explanation >>> explanation run queued, "
        "trace_id=%s, analysis_run_id=%s, fund_code=%s",
        get_trace_id(),
        payload.analysis_run_id,
        request.fund_code,
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
