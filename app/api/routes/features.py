"""特征相关HTTP入口；内部Token认证后可读取特征状态或预览历史样本。

用户学习时从GET preview_stored_historical_nav开始读：传基金代码和日期，由服务自己查数据库。
同路径POST是可选的自备净值测试方式。完整前缀/internal/v1/features由上层路由统一挂载。
"""

from datetime import date
from time import perf_counter
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.exc import SQLAlchemyError

from app.api.dependencies import require_service_token
from app.core.logging import get_logger
from app.core.middleware import get_trace_id
from app.repositories.historical_nav import HistoricalNavPreviewReadError
from app.schemas.feature import InternalFeatureStatus
from app.schemas.historical_nav import (
    HistoricalNavBatchPreviewRequest,
    HistoricalNavBatchPreviewResponse,
    HistoricalNavPreviewRequest,
    HistoricalNavPreviewResponse,
)
from app.services.feature_read import get_latest_stock_feature_status
from app.services.historical_nav_preview import (
    HistoricalNavBatchTimeoutError,
    preview_stored_historical_nav_batch,
    preview_stored_historical_nav_sample,
)
from app.services.historical_nav_samples import (
    HistoricalNavPoint,
    HistoricalNavSample,
    HistoricalNavSampleInput,
    build_historical_nav_samples,
)

# router用于登记本文件的URL；logger记录请求摘要，不打印Token或整组净值。
router = APIRouter()
logger = get_logger(__name__)


@router.get("/latest", response_model=InternalFeatureStatus, dependencies=[Depends(require_service_token)])
def get_latest_internal_feature(
    fund_code: Annotated[str, Query(alias="fundCode", min_length=6, max_length=6, pattern=r"^\d{6}$")],
) -> InternalFeatureStatus:
    """返回已持久化特征或正常的不可用状态；不运行模型或数据同步。"""
    payload = get_latest_stock_feature_status(fund_code)
    logger.info(
        "features.get_latest_internal_feature >>> returned feature status, trace_id=%s, fund_code=%s, status=%s",
        get_trace_id(),
        fund_code,
        payload.status,
    )
    return payload


@router.get(
    "/historical-nav-samples/preview",
    response_model=HistoricalNavSample,
    dependencies=[Depends(require_service_token)],
)
def preview_stored_historical_nav(
    # alias表示URL使用fundCode，Python函数里使用fund_code；格式固定6位数字字符串。
    fund_code: Annotated[str, Query(alias="fundCode", pattern=r"^[0-9]{6}$", min_length=6, max_length=6)],
    # URL里的asOfDate会自动解析为date；这是净值业务日，公告日由数据库中的记录确定。
    as_of_date: Annotated[date, Query(alias="asOfDate")],
) -> HistoricalNavSample:
    """GET入口：接收基金和业务日，返回数据库中该日的历史样本，无需Body。

    示例：?fundCode=008888&asOfDate=2025-08-07。
    上面的Depends(require_service_token)会先检查X-Service-Token并拒绝浏览器Origin，
    通过后才会进入本函数。操作说明见historical-nav-http-preview.md。
    """
    # 用单调计时器统计执行耗时；不是样本的业务日期或数据可得时间。
    started_at = perf_counter()
    try:
        # 本层只负责HTTP，不写SQL或收益公式；交给服务层串起读取和计算。
        sample = preview_stored_historical_nav_sample(fund_code=fund_code, as_of_date=as_of_date)
    except HistoricalNavPreviewReadError as error:
        # 正常业务拒绝：缺基金/缺指定日净值为404；品类不适用/来源未就绪为409。
        logger.warning(
            "features.preview_stored_historical_nav >>> unavailable, trace_id=%s, fund_code=%s, as_of_date=%s, code=%s",
            get_trace_id(), fund_code, as_of_date, error.code,
        )
        status_code = 404 if error.code in {"FUND_NOT_FOUND", "NAV_NOT_FOUND"} else 409
        raise HTTPException(status_code=status_code, detail={"code": error.code, "message": str(error)}) from error
    except SQLAlchemyError as error:
        # 数据库连接失败、超时等暂时性故障：日志保留堆栈，HTTP只返回简洁原因，不暴露连接细节。
        logger.exception(
            "features.preview_stored_historical_nav >>> database read failed, trace_id=%s, fund_code=%s, as_of_date=%s",
            get_trace_id(), fund_code, as_of_date,
        )
        raise HTTPException(status_code=503, detail="数据库暂时不可用，请稍后重试。") from error
    except (ValueError, ArithmeticError) as error:
        # 已存数据进入计算后仍可能数值异常；转为422，让调用者知道是数据问题而非预测结论。
        logger.exception(
            "features.preview_stored_historical_nav >>> invalid stored NAV, trace_id=%s, fund_code=%s, as_of_date=%s",
            get_trace_id(), fund_code, as_of_date,
        )
        raise HTTPException(status_code=422, detail="该日期净值无法计算，请检查数据质量。") from error
    # TraceID用于把这次请求与日志对应起来；仅记录身份、日期、状态和耗时。
    logger.info(
        "features.preview_stored_historical_nav >>> preview completed, trace_id=%s, fund_code=%s, "
        "as_of_date=%s, status=%s, elapsed_ms=%.2f",
        get_trace_id(), fund_code, as_of_date, sample.eligibility_status, (perf_counter() - started_at) * 1000,
    )
    # FastAPI按response_model输出JSON：date成为日期字符串，Decimal成为小数字符串。
    # HTTP 200只表示请求已处理；样本是否可用还要看eligibility_status。
    return sample


@router.get(
    "/historical-nav-samples/dry-run",
    response_model=HistoricalNavBatchPreviewResponse,
    dependencies=[Depends(require_service_token)],
)
def dry_run_historical_nav_samples(
    # Query() 表示从 URL 读取四个参数；不需要 JSON 请求体，字段解释见这个请求类。
    request: Annotated[HistoricalNavBatchPreviewRequest, Query()],
) -> HistoricalNavBatchPreviewResponse:
    """批量 GET：一只基金、一小段日期的只读试跑；pageSize 控制内部读取批次。"""
    started_at = perf_counter()
    try:
        result = preview_stored_historical_nav_batch(request)
    except HistoricalNavPreviewReadError as error:
        logger.warning(
            "features.dry_run_historical_nav_samples >>> unavailable, trace_id=%s, fund_code=%s, code=%s",
            get_trace_id(), request.fund_code, error.code,
        )
        status_code = 404 if error.code == "FUND_NOT_FOUND" else 409
        raise HTTPException(status_code=status_code, detail={"code": error.code, "message": str(error)}) from error
    except (SQLAlchemyError, HistoricalNavBatchTimeoutError) as error:
        # 查询故障或超时均整次失败，不返回已经算好的一部分来冒充完整结果。
        logger.exception(
            "features.dry_run_historical_nav_samples >>> read failed, trace_id=%s, fund_code=%s, "
            "start_date=%s, end_date=%s, page_size=%s, elapsed_ms=%.2f",
            get_trace_id(), request.fund_code, request.start_date, request.end_date, request.page_size,
            (perf_counter() - started_at) * 1000,
        )
        raise HTTPException(status_code=503, detail="批量预览暂时不可用或超时，请缩短日期范围后重试。") from error
    except (ValueError, ArithmeticError) as error:
        logger.exception(
            "features.dry_run_historical_nav_samples >>> invalid NAV, trace_id=%s, fund_code=%s",
            get_trace_id(), request.fund_code,
        )
        raise HTTPException(status_code=422, detail="该范围净值无法计算，请检查数据质量。") from error
    # 一次请求只记一条完成摘要，不逐日打印 INFO，不记录净值明细或服务 Token。
    logger.info(
        "features.dry_run_historical_nav_samples >>> completed, trace_id=%s, fund_code=%s, "
        "start_date=%s, end_date=%s, page_size=%s, page_count=%s, samples=%s, "
        "scorable=%s, insufficient=%s, pending=%s, elapsed_ms=%.2f",
        get_trace_id(), request.fund_code, request.start_date, request.end_date, result.page_size,
        result.page_count, result.sample_count, result.scorable_count, result.data_insufficient_count,
        result.label_not_matured_count, (perf_counter() - started_at) * 1000,
    )
    return result


@router.post(
    "/historical-nav-samples/preview",
    response_model=HistoricalNavPreviewResponse,
    dependencies=[Depends(require_service_token)],
)
def preview_historical_nav_samples(request: HistoricalNavPreviewRequest) -> HistoricalNavPreviewResponse:
    """计算请求中的历史样本；不读写数据库、不采集、不训练、不发布。

    调用说明见 docs_zhx/implementation/historical-nav-http-preview.md；
    跨端设计与验收见 Vue 文档仓的 free-data-prediction-v1.md。
    """
    # POST的request已由Pydantic校验，调用方最多提交512条净值；这个入口不读取数据库。
    started_at = perf_counter()
    try:
        samples = build_historical_nav_samples(
            HistoricalNavSampleInput(
                fund_code=request.fund_code,
                fund_type=request.fund_type,
                source_code=request.source_code,
                source_sync_run_id=request.source_sync_run_id,
                # model_dump取出已校验的字段，**按字段名组装纯计算对象，date/Decimal类型会保留。
                nav_points=tuple(HistoricalNavPoint(**point.model_dump()) for point in request.nav_points),
            )
        )
    except (ValueError, ArithmeticError) as error:
        logger.exception(
            "features.preview_historical_nav_samples >>> calculation rejected, trace_id=%s, fund_code=%s, nav_count=%s",
            get_trace_id(), request.fund_code, len(request.nav_points),
        )
        raise HTTPException(status_code=422, detail="NAV sample calculation failed; check dates and values.") from error
    # 这里只筛选展示结果，不改计算时的公告截止日；未指定日期则保留全部样本。
    if request.as_of_date is not None:
        samples = tuple(sample for sample in samples if sample.as_of_date == request.as_of_date)
    # 计数基于筛选后的样本。例如输入81条、只看一天，input_nav_count=81，sample_count=1。
    # sum(条件)利用True记1、False记0，累计不同状态的条数。
    payload = HistoricalNavPreviewResponse(
        input_nav_count=len(request.nav_points),
        sample_count=len(samples),
        scorable_count=sum(sample.eligibility_status == "SCORABLE" for sample in samples),
        data_insufficient_count=sum(sample.eligibility_status == "DATA_INSUFFICIENT" for sample in samples),
        label_not_matured_count=sum(sample.eligibility_status == "LABEL_NOT_MATURED" for sample in samples),
        items=samples,
    )
    logger.info(
        "features.preview_historical_nav_samples >>> preview completed, trace_id=%s, fund_code=%s, "
        "nav_count=%s, sample_count=%s, scorable=%s, insufficient=%s, label_pending=%s, elapsed_ms=%.2f",
        get_trace_id(), request.fund_code, payload.input_nav_count, payload.sample_count,
        payload.scorable_count, payload.data_insufficient_count, payload.label_not_matured_count,
        (perf_counter() - started_at) * 1000,
    )
    return payload
