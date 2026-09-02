"""基金分析摘要的只读内部接口。"""

from typing import Annotated

from fastapi import APIRouter, Depends, Query

from app.api.dependencies import require_service_token
from app.core.logging import get_logger
from app.core.middleware import get_trace_id
from app.schemas.analysis_summary import InternalFundAnalysisSummary
from app.services.analysis_summary import get_fund_analysis_summary

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
    """返回已持久化的受限摘要；读取不会创建回测、评分或模型发布。"""
    payload = get_fund_analysis_summary(fund_code)
    logger.info(
        "analysis.get_internal_fund_analysis_summary >>> returned persisted summary, "
        "trace_id=%s, fund_code=%s, status=%s",
        get_trace_id(),
        fund_code,
        payload.availability_status,
    )
    return payload
