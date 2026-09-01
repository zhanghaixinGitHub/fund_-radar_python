"""供 Java 核心服务认证读取的 M3 特征状态接口。"""

from typing import Annotated

from fastapi import APIRouter, Depends, Query

from app.api.dependencies import require_service_token
from app.core.logging import get_logger
from app.core.middleware import get_trace_id
from app.schemas.feature import InternalFeatureStatus
from app.services.feature_read import get_latest_stock_feature_status

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
