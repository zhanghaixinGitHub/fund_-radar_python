"""供 Java 核心服务读取的受限数据源治理诊断接口。"""

from fastapi import APIRouter, Depends

from app.api.dependencies import require_service_token
from app.core.logging import get_logger
from app.core.middleware import get_trace_id
from app.schemas.source import SourceDiagnostic
from app.services.source_diagnostics import list_source_diagnostics

router = APIRouter()
logger = get_logger(__name__)


@router.get("", response_model=tuple[SourceDiagnostic, ...], dependencies=[Depends(require_service_token)])
def list_internal_source_diagnostics() -> tuple[SourceDiagnostic, ...]:
    """返回已配置数据源的安全状态，不返回凭据、原始内容，也不发起外部调用。"""
    diagnostics = list_source_diagnostics()
    logger.info(
        "sources.list_internal_source_diagnostics >>> returned source diagnostics, trace_id=%s, count=%s",
        get_trace_id(),
        len(diagnostics),
    )
    return diagnostics
