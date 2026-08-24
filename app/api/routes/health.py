"""Health endpoint available only to the Java core service."""

from datetime import UTC, datetime

from fastapi import APIRouter, Depends, Response

from app.api.dependencies import require_service_token
from app.core.logging import get_logger
from app.core.middleware import get_trace_id
from app.schemas.system import InternalHealthResponse

router = APIRouter()
logger = get_logger(__name__)


@router.get("/health", response_model=InternalHealthResponse, dependencies=[Depends(require_service_token)])
async def get_internal_health(response: Response) -> InternalHealthResponse:
    """Return a minimal authenticated health response for Java service probes."""
    trace_id = get_trace_id()
    response.headers["X-Trace-Id"] = trace_id
    logger.info("health.get_internal_health >>> internal health probe succeeded, trace_id=%s", trace_id)
    return InternalHealthResponse(service="fund-ai", status="UP", time=datetime.now(UTC))
