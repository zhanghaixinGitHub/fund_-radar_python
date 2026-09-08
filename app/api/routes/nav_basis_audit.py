"""内部净值口径审计HTTP；沿用服务身份校验，错误不暴露数据库细节。"""

from time import perf_counter
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.exc import SQLAlchemyError

from app.api.dependencies import require_service_token
from app.core.logging import get_logger
from app.core.middleware import get_trace_id
from app.repositories.historical_nav import HistoricalNavPreviewReadError
from app.schemas.nav_basis_audit import NavBasisAuditRequest, NavBasisAuditResponse
from app.services.nav_basis_audit import audit_nav_basis
from app.services.trading_calendar import CalendarCoverageError

router = APIRouter(dependencies=[Depends(require_service_token)])
logger = get_logger(__name__)


@router.get("/historical-nav-samples/nav-basis-audit", response_model=NavBasisAuditResponse)
def get_nav_basis_audit(request: Annotated[NavBasisAuditRequest, Query()]) -> NavBasisAuditResponse:
    started = perf_counter()
    try:
        result = audit_nav_basis(request)
    except (HistoricalNavPreviewReadError, CalendarCoverageError) as error:
        code = error.code if isinstance(error, HistoricalNavPreviewReadError) else "CALENDAR_COVERAGE_INSUFFICIENT"
        logger.warning(
            "nav_basis_audit.get_nav_basis_audit >>> rejected, trace_id=%s, fund=%s, code=%s",
            get_trace_id(),
            request.fund_code,
            code,
        )
        raise HTTPException(
            status_code=404 if code == "FUND_NOT_FOUND" else 409, detail={"code": code, "message": str(error)}
        ) from error
    except (SQLAlchemyError, OSError, ValueError, ArithmeticError, TypeError) as error:
        logger.exception(
            "nav_basis_audit.get_nav_basis_audit >>> failed, trace_id=%s, fund=%s", get_trace_id(), request.fund_code
        )
        raise HTTPException(
            status_code=503,
            detail={
                "code": "NAV_BASIS_AUDIT_UNAVAILABLE",
                "message": "净值口径核验暂不可用，未修改数据，请联系维护人员。",
            },
        ) from error
    logger.info(
        "nav_basis_audit.get_nav_basis_audit >>> completed, trace_id=%s, fund=%s, start=%s, end=%s, "
        "status=%s, elapsed_ms=%.2f",
        get_trace_id(),
        request.fund_code,
        request.start_date,
        request.end_date,
        result.status,
        (perf_counter() - started) * 1000,
    )
    return result
