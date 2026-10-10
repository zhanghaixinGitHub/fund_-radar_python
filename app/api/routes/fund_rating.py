"""基金评级内部契约：服务令牌保护，查询不触发任务。"""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Path, Query, Response
from sqlalchemy.exc import SQLAlchemyError

from app.api.dependencies import require_service_token
from app.api.routes.funds import _to_internal_sync_job_status
from app.core.logging import get_logger
from app.schemas.fund import InternalSyncJobStatus
from app.schemas.fund_rating import RatingDetail, RatingPage, RatingSingleRequest
from app.services.fund_rating import read
from app.services.sync_jobs import SyncJobInProgressError, get_sync_job_manager

router = APIRouter(dependencies=[Depends(require_service_token)])
logger = get_logger(__name__)


@router.get("/ratings", response_model=RatingPage)
def ratings(
    response: Response, fund_codes: Annotated[str, Query(alias="fundCodes", pattern=r"^[0-9]{6}(,[0-9]{6}){0,99}$")]
):
    response.headers["Cache-Control"] = "private, no-store"
    try:
        return read(fund_codes.split(","))
    except (ValueError, SQLAlchemyError, OSError):
        logger.exception("fund_rating.read >>> batch read failed")
        raise HTTPException(503, "评级暂不可用，请稍后重试。") from None


@router.get("/{fund_code}/rating", response_model=RatingDetail)
def rating(
    response: Response,
    fund_code: Annotated[str, Path(pattern=r"^[0-9]{6}$")],
    rating_ref: Annotated[str | None, Query(alias="ratingRef", pattern=r"^[a-f0-9]{64}$")] = None,
):
    response.headers["Cache-Control"] = "private, no-store"
    try:
        return read([fund_code], rating_ref=rating_ref, detail=True)
    except LookupError:
        raise HTTPException(404, "基金或所选评级不存在。") from None
    except (ValueError, SQLAlchemyError, OSError):
        logger.exception("fund_rating.read >>> detail read failed, fund_code=%s", fund_code)
        raise HTTPException(503, "评级暂不可用，请稍后重试。") from None


def _start(code=None):
    try:
        return _to_internal_sync_job_status(get_sync_job_manager().start_fund_ratings(code))
    except SyncJobInProgressError:
        raise HTTPException(409, "已有同步任务正在执行。") from None
    except LookupError:
        raise HTTPException(404, "基金不存在。") from None


@router.post("/sync-jobs/fund-ratings/all", response_model=InternalSyncJobStatus, status_code=202)
def start_all():
    return _start()


@router.post("/sync-jobs/fund-ratings/single", response_model=InternalSyncJobStatus, status_code=202)
def start_single(body: RatingSingleRequest):
    return _start(body.fundCode)


@router.get("/sync-jobs/fund-ratings/latest", response_model=InternalSyncJobStatus | None)
def latest():
    result = get_sync_job_manager().get_latest_job("FUND_RATINGS")
    return _to_internal_sync_job_status(result) if result else None
