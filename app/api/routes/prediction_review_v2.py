"""第二版正式分析页面接口；读取不生成新方向。"""

from datetime import datetime
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, Path

from app.api.dependencies import require_service_token
from app.api.routes.multi_predictions import PredictionRoute
from app.services.prediction_review_v2_publication import current_review

router = APIRouter(dependencies=[Depends(require_service_token)], route_class=PredictionRoute)
FundCode = Annotated[str, Path(pattern=r"^[0-9]{6}$")]


@router.get("/funds/{fund_code}/review-v2")
def read_review(fund_code: FundCode):
    return current_review(fund_code)


@router.get("/funds/{fund_code}/review-v2/history")
def read_history(fund_code: FundCode, before: datetime | None = None, beforeId: UUID | None = None):
    from fastapi import HTTPException

    from app.services.prediction_review_v2_publication import review_history

    try:
        return review_history(fund_code, before=before, before_id=beforeId)
    except ValueError as error:
        raise HTTPException(422, "分析历史分页参数无效。") from error


@router.post("/funds/{fund_code}/review-v2/generate")
def request_review(fund_code: FundCode):
    from app.services.prediction_review_v2_requests import request_review as enqueue

    return enqueue(fund_code)
