"""第二版正式分析页面接口；读取不生成新方向。"""

from typing import Annotated

from fastapi import APIRouter, Depends, Path

from app.api.dependencies import require_service_token
from app.api.routes.multi_predictions import PredictionRoute
from app.services.prediction_review_v2_publication import current_review

router = APIRouter(dependencies=[Depends(require_service_token)], route_class=PredictionRoute)
FundCode = Annotated[str, Path(pattern=r"^[0-9]{6}$")]


@router.get("/funds/{fund_code}/review-v2")
def read_review(fund_code: FundCode):
    return current_review(fund_code)


@router.post("/funds/{fund_code}/review-v2/generate")
def request_review(fund_code: FundCode):
    from app.services.prediction_review_v2_requests import request_review as enqueue

    return enqueue(fund_code)
