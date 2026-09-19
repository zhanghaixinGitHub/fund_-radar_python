"""Java 专用费率档案接口：抓取并解析天天基金 f10 费率页，不落库、不接收个人数据。"""

from dataclasses import asdict
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Path

from app.api.dependencies import require_service_token
from app.schemas.simulation_fee import FundFeeProfile
from app.services.eastmoney_fee import fetch_fee_profile

router = APIRouter(dependencies=[Depends(require_service_token)])


@router.get("/fees/{fund_code}", response_model=FundFeeProfile)
def fund_fee(fund_code: Annotated[str, Path(pattern=r"^[0-9]{6}$")]) -> FundFeeProfile:
    try:
        profile = fetch_fee_profile(fund_code)
    except ValueError as error:
        # 页面存在但费率结构无法可靠解析时要求 Java 侧转人工维护；网络与页面异常按上游故障返回。
        if str(error).startswith("MANUAL_REQUIRED"):
            raise HTTPException(422, str(error)) from error
        raise HTTPException(502, str(error)) from error
    return FundFeeProfile(**asdict(profile))
