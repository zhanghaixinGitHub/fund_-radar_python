"""持仓建议的公共回报核验；服务令牌与来源限制沿用内部接口。"""

from datetime import date
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Path, Query

from app.api.dependencies import require_service_token
from app.repositories.historical_nav import HistoricalNavPreviewReadError
from app.schemas.portfolio_advice import AdviceOutcome
from app.services.portfolio_advice import get_advice_outcome

router = APIRouter(dependencies=[Depends(require_service_token)])


@router.get("/{fund_code}/outcome", response_model=AdviceOutcome)
def outcome(
    fund_code: Annotated[str, Path(pattern=r"^[0-9]{6}$")],
    start_date: Annotated[date, Query(alias="startDate")],
    end_date: Annotated[date, Query(alias="endDate")],
) -> AdviceOutcome:
    try:
        return get_advice_outcome(fund_code, start_date, end_date)
    except HistoricalNavPreviewReadError as error:
        raise HTTPException(503, "基金来源资料暂时无法核验。") from error
    except ValueError as error:
        raise HTTPException(422, "观察区间不合法或资料未通过校验。") from error
