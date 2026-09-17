"""持仓建议的公共回报核验；服务令牌与来源限制沿用内部接口。"""

from datetime import date
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Path, Query

from app.api.dependencies import require_service_token
from app.repositories.historical_nav import HistoricalNavPreviewReadError
from app.schemas.portfolio_advice import AdviceOutcome, DiagnosisFacts, DraftStats
from app.services.portfolio_advice import get_advice_outcome, get_diagnosis_facts, get_draft_stats

router = APIRouter(dependencies=[Depends(require_service_token)])


@router.get("/{fund_code}/diagnosis-facts", response_model=DiagnosisFacts)
def diagnosis_facts(
    fund_code: Annotated[str, Path(pattern=r"^[0-9]{6}$")],
    as_of_date: Annotated[date | None, Query(alias="asOfDate")] = None,
) -> DiagnosisFacts:
    """七项持仓诊断的公共事实；只读、不含用户身份，基线对比由 Java 报告侧完成。"""
    try:
        return get_diagnosis_facts(fund_code, as_of_date)
    except HistoricalNavPreviewReadError as error:
        if error.code == "FUND_NOT_FOUND":
            raise HTTPException(404, "数据库中没有这只基金。") from error
        raise HTTPException(503, "基金来源资料暂时无法核验。") from error


@router.get("/{fund_code}/draft-stats", response_model=DraftStats)
def draft_stats(
    fund_code: Annotated[str, Path(pattern=r"^[0-9]{6}$")],
) -> DraftStats:
    """规则草案统计；历史不足或不适用返回明确状态与原因，不给阈值数字。只读、不含用户身份。"""
    try:
        return get_draft_stats(fund_code)
    except HistoricalNavPreviewReadError as error:
        if error.code == "FUND_NOT_FOUND":
            raise HTTPException(404, "数据库中没有这只基金。") from error
        raise HTTPException(503, "基金来源资料暂时无法核验。") from error


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
