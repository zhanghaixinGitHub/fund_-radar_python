"""Java 专用模拟结算公共行情接口；不接收个人订单。"""

from datetime import date, datetime
from typing import Annotated
from zoneinfo import ZoneInfo

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Path, Query

from app.api.dependencies import require_service_token
from app.schemas.simulation_market import SimulationCalendar, SimulationMarket, SimulationRefresh
from app.services.simulation_market import read_market, refresh_market, validate_registered_codes
from app.services.trading_calendar import load_current_calendar

router = APIRouter(dependencies=[Depends(require_service_token)])


@router.get("/calendar", response_model=SimulationCalendar)
def calendar() -> SimulationCalendar:
    value = load_current_calendar()
    return SimulationCalendar(
        version=value.definition.version,
        coverage_start=value.definition.coverage_start,
        coverage_end=value.definition.coverage_end,
        sessions=value.sessions,
    )


@router.get("/funds/{fund_code}", response_model=SimulationMarket)
def market(
    fund_code: Annotated[str, Path(pattern=r"^[0-9]{6}$")],
    start_date: Annotated[date, Query(alias="startDate")],
    end_date: Annotated[date, Query(alias="endDate")],
) -> SimulationMarket:
    today = datetime.now(ZoneInfo("Asia/Shanghai")).date()
    if start_date > end_date or (end_date - start_date).days > 3660 or end_date > today:
        raise HTTPException(422, "净值查询范围不合法，最多覆盖十年且不能查询未来。")
    value = read_market(fund_code, start_date, end_date)
    if value is None:
        raise HTTPException(404, "未找到基金。")
    return value


@router.post("/refresh", status_code=202)
def refresh(request: SimulationRefresh, tasks: BackgroundTasks) -> dict[str, str]:
    try:
        codes = validate_registered_codes(request.fund_codes)
    except ValueError as error:
        raise HTTPException(422, str(error)) from error
    tasks.add_task(refresh_market, codes)
    return {"status": "ACCEPTED"}
