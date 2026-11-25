"""正式四周期使用北京时间日期；旧版日线研究仍使用自己的收盘窗口。"""

from bisect import bisect_right
from datetime import date, datetime, time, timedelta

from app.services.direction_1d_protocol import ZONE, calendar

POLICY = "CN_NAV_NEXT_VALUATION_MIDNIGHT_V1"


def is_midnight(window):
    return (window or {}).get("generation_policy") == POLICY


def window(now: datetime) -> dict:
    if now.tzinfo is None:
        raise ValueError("TIMEZONE_REQUIRED")
    now = now.astimezone(ZONE)
    days, version = calendar()
    index = bisect_right(days, now.date())
    if not days[0] <= now.date() <= days[-1] or not 0 < index < len(days):
        raise ValueError("CALENDAR_UNAVAILABLE")
    opened = datetime.combine(now.date(), time.min, ZONE)
    deadline = opened + timedelta(days=1)
    return {
        "generation_policy": POLICY,
        "base_nav_date": str(days[index - 1]),
        "target_nav_date": str(days[index]),
        "window_open_at": opened.isoformat(),
        "deadline_at": deadline.isoformat(),
        "calendar_version": version,
        "status": "OPEN",
        "next_window_open_at": deadline.isoformat(),
    }


def fund_window(planning, valuations):
    """只从本基金已经核实的估值日历选择紧邻当前北京时间日期的两日。"""
    day = datetime.fromisoformat(planning["window_open_at"]).date()
    index = bisect_right(valuations.sessions, day)
    if not valuations.coverage_start <= day <= valuations.coverage_end or not 0 < index < len(valuations.sessions):
        valuations.missing(day)
    base, target = valuations.sessions[index - 1:index + 1]
    return {**planning, "base_nav_date": str(base), "target_nav_date": str(target)}


def public_dates(fund_code, clock, planning):
    """仅使用已有基金档案和日历。净值披露情况不会改变页面上的计划日期。"""
    from pathlib import Path

    from app.db.session import get_engine
    from app.repositories import direction_1d as repo
    from app.services.comprehensive_multi_prediction import review_calendar
    from app.services.fund_comprehensive_materials import checked_documents
    from app.services.fund_exposure_common import read
    from app.services.fund_workflow_materials import registered_materials
    from app.services.prediction_contract import PredictionFailure, target_dates
    from app.services.prediction_features import public_calendar
    from app.services.prediction_natural_month import natural_month_target

    with get_engine().connect() as connection:
        profiles = repo.profiles(connection, [fund_code])
    if len(profiles) != 1 or profiles[0]["status"] != "ACTIVE":
        return {}
    try:
        valuations = public_calendar(profiles[0])
    except PredictionFailure:
        try:
            handoff = registered_materials(fund_code, at=clock)
            if not handoff:
                return {}
            material = read(Path(handoff["materials_path"]))
            data = {**material, "fund_name": profiles[0]["fund_name"], "window": planning,
                    "as_of": clock.isoformat(), "documents": checked_documents(clock, fund_code, material)}
            valuations = review_calendar(data, fund_code, clock)
        except (ValueError, KeyError, TypeError, FileNotFoundError):
            return {}
    try:
        actual = fund_window(planning, valuations)
    except PredictionFailure:
        return {}
    base, start = actual["base_nav_date"], actual["target_nav_date"]
    periods = [{"horizonId": "T1", "endDate": start}]
    for horizon in ("T5_V1", "T20_V1"):
        try:
            end = target_dates(valuations, clock, horizon, valuation_window=actual)["endDate"]
        except PredictionFailure:
            end = None
        periods.append({"horizonId": horizon, "endDate": end})
    nominal = natural_month_target(base, start)["nominalEndDate"]
    endpoint = date.fromisoformat(nominal)
    end = None
    if valuations.coverage_start <= endpoint <= valuations.coverage_end:
        index = bisect_right(valuations.sessions, endpoint)
        if index:
            end = str(valuations.sessions[index - 1])
    periods.append({"horizonId": "M6_V1", "endDate": end, "nominalEndDate": nominal})
    return {"baseNavDate": base, "targetNavDate": start, "periods": periods}


def publication_deadline(identity):
    """旧归档延用原发布时限，当前轮次以冻结的北京时间零点为准。"""
    if identity.get("deadline_at"):
        return datetime.fromisoformat(identity["deadline_at"])
    return datetime.combine(date.fromisoformat(identity["target_date"]), time(15), ZONE)
