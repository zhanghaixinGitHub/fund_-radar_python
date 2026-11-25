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


def publication_deadline(identity):
    """旧归档延用原发布时限，当前轮次以冻结的北京时间零点为准。"""
    if identity.get("deadline_at"):
        return datetime.fromisoformat(identity["deadline_at"])
    return datetime.combine(date.fromisoformat(identity["target_date"]), time(15), ZONE)
