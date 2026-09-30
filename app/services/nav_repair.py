"""境内净值缺口计算；特殊或跨境基金沿用已有同步范围，不猜测境外日历。"""

import re
from datetime import datetime, time
from zoneinfo import ZoneInfo

ZONE = ZoneInfo("Asia/Shanghai")


def domestic_calendar_supported(fund):
    description = " ".join(
        str(fund.get(k) or "") for k in ("fund_name", "fund_type", "source_fund_type", "benchmark", "invest_type")
    )
    return fund.get("fund_type") in {"STOCK", "MIXED", "BOND"} and not re.search(
        r"QDII|FOF|REIT|货币|黄金|原油|商品|美元|港元|港股|恒生|海外|越南|日本|纳斯达克|全球|标普", description, re.I
    )


def expected_dates(
    sessions, existing, target_date, *, now=None, found_date=None,
    purchase_start_date=None, redemption_start_date=None,
):
    """核对开放后逐日净值，保留中间缺口；成立至首次开放期间不强制逐交易日公布。

    首次开放取来源登记的申购/赎回起始日中较早者，不依据缺失形状猜日期。
    缺少元数据时维持原保守范围，异常日期明确报错，已有封闭期净值不删除。
    """
    now = (now or datetime.now(ZONE)).astimezone(ZONE)
    if target_date > sessions[-1] and target_date.year > sessions[-1].year:
        raise ValueError("CALENDAR_UNAVAILABLE")
    start = min(existing) if existing else found_date
    if start is None or start < sessions[0]:
        raise ValueError("HISTORICAL_BASELINE_REQUIRED")
    opening_dates = [day for day in (purchase_start_date, redemption_start_date) if day is not None]
    if opening_dates:
        opened = min(opening_dates)
        if found_date is not None and opened < found_date:
            raise ValueError("FUND_OPEN_DATE_INVALID")
        start = max(start, opened)
    end = min(target_date, now.date())
    return tuple(d for d in sessions if start <= d <= end and not (d == now.date() and now.time() < time(15)))


def contiguous_ranges(missing, sessions):
    """只请求实际缺失的连续估值日段，单次最多约一年，避免重复拉取整段已完整历史。"""
    if not missing:
        return ()
    positions = {d: i for i, d in enumerate(sessions)}
    ranges, start, previous = [], missing[0], missing[0]
    for day in missing[1:]:
        if positions[day] != positions[previous] + 1 or (day - start).days > 365:
            ranges.append((start, previous))
            start = day
        previous = day
    ranges.append((start, previous))
    return tuple(ranges)
