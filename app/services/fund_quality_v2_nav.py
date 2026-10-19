"""从已归档本份额净值行重建真实覆盖事实，独立于旧评级八维准入。

净值覆盖与合同基准覆盖分别记录。本模块只生成来源事实和净值曲线，
不会因为净值齐全便生成相对收益、治理分或V2总分。
"""

from datetime import date

from app.services.fund_rating_rules import decimal


def rebuild_nav_coverage(snapshot: dict, *, fund_code: str, as_of: date) -> dict:
    """核对份额、日期、正值、37个月末及本机既有估值日历，不补空净值。

    ann_date是原始公布日，updated_at仅是本地维护时间；两者不混淆。
    晚于经济期末公布的净值单独列出，不能称为经济期末当时已经知道。
    跨境产品的实际估值日历仍须另核，不能因能套上境内日历便宣称齐全。
    """
    from app.services.direction_1d_protocol import calendar

    if snapshot.get("fund_code") != fund_code:
        raise ValueError("QUALITY_NAV_SHARE_IDENTITY")
    history = snapshot.get("history") or {}
    rows, requested, ends = history.get("rows", []), history.get("dates", []), history.get("month_ends", [])
    if not rows or len(rows) > 5000 or not requested or requested != sorted(set(requested)):
        raise ValueError("QUALITY_NAV_HISTORY_SHAPE")
    if len(ends) != 37 or ends != sorted(set(ends)) or any(day not in requested for day in ends):
        raise ValueError("QUALITY_NAV_MONTH_ENDS")
    months = [date.fromisoformat(day).year * 12 + date.fromisoformat(day).month for day in ends]
    if months != list(range(months[0], months[0] + 37)) or requested[0] != ends[0] or requested[-1] != ends[-1]:
        raise ValueError("QUALITY_NAV_WINDOW")
    if date.fromisoformat(requested[-1]) > as_of:
        raise ValueError("QUALITY_NAV_FUTURE_OBSERVATION")
    by_day = {}
    late, unknown_publication = [], []
    requested_set = set(requested)
    for row in rows:
        day = row.get("nav_date")
        if not day or day in by_day or date.fromisoformat(day) > as_of:
            raise ValueError("QUALITY_NAV_DUPLICATE_OR_FUTURE_ROW")
        by_day[day] = row
        # 经济观察日不等于资料可得日。只检查实际计算要用的估值行；
        # 原响应中未使用的其他行不据此被当作已知，也不污染当前窗口。
        if day in requested_set:
            if not row.get("ann_date"):
                unknown_publication.append(day)
            elif date.fromisoformat(row["ann_date"]) > as_of:
                late.append({"nav_date": day, "announced_on": row["ann_date"]})
    missing = [day for day in requested if day not in by_day or by_day[day].get("adjusted_nav") is None]
    for day in requested:
        if day not in missing and (decimal(by_day[day]["adjusted_nav"]) <= 0 or not by_day[day].get("content_hash")):
            raise ValueError("QUALITY_NAV_VALUE_OR_ROW_LINEAGE")
    sessions, calendar_hash = calendar()
    same_calendar = history.get("calendar_hash") == calendar_hash and requested == [
        str(day) for day in sessions if date.fromisoformat(requested[0]) <= day <= as_of
    ]
    present_ends = [day for day in ends if day not in missing]
    facts = {
        "fund_code": fund_code,
        "requested_from": requested[0],
        "requested_to": requested[-1],
        "available_from": min(by_day),
        "available_to": max(by_day),
        "raw_nav_rows": len(rows),
        "requested_valuation_days": len(requested),
        "missing_adjusted_nav_dates": missing,
        "available_month_end_count": len(present_ends),
        "domestic_calendar_matches": same_calendar,
        "late_announcements": late,
        "unknown_announcement_dates": unknown_publication,
        "valuation_end": requested[-1],
        "availability_cutoff": str(as_of),
        "status": "OWN_NAV_PUBLICATION_AFTER_AS_OF" if late
        else "OWN_NAV_PUBLICATION_DATE_UNVERIFIED" if unknown_publication
        else "OWN_ADJUSTED_NAV_WINDOW_REBUILT" if not missing and same_calendar
        else "OWN_NAV_WINDOW_INCOMPLETE_OR_CALENDAR_UNVERIFIED",
        "benchmark_status": "NOT_ASSESSED_SEPARATELY_REQUIRES_CONTRACT_CURVE",
    }
    if not missing and same_calendar and not late and not unknown_publication:
        facts["curve"] = {
            "dates": requested,
            "month_ends": ends,
            "adjusted_nav": [str(decimal(by_day[day]["adjusted_nav"])) for day in requested],
        }
    return facts
