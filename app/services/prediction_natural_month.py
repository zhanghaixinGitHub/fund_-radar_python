"""第二版半年自然月目标；目标日期与到期实际净值日分别保存。"""

from datetime import date

from app.services.prediction_contract import PredictionFailure, add_months

RULE = "BASE_PLUS_6_CALENDAR_MONTHS_PREVIOUS_VALID_NAV_V1"


def natural_month_target(base_date, start_date):
    base, start = date.fromisoformat(base_date), date.fromisoformat(start_date)
    if start <= base:
        raise ValueError("M6_BASE_START_ORDER_INVALID")
    target = add_months(base, 6)
    return {
        "horizonId": "M6_V1",
        "calendarId": None,
        "calendarHash": None,
        "baseNavDate": str(base),
        "startDate": str(start),
        "endDate": str(target),
        "nominalEndDate": str(target),
        "endDateStatus": "RESOLVED",
        "endDateBasis": "CALENDAR_MONTH",
        "outcomeRule": RULE,
    }


def actual_month_endpoint(target, calendar, nav_dates):
    """到期使用目标日或之前最近实际估值日；该日缺净值则等待，不跳过缺失日。"""
    target = date.fromisoformat(target) if isinstance(target, str) else target
    if not calendar.coverage_start <= target <= calendar.coverage_end:
        calendar.missing(target)
    candidates = [day for day in calendar.sessions if day <= target]
    if not candidates:
        calendar.missing(target)
    actual = candidates[-1]
    if actual not in nav_dates:
        raise PredictionFailure("NAV_GAP", "OUTCOME", "半年到期实际估值日的净值尚未公布")
    return {"targetDate": str(target), "actualNavDate": str(actual), "calendarDayGap": (target - actual).days}


def reconcile_due_endpoints(backend):
    """接入既有每日核对，只追加半年实际取值日凭据，不生成方向或准确率。"""
    from pathlib import Path

    from sqlalchemy import text

    from app.db.session import get_engine
    from app.services.comprehensive_multi_prediction import review_calendar
    from app.services.direction_1d_protocol import digest
    from app.services.fund_exposure_common import now, read
    from app.services.fund_information_snapshot import prepare
    from app.services.fund_materials_store import versioned_save
    from app.services.fund_workflow_materials import registered_materials

    at = now()
    codes = tuple(getattr(backend, "scope", ()))
    if not codes:
        return {"status": "EMPTY_HOLDINGS", "checked": 0, "items": []}
    cursor_path = backend.directory.parent / "natural-month-endpoint-cursor.json"
    cursor = read(cursor_path) if cursor_path.exists() else {}
    offset = cursor.get("offset", 0) if cursor.get("owner_hash") == digest(backend.owner) else 0
    with get_engine().connect() as connection:
        records = connection.execute(text("""
            SELECT r.release_id, item->'case' AS original FROM prediction_review_v2_release r,
              jsonb_array_elements(r.archive->'cases') item
            WHERE r.archive->'owner_audit'->>'owner_hash'=:owner
              AND item->'case'->>'fund_code'=ANY(CAST(:codes AS text[]))
              AND item->'case'->'period'->>'outcome_rule'=:rule
              AND item->'case'->'period'->>'end_date'<:today
            ORDER BY r.adopted_at,r.release_id,item->'case'->>'fund_code' LIMIT 201 OFFSET :offset
        """), {"owner": digest(backend.owner), "codes": list(codes), "rule": RULE,
               "today": at.date().isoformat(), "offset": offset}).mappings().all()
    items = []
    for row in records[:200]:
        original = row["original"]
        code, period = original["fund_code"], original["period"]
        item = {"release_id": str(row["release_id"]), "fund_code": code,
                "targetDate": period["end_date"], "excludedFromAccuracy": True}
        try:
            handoff = registered_materials(code, at=at)
            if not handoff:
                raise ValueError("M6_CURRENT_VALUATION_MATERIALS_MISSING")
            data = prepare(at, code, Path(handoff["materials_path"]))
            calendar = review_calendar(data, code, at)
            actual = actual_month_endpoint(period["end_date"], calendar,
                                           {date.fromisoformat(str(nav["nav_date"])) for nav in data["nav"]})
            item.update(status="RESOLVED", calendar_id=calendar.calendar_id,
                        calendar_hash=calendar.source_hash, **actual)
        except (ValueError, KeyError, FileNotFoundError) as error:
            item.update(status="WAITING_DATA", reason=str(error)[:240])
        versioned_save(backend.directory / "natural-month-endpoints" / (digest(item) + ".json"), item)
        items.append(item)
    versioned_save(cursor_path, {"owner_hash": digest(backend.owner),
                                "offset": offset + 200 if len(records) > 200 else 0})
    return {"status": "PARTIAL_SUCCESS" if len(records) > 200 or any(i["status"] != "RESOLVED" for i in items)
            else "SUCCEEDED", "checked": len(items), "has_more": len(records) > 200, "items": items}
