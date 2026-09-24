"""净值补拉的持久水位：逐基金记录缺口、失败、下一次重试，不把空响应当作未公布。"""

import json
from datetime import UTC, datetime, timedelta

from sqlalchemy import text


def read_inputs(session, source_id, codes, start, end):
    """每批最多200只，日期限于已核验日历；只读取日期而非整份净值，避免全市场N+1。"""
    metadata = {
        r["fund_code"]: dict(r)
        for r in session.execute(
            text("""
          SELECT f.fund_code,f.fund_name,f.fund_type,p.benchmark,p.source_fund_type,p.invest_type,p.found_date,
                 s.status AS repair_status,s.next_retry_at,s.attempts
          FROM fund_share_class f LEFT JOIN fund_profile p ON p.fund_code=f.fund_code AND p.source_id=:source
          LEFT JOIN nav_sync_state s ON s.fund_code=f.fund_code AND s.source_id=:source
          WHERE f.fund_code=ANY(:codes)
        """),
            {"codes": list(codes), "source": source_id},
        ).mappings()
    }
    dates = {code: set() for code in codes}
    for code, day in session.execute(
        text("""
        SELECT fund_code,nav_date FROM nav_daily WHERE source_id=:source AND fund_code=ANY(:codes)
        AND nav_date BETWEEN :start AND :end
    """),
        {"source": source_id, "codes": list(codes), "start": start, "end": end},
    ).yield_per(5000):
        dates[code].add(day)
    return metadata, dates


def save_state(session, *, source_id, code, run_id, status, missing, reason, retryable=True):
    """失败连续次数驱动30分钟至6小时退避；等待未知每30分钟复查，配置错误仅手动重试。"""
    previous = (
        session.execute(
            text("SELECT attempts FROM nav_sync_state WHERE source_id=:s AND fund_code=:f"), {"s": source_id, "f": code}
        ).scalar()
        or 0
    )
    attempts = previous + 1 if status == "SYNC_FAILED" else 0
    delay = min(360, 30 * 2 ** min(max(attempts - 1, 0), 4))
    retry_at = datetime.now(UTC) + timedelta(minutes=delay) if retryable else None
    session.execute(
        text("""
        INSERT INTO nav_sync_state(source_id,fund_code,sync_run_id,status,missing_dates,reason,attempts,next_retry_at)
        VALUES(:s,:f,:run,:status,CAST(:missing AS jsonb),:reason,:attempts,:retry)
        ON CONFLICT(source_id,fund_code) DO UPDATE SET sync_run_id=excluded.sync_run_id,status=excluded.status,
          missing_dates=excluded.missing_dates,reason=excluded.reason,attempts=excluded.attempts,
          next_retry_at=excluded.next_retry_at,checked_at=clock_timestamp()
    """),
        {
            "s": source_id,
            "f": code,
            "run": run_id,
            "status": status,
            "missing": json.dumps([str(d) for d in missing]),
            "reason": reason[:512],
            "attempts": attempts,
            "retry": retry_at,
        },
    )
