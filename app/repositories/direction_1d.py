"""1日公共证据仓储；不访问fund_core或用户身份。"""

import json
from datetime import datetime, timedelta
from uuid import uuid4

from sqlalchemy import text

from app.db.session import get_engine
from app.services.direction_1d_protocol import ZONE, canonical, digest


def clock() -> datetime:
    with get_engine().connect() as c:
        return c.execute(text("SELECT clock_timestamp()")).scalar_one()


def source(c):
    row = c.execute(text("SELECT * FROM source_registry WHERE source_code='TUSHARE_PRO_FUND'")).mappings().one_or_none()
    if not row or not row["enabled"] or "fund_nav" not in row["authorized_api_names"] or row["retention_days"] <= 0:
        raise ValueError("SOURCE_UNAVAILABLE")
    return row


def profiles(c, codes):
    return [
        dict(r)
        for r in c.execute(
            text("""
      SELECT s.*,p.benchmark,p.invest_type,p.source_fund_type,p.market,p.content_hash AS profile_hash,
             m.manager_name,m.fund_name AS master_name,p.found_date
      FROM fund_share_class s JOIN fund_master m USING(fund_master_id)
      LEFT JOIN fund_profile p ON p.fund_code=s.fund_code
        AND p.source_id=(SELECT source_id FROM source_registry WHERE source_code=s.source_code)
      WHERE s.fund_code=ANY(:codes) ORDER BY s.fund_code
    """),
            {"codes": list(codes)},
        ).mappings()
    ]


def navs(c, code, source_id, start, end):
    return [
        dict(r)
        for r in c.execute(
            text("""
      SELECT nav_date,unit_nav,ann_date,content_hash,updated_at FROM nav_daily
      WHERE fund_code=:code AND source_id=:source AND nav_date BETWEEN :start AND :end ORDER BY nav_date
    """),
            {"code": code, "source": source_id, "start": start, "end": end},
        ).mappings()
    ]


def observe(c, code, source_row, rows, now):
    """旧库首次加入时只记现在读取；同值版本重用且不续期，后来修订追加。"""
    result = []
    for row in rows:
        if row["ann_date"] and row["ann_date"] > now.astimezone(ZONE).date():
            raise ValueError("SOURCE_FUTURE_ANNOUNCEMENT")
        payload = {
            "nav_date": str(row["nav_date"]),
            "unit_nav": str(row["unit_nav"]),
            "ann_date": str(row["ann_date"]) if row["ann_date"] else None,
            "upstream_hash": row["content_hash"],
            "source_id": str(source_row["source_id"]),
        }
        h = digest(payload)
        c.execute(
            text("""
          INSERT INTO
          direction_1d_source_version(version_id,fund_code,source_id,kind,business_date,content_hash,payload,received_at,expires_at)
          VALUES(:id,:code,:source,'NAV',:day,:hash,CAST(:payload AS jsonb),:now,:expires)
          ON CONFLICT(fund_code,source_id,kind,business_date,content_hash) DO NOTHING
        """),
            {
                "id": uuid4(),
                "code": code,
                "source": source_row["source_id"],
                "day": row["nav_date"],
                "hash": h,
                "payload": canonical(payload),
                "now": now,
                "expires": now + timedelta(days=source_row["retention_days"]),
            },
        )
        version = (
            c.execute(
                text("""SELECT version_id,received_at,expires_at FROM direction_1d_source_version
          WHERE fund_code=:code AND source_id=:source AND kind='NAV' AND business_date=:day AND content_hash=:hash"""),
                {"code": code, "source": source_row["source_id"], "day": row["nav_date"], "hash": h},
            )
            .mappings()
            .one()
        )
        if version["expires_at"] <= now:
            raise ValueError("EVIDENCE_EXPIRED")
        result.append(
            {
                **payload,
                "version_id": str(version["version_id"]),
                "content_hash": h,
                "observed_at": version["received_at"].isoformat(),
                "expires_at": version["expires_at"].isoformat(),
            }
        )
    return result


def save_snapshot(c, kind, key, payload, now, expires):
    sid, h = uuid4(), digest(payload)
    c.execute(
        text("""INSERT INTO direction_1d_snapshot(snapshot_id,kind,task_key,as_of,payload,content_hash,expires_at)
      VALUES(:id,:kind,:key,:now,CAST(:payload AS jsonb),:hash,:expires)"""),
        {"id": sid, "kind": kind, "key": key, "now": now, "payload": canonical(payload), "hash": h, "expires": expires},
    )
    return str(sid), h


def models(c):
    return [
        dict(r) for r in c.execute(text("SELECT * FROM direction_1d_model ORDER BY registered_at,model_id")).mappings()
    ]


def get_job(job_id):
    with get_engine().connect() as c:
        r = c.execute(text("SELECT * FROM direction_1d_job WHERE job_id=:id"), {"id": job_id}).mappings().first()
        return json.loads(canonical(dict(r))) if r else None
