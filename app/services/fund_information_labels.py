"""新分析的单位净值核对：基准未公布时仍保持原比较日期，答案不进入训练。"""

from datetime import date, datetime

from sqlalchemy import text

from app.db.session import get_engine
from app.repositories import direction_1d as repo
from app.services.direction_1d_protocol import ZONE, canonical, digest, label


def labels(body):
    now = repo.clock()
    target, base = date.fromisoformat(body["target_nav_date"]), date.fromisoformat(body["base_nav_date"])
    if target > now.astimezone(ZONE).date():
        return {"status": "PENDING_TARGET"}
    with get_engine().begin() as c:
        source = repo.source(c)
        if str(source["source_id"]) != body["input"]["source_id"]:
            raise ValueError("LABEL_SOURCE_CHANGED")
        rows = repo.navs(c, "002112", source["source_id"], base, target)
        by_day = {r["nav_date"]: r for r in rows}
        if base not in by_day or target not in by_day:
            return {"status": "PENDING_NAV"}
        a, b = repo.observe(c, "002112", source, [by_day[base], by_day[target]], now)
        key_hash = digest({"base": a["content_hash"], "target": b["content_hash"]})
        previous = (
            c.execute(
                text(
                    "SELECT snapshot_id,payload,content_hash FROM direction_1d_snapshot "
                    "WHERE kind='LABEL' AND task_key=:key AND payload->>'revision_key'=:hash ORDER BY as_of LIMIT 1"
                ),
                {"key": body["task_key"], "hash": key_hash},
            )
            .mappings()
            .first()
        )
        if previous:
            return {
                "snapshot_id": str(previous["snapshot_id"]),
                "content_hash": previous["content_hash"],
                "payload": previous["payload"],
                "payload_json": canonical(previous["payload"]),
            }
        frozen = next((v for v in body["input"]["values"] if v["nav_date"] == str(base)), None)
        payload = {
            **label(a["unit_nav"], b["unit_nav"]),
            "task_key": body["task_key"],
            "input_snapshot_id": body["input_snapshot_id"],
            "target_nav_date": str(target),
            "target_definition": body["target_definition"],
            "kind": "FORWARD_ORIGINAL",
            "status": "AVAILABLE",
            "base_source": a,
            "target_source": b,
            "label_observed_at": now.isoformat(),
            "base_revised": bool(frozen and a["content_hash"] != frozen["content_hash"]),
            "base_initially_missing": frozen is None,
            "training_eligible": False,
            "revision_key": key_hash,
            "event_status": "UNKNOWN",
            "events": [],
        }
        sid, content_hash = repo.save_snapshot(
            c, "LABEL", body["task_key"], payload, now, datetime.fromisoformat(body["expires_at"])
        )
    return {"snapshot_id": sid, "content_hash": content_hash, "payload": payload, "payload_json": canonical(payload)}
