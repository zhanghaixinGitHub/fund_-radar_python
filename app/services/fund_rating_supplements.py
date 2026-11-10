"""补件待办独立留痕。只有资料核验回执能改变状态。"""

import json
import re
import sqlite3
from contextlib import closing
from datetime import datetime
from pathlib import Path

from app.services.direction_1d_protocol import canonical, digest
from app.services.fund_exposure_common import now
from app.services.material_content import content_digest, source_bytes, source_size

ACTIONABLE = {"READY_TO_PARSE", "RETRY"}
WAITING = "WAITING_EVIDENCE"
CLOSED = {"VERIFIED", "NOT_APPLICABLE", "NO_HISTORY"}


def record(database: Path, receipt: dict, *, sources_root: Path) -> bool:
    """一个份额、月份、字段下追加核验记录。重复回执不覆盖更新记录。"""
    code, period, field = (receipt[key] for key in ("fund_code", "period", "field"))
    state = receipt["state"]
    if not re.fullmatch(r"[0-9]{6}", code) or not re.fullmatch(r"[a-z][a-z0-9_]{0,79}", field):
        raise ValueError("RATING_SUPPLEMENT_IDENTITY")
    datetime.strptime(period, "%Y-%m")
    if not re.fullmatch(r"[0-9]{4}-[0-9]{2}", period):
        raise ValueError("RATING_SUPPLEMENT_PERIOD")
    if state not in ACTIONABLE | CLOSED | {WAITING} or not all(receipt[k].strip() for k in ("owner", "note")):
        raise ValueError("RATING_SUPPLEMENT_RECEIPT")
    next_at = None
    if state in ACTIONABLE:
        due = datetime.fromisoformat(receipt["next_at"])
        if due.tzinfo is None:
            raise ValueError("RATING_SUPPLEMENT_NEXT_AT")
        next_at = due.timestamp()
    if state == WAITING and not receipt.get("resume_when", "").strip():
        raise ValueError("RATING_SUPPLEMENT_RESUME_WHEN")
    evidence = receipt.get("sources", [])
    if state in CLOSED | {"READY_TO_PARSE", WAITING} and not evidence:
        raise ValueError("RATING_SUPPLEMENT_EVIDENCE_REQUIRED")
    root = sources_root.resolve()
    for ref in evidence:
        path = (root / ref["file"]).resolve()
        if not path.is_relative_to(root) or not 0 < source_size(path) <= 25_000_000:
            raise ValueError("RATING_SUPPLEMENT_SOURCE_SCOPE")
        if content_digest(source_bytes(path)) != ref["sha256"]:
            raise ValueError("RATING_SUPPLEMENT_SOURCE_HASH")

    database.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(database)) as connection, connection:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS receipt (
                id INTEGER PRIMARY KEY, hash TEXT UNIQUE NOT NULL,
                fund_code TEXT NOT NULL, period TEXT NOT NULL, field TEXT NOT NULL,
                checked_at TEXT NOT NULL, next_at REAL, payload TEXT NOT NULL
            )
        """)
        cursor = connection.execute(
            """INSERT INTO receipt (hash, fund_code, period, field, checked_at, next_at, payload)
               VALUES (?, ?, ?, ?, ?, ?, ?) ON CONFLICT(hash) DO NOTHING""",
            (digest(receipt), code, period, field, now().isoformat(), next_at, canonical(receipt)),
        )
        return cursor.rowcount == 1


def report(database: Path) -> dict:
    if database.exists():
        with closing(sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True)) as connection:
            rows = connection.execute("""
                SELECT payload, checked_at, next_at FROM receipt
                WHERE id IN (SELECT max(id) FROM receipt GROUP BY fund_code, period, field)
                ORDER BY fund_code, period, field
            """).fetchall()
    else:
        rows = []
    timestamp = now().timestamp()
    items = [
        {**json.loads(payload), "checked_at": checked_at, "due": next_at is not None and next_at <= timestamp}
        for payload, checked_at, next_at in rows
    ]
    return {
        "ready_to_parse": sum(item["state"] == "READY_TO_PARSE" for item in items),
        "due": sum(item["due"] for item in items),
        "open": sum(item["state"] not in CLOSED for item in items),
        "items": items,
    }
