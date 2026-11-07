"""对D盘全部内容寻址原件逐份核对数据库长度和SHA256；不解析或改写原件。"""

import argparse
import json
import os
import sqlite3
import time
from pathlib import Path

from app.core.config import get_settings
from app.core.workspace_paths import LEGACY_SOURCE_ROOT, SOURCE_ROOT
from sqlalchemy import create_engine, text

from scripts import verify_old_project_retirement as guard


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initialize-only", action="store_true")
    parser.add_argument("--retry-errors", action="store_true")
    parser.add_argument("--revalidate-storage", action="store_true")
    args = parser.parse_args()
    control = guard.WORKSPACE / "03-python/.local-runs/retirement-control-current"
    control.mkdir(parents=True, exist_ok=True)
    database = sqlite3.connect(control / "archive-verification.sqlite3")
    database.execute("PRAGMA journal_mode=WAL")
    database.execute("PRAGMA cache_size=-8192")
    database.execute("CREATE TABLE IF NOT EXISTS objects (hash TEXT PRIMARY KEY, relative TEXT, size INTEGER, "
                     "state TEXT, actual TEXT, error TEXT) WITHOUT ROWID")
    database.execute("CREATE TABLE IF NOT EXISTS metadata (name TEXT PRIMARY KEY, value TEXT)")
    settings = get_settings()
    if not database.execute("SELECT 1 FROM metadata WHERE name='initialized'").fetchone():
        engine = create_engine(settings.ai_database_url, connect_args={
            "connect_timeout": 5, "options": "-c default_transaction_read_only=on -c statement_timeout=30000",
        })
        with engine.connect() as connection:
            result = connection.execution_options(stream_results=True).execute(text(
                "SELECT l.location,l.source_hash,c.source_size FROM material_source_location l "
                "JOIN material_extracted_content c ON c.source_hash=l.source_hash "
                "WHERE starts_with(l.location,:old) OR starts_with(l.location,:new)"
            ), {"old": str(LEGACY_SOURCE_ROOT / "originals") + os.sep,
                "new": str(SOURCE_ROOT / "originals") + os.sep})
            batch = []
            for location, source_hash, size in result:
                path = Path(location)
                base = LEGACY_SOURCE_ROOT if path.is_relative_to(LEGACY_SOURCE_ROOT) else SOURCE_ROOT
                batch.append((source_hash, path.relative_to(base).as_posix(), int(size)))
                if len(batch) == 2000:
                    database.executemany("INSERT OR IGNORE INTO objects(hash,relative,size) VALUES(?,?,?)", batch)
                    database.commit()
                    batch.clear()
            database.executemany("INSERT OR IGNORE INTO objects(hash,relative,size) VALUES(?,?,?)", batch)
            connection.rollback()
        engine.dispose()
        database.execute("INSERT INTO metadata VALUES('initialized','true')")
        database.commit()
    total = database.execute("SELECT count(*) FROM objects").fetchone()[0]
    print(json.dumps({"archive_expected_objects": total, "initialize_only": args.initialize_only}), flush=True)
    if args.initialize_only:
        database.close()
        return
    main_index = sqlite3.connect(control / "file-verification.sqlite3")
    expected = main_index.execute("SELECT value FROM metadata WHERE name='workspace_device'").fetchone()
    main_index.close()
    if expected:
        guard.EXPECTED_WORKSPACE_DEVICE = int(expected[0])
    guard.require_workspace()
    if args.revalidate_storage:
        # 断连后只复用数据库期望索引，原件实际摘要必须在当前连接重新逐份读取。
        database.execute("UPDATE objects SET state=NULL,actual=NULL,error=NULL")
        database.execute("INSERT OR REPLACE INTO metadata VALUES('storage_revalidation_started',?)",
                         (str(time.time()),))
        database.commit()
    elif args.retry_errors:
        database.execute("UPDATE objects SET state=NULL,error=NULL WHERE state='error'")
        database.commit()
    last, cursor = time.monotonic(), ""
    while True:
        rows = database.execute("SELECT hash,relative,size FROM objects WHERE state IS NULL AND hash>? "
                                "ORDER BY hash LIMIT 256", (cursor,)).fetchall()
        if not rows:
            break
        guard.require_workspace()
        for source_hash, relative, size in rows:
            path = SOURCE_ROOT / relative
            try:
                before = guard.io_path(path).stat()
                actual = guard.digest(path).hex()
                after = guard.io_path(path).stat()
                if before.st_size != size or actual != source_hash:
                    raise ValueError("ARCHIVE_OBJECT_HASH_OR_SIZE_MISMATCH")
                if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                    raise ValueError("ARCHIVE_OBJECT_CHANGED_DURING_READ")
                database.execute("UPDATE objects SET state='verified',actual=?,error=NULL WHERE hash=?",
                                 (actual, source_hash))
            except Exception as error:
                database.execute("UPDATE objects SET state='error',error=? WHERE hash=?",
                                 (type(error).__name__ + ":" + str(error), source_hash))
                if guard.storage_failure(error):
                    database.commit()
                    raise
        database.commit()
        cursor = rows[-1][0]
        if time.monotonic() - last > 25:
            states = dict(database.execute("SELECT coalesce(state,'pending'),count(*) FROM objects GROUP BY state"))
            print(json.dumps({"archive_states": states}), flush=True)
            last = time.monotonic()
    actual_count = 0
    extra = []
    with os.scandir(SOURCE_ROOT / "originals") as groups:
        for group in groups:
            if not group.is_dir(follow_symlinks=False):
                extra.append(group.name)
                continue
            with os.scandir(group.path) as entries:
                for item in entries:
                    actual_count += 1
                    if not database.execute("SELECT 1 FROM objects WHERE hash=?", (Path(item.name).stem,)).fetchone():
                        extra.append(group.name + "/" + item.name)
    states = dict(database.execute("SELECT coalesce(state,'pending'),count(*) FROM objects GROUP BY state"))
    report = {"complete": states.get("verified", 0) == total and actual_count == total and not extra,
              "expected_objects": total, "physical_objects": actual_count, "states": states, "extra": extra,
              "originals_changed_or_deleted_by_verifier": False}
    (control / "archive-verification.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report), flush=True)
    database.close()
    if not report["complete"]:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
