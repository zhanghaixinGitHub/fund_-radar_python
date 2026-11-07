"""用真实只读库及一份小型归档PDF验证混合读取，不重新采集、不写资料表或原件。"""

import hashlib
import json
import os
import sys
from pathlib import Path

from app.core.config import get_settings
from app.core.workspace_paths import LEGACY_PYTHON_ROOT, PYTHON_ROOT, SOURCE_ROOT
from sqlalchemy import text
from sqlalchemy.engine import make_url


def main():
    settings = get_settings()
    url = make_url(settings.ai_database_url).update_query_dict({
        "options": "-c default_transaction_read_only=on -c statement_timeout=10000",
    })
    os.environ["AI_DATABASE_URL"] = url.render_as_string(hide_password=False)
    get_settings.cache_clear()
    from app.db.session import get_engine
    from app.services import material_content as store
    from app.services.material_file_access import original_bytes

    phase = "cache"
    forbidden = []

    def audit(event, args):
        if event != "open" or not args or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        path = os.path.abspath(os.fsdecode(args[0])).removeprefix("\\\\?\\").casefold()
        if path.startswith(str(LEGACY_PYTHON_ROOT).casefold() + os.sep):
            forbidden.append("old_project")
            raise PermissionError("OLD_PROJECT_READ_BLOCKED")
        if phase == "cache" and path.startswith(str(SOURCE_ROOT).casefold() + os.sep):
            forbidden.append("original_during_cache_hit")
            raise PermissionError("CACHE_HIT_ORIGINAL_READ_BLOCKED")
        mode, flags = args[1:3]
        writing = isinstance(mode, str) and any(c in mode for c in "wax+")
        writing = writing or isinstance(flags, int) and bool(
            flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND)
        )
        if writing and (path.startswith("c:") or path.startswith(str(SOURCE_ROOT).casefold() + os.sep)):
            forbidden.append("source_or_C_write")
            raise PermissionError("SOURCE_OR_C_WRITE_BLOCKED")

    sys.addaudithook(audit)
    with get_engine().connect() as connection:
        assert connection.exec_driver_sql("SHOW transaction_read_only").scalar() == "on"
        row = connection.execute(text(
            "SELECT l.location,c.source_hash,c.source_size FROM material_source_location l "
            "JOIN material_extracted_content c ON c.source_hash=l.source_hash "
            "WHERE starts_with(l.location,:root) AND c.format='PDF' AND c.source_size BETWEEN 1000 AND 500000 "
            "AND EXISTS (SELECT 1 FROM material_page_representation p WHERE p.source_hash=c.source_hash "
            "AND p.engine='SOURCE_ARCHIVE_ROUTE_V1') ORDER BY c.source_size LIMIT 1"
        ), {"root": str(LEGACY_PYTHON_ROOT) + os.sep}).first()
        connection.rollback()
    if row is None:
        raise ValueError("SMALL_ARCHIVED_PDF_NOT_AVAILABLE")
    legacy, identity, size = row
    path = PYTHON_ROOT / Path(legacy).relative_to(LEGACY_PYTHON_ROOT)
    cached = store.source_bytes(path)
    assert isinstance(cached, store.SourceContent) and cached.source_hash == identity
    with store.database_only(), store.document_reader(cached) as reader:
        cached_pages = len(reader.pages)
        reader.pages[0].extract_text()
    phase = "original"
    raw = original_bytes(path, expected_hash=identity)
    assert len(raw) == size and hashlib.sha256(raw).hexdigest() == identity
    with store.document_reader(raw) as reader:
        original_pages = len(reader.pages)
        reader.pages[0].extract_text()
    assert original_pages == cached_pages
    report = {"database_read_only": True, "cache_hit_did_not_open_archive": True,
              "original_read_on_demand_hash_and_size_verified": True,
              "source_hash": identity, "source_size": size, "pages": original_pages,
              "forbidden_attempts": forbidden, "new_source_or_database_content_written": False}
    output = PYTHON_ROOT / ".local-runs/mixed-reading-20261107"
    output.mkdir(parents=True, exist_ok=True)
    (output / "real-mixed-readback.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
