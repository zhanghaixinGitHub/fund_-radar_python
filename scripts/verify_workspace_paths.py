"""在数据库强制只读、禁止打开旧工程与原件的条件下验证迁移兼容。"""

import builtins
import io
import json
import os
from pathlib import Path
from unittest.mock import patch

from app.core.config import get_settings
from app.core.workspace_paths import LEGACY_PYTHON_ROOT, LEGACY_SOURCE_ROOT, PYTHON_ROOT
from app.services import material_content as store
from app.services import material_source_archive as archive
from sqlalchemy import create_engine, text


def main():
    """验证实际资料底稿与不可变路由；输出只包含计数、摘要和工程路径。"""
    settings = get_settings()
    engine = create_engine(settings.ai_database_url, connect_args={
        "connect_timeout": 5, "options": "-c default_transaction_read_only=on -c statement_timeout=30000",
    })
    repo = store.MaterialRepository(engine)
    with engine.connect() as connection:
        aliases = connection.execute(text(
            "SELECT location,source_hash FROM material_source_location WHERE starts_with(location,:root) LIMIT 600"
        ), {"root": str(LEGACY_PYTHON_ROOT / "data") + os.sep}).all()
        routes = connection.execute(text(
            "SELECT source_hash,content_hash FROM material_page_representation WHERE engine=:engine LIMIT 8"
        ), {"engine": archive.ROUTE_ENGINE}).all()
        readonly = connection.exec_driver_sql("SHOW transaction_read_only").scalar()
        connection.rollback()
    candidates = []
    for old, source_hash in aliases:
        new = PYTHON_ROOT / Path(old).relative_to(LEGACY_PYTHON_ROOT)
        if new.is_file():
            candidates.append((Path(old), new, source_hash))
        if len(candidates) == 3:
            break
    if not candidates or not routes:
        raise ValueError("WORKSPACE_VERIFICATION_NO_REAL_SAMPLES")
    blocked = [str(Path(p).absolute()).casefold() for p in (
        LEGACY_PYTHON_ROOT, LEGACY_SOURCE_ROOT, Path(settings.fund_source_archive_root),
        Path("C:/ideaProject/workSpace12"), Path("C:/WebStormProject/workSpace05"),
    )]
    attempted = []

    def guard(original):
        def opened(file, *args, **kwargs):
            if isinstance(file, (str, bytes, os.PathLike)):
                value = os.path.abspath(os.fsdecode(file)).casefold()
                if any(value == root or value.startswith(root + os.sep) for root in blocked):
                    attempted.append(value)
                    raise AssertionError("WORKSPACE_VERIFICATION_ORIGINAL_OPEN_FORBIDDEN")
            return original(file, *args, **kwargs)
        return opened

    route_hashes = []
    with (
        patch("builtins.open", guard(builtins.open)),
        patch("io.open", guard(io.open)),
        patch("os.open", guard(os.open)),
    ):
        with store.using_repository(repo), store.database_only():
            for old, new, expected in candidates:
                left, right = store.source_bytes(old), store.source_bytes(new)
                if left.source_hash != expected or right.source_hash != expected:
                    raise ValueError("WORKSPACE_VERIFICATION_SOURCE_IDENTITY_CHANGED")
                if store.encoded(left.extracted) != store.encoded(right.extracted):
                    raise ValueError("WORKSPACE_VERIFICATION_CONTENT_CHANGED")
            for source_hash, receipt_hash in routes:
                before = repo.page(source_hash, archive.ROUTE_ENGINE, 0)
                route = archive.archive_route(source_hash, repo)
                value = store.source_bytes(route["archive_path"])
                if value.source_hash != source_hash or store.checksum(before.encode()) != receipt_hash:
                    raise ValueError("WORKSPACE_VERIFICATION_ARCHIVE_IDENTITY_CHANGED")
                if repo.page(source_hash, archive.ROUTE_ENGINE, 0) != before:
                    raise ValueError("WORKSPACE_VERIFICATION_RECEIPT_CHANGED")
                route_hashes.append(source_hash)
    if attempted:
        raise ValueError("WORKSPACE_VERIFICATION_ORIGINAL_OPEN_ATTEMPTED")
    engine.dispose()
    result = {"transaction_read_only": readonly, "project": str(PYTHON_ROOT),
              "archive_root": settings.fund_source_archive_root, "real_copied_file_pairs": len(candidates),
              "real_archive_routes": len(routes), "old_project_or_original_open_attempts": len(attempted),
              "route_source_hashes": route_hashes, "immutable_receipts_unchanged": True}
    output = PYTHON_ROOT / ".local-runs/workspace-relocation/readback-verification.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
