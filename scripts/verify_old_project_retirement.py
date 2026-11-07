"""逐文件核验旧工程的不可重建内容；只补齐D盘缺项，差异另存，不改写或删除C盘。"""

import argparse
import hashlib
import json
import os
import shutil
import sqlite3
import stat
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from uuid import uuid4

WORKSPACE = Path(__file__).resolve().parents[2]
CONTROL = WORKSPACE / "03-python/.local-runs/old-project-retirement"
PAIRS = (
    ("01-vue", Path("C:/WebStormProject/workSpace05")),
    ("02-java", Path("C:/ideaProject/workSpace12")),
    ("03-python", Path("C:/pythonProject/workSpace06")),
)
# 只排除工程根部可重建的依赖/构建产物，以及任意层级Python缓存。
# data、tmp、.local-runs、.idea、.env、Git元数据和隐藏业务文件均纳入。
REBUILDABLE = {"node_modules", ".venv", ".tools", "target", "dist", ".pytest_cache", ".ruff_cache"}
ROOTS = dict(PAIRS)
EXPECTED_WORKSPACE_DEVICE = None


class WorkspaceStorageUnavailable(OSError):
    """外接盘掉线或卷身份变化；立即停止，不按普通文件缺失继续补写。"""


def write_report(path, value):
    """元数据先写独立临时名再替换；仅短暂共享占用重试，真实设备错误直接上抛。"""
    target = io_path(path)
    temporary = target.with_name(target.name + "." + uuid4().hex + ".tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(json.dumps(value, ensure_ascii=False, indent=2).encode("utf-8"))
            stream.flush()
            os.fsync(stream.fileno())
        for attempt in range(4):
            try:
                os.replace(temporary, target)
                return
            except PermissionError:
                if attempt == 3:
                    raise
                time.sleep(0.05 * (attempt + 1))
    finally:
        temporary.unlink(missing_ok=True)


def require_workspace():
    try:
        device = io_path(WORKSPACE).stat().st_dev
    except OSError as error:
        raise WorkspaceStorageUnavailable("WORKSPACE_DRIVE_UNAVAILABLE") from error
    if EXPECTED_WORKSPACE_DEVICE is not None and device != EXPECTED_WORKSPACE_DEVICE:
        raise WorkspaceStorageUnavailable("WORKSPACE_VOLUME_IDENTITY_CHANGED")
    return device


def storage_failure(error):
    if isinstance(error, WorkspaceStorageUnavailable):
        return True
    if isinstance(error, OSError) and getattr(error, "winerror", None) in {21, 23, 55, 121, 433, 995, 1117, 1167}:
        return True
    try:
        require_workspace()
    except WorkspaceStorageUnavailable:
        return True
    return False


def io_path(path):
    """Windows扩展长度路径，防止迁往更深的D盘目录后漏掉历史长文件名。"""
    value = os.path.abspath(path)
    return Path("\\\\?\\" + value) if os.name == "nt" and not value.startswith("\\\\?\\") else Path(value)


def digest(path):
    """流式读取原文件；不写正文，不把大文件整体放进内存。"""
    with io_path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").digest()


def inventory(source):
    """按路径顺序枚举，减少外接机械盘随机寻道；不跟随重解析目录。"""
    stack = [source]
    while stack:
        directory = stack.pop()
        with os.scandir(io_path(directory)) as items:
            entries = sorted(items, key=lambda item: item.name.casefold())
        children = []
        for item in entries:
            relative = directory.relative_to(source) / item.name
            if item.is_symlink():
                raise ValueError("SOURCE_LINK_REQUIRES_REVIEW:" + str(relative))
            if item.is_dir(follow_symlinks=False):
                if item.name == "__pycache__" or (len(relative.parts) == 1 and item.name in REBUILDABLE):
                    continue
                children.append(directory / item.name)
            elif item.is_file(follow_symlinks=False) and not item.name.endswith((".pyc", ".pyo")):
                stat = item.stat(follow_symlinks=False)
                yield relative.as_posix(), stat.st_size, stat.st_mtime_ns
        stack.extend(reversed(children))


def destination(repo, relative):
    """旧Git树独立保留，避免覆盖D盘新的分支、检查点和索引。"""
    return WORKSPACE / repo / relative


def checked_copy(source, target, expected):
    """只创建缺失目标并独立读回；既有目标不覆盖、不删除。"""
    require_workspace()
    source, target = io_path(source), io_path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if digest(target) != expected:
            raise ValueError("RETIREMENT_EXISTING_BACKUP_CHANGED")
        return
    # xb防止其他进程在存在性检查后创建同名文件而被覆盖。
    with source.open("rb") as left, target.open("xb") as right:
        shutil.copyfileobj(left, right, 1024 * 1024)
        right.flush()
        os.fsync(right.fileno())
    if digest(target) != expected:
        raise ValueError("RETIREMENT_COPY_READBACK_FAILED")
    shutil.copystat(source, target)


def compare(row):
    require_workspace()
    repo, relative, size, modified = row[:4]
    source = ROOTS[repo] / relative
    initial = io_path(source).stat()
    if (initial.st_size, initial.st_mtime_ns) != (size, modified):
        raise ValueError("SOURCE_CHANGED_AFTER_FREEZE")
    # 换接口后仅复用已冻结且元数据未变化的C盘摘要，D盘内容仍重新完整读取。
    before = row[4] if len(row) > 4 and row[4] is not None else digest(source)
    target = destination(repo, relative)
    try:
        target_info = io_path(target).stat()
    except FileNotFoundError:
        target_info = None
    if target_info is not None and target_info.st_dev != require_workspace():
        raise ValueError("TARGET_NOT_ON_WORKSPACE_VOLUME")
    if target_info is None:
        if relative == ".git" or relative.startswith(".git/"):
            target = CONTROL / "retained-git" / repo / relative
        checked_copy(source, target, before)
        state, retained = "git-retained" if relative.startswith(".git/") else "restored", str(target)
    elif stat.S_ISREG(target_info.st_mode) and target_info.st_size == size and digest(target) == before:
        state, retained = "equal", ""
    else:
        retained = str(CONTROL / "retained-source-versions" / repo / before.hex() / relative)
        checked_copy(source, Path(retained), before)
        state = "different-retained"
    final = io_path(source).stat()
    if (final.st_size, final.st_mtime_ns) != (size, modified):
        raise ValueError("SOURCE_CHANGED_DURING_HASH")
    return repo, relative, state, before, retained


def main():
    global EXPECTED_WORKSPACE_DEVICE
    parser = argparse.ArgumentParser(description=__doc__)
    # 当前新检查点只写D盘；早期C盘故障断点先保留，不静默选用D盘旧的同名索引。
    parser.add_argument("--control-dir", type=Path,
                        default=WORKSPACE / "03-python/.local-runs/retirement-control-current")
    parser.add_argument("--workers", type=int, choices=(1, 2), default=1)
    parser.add_argument("--retry-errors", action="store_true")
    parser.add_argument("--revalidate-storage", action="store_true")
    args = parser.parse_args()
    index_root = args.control_dir.absolute()
    if os.name == "nt" and index_root.drive.upper() != "D:":
        raise ValueError("RETIREMENT_CONTROL_MUST_BE_ON_D_DRIVE")
    index_root.mkdir(parents=True, exist_ok=True)
    CONTROL.mkdir(parents=True, exist_ok=True)
    database = sqlite3.connect(index_root / "file-verification.sqlite3")
    database.execute("PRAGMA journal_mode=WAL")
    database.execute("PRAGMA synchronous=FULL")
    database.execute("PRAGMA cache_size=-16384")
    database.execute("CREATE TABLE IF NOT EXISTS files (repo TEXT, path TEXT, size INTEGER, modified INTEGER, "
                     "state TEXT, sha BLOB, retained TEXT, error TEXT, PRIMARY KEY(repo,path)) WITHOUT ROWID")
    database.execute("CREATE TABLE IF NOT EXISTS metadata (name TEXT PRIMARY KEY, value TEXT)")
    current_device = require_workspace()
    saved_device = database.execute("SELECT value FROM metadata WHERE name='workspace_device'").fetchone()
    if saved_device is not None and int(saved_device[0]) != current_device:
        raise WorkspaceStorageUnavailable("WORKSPACE_VOLUME_IDENTITY_CHANGED")
    EXPECTED_WORKSPACE_DEVICE = current_device
    database.execute("INSERT OR IGNORE INTO metadata VALUES('workspace_device',?)", (str(current_device),))
    if args.retry_errors:
        database.execute("UPDATE files SET state=NULL,error=NULL WHERE state='error'")
    if args.revalidate_storage:
        database.execute("UPDATE files SET state=NULL,error=NULL")
        database.execute("INSERT OR REPLACE INTO metadata VALUES('storage_revalidation_started',?)",
                         (str(time.time()),))
    database.commit()
    started = time.monotonic()
    last_progress = 0.0
    counts_cache = None

    def progress(stage):
        nonlocal last_progress
        counts = counts_cache if counts_cache is not None else {str(state): count for state, count in database.execute(
            "SELECT coalesce(state,'pending'),count(*) FROM files GROUP BY state"
        )}
        result = {"stage": stage, "states": counts, "elapsed_seconds": round(time.monotonic() - started, 1)}
        if index_root.drive.upper() == "C:":
            control_bytes = sum(p.stat().st_size for p in index_root.iterdir() if p.is_file())
            if control_bytes > 512 * 1024 * 1024:
                raise ValueError("CONTROL_DIRECTORY_BUDGET_EXCEEDED")
        write_report(index_root / "progress.json", result)
        print(json.dumps(result, ensure_ascii=False), flush=True)
        last_progress = time.monotonic()

    if not database.execute("SELECT 1 FROM metadata WHERE name='inventory_complete'").fetchone():
        for repo, source in PAIRS:
            batch = []
            for relative, size, modified in inventory(source):
                batch.append((repo, relative, size, modified))
                if len(batch) == 2000:
                    database.executemany("INSERT OR IGNORE INTO files(repo,path,size,modified) VALUES(?,?,?,?)", batch)
                    database.commit()
                    batch.clear()
                    if time.monotonic() - last_progress > 25:
                        progress("inventory")
            database.executemany("INSERT OR IGNORE INTO files(repo,path,size,modified) VALUES(?,?,?,?)", batch)
            database.commit()
            progress("inventory:" + repo)
        database.execute("INSERT OR REPLACE INTO metadata VALUES('inventory_complete','true')")
        database.commit()
    progress("hashing")
    counts_cache = dict(database.execute("SELECT coalesce(state,'pending'),count(*) FROM files GROUP BY state"))
    # 两个工作线程、有界批次，避免在内存有限的机器上加载百万条路径或整批正文。
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        cursor_repo, cursor_path = "", ""
        while True:
            batch = database.execute("SELECT repo,path,size,modified,sha FROM files WHERE state IS NULL "
                                     "AND (repo,path)>(?,?) ORDER BY repo,path LIMIT 256",
                                     (cursor_repo, cursor_path)).fetchall()
            if not batch:
                break
            futures = [pool.submit(compare, row) for row in batch]
            for row, future in zip(batch, futures, strict=True):
                try:
                    repo, relative, state, sha, retained = future.result()
                    database.execute("UPDATE files SET state=?,sha=?,retained=?,error=NULL WHERE repo=? AND path=?",
                                     (state, sha, retained, repo, relative))
                    counts_cache[state] = counts_cache.get(state, 0) + 1
                except Exception as error:
                    database.execute("UPDATE files SET state='error',error=? WHERE repo=? AND path=?",
                                     (type(error).__name__ + ":" + str(error), row[0], row[1]))
                    counts_cache["error"] = counts_cache.get("error", 0) + 1
                    if storage_failure(error):
                        counts_cache["pending"] -= 1
                        for pending in futures:
                            pending.cancel()
                        pool.shutdown(wait=True, cancel_futures=True)
                        database.commit()
                        progress("awaiting-stable-storage")
                        result = {"complete": False, "reason": "DEVICE_IO_FAILURE",
                                  "source_projects_preserved": True, "control_database": str(index_root)}
                        write_report(index_root / "historical-verification.json", result)
                        database.close()
                        raise SystemExit(2) from None
                counts_cache["pending"] -= 1
            database.commit()
            cursor_repo, cursor_path = batch[-1][:2]
            if time.monotonic() - last_progress > 25:
                progress("hashing")
    progress("hashing-complete")
    # 全部内容已核后，重枚举C盘文件元数据，识别核验期间新生成或变化的文件。
    changed = []
    for repo, source in PAIRS:
        for relative, size, modified in inventory(source):
            existing = database.execute("SELECT size,modified FROM files WHERE repo=? AND path=?",
                                        (repo, relative)).fetchone()
            if existing != (size, modified):
                changed.append((repo, relative, size, modified))
    if changed:
        database.executemany("INSERT INTO files(repo,path,size,modified) VALUES(?,?,?,?) ON CONFLICT(repo,path) "
                             "DO UPDATE SET size=excluded.size,modified=excluded.modified,"
                             "state=NULL,sha=NULL,error=NULL",
                             changed)
        database.commit()
    errors = database.execute("SELECT count(*) FROM files WHERE state='error'").fetchone()[0]
    result = {"complete": not changed and errors == 0, "source_changes_requiring_resume": len(changed),
              "errors": errors, "files": database.execute("SELECT count(*) FROM files").fetchone()[0],
              "bytes": database.execute("SELECT sum(size) FROM files").fetchone()[0],
              "states": dict(database.execute("SELECT coalesce(state,'pending'),count(*) FROM files GROUP BY state")),
              "excluded_rebuildable_root_directories": sorted(REBUILDABLE),
              "source_roots_preserved": True}
    write_report(index_root / "historical-verification.json", result)
    print(json.dumps(result), flush=True)
    database.close()
    if not result["complete"]:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
