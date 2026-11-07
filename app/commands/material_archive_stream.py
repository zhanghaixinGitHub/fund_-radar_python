"""全基金原件迁移：D盘顺序凭据及内存索引，避免外接盘SQLite随机读写与临时排序。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import stat
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import contextmanager
from contextvars import copy_context
from pathlib import Path

from sqlalchemy import select, text

from app.commands.material_source_archive import save
from app.models.material_content import MaterialContent, MaterialLocation
from app.services import material_archive_scope as scope
from app.services import material_content as store
from app.services import material_source_archive as archive
from app.services.material_archive_batch import capture_unindexed, group_destination, move_group
from app.services.material_archive_guard import material_writes_only
from app.services.material_archive_io import assert_deletion_allowed, control_path, critical_io, record_fault
from app.services.material_file_cleanup import physical_path

CONTROL_BUDGET_BYTES = 512 * 1024 * 1024


@contextmanager
def run_lock(run, action):
    """同批同动作只有一个写者；本机工程锁不受D盘掉线影响，退出即释放。

    只把批次路径/动作的摘要和一个锁字节放在控制目录，不保存原件或正文。
    新旧执行模式使用同一锁位置，不能因切换工程流水位置而并发执行迁移。
    """
    import msvcrt

    key = store.checksum((os.path.normcase(str(Path(run).resolve()))+"|"+action).encode())
    path = control_path().parent / "writer-locks" / (key+".lock")
    if path.parent.resolve() != path.parent:
        raise ValueError("MATERIAL_WRITER_LOCK_DIRECTORY_REDIRECTED")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        if handle.seek(0, os.SEEK_END) == 0:
            handle.write(b"1")
            handle.flush()
        handle.seek(0)
        try:
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError as error:
            raise RuntimeError("MATERIAL_BATCH_ACTION_ALREADY_RUNNING:"+action) from error
        try:
            yield
        finally:
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)


def journal_items(path, *, repair=True):
    """恢复本进程独占的顺序凭据：仅未刷完的尾行可截断，完整坏行必须中止。"""
    path = Path(path)
    if not path.exists():
        return
    valid = 0
    with path.open("rb") as lines:
        for line in lines:
            try:
                item = json.loads(line)
            except (json.JSONDecodeError, UnicodeDecodeError):
                if line.endswith(b"\n"):
                    raise ValueError("MATERIAL_BATCH_JOURNAL_CORRUPTED:"+path.name) from None
                if repair:
                    path.with_suffix(".incomplete-tail.bin").write_bytes(line)
                break
            valid += len(line)
            yield item
    if repair and valid != path.stat().st_size:
        with path.open("r+b") as output:
            output.truncate(valid)
            output.flush()
            os.fsync(output.fileno())


def state_root(run, control_dir=None):
    """冻结资格清单仍在D盘；可选本机控制目录只放工程事件，严格绑定同一批次。

    D盘反复在追加日志时失联后，允许用本机有界工程流水减少外接盘元数据写入。
    不复制原件、正文或整套冻结清单到C盘，不自动解除存储故障闸门。
    """
    run = Path(run)
    if control_dir is None:
        return run
    control = Path(control_dir).absolute()
    allowed = store.PROJECT / ".local-runs/material-migration-control"
    if not control.is_relative_to(allowed) or control == allowed or control.resolve() != control:
        raise ValueError("MATERIAL_CONTROL_DIRECTORY_OUTSIDE_ENGINEERING_SCOPE")
    control.mkdir(parents=True, exist_ok=True)
    plan_data = json.loads((run / "stream-plan.json").read_text(encoding="utf-8"))
    identity = {"source_plan_run": str(run.absolute()),
                "source_manifest_sha256": plan_data["source_manifest_sha256"],
                "metadata_budget_bytes": CONTROL_BUDGET_BYTES, "source_bytes_saved_on_C": False}
    binding = control / "control-run.json"
    if binding.exists():
        if json.loads(binding.read_text(encoding="utf-8")) != identity:
            raise ValueError("MATERIAL_CONTROL_RUN_IDENTITY_CHANGED")
    else:
        save(binding, identity)
    return control


def event_journals(run, output, name):
    """旧D盘事件只读，未刷完尾部不修写；本机新事件可按既有数据库凭据恢复。"""
    if output != run:
        yield run / name, False
    yield output / name, True


def control_metadata_bytes(output):
    """统计本机工程文件的逻辑占用，兼容报告原子替换时临时名字消失。

    报告写者会在枚举之后将临时文件改名；该路径已不存在时不再计入本次快照，
    下一组仍重新检查预算。只忽略这一种竞争，权限或设备I/O错误继续向上抛出。
    清单、来源和数据库凭据另有身份校验，不能用本统计跳过其缺失检查。
    """
    used = 0
    for path in output.iterdir():
        try:
            details = path.stat()
        except FileNotFoundError:
            continue
        if stat.S_ISREG(details.st_mode):
            used += details.st_size
    return used


def check_control_budget(output, run):
    used = control_metadata_bytes(output) if output != run else 0
    if used >= CONTROL_BUDGET_BYTES:
        raise RuntimeError("MATERIAL_ENGINEERING_METADATA_BUDGET_EXCEEDED")


def physical_files(root):
    """直接使用目录枚举已有的大小/时间信息，避免对百万文件逐个重新打开属性句柄。

    Windows DirEntry的inode字段为0，不能据此计算硬链接；硬链接和实体身份只在
    真正迁移时由LockedOriginal句柄读取。本阶段只使用FindData已有的长度/时间。
    """
    root = physical_path(root)
    try:
        root.stat()
    except FileNotFoundError:
        return
    pending = [root]
    while pending:
        directory = pending.pop()
        with os.scandir(directory) as entries:
            for entry in entries:
                stat = entry.stat(follow_symlinks=False)
                if stat.st_file_attributes & 0x400:
                    continue
                if entry.is_dir(follow_symlinks=False):
                    if entry.name not in scope.SKIP_DIRECTORIES and not entry.name.startswith("deps") \
                            and not entry.name.endswith("-deps"):
                        pending.append(Path(entry.path))
                else:
                    yield Path(str(entry.path).removeprefix("\\\\?\\")), stat


def plan(run):
    """全量盘点只顺序写D盘，正文不装内存；按用途保护全部快照和实验结果。"""
    run = Path(run).absolute()
    if not run.is_relative_to(archive.root_path() / "runs") or run.resolve() != run:
        raise ValueError("MATERIAL_ARCHIVE_RUN_OUTSIDE_EXTERNAL_DISK")
    run.mkdir(parents=True, exist_ok=True)
    if (run / "stream-plan.json").exists():
        return json.loads((run / "stream-plan.json").read_text(encoding="utf-8"))
    repo = store.repository()
    with repo.engine.connect() as c:
        if repo.engine.dialect.name == "postgresql":
            c.execute(text("SET TRANSACTION READ ONLY"))
        query = select(MaterialLocation.location.label("path"), MaterialLocation.source_hash,
                       MaterialContent.format.label("kind"), MaterialContent.content_hash,
                       MaterialContent.source_size.label("size")).join(
                           MaterialContent, MaterialContent.source_hash == MaterialLocation.source_hash
                       ).where(MaterialLocation.location.startswith(str(store.PROJECT) + os.sep, autoescape=True))
        records = c.execute(query.execution_options(stream_results=True)).mappings()
        aliases = {row["path"]: row for row in records}
    print(json.dumps({"source_metadata_loaded": len(aliases)}), flush=True)
    counters = defaultdict(lambda: [0, 0])
    kinds, hashes, unindexed, count = Counter(), set(), 0, 0
    source_digest, protected_digest = hashlib.sha256(), hashlib.sha256()
    source_file, protected_file = run / "source-plan-v3.jsonl", run / "protected-plan-v3.jsonl"
    if source_file.exists() or protected_file.exists():
        raise ValueError("MATERIAL_STREAM_PARTIAL_PLAN_EXISTS_REVIEW_CHECKPOINT")
    started = time.monotonic()
    with source_file.open("xb") as sources, protected_file.open("xb") as protected:
        for top in ("data", ".local-runs"):
            for path, stat in physical_files(store.PROJECT / top):
                    possible, reason = scope.classify(path, store.PROJECT, indexed=True)
                    indexed = aliases.get(str(path)) if possible else None
                    eligible, reason = scope.classify(path, store.PROJECT, indexed=indexed is not None)
                    if eligible and path.suffix.lower() in {".json", ".bin", ".raw", ""}:
                        with physical_path(path).open("rb") as body:
                            eligible, reason = scope.classify(path, store.PROJECT, indexed=indexed is not None,
                                                              prefix=body.read(8192))
                    item = {"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns, "reason": reason}
                    if eligible:
                        item.update({"source_hash": indexed["source_hash"] if indexed else None,
                                     "kind": indexed["kind"] if indexed else None,
                                     "content_hash": indexed["content_hash"] if indexed else None})
                        raw = store.encoded(item) + b"\n"
                        sources.write(raw)
                        source_digest.update(raw)
                        counters["SOURCE"][0] += 1
                        counters["SOURCE"][1] += stat.st_size
                        kinds[item["kind"] or "UNINDEXED"] += 1
                        if item["source_hash"]:
                            hashes.add(item["source_hash"])
                        else:
                            unindexed += 1
                    else:
                        raw = store.encoded(item) + b"\n"
                        protected.write(raw)
                        protected_digest.update(raw)
                        counters["PROTECTED"][0] += 1
                        counters["PROTECTED"][1] += stat.st_size
                    count += 1
                    if count % 100000 == 0:
                        print(json.dumps({"scanned": count, "sources": counters["SOURCE"][0],
                                          "seconds": round(time.monotonic()-started)}), flush=True)
        for output in (sources, protected):
            output.flush()
            os.fsync(output.fileno())
    result = {"totals": dict(counters), "kinds": dict(kinds), "known_unique_source_hashes": len(hashes),
              "unindexed_sources": unindexed, "source_manifest_sha256": source_digest.hexdigest(),
              "protected_manifest_sha256": protected_digest.hexdigest(),
              "metadata_index": "MEMORY_ONLY; durable sequential manifests on D",
              "protected_snapshots_and_experiment_outputs_unchanged": True}
    save(run / "stream-plan.json", result)
    return result


def supplemental_manifests(run):
    """补查可追加新批号；旧补充清单与摘要不覆盖，不靠修改原计划扩大授权路径。"""
    for marker in sorted(Path(run).glob("stream-supplement*-plan.json")):
        approved = json.loads(marker.read_text(encoding="utf-8"))
        suffix = marker.name.removeprefix("stream-supplement").removesuffix("-plan.json")
        if suffix and not (suffix.startswith("-") and suffix[1:].isdigit()):
            raise ValueError("MATERIAL_SUPPLEMENT_MANIFEST_NAME_INVALID")
        manifest = Path(run)/("source-plan-supplement"+suffix+".jsonl")
        with manifest.open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != approved["source_manifest_sha256"]:
                raise ValueError("MATERIAL_STREAM_SUPPLEMENT_CHANGED")
        yield manifest


def memory_index(run):
    """只把迁移资格与来源身份放内存索引；没有磁盘SQLite文件、WAL或C盘临时排序。"""
    run = Path(run)
    planned = json.loads((run / "stream-plan.json").read_text(encoding="utf-8"))
    manifest = run / "source-plan-v3.jsonl"
    with manifest.open("rb") as source:
        if hashlib.file_digest(source, "sha256").hexdigest() != planned["source_manifest_sha256"]:
            raise ValueError("MATERIAL_STREAM_PLAN_CHANGED")
    db = sqlite3.connect(":memory:")
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA temp_store=MEMORY")
    db.execute("CREATE TABLE files(path TEXT PRIMARY KEY,source_hash TEXT,kind TEXT,size INTEGER,mtime_ns INTEGER,"
               "content_hash TEXT,status TEXT DEFAULT 'PENDING')")
    with manifest.open(encoding="utf-8") as source:
        for line in source:
            item = json.loads(line)
            db.execute("INSERT INTO files(path,source_hash,kind,size,mtime_ns,content_hash) VALUES(?,?,?,?,?,?)",
                       tuple(item[k] for k in ["path", "source_hash", "kind", "size", "mtime_ns", "content_hash"]))
    for manifest in supplemental_manifests(run):
        with manifest.open(encoding="utf-8") as lines:
            for line in lines:
                item = json.loads(line)
                db.execute("INSERT INTO files(path,source_hash,kind,size,mtime_ns,content_hash) VALUES(?,?,?,?,?,?)",
                           tuple(item[k] for k in ["path", "source_hash", "kind", "size", "mtime_ns", "content_hash"]))
    db.execute("CREATE INDEX hashes ON files(source_hash,status)")
    db.commit()
    return db


def supplement(run):
    """复核初筛保护项中的已确认业务命名原件，另封补充清单，绝不修改原始冻结清单。"""
    run, repo = Path(run), store.repository()
    existing, versions = set(), list(supplemental_manifests(run))
    for manifest in versions:
        with manifest.open(encoding="utf-8") as lines:
            existing.update(json.loads(line)["path"] for line in lines)
    suffix = "-"+str(len(versions)+1) if versions else ""
    marker = run/("stream-supplement"+suffix+"-plan.json")
    planned = json.loads((run/"stream-plan.json").read_text(encoding="utf-8"))
    manifest = run/"protected-plan-v3.jsonl"
    with manifest.open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != planned["protected_manifest_sha256"]:
            raise ValueError("MATERIAL_STREAM_PROTECTION_PLAN_CHANGED")
    selected, digest = [], hashlib.sha256()
    with manifest.open(encoding="utf-8") as lines:
        for line in lines:
            item = json.loads(line)
            if item["path"] in existing:
                continue
            path = Path(item["path"])
            if path.suffix.lower() not in scope.DOCUMENT_SUFFIXES:
                continue
            candidate, reason = scope.classify(path, store.PROJECT, indexed=True)
            if not candidate:
                continue
            metadata = repo.metadata(path=path)
            eligible, reason = scope.classify(path, store.PROJECT, indexed=metadata is not None)
            if not eligible:
                continue
            actual = physical_path(path).stat()
            if actual.st_size != item["size"] or actual.st_mtime_ns != item["mtime_ns"]:
                raise ValueError("MATERIAL_SUPPLEMENT_SOURCE_CHANGED:"+str(path))
            item.update({"source_hash": metadata.source_hash if metadata else None,
                         "kind": metadata.kind if metadata else None,
                         "content_hash": metadata.content_hash if metadata else None, "reason": reason})
            selected.append(item)
    if not selected:
        return {"additional_files": 0, "previous_supplemental_files": len(existing), "old_manifests_unchanged": True}
    target = run/("source-plan-supplement"+suffix+".jsonl")
    with target.open("xb") as stream:
        for item in selected:
            body = store.encoded(item)+b"\n"
            stream.write(body)
            digest.update(body)
        stream.flush()
        os.fsync(stream.fileno())
    result = {"files": len(selected), "bytes": sum(x["size"] for x in selected),
              "unindexed": sum(x["source_hash"] is None for x in selected),
              "source_manifest_sha256": digest.hexdigest(), "old_manifests_unchanged": True,
              "historical_snapshots_experiment_outputs_and_frozen_evidence_kept": True}
    save(marker, result)
    return result


def write_event(stream, value):
    stream.write(json.dumps(value, ensure_ascii=False) + "\n")
    stream.flush()
    os.fsync(stream.fileno())


def move_source_group(items, repo, *, external_manifest=True):
    """同一来源串行退出；不同来源可有界并发。工作线程不修改内存索引或顺序日志。

    每组自身仍持有C/D实体锁和数据库事务。线程意外退出时，已提交的路径凭据
    可由下一次运行恢复；没有已提交凭据就绝不把缺文件当作迁移成功。
    """
    events, live = [], []
    try:
        for item in items:
            if physical_path(item["path"]).exists():
                live.append(item)
                continue
            with repo.engine.connect() as c:
                row = c.execute(text("SELECT source_hash,deleted_at FROM material_source_location WHERE location=:p"),
                                {"p": item["path"]}).mappings().one_or_none()
            target = archive.root_path()/"originals"/item["source_hash"][:2]/(
                item["source_hash"]+archive.SUFFIX.get(item["kind"], ".bin"))
            if row is None or not row["deleted_at"] or row["source_hash"] != item["source_hash"]:
                raise ValueError("MISSING_WITHOUT_COMMITTED_MOVE:"+item["path"])
            archive._verify(target, item["source_hash"], item["size"])
            events.append({"event": "RECOVERED", "path": item["path"], "source_hash": item["source_hash"],
                           "allocation_unknown": True})
        ids = [(s.st_dev, s.st_ino) for s in (os.stat(physical_path(x["path"])) for x in live)]
        step = 1 if len(set(ids)) != len(ids) else 64
        if live:
            with group_destination(live, repo) as destination:
                for offset in range(0, len(live), step):
                    chunk = live[offset:offset+step]
                    try:
                        result = move_group(chunk, approved_paths=[x["path"] for x in chunk], repo=repo,
                                            destination=destination, external_manifest=external_manifest)
                        events.append({"event": "MOVED", **result})
                    except Exception as error:
                        events.append({"event": "ERROR", "paths": [x["path"] for x in chunk],
                                       "fatal_storage_failure": critical_io(error),
                                       "error": type(error).__name__+":"+str(error)[:300]})
                        if critical_io(error):
                            # 同一来源可能有很多硬链接；设备故障后立即交回主流程落闸，
                            # 不能先继续尝试后续别名，再等整组处理完才停止。
                            break
    except Exception as error:
        events.append({"event": "ERROR", "paths": [x["path"] for x in items],
                       "fatal_storage_failure": critical_io(error),
                       "error": type(error).__name__+":"+str(error)[:300]})
    return events


def group_events(db, groups, repo, *, workers, external_manifest=True, before_group=None):
    """最多同时提交workers个来源，避免把十几万个Future/正文装进内存。"""
    def items(group):
        return [dict(row) for row in db.execute("SELECT * FROM files WHERE source_hash=? AND status='PENDING'",
                                                (group["source_hash"],))]
    if workers == 1:
        for group in groups:
            if before_group is not None:
                before_group()
            yield move_source_group(items(group), repo, external_manifest=external_manifest)
        return
    iterator = iter(groups)
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="material-migrate") as pool:
        pending = set()
        while True:
            while len(pending) < workers:
                try:
                    group = next(iterator)
                except StopIteration:
                    break
                pending.add(pool.submit(copy_context().run, move_source_group, items(group), repo,
                                        external_manifest=external_manifest))
            if not pending:
                break
            finished, pending = wait(pending, return_when=FIRST_COMPLETED)
            for future in finished:
                yield future.result()


def migrate(run, *, limit=None, workers=1, control_dir=None):
    """根据冻结顺序清单分组迁移，事件逐组刷盘；中断后由数据库凭据恢复，不猜分配量。"""
    run = Path(run)
    assert_deletion_allowed(run)
    output = state_root(run, control_dir)
    if control_dir is not None and workers != 1:
        raise ValueError("MATERIAL_LOCAL_CONTROL_REQUIRES_SINGLE_WORKER")
    db, repo = memory_index(run), store.repository()
    moved, reclaimed, failures = 0, 0, []
    archived = set()
    event_path = output / "stream-migration-events.jsonl"
    for journal_path, repair in event_journals(run, output, event_path.name):
        for obj in journal_items(journal_path, repair=repair):
            if obj.get("event") == "RECOVERED":
                db.execute("UPDATE files SET status='MOVED' WHERE path=?", (obj["path"],))
                archived.add(obj["source_hash"])
                continue
            if obj.get("event") != "MOVED":
                continue
            archived.add(obj["source_hash"])
            for item in obj["removed"]:
                db.execute("UPDATE files SET status='MOVED' WHERE path=?", (item["path"],))
                reclaimed += item["reclaimed_allocated_bytes"]
    if not (output / "stream-space-before.json").exists():
        import shutil

        save(output / "stream-space-before.json", {"C_free": shutil.disk_usage("C:/").free,
                                                "D_free": shutil.disk_usage(archive.root_path()).free})
    if not 1 <= workers <= 4:
        raise ValueError("MATERIAL_MIGRATION_WORKERS_OUTSIDE_BOUND")
    # 有界小批只处理已登记来源；无界执行时逐条识别未登记来源并留凭据。
    # 本机流水模式遇到首个异常即停止待查；不能跳过错误后把全量迁移报告为成功。
    if limit is None:
        identification = output / "stream-identification-events.jsonl"
        for source_journal, repair in event_journals(run, output, identification.name):
            for item in journal_items(source_journal, repair=repair):
                if item.get("event") == "IDENTIFIED":
                    db.execute("UPDATE files SET source_hash=?,kind=?,content_hash=? WHERE path=?",
                               tuple(item[k] for k in ["source_hash", "kind", "content_hash", "path"]))
        unknown = db.execute("SELECT * FROM files WHERE source_hash IS NULL AND status='PENDING'").fetchall()
        started_identification = time.monotonic()
        with identification.open("a", encoding="utf-8") as journal:
            for row in unknown:
                check_control_budget(output, run)
                try:
                    identified = capture_unindexed(dict(row), repo=repo)
                    write_event(journal, {"event": "IDENTIFIED", **identified})
                    db.execute("UPDATE files SET source_hash=?,kind=?,content_hash=? WHERE path=?",
                               tuple(identified[k] for k in ["source_hash", "kind", "content_hash", "path"]))
                except Exception as error:
                    failure = {"event": "IDENTIFICATION_ERROR", "path": row["path"],
                               "error": type(error).__name__+":"+str(error)[:300]}
                    failures.append(failure)
                    write_event(journal, failure)
                    if critical_io(error):
                        record_fault(run, error, evidence=failure)
                        raise RuntimeError("MATERIAL_ARCHIVE_FATAL_STORAGE_FAILURE") from error
                    if output != run:
                        raise RuntimeError("MATERIAL_LOCAL_CONTROL_IDENTIFICATION_REQUIRES_REVIEW") from error
                if time.monotonic()-started_identification > 45:
                    print(json.dumps({"identification_remaining": db.execute(
                        "SELECT count(*) FROM files WHERE source_hash IS NULL").fetchone()[0]}), flush=True)
                    started_identification = time.monotonic()
    groups = db.execute("SELECT source_hash,sum(size) bytes FROM files WHERE status='PENDING' "
                        "AND source_hash IS NOT NULL GROUP BY source_hash ORDER BY bytes DESC").fetchall()
    if limit is not None:
        # 每组至少一个路径；多副本组整体退出，因此数量可略超limit。
        selected, total = [], 0
        for group in groups:
            if total >= limit:
                break
            selected.append(group)
            total += db.execute("SELECT count(*) FROM files WHERE source_hash=? AND status='PENDING'",
                                (group["source_hash"],)).fetchone()[0]
        groups = selected
    started = time.monotonic()
    with event_path.open("a", encoding="utf-8") as journal:
        for events in group_events(db, groups, repo, workers=workers, external_manifest=output == run,
                                   before_group=lambda: check_control_budget(output, run)):
            for result in events:
                assert_deletion_allowed(run)
                if result.get("fatal_storage_failure"):
                    error = OSError(result["error"])
                    record_fault(run, error, evidence=result)
                    raise RuntimeError("MATERIAL_ARCHIVE_FATAL_STORAGE_FAILURE") from error
                write_event(journal, result)
                if result["event"] == "MOVED":
                    archived.add(result["source_hash"])
                    for item in result["removed"]:
                        db.execute("UPDATE files SET status='MOVED' WHERE path=?", (item["path"],))
                        reclaimed += item["reclaimed_allocated_bytes"]
                    moved += len(result["removed"])
                elif result["event"] == "RECOVERED":
                    db.execute("UPDATE files SET status='MOVED' WHERE path=?", (result["path"],))
                    archived.add(result["source_hash"])
                else:
                    failures.append(result)
                    if output != run:
                        raise RuntimeError("MATERIAL_LOCAL_CONTROL_SOURCE_ERROR; C originals retained")
                if time.monotonic()-started > 45:
                    snapshot = {"moved_this_process": moved, "archived_unique_sources": len(archived),
                                "C_reclaimed_allocated_bytes": reclaimed, "errors": len(failures)}
                    save(output/"stream-progress.json", snapshot)
                    print(json.dumps(snapshot), flush=True)
                    started = time.monotonic()
    result = {"moved_this_process": moved, "archived_unique_sources": len(archived),
              "C_reclaimed_allocated_bytes": reclaimed, "errors": failures,
              "remaining": [dict(row) for row in db.execute("SELECT status,count(*) files FROM files GROUP BY status")]}
    save(output/"stream-migration-result.json", result)
    db.close()
    return result


def archived_sources(run, *, control_dir=None):
    """只选择有完成迁移凭据的D盘对象；不把清单里的预期路径冒充已归档原件。"""
    found = {}
    pilot = run / "preflight-pilot-result.json"
    if pilot.exists():
        for item in json.loads(pilot.read_text(encoding="utf-8"))["results"]:
            found[item["source_hash"]] = item
    output = state_root(run, control_dir)
    for journal, _ in event_journals(run, output, "stream-migration-events.jsonl"):
        if not journal.exists():
            continue
        with journal.open("rb") as lines:
            for line in lines:
                if not line.endswith(b"\n"):
                    break  # 另一个迁移进程尚未刷完的尾行不能作为原件资格。
                item = json.loads(line)
                if item.get("event") == "MOVED":
                    found[item["source_hash"]] = item
                elif item.get("event") == "RECOVERED" and item["source_hash"] not in found:
                    # 日志刷新失败时，来源可能只有按已提交数据库凭据恢复的退出事件。
                    # 不能因此永远漏掉该原件的后续文本补提；格式和D盘路由仍须读库校验。
                    repo = store.repository()
                    metadata = repo.metadata(source_hash=item["source_hash"])
                    route = archive.archive_route(item["source_hash"], repo)
                    if metadata is None or route is None:
                        raise ValueError("MATERIAL_RECOVERED_ARCHIVE_ROUTE_MISSING")
                    found[item["source_hash"]] = {
                        "source_hash": item["source_hash"], "archive_path": route["archive_path"],
                        "kind": metadata.kind, "size": metadata.source_size, "content_hash": metadata.content_hash,
                    }
    return found


def parse_archived(run, *, limit=None, kinds=None, control_dir=None):
    """可中断的逐来源文本补提，失败和图表未决独立保留，不用归档数量代表完整性。"""
    from app.services.material_archive_text import capture

    run, repo = Path(run), store.repository()
    output = state_root(run, control_dir)
    sources, done = archived_sources(run, control_dir=control_dir), {}
    journal = output/"stream-text-events.jsonl"
    for source_journal, repair in event_journals(run, output, journal.name):
        for item in journal_items(source_journal, repair=repair):
            if item.get("event") == "TEXT_CAPTURED":
                done[item["source_hash"]] = item
    count, started = 0, time.monotonic()
    with journal.open("a", encoding="utf-8") as event_log:
        ordered = sorted(sources.values(), key=lambda item: (item["kind"] not in {"JSON", "TEXT"}, item["size"]))
        for item in ordered:
            if item["source_hash"] in done or kinds and item["kind"] not in kinds:
                continue
            if limit is not None and count >= limit:
                break
            # 只在即将补提的来源前复查预算；跳过的其他格式不读原件、不产生正文或事件。
            # 十几万来源按格式分阶段执行时，避免为每条跳过项反复枚举同一控制目录。
            check_control_budget(output, run)
            try:
                captured = capture(item["archive_path"], repo=repo)
                result = {"event": "TEXT_CAPTURED", "source_hash": item["source_hash"], "kind": item["kind"],
                          "status": captured["status"], "pages": captured.get("pages", 0),
                          "tables": captured.get("tables", 0), "pending_count": len(captured.get("pending", []))}
                done[item["source_hash"]] = result
            except Exception as error:
                result = {"event": "TEXT_ERROR", "source_hash": item["source_hash"], "kind": item["kind"],
                          "error": type(error).__name__+":"+str(error)[:300]}
                if critical_io(error) or control_dir is not None and isinstance(error, OSError):
                    # 本机日志仍可写并不表示D盘可读；原件失联后停止补提，不能对剩余
                    # 数万份资料反复访问失联设备，也不能把失败记成已经完成解析。
                    write_event(event_log, result)
                    record_fault(run, error, evidence=result)
                    raise RuntimeError("MATERIAL_ARCHIVE_TEXT_IO_FAILURE") from error
            write_event(event_log, result)
            count += 1
            if time.monotonic()-started > 45:
                print(json.dumps({"text_sources_processed_this_run": count, "archived_sources": len(sources),
                                  "status_counts": dict(Counter(x["status"] for x in done.values()))}), flush=True)
                started = time.monotonic()
    result = {"archived_sources": len(sources), "attempted_this_process": count,
              "status_counts": dict(Counter(x["status"] for x in done.values())),
              "not_yet_captured": len(set(sources)-set(done)), "binary_payload_saved": False}
    save(output/"stream-text-result.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["plan", "supplement", "migrate", "parse"])
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--tool-path", type=Path, action="append", default=[])
    parser.add_argument("--kinds", help="仅补提指定格式，逗号分隔；省略即处理全部格式")
    parser.add_argument("--control-dir", type=Path, help="本机有界工程流水目录；原件及冻结资格清单仍在D盘")
    args = parser.parse_args()
    from app.services.material_file_access import legacy_workflow_error

    legacy_workflow_error()
    for path in reversed(args.tool_path):
        sys.path.insert(0, str(path.resolve()))
    if args.action == "migrate":
        assert_deletion_allowed(args.run)
    if args.control_dir is not None and args.action not in {"migrate", "parse"}:
        raise ValueError("MATERIAL_LOCAL_CONTROL_ONLY_FOR_MIGRATE_OR_PARSE")
    output = state_root(args.run, args.control_dir)
    with run_lock(args.run, args.action), material_writes_only(store.repository().engine) as writes:
        try:
            if args.action == "plan":
                result = plan(args.run)
            elif args.action == "supplement":
                result = supplement(args.run)
            elif args.action == "migrate":
                result = migrate(args.run, limit=args.limit, workers=args.workers, control_dir=args.control_dir)
            else:
                result = parse_archived(args.run, limit=args.limit,
                                        kinds=set(args.kinds.split(",")) if args.kinds else None,
                                        control_dir=args.control_dir)
            print(json.dumps(result, ensure_ascii=False), flush=True)
        except OSError as error:
            if args.action == "migrate" and critical_io(error):
                record_fault(args.run, error)
            raise
        finally:
            # D盘失联时，审计报告写入也可能失败；保留首次业务异常，不能被收尾的
            # FileNotFoundError覆盖。本机只落小型工程凭据，不回写原件或解析正文。
            primary_error = sys.exc_info()[1]
            scope = {"allowed_material_table_writes": writes, "business_table_writes": 0}
            try:
                save(output/(args.action+"-write-scope.json"), scope)
            except OSError as audit_error:
                if args.action == "migrate" and not control_path().is_file():
                    record_fault(args.run, primary_error or audit_error)
                save(control_path().with_name(args.action+"-write-scope-D-unavailable.json"), {
                    **scope, "run": str(args.run), "source_file_bytes_saved_on_C": False,
                    "primary_error": str(primary_error)[:500] if primary_error else None,
                    "audit_write_error": type(audit_error).__name__+":"+str(audit_error)[:500],
                })
                if primary_error is None:
                    raise


if __name__ == "__main__":
    main()
