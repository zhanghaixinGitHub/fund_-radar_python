"""经冻结清单授权的同源原件分组归档；不迁移历史快照或实验结果。"""

from __future__ import annotations

import hashlib
import os
from contextlib import ExitStack, contextmanager
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import select

from app.models.material_content import MaterialLocation
from app.services import material_archive_scope as scope
from app.services import material_content as store
from app.services import material_source_archive as archive
from app.services.material_file_cleanup import LockedOriginal, normalized, physical_path


def capture_unindexed(item, *, repo=None):
    """无索引原件先复制到D盘再解析；真实格式确认后只保留一个内容寻址对象。

    本函数不删除C盘文件；登记和独立读回通过后，由move_group再次锁定源实体退出。
    未适配格式仅存异常文字和D盘原件，不把二进制或图片包装进数据库。
    """
    import io

    repo = repo or store.repository()
    path = Path(item["path"]).absolute()
    with physical_path(path).open("rb") as source:
        source_hash = hashlib.file_digest(source, "sha256").hexdigest()
    existing = repo.metadata(source_hash=source_hash)
    if existing is not None:
        repo.link(path, source_hash)
        return {**item, "source_hash": source_hash, "kind": existing.kind,
                "content_hash": existing.content_hash}
    with LockedOriginal(path, root=store.PROJECT, expected_hash=source_hash, expected_size=item["size"]) as source:
        source.stream.seek(0)
        eligible, reason = scope.classify(path, store.PROJECT, prefix=source.stream.read(8192))
        if not eligible:
            raise ValueError("MATERIAL_BATCH_PROTECTED_FILE:" + reason)
        provisional = archive._copy(source.stream, source_hash=source_hash, size=item["size"], kind="UNSUPPORTED")
        if item["size"] <= store.MAX_BYTES:
            raw = provisional.read_bytes()
            if store.checksum(raw) != source_hash:
                raise ValueError("MATERIAL_BATCH_ARCHIVE_SOURCE_CHANGED")
            try:
                value = store.extract(raw, path.suffix)
            except Exception as error:
                value = store.opaque_content(raw, path.suffix, type(error).__name__)
            target = archive._copy(io.BytesIO(raw), source_hash=source_hash, size=item["size"], kind=value.kind)
        else:
            data = {"unsupported_format": path.suffix, "extraction_issue": "SOURCE_SIZE_LIMIT"}
            value = store.SourceContent(source_hash, item["size"], "UNSUPPORTED", data,
                                        store.checksum(store.encoded(data)), False, "SOURCE_SIZE_LIMIT")
            target = provisional
        repo.put(target, value)
        archive.register_archive(archive.fund_for(path) or "shared", target, source_hash, repo)
        if target != provisional:
            # 只删除本函数自己的同源临时扩展名，并同时锁住已核验的最终D盘对象。
            with LockedOriginal(target, root=archive.root_path(), expected_hash=source_hash,
                                expected_size=item["size"]):
                with LockedOriginal(provisional, root=archive.root_path(), expected_hash=source_hash,
                                    expected_size=item["size"]) as redundant:
                    redundant.mark_delete()
        repo.link(path, source_hash)
        return {**item, "source_hash": source_hash, "kind": value.kind, "content_hash": value.content_hash}


@contextmanager
def group_destination(items, repo):
    """同一来源的D盘原件只核验并锁定一次，持续保护全部硬链接逐条退出。

    不能同时锁定C盘同一实体的多个DELETE句柄；D盘保护锁不受此限制，逐条退出
    期间始终持有。这样既不反复读取外接盘大PDF，也不引入按时间缓存的校验漏洞。
    """
    item = items[0]
    metadata = repo.metadata(source_hash=item["source_hash"])
    if metadata is None:
        raise ValueError("MATERIAL_BATCH_UNINDEXED_SOURCE_REQUIRES_PREFLIGHT")
    target = archive.root_path()/"originals"/item["source_hash"][:2]/(
        item["source_hash"]+archive.SUFFIX.get(metadata.kind, ".bin"))
    if not target.exists():
        with LockedOriginal(item["path"], root=store.PROJECT, expected_hash=item["source_hash"],
                            expected_size=item["size"]) as source:
            archive._copy(source.stream, source_hash=item["source_hash"], size=item["size"], kind=metadata.kind)
    with LockedOriginal(target, root=archive.root_path(), expected_hash=item["source_hash"],
                        expected_size=item["size"]) as destination:
        yield destination


def move_group(items, *, approved_paths, repo=None, destination=None, external_manifest=True):
    """先核D盘完整原件及库内底稿，再按组移除C盘精确实体，失败撤销删除标记。

    items为本次冻结清单中的同一来源路径、长度和摘要。仅调用者提供的精确清单
    有删除资格；内容解析不完整的原件仍可迁往D盘，结果明确标记TEXT_REVIEW_PENDING。
    该函数从D盘锁定流补提新来源，不从C盘解析，不改变评分/预测或路径保留标记。
    """
    repo = repo or store.repository()
    if not items:
        return None
    approved = {normalized(p) for p in approved_paths}
    paths = [Path(item["path"]).absolute() for item in items]
    if any(normalized(p) not in approved for p in paths):
        raise ValueError("MATERIAL_BATCH_PATH_NOT_APPROVED")
    identities = [(s.st_dev, s.st_ino) for s in (os.stat(physical_path(p), follow_symlinks=False) for p in paths)]
    if len(set(identities)) != len(identities):
        # Windows同一实体的多个硬链接不能同时持有当前DELETE保护锁；调用者逐链接退出。
        raise ValueError("MATERIAL_BATCH_HARDLINKS_REQUIRE_SEQUENTIAL_EXIT")
    source_hash = items[0].get("source_hash")
    if not source_hash:
        with paths[0].open("rb") as source:
            source_hash = hashlib.file_digest(source, "sha256").hexdigest()
    if any(item.get("source_hash") not in {None, source_hash} for item in items):
        raise ValueError("MATERIAL_BATCH_SOURCE_GROUP_CHANGED")
    metadata = repo.metadata(source_hash=source_hash)
    if metadata is None:
        raise ValueError("MATERIAL_BATCH_UNINDEXED_SOURCE_REQUIRES_PREFLIGHT")
    kind = metadata.kind if metadata else {
        ".pdf": "PDF", ".json": "JSON", ".html": "HTML", ".htm": "HTML", ".docx": "DOCX",
        ".zip": "ARCHIVE", ".txt": "TEXT", ".csv": "TEXT", ".js": "TEXT",
    }.get(paths[0].suffix.lower(), "UNSUPPORTED")
    size = items[0]["size"]
    if metadata is not None and metadata.source_size != size:
        raise ValueError("MATERIAL_BATCH_DATABASE_SOURCE_SIZE_CHANGED")
    # 先锁住全部C盘实体；D盘保护放在外层，确保C盘实际删除期间D盘不能被替换。
    with ExitStack() as destination_guard:
        with ExitStack() as originals:
            locked = []
            for path, item in zip(paths, items, strict=True):
                if item["size"] != size:
                    raise ValueError("MATERIAL_BATCH_SOURCE_SIZE_CHANGED")
                source = originals.enter_context(LockedOriginal(
                    path, root=store.PROJECT, expected_hash=source_hash, expected_size=size,
                    expected_file_id=item.get("file_id"),
                ))
                source.stream.seek(0)
                eligible, reason = scope.classify(path, store.PROJECT, indexed=repo.metadata(path=path) is not None,
                                                  prefix=source.stream.read(8192))
                if not eligible:
                    raise ValueError("MATERIAL_BATCH_PROTECTED_FILE:" + reason)
                locked.append(source)
            if destination is None:
                target = archive._copy(locked[0].stream, source_hash=source_hash, size=size, kind=kind)
                destination = destination_guard.enter_context(LockedOriginal(
                    target, root=archive.root_path(), expected_hash=source_hash, expected_size=size,
                ))
            else:
                target = archive.root_path()/"originals"/source_hash[:2]/(
                    source_hash+archive.SUFFIX.get(kind, ".bin"))
                if (not isinstance(destination, LockedOriginal) or destination.stream is None
                        or destination.stream.closed or destination.path != target
                        or destination.expected_hash != source_hash or destination.size != size
                        or destination.pending):
                    raise ValueError("MATERIAL_BATCH_DESTINATION_GUARD_REQUIRED")
            value = repo.get(source_hash=source_hash, fresh=True)
            if value is None:
                destination.stream.seek(0)
                if size <= store.MAX_BYTES:
                    raw = destination.stream.read()
                    try:
                        value = store.extract(raw, paths[0].suffix)
                    except Exception as error:
                        value = store.opaque_content(raw, paths[0].suffix, type(error).__name__)
                else:
                    data = {"unsupported_format": paths[0].suffix, "extraction_issue": "SOURCE_SIZE_LIMIT"}
                    value = store.SourceContent(source_hash, size, "UNSUPPORTED", data,
                                                store.checksum(store.encoded(data)), False, "SOURCE_SIZE_LIMIT")
                # 未知后缀按占位归档时，提取后的真实格式可能不同。原对象不复制第二份；
                # 此类来源应先完成类型识别，再进入本组，禁止留下同哈希两个扩展名。
                if value.kind != kind:
                    raise ValueError("MATERIAL_BATCH_UNINDEXED_FORMAT_REQUIRES_PREFLIGHT")
                repo.put(target, value)
            else:
                repo.link(target, source_hash)
            archive._register_verified_archive(target, value, repo)
            for path in paths:
                old = repo.metadata(path=path)
                if old is not None and old.source_hash != source_hash:
                    raise ValueError("MATERIAL_BATCH_DATABASE_ALIAS_CHANGED")
                repo.link(path, source_hash)
            readback = repo.get(target, fresh=True)
            if readback is None or readback.content_hash != value.content_hash or readback.source_hash != source_hash:
                raise ValueError("MATERIAL_BATCH_DATABASE_READBACK_FAILED")
            # 分组凭据保留全部原路径及真实基金目录关系；清单不是删除完成标记。
            manifest = {
                "source_hash": source_hash, "archive_path": str(target), "format": value.kind,
                "base_content_hash": value.content_hash, "source_size": size,
                "aliases": [{"path": str(p), "fund": archive.fund_for(p) or "shared"} for p in paths],
                "semantic_completion_is_not_implied": True,
            }
            from app.commands.material_source_archive import save

            identity = store.checksum(store.encoded(manifest["aliases"]))
            if external_manifest:
                directory = archive._directory(archive.root_path(), Path("manifests/shared") / source_hash)
                manifest_path = directory / ("batch-" + identity + ".json")
                if manifest_path.exists():
                    import json

                    if json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
                        raise ValueError("MATERIAL_BATCH_MANIFEST_CONFLICT")
                else:
                    save(manifest_path, manifest)
            # 故障降载模式不再向D盘追加分组清单；上述原路径别名、D盘路由和底稿
            # 已分别入库并独立读回，下面仍逐条锁定数据库身份后才允许C盘实体退出。
            # 工程顺序流水由调用者写本机控制目录，不含原件或解析正文。
            marker = store.checksum(store.encoded({"archive": str(target), "source_hash": source_hash}))
            with repo.engine.begin() as c:
                rows = c.execute(select(MaterialLocation).where(
                    MaterialLocation.location.in_([str(p) for p in paths])
                ).with_for_update()).mappings().all()
                if len(rows) != len(paths) or any(row["source_hash"] != source_hash for row in rows):
                    raise ValueError("MATERIAL_BATCH_DATABASE_ALIAS_CHANGED")
                for source in locked:
                    source.mark_delete()
                c.execute(MaterialLocation.__table__.update().where(
                    MaterialLocation.location.in_([str(p) for p in paths])
                ).values(deleted_at=datetime.now(UTC), verification=marker))
            result = {"source_hash": source_hash, "archive_path": str(target), "kind": value.kind,
                      "size": size, "content_hash": value.content_hash,
                      "parse_status": "TEXT_REVIEW_PENDING",
                      "removed": [{"path": str(path), "allocated_bytes": source.allocated_bytes,
                                   "hardlinks_at_delete": source.hardlinks, "file_id": source.file_id}
                                  for path, source in zip(paths, locked, strict=True)]}
        # 所有同源硬链接若在本组删除，最后一个关闭后才释放一次；其余有未删除链接则计零。
        identities = {}
        for item in result["removed"]:
            identities.setdefault(tuple(item["file_id"]), []).append(item)
        for items_for_id in identities.values():
            reclaimed = (
                items_for_id[0]["allocated_bytes"]
                if len(items_for_id) == items_for_id[0]["hardlinks_at_delete"] else 0
            )
            for item in items_for_id:
                item["reclaimed_allocated_bytes"] = 0
            items_for_id[-1]["reclaimed_allocated_bytes"] = reclaimed
        return result
