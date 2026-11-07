"""既有外接盘归档的身份和路由兼容；2026-11-07起不再统一截获新下载。

以下迁移/清理实现保留历史追溯，旧CLI及删除入口已停用；普通业务按需读取D盘原件，
新资料按调用方D盘目录保存，现存原件和数据保留，不以旧迁移授权继续删除。
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import time
from contextlib import ExitStack
from datetime import UTC, datetime
from pathlib import Path
from threading import RLock
from uuid import uuid4

from sqlalchemy import select

from app.core.config import get_settings
from app.core.workspace_paths import material_location_aliases, runtime_material_path
from app.models.material_content import MaterialLocation
from app.services import material_content as store

SUFFIX = {"PDF": ".pdf", "JSON": ".json", "HTML": ".html", "TEXT": ".txt", "DOCX": ".docx"}
ROUTE_ENGINE = "SOURCE_ARCHIVE_ROUTE_V1"
_directory_lock = RLock()


def fund_for(path):
    """识别明确的基金目录或文件名前缀；新采集已全基金归档，未知归属记shared。"""
    path = Path(path)
    if not path.absolute().is_relative_to(store.PROJECT.resolve()):
        return None
    if not store.original_location(path):
        from app.services.material_archive_scope import classify

        # 旧业务命名原件与primary-ir目录也保留明确基金归属；快照/结果不会据此获得迁移资格。
        if not classify(path, store.PROJECT, indexed=True)[0]:
            return None
    for index, part in enumerate(path.parts[:-1]):
        if part == "funds" and re.fullmatch(r"\d{6}", path.parts[index + 1]):
            return path.parts[index + 1]
    if any(part in {"company-sources", "companies", "stock-sources", "stock-days"} for part in path.parts):
        # 公司材料末级的600xxx等是股票主体代码，不得误写成基金归属；原关系保留在完整别名中。
        return None
    codes = [part for part in path.parts if re.fullmatch(r"\d{6}", part)]
    prefix = re.match(r"^(\d{6})-", path.name)
    return codes[-1] if codes else prefix.group(1) if prefix else None


def reviewed_fund_for(path, source_hash, repo):
    """共享路径复用已核验的D盘归档登记；方法名兼容旧调用，不表示内容解析完整。"""
    path = Path(path).absolute()
    if not path.is_relative_to(store.PROJECT.resolve()):
        return None
    if archive_route(source_hash, repo) is not None:
        return "shared"
    from app.services.material_important_text import release

    return "008888" if release(source_hash, repo) is not None else None


def archive_route(source_hash, repo):
    """仅从库内读取不可变归档凭据；外接盘离线时也不检查或打开原文件。"""
    body = repo.page(source_hash, ROUTE_ENGINE, 0)
    if body is None:
        return None
    item = json.loads(body)
    root = Path(get_settings().fund_source_archive_root).absolute()
    expected = root / "originals" / source_hash[:2] / (source_hash + SUFFIX.get(item["format"], ".bin"))
    # 旧回执不可改写；仅接受已知旧根目录下同一哈希、分桶和扩展名的精确位置。
    accepted = material_location_aliases(expected, project_root=store.PROJECT, archive_root=root)
    if item["source_hash"] != source_hash or str(Path(item["archive_path"])) not in accepted:
        raise ValueError("MATERIAL_ARCHIVE_ROUTE_IDENTITY_CHANGED")
    return {**item, "archive_path": str(expected)}


def register_archive(fund, target, source_hash, repo):
    """核对D盘实体与库内别名后登记来源路由；不附带任何图表/语义完整性声明。

    同一来源可被多个基金引用，归档路由和原件都只有一个哈希对象，基金别名在清单中另记。
    不记录易变时间或别名列表，重复执行只读回同一凭据，不覆盖旧来源版本。
    """
    if fund != "shared" and not re.fullmatch(r"\d{6}", fund):
        raise ValueError("MATERIAL_ARCHIVE_INVALID_FUND")
    value = repo.metadata(path=target)
    if value is None or value.source_hash != source_hash:
        raise ValueError("MATERIAL_ARCHIVE_DESTINATION_ALIAS_MISSING_OR_CHANGED")
    target = Path(target).absolute()
    expected = root_path() / "originals" / source_hash[:2] / (source_hash + SUFFIX.get(value.kind, ".bin"))
    if target != expected:
        raise ValueError("MATERIAL_ARCHIVE_ROUTE_PATH_CHANGED")
    from app.services.material_file_cleanup import LockedOriginal

    with LockedOriginal(target, root=root_path(), expected_hash=source_hash, expected_size=value.source_size):
        return _register_verified_archive(target, value, repo)


def _register_verified_archive(target, value, repo):
    """调用者已持有D盘实体校验锁，避免Windows锁定期间用另一普通句柄重复打开。"""
    item = {"source_hash": value.source_hash, "source_size": value.source_size,
            "format": value.kind, "archive_path": str(target)}
    previous = archive_route(value.source_hash, repo)
    if previous is not None:
        if previous != item:
            raise ValueError("MATERIAL_ARCHIVE_ROUTE_IDENTITY_CHANGED")
        return previous
    repo.page(value.source_hash, ROUTE_ENGINE, 0, store.encoded(item).decode())
    return archive_route(value.source_hash, repo)


def archive_path(path):
    """只判断路径归属，不访问外接盘；外接盘离线也不影响数据库读库。"""
    root = Path(get_settings().fund_source_archive_root).absolute()
    current = runtime_material_path(path, project_root=store.PROJECT, archive_root=root)
    return current.is_relative_to(root / "originals")


def root_path():
    """外接盘必须已挂载且根目录真实存在；不可退回C盘，也不创建不存在的盘符。"""
    root = Path(get_settings().fund_source_archive_root).absolute()
    if not root.is_dir() or root.is_symlink() or root.resolve() != root:
        raise ValueError("MATERIAL_ARCHIVE_DRIVE_UNAVAILABLE_OR_REDIRECTED")
    if root.is_relative_to(store.PROJECT.resolve()):
        raise ValueError("MATERIAL_ARCHIVE_ROOT_MUST_BE_EXTERNAL")
    if store._override.get() is None and root.drive.upper() == "C:":
        raise ValueError("MATERIAL_ARCHIVE_ROOT_MUST_NOT_BE_C_DRIVE")
    return root


def _directory(root, relative):
    # 同进程并发首次创建originals/staging时，Windows的resolve可能跨越不存在到
    # 已存在状态；目录检查与创建串行，正文复制和不同来源的实体锁仍可并发。
    with _directory_lock:
        path = root / relative
        if path.resolve() != path:
            raise ValueError("MATERIAL_ARCHIVE_DIRECTORY_REDIRECTED")
        path.mkdir(parents=True, exist_ok=True)
        if path.resolve() != path or not path.is_relative_to(root):
            raise ValueError("MATERIAL_ARCHIVE_DIRECTORY_REDIRECTED")
        return path


def _verify(path, expected_hash, size):
    from app.services.material_file_cleanup import checked_target

    checked_target(path, root_path())
    with path.open("rb") as stream:
        actual = hashlib.file_digest(stream, "sha256").hexdigest()
    if actual != expected_hash or path.stat().st_size != size:
        raise ValueError("MATERIAL_ARCHIVE_COPY_MISMATCH")


def _flush_published_target(target):
    """最终名字公开后刷新文件信息；Windows短暂扫描占用只做有界重试。

    不写入或改变原件字节。持续占用仍报错保留C盘原件，设备失联/坏句柄等I/O
    异常不按占用吞掉，继续交给批次故障闸门阻止删除。
    """
    for attempt in range(9):
        try:
            with target.open("r+b") as durable:
                durable.flush()
                os.fsync(durable.fileno())
            return
        except PermissionError:
            if attempt == 8:
                raise
            time.sleep(0.1*(attempt+1))


def _copy(stream, *, source_hash, size, kind):
    """有界流式写盘、fsync、独立重读后原子公开；既有对象只校验，不覆盖。"""
    root = root_path()
    directory = _directory(root, Path("originals") / source_hash[:2])
    target = directory / (source_hash + SUFFIX.get(kind, ".bin"))
    if target.exists():
        _verify(target, source_hash, size)
        _flush_published_target(target)
        return target
    staging = _directory(root, Path("staging")) / (uuid4().hex + ".part")
    try:
        stream.seek(0)
        with staging.open("xb") as output:
            shutil.copyfileobj(stream, output, 1024 * 1024)
            output.flush()
            os.fsync(output.fileno())
        _verify(staging, source_hash, size)
        try:
            os.link(staging, target)
        except FileExistsError:
            pass
        _verify(target, source_hash, size)
        _flush_published_target(target)
    finally:
        staging.unlink(missing_ok=True)
    return target


def _manifest(fund, path, target, value, *, delete_local):
    """每条别名单独写不可变凭据，避免并行更新一份总清单丢失历史引用。"""
    directory = _directory(root_path(), Path("manifests") / fund / value.source_hash)
    alias_hash = store.checksum(str(Path(path).absolute()).encode("utf-8"))
    item = {
        "fund_code": fund,
        "source_hash": value.source_hash,
        "source_size": value.source_size,
        "source_alias": str(Path(path).absolute()),
        "archive_path": str(target),
        "format": value.kind,
        "base_content_hash": value.content_hash,
        "local_copy_removal_requested": delete_local,
        "content_complete_is_not_implied_by_archive": True,
    }
    name = directory / (alias_hash + ("-moved.json" if delete_local else ".json"))
    raw = store.encoded(item)
    try:
        with name.open("xb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError:
        if name.read_bytes() != raw:
            raise ValueError("MATERIAL_ARCHIVE_MANIFEST_CONFLICT") from None
    return name


def publish_source(path, raw, *, replace=False, explicit_source=False):
    """停止统一截获下载；返回False让已有发布器按原有D盘业务目录原子保存。"""
    return False


def relocate(path, *, source_hash, delete_local=False, repo=None):
    """单个现存原件归档；删除只在锁住同一实体、D盘核验和DB事务提交后生效。"""
    from app.services.material_file_cleanup import LockedOriginal

    path = Path(path).absolute()
    fund = fund_for(path)
    if fund is None or not path.is_relative_to(store.PROJECT.resolve()):
        raise ValueError("MATERIAL_ARCHIVE_PATH_OUTSIDE_PILOT_SCOPE")
    repo = repo or store.repository()
    value = repo.get(path, fresh=True)
    if value is None or value.source_hash != source_hash:
        raise ValueError("MATERIAL_ARCHIVE_DATABASE_ALIAS_CHANGED")
    # D盘保护锁最后释放，保证C盘文件句柄实际关闭/删除期间D盘对象不能被替换。
    with ExitStack() as destination_guards:
        with LockedOriginal(
            path, root=store.PROJECT, expected_hash=source_hash, expected_size=value.source_size
        ) as locked:
            target = _copy(locked.stream, source_hash=source_hash, size=value.source_size, kind=value.kind)
            destination_guards.enter_context(
                LockedOriginal(target, root=root_path(), expected_hash=source_hash, expected_size=value.source_size)
            )
            repo.link(target, source_hash)
            _manifest(fund, path, target, value, delete_local=delete_local)
            _register_verified_archive(target, value, repo)
            marker = store.checksum(store.encoded({"archive": str(target), "source_hash": source_hash}))
            if delete_local:
                with repo.engine.begin() as c:
                    current = (
                        c.execute(
                            select(MaterialLocation).where(MaterialLocation.location == str(path)).with_for_update()
                        )
                        .mappings()
                        .one()
                    )
                    if current["source_hash"] != source_hash:
                        raise ValueError("MATERIAL_ARCHIVE_DATABASE_ALIAS_CHANGED")
                    locked.mark_delete()
                    c.execute(
                        MaterialLocation.__table__.update()
                        .where(MaterialLocation.location == str(path))
                        .values(deleted_at=datetime.now(UTC), verification=marker)
                    )
            return {
                "source_alias": str(path),
                "archive_path": str(target),
                "source_hash": source_hash,
                "format": value.kind,
                "size": value.source_size,
                "removed_C_copy": delete_local,
                "reclaimed_C_allocated_bytes": locked.allocated_bytes if delete_local and locked.hardlinks == 1 else 0,
            }


def remove_reviewed_copy(path, *, source_hash, approved_paths, repo=None):
    """退出本批已核验来源的其他C盘副本；必须给出精确路径清单，绝不自动遍历目录。

    用于同一原件散落在共享采集目录的情形。重要内容已发布、D盘实体和数据库
    别名都通过后，只删除被锁住的那个C盘实体，保留历史别名与retention_hold。
    """
    from app.services.material_document import get_review
    from app.services.material_important_text import POLICY, release

    repo = repo or store.repository()
    published = release(source_hash, repo)
    review = get_review(source_hash, repo=repo, policy=POLICY)
    if not published or not review or not review["ready"] or review["document_hash"] != published["document_hash"]:
        raise ValueError("MATERIAL_ARCHIVE_IMPORTANT_TEXT_NOT_READY")
    return remove_archived_copy(path, source_hash=source_hash, approved_paths=approved_paths, repo=repo)


def remove_archived_copy(path, *, source_hash, approved_paths, repo=None):
    """按单次授权清单移除C盘同源实体；完整原件继续保留在D盘。

    归档迁移不授予内容完整资格。用户允许共享路径退出时，仍须锁住并核对C/D
    同一来源、数据库别名和正文读回；数据库提交失败则撤销文件删除标记。
    不遍历目录，不删除底稿、历史别名、其他来源或路径保留标记。
    """
    from app.services.material_file_cleanup import LockedOriginal, normalized

    path = Path(path).absolute()
    if (
        normalized(path) not in {normalized(p) for p in approved_paths}
        or not path.is_relative_to(store.PROJECT.resolve())
        or not store.original_location(path)
    ):
        raise ValueError("MATERIAL_ARCHIVE_COPY_NOT_IN_EXPLICIT_SCOPE")
    repo = repo or store.repository()
    value = repo.get(source_hash=source_hash, fresh=True)
    if value is None:
        raise ValueError("MATERIAL_ARCHIVE_DATABASE_CONTENT_MISSING")
    target = root_path() / "originals" / source_hash[:2] / (source_hash + SUFFIX.get(value.kind, ".bin"))
    with ExitStack() as destination_guards:
        with LockedOriginal(
            path, root=store.PROJECT, expected_hash=source_hash, expected_size=value.source_size
        ) as locked:
            destination_guards.enter_context(
                LockedOriginal(target, root=root_path(), expected_hash=source_hash, expected_size=value.source_size)
            )
            archived = repo.metadata(path=target)
            if archived is None or archived.source_hash != source_hash:
                raise ValueError("MATERIAL_ARCHIVE_DESTINATION_ALIAS_MISSING_OR_CHANGED")
            metadata = repo.metadata(path=path)
            if metadata is None:
                repo.link(path, source_hash)
            elif metadata.source_hash != source_hash:
                raise ValueError("MATERIAL_ARCHIVE_DATABASE_ALIAS_CHANGED")
            with store.database_only():
                readback = store.source_bytes(path)
                if readback.source_hash != source_hash or store.encoded(readback.extracted) != store.encoded(
                    value.extracted
                ):
                    raise ValueError("MATERIAL_ARCHIVE_COPY_DATABASE_READBACK_CHANGED")
            marker = store.checksum(store.encoded({"archive": str(target), "source_hash": source_hash}))
            with repo.engine.begin() as c:
                row = (
                    c.execute(select(MaterialLocation).where(MaterialLocation.location == str(path)).with_for_update())
                    .mappings()
                    .one()
                )
                if row["source_hash"] != source_hash:
                    raise ValueError("MATERIAL_ARCHIVE_DATABASE_ALIAS_CHANGED")
                locked.mark_delete()
                c.execute(
                    MaterialLocation.__table__.update()
                    .where(MaterialLocation.location == str(path))
                    .values(deleted_at=datetime.now(UTC), verification=marker)
                )
            return {
                "path": str(path),
                "source_hash": source_hash,
                "archive_path": str(target),
                "reclaimed_allocated_bytes": locked.allocated_bytes if locked.hardlinks == 1 else 0,
            }
