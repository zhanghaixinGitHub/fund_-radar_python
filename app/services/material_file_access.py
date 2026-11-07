"""资料混合读取：复用同版本解析结果，按需读D盘原件，可变业务文件保持文件语义。"""

import hashlib
import logging
import os
import re
from functools import lru_cache
from pathlib import Path

from sqlalchemy.exc import SQLAlchemyError

logger = logging.getLogger(__name__)


def physical_path(path):
    """只映射已知旧根目录，不访问旧C盘工程；Windows使用扩展路径支持历史长文件名。"""
    from app.core.config import get_settings
    from app.core.workspace_paths import runtime_material_path
    from app.services import material_content as store

    path = runtime_material_path(path, project_root=store.PROJECT,
                                 archive_root=get_settings().fund_source_archive_root)
    if os.name == "nt" and path.drive.upper().removeprefix("\\\\?\\") != "D:":
        raise ValueError("MATERIAL_PROJECT_FILE_MUST_BE_ON_D_DRIVE")
    if os.name == "nt" and not str(path).startswith("\\\\?\\"):
        return Path("\\\\?\\" + str(path))
    return path


def archived_bytes(metadata, route):
    """已登记归档必须实际存在且内容一致；故障或损坏不以数据库旧正文掩盖。"""
    from app.core.config import get_settings

    root = Path(get_settings().fund_source_archive_root).absolute()
    target = Path(route["archive_path"])
    if not root.is_dir():
        raise OSError("MATERIAL_ARCHIVE_DRIVE_UNAVAILABLE")
    if root.resolve() != root or target.resolve() != target or not target.is_relative_to(root / "originals"):
        raise ValueError("MATERIAL_ARCHIVE_ROUTE_REDIRECTED")
    if route["source_size"] != metadata.source_size:
        raise ValueError("MATERIAL_ARCHIVE_SOURCE_SIZE_CHANGED")
    with physical_path(target).open("rb") as stream:
        before = os.fstat(stream.fileno())
        if before.st_size != metadata.source_size:
            raise ValueError("MATERIAL_ARCHIVE_SOURCE_SIZE_CHANGED")
        raw = stream.read()
        after = os.fstat(stream.fileno())
    if (before.st_size, before.st_mtime_ns, before.st_ino) != (after.st_size, after.st_mtime_ns, after.st_ino):
        raise ValueError("MATERIAL_ARCHIVE_SOURCE_CHANGED_DURING_READ")
    if hashlib.sha256(raw).hexdigest() != metadata.source_hash:
        raise ValueError("MATERIAL_ARCHIVE_SOURCE_HASH_CHANGED")
    return raw


def original_bytes(path, *, expected_hash=None, repo=None, metadata=None):
    """只读取真实原件；查历史路由时只读身份，不以库内提取结果冒充二进制。"""
    from app.services import material_content as store
    from app.services.material_source_archive import archive_route

    path = Path(path)
    try:
        raw = physical_path(path).read_bytes()
    except FileNotFoundError:
        # 仅“文件确实不存在”可查兼容来源；权限、设备失联等OSError必须原样失败。
        if not store.managed(path):
            raise
    else:
        from app.services.material_source_archive import archive_path

        if archive_path(path):
            # 内容寻址对象可直接从文件名核对身份，无需为读取现存原件依赖资料库。
            identity = path.stem
            if not re.fullmatch(r"[0-9a-f]{64}", identity) or path.parent.name != identity[:2]:
                raise ValueError("MATERIAL_ARCHIVE_OBJECT_IDENTITY_INVALID")
            if hashlib.sha256(raw).hexdigest() != identity:
                raise ValueError("MATERIAL_ARCHIVE_SOURCE_HASH_CHANGED")
        if expected_hash is not None and hashlib.sha256(raw).hexdigest() != expected_hash:
            raise ValueError("MATERIAL_SOURCE_VERSION_CHANGED")
        return raw
    repo = repo or store.repository()
    metadata = metadata or repo.metadata(path=path)
    if metadata is None:
        raise FileNotFoundError("MATERIAL_SOURCE_NOT_FOUND:" + str(path))
    if expected_hash is not None and metadata.source_hash != expected_hash:
        raise ValueError("MATERIAL_SOURCE_VERSION_CHANGED")
    route = archive_route(metadata.source_hash, repo)
    if route is not None:
        try:
            return archived_bytes(metadata, route)
        except FileNotFoundError as error:
            raise ValueError("MATERIAL_ARCHIVE_ORIGINAL_MISSING") from error
    raise FileNotFoundError("MATERIAL_ORIGINAL_NOT_RETAINED:" + str(path))


@lru_cache(maxsize=512)
def _file_hash(path, device, inode, size, modified, changed):
    """只在文件实体/长度/时间变化时重读摘要，避免同一原件在每个请求重复全文读取。"""
    with Path(path).open("rb") as stream:
        before = os.fstat(stream.fileno())
        identity = hashlib.file_digest(stream, "sha256").hexdigest()
        after = os.fstat(stream.fileno())
    # Windows的路径stat与句柄fstat对ctime口径不同，跨调用只核对实体、长度、mtime；
    # 同一句柄读取前后的ctime仍要一致，避免把正常覆盖写入误判为读取中发生变化。
    expected = (device, inode, size, modified)
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != expected:
        raise ValueError("MATERIAL_SOURCE_CHANGED_DURING_READ")
    if (before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
        after.st_size, after.st_mtime_ns, after.st_ctime_ns
    ):
        raise ValueError("MATERIAL_SOURCE_CHANGED_DURING_READ")
    return identity


def _same_version(path, metadata):
    target = physical_path(path)
    try:
        info = target.stat()
    except FileNotFoundError:
        # 已迁空的逻辑路径仍可复用绑定来源的缓存，不能据此宣称原件实体已经核验。
        return True
    if info.st_size != metadata.source_size:
        return False
    return _file_hash(str(target), info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns,
                      info.st_ctime_ns) == metadata.source_hash


def _stored_value(value, path, repo):
    from app.services import material_content as store

    if value is None or value.kind == "UNSUPPORTED":
        raise ValueError("MATERIAL_HISTORICAL_CONTENT_UNAVAILABLE:" + str(path))
    # 每次绑定独立读取视图，避免修改请求缓存对象后污染显式数据库验收的访问边界。
    value = store.SourceContent(value.source_hash, value.source_size, value.kind, value.extracted,
                                value.content_hash, value.complete, value.issue)
    value.database_only_source = False
    value.source_path = str(path.absolute())
    if value.kind in {"PDF", "HTML"}:
        from app.services.material_important_text import release

        value.important_release = release(value.source_hash, repo)
    return value


def read_material(path):
    """源资料命中同版本解析缓存时复用；运行文件读实际新状态，无缓存则按需读原件。"""
    from app.services import material_content as store
    from app.services.material_source_archive import archive_path

    path = Path(path)
    source = store.original_location(path) or archive_path(path) or path.suffix.lower() in {".pdf", ".html", ".docx"}
    repo = metadata = None
    if source and store.managed(path):
        try:
            repo = store.repository()
            metadata = repo.metadata(path=path)
            if metadata is not None and metadata.kind != "UNSUPPORTED" and _same_version(path, metadata):
                return _stored_value(repo.get(path), path, repo)
        except SQLAlchemyError:
            # 资料库不可用不阻止读取现存原件；原件也缺失时保留数据库异常，不虚报成功。
            if not physical_path(path).is_file():
                raise
            logger.warning("material_file_access.read_material   >>> 资料缓存不可用，本次读取原件")
            return original_bytes(path)
    try:
        return original_bytes(path, repo=repo, metadata=metadata)
    except FileNotFoundError:
        if not store.managed(path):
            raise
        repo = repo or store.repository()
        return _stored_value(repo.get(path), path, repo)


def legacy_workflow_error():
    """旧空间优化批处理已撤回；保留模块用于追溯，不继续迁移、解析或清理。"""
    raise RuntimeError("MATERIAL_SPACE_OPTIMIZATION_RETIRED: 2026-11-07已改为混合读取，旧批量入库/迁移/清理流程停止")
