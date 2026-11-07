"""外接盘实际I/O故障的持久化删除闸门；控制信息不包含任何源文件字节。"""

from __future__ import annotations

import errno
from datetime import UTC, datetime
from pathlib import Path

from app.services import material_content as store


def control_path():
    """D盘失联时仍须记住禁止删除；C盘仅保留数KB工程控制信息，不保留原件副本。"""
    return store.PROJECT/".local-runs/material-migration-control/external-drive-io-fault.json"


def assert_deletion_allowed(run):
    """挂载恢复、少量文件可读或Volume Healthy均不能自动解除已经发生的存储故障。"""
    if control_path().is_file() or (Path(run)/"external-drive-io-fault.json").is_file():
        raise RuntimeError("MATERIAL_ARCHIVE_IO_FAULT_REQUIRES_FULL_VERIFICATION_AND_STABLE_CONNECTION")


def critical_io(error):
    """普通共享占用逐项保留，设备失联/坏句柄/读写失败/空间耗尽停止整批删除。"""
    return isinstance(error, OSError) and (
        error.errno in {errno.EIO, errno.EBADF, errno.ENODEV, errno.ENXIO, errno.ENOSPC}
        or getattr(error, "winerror", None) in {21, 23, 55, 64, 1117, 1167}
    )


def record_fault(run, error, *, evidence=None):
    """优先刷盘本机小型故障闸门，再尽力保存D盘副本；原件不回写C盘。"""
    from app.commands.material_source_archive import save

    value = {"at": datetime.now(UTC).isoformat(), "run": str(run), "C_deletion_blocked": True,
             "error": type(error).__name__+":"+str(error)[:500], "evidence": evidence,
             "source_file_bytes_saved_on_C": False}
    save(control_path(), value)
    try:
        save(Path(run)/"external-drive-io-fault.json", value)
    except OSError:
        pass  # 已有C盘持久化闸门；D盘不可写时不得让异常遮盖首次故障。
    return value
