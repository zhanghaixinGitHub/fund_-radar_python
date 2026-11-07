"""已核验原件的Windows实体锁与单文件清理基础设施。

复用前序经失败保护测试的句柄实现；调用者必须另行验证范围、完整文本、
引用和用户授权。本模块不遍历目录、不打开清理开关，也不自动删除任何文件。
"""

from __future__ import annotations

import ctypes
import hashlib
import os
import stat
from pathlib import Path


def normalized(path):
    return os.path.normcase(os.path.abspath(path))


def physical_path(path):
    """Windows长路径只用于实体I/O；数据库和授权清单继续保存原来的普通绝对路径。"""
    path = Path(path).absolute()
    if os.name == "nt" and not str(path).startswith("\\\\?\\"):
        return Path("\\\\?\\" + str(path))
    return path


def checked_target(path, root):
    """同时检查绝对路径、最终物理位置和符号链接，拒绝作用域外目标。"""
    path, root = Path(path), Path(root).resolve()
    if not path.is_absolute():
        raise ValueError("TARGET_NOT_ABSOLUTE_OR_IS_SYMLINK")
    attributes = physical_path(path).lstat()
    if stat.S_ISLNK(attributes.st_mode) or getattr(attributes, "st_file_attributes", 0) & 0x400:
        raise ValueError("TARGET_NOT_ABSOLUTE_OR_IS_SYMLINK")
    target = Path(str(physical_path(path).resolve(strict=True)).removeprefix("\\\\?\\"))
    if not target.is_relative_to(root) or target == root or normalized(target) != normalized(path):
        raise ValueError("TARGET_ESCAPES_APPROVED_ROOT")
    if not stat.S_ISREG(attributes.st_mode):
        raise ValueError("TARGET_NOT_REGULAR_FILE")
    return target


class LockedOriginal:
    """Windows只允许其他读者共享，阻止核验期间的写入、重命名及替换。

    句柄关联的实体通过全部摘要检查后才可置删除标记；发生异常时在关闭句柄
    前撤销标记。成功提交数据库记录后关闭句柄，删除的始终是已核验实体。
    """

    class StandardInfo(ctypes.Structure):
        _fields_ = [
            ("AllocationSize", ctypes.c_longlong),
            ("EndOfFile", ctypes.c_longlong),
            ("NumberOfLinks", ctypes.c_uint32),
            ("DeletePending", ctypes.c_ubyte),
            ("Directory", ctypes.c_ubyte),
        ]

    class DispositionInfo(ctypes.Structure):
        _fields_ = [("DeleteFile", ctypes.c_ubyte)]

    def __init__(self, path, *, root, expected_hash, expected_size, expected_file_id=None):
        self.path = checked_target(path, root)
        self.root = Path(root).resolve()
        self.expected_hash, self.expected_size = expected_hash, expected_size
        self.expected_file_id = expected_file_id
        self.stream = None
        self.pending = False
        self.kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        k = self.kernel
        k.CreateFileW.argtypes = [
            ctypes.c_wchar_p,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_void_p,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_void_p,
        ]
        k.CreateFileW.restype = ctypes.c_void_p
        k.CloseHandle.argtypes = [ctypes.c_void_p]
        k.GetFinalPathNameByHandleW.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p, ctypes.c_uint32, ctypes.c_uint32]
        k.GetFinalPathNameByHandleW.restype = ctypes.c_uint32
        k.GetFileInformationByHandleEx.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_uint32]
        k.SetFileInformationByHandle.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_uint32]
        k.GetCompressedFileSizeW.argtypes = [ctypes.c_wchar_p, ctypes.POINTER(ctypes.c_uint32)]
        k.GetCompressedFileSizeW.restype = ctypes.c_uint32

    def __enter__(self):
        import msvcrt

        name = "\\\\?\\" + str(self.path)
        # GENERIC_READ|DELETE，FILE_SHARE_READ，OPEN_EXISTING；不允许并发写入或替换。
        self.handle = self.kernel.CreateFileW(name, 0x80010000, 1, None, 3, 0x08000000, None)
        if self.handle == ctypes.c_void_p(-1).value:
            raise ctypes.WinError(ctypes.get_last_error())
        try:
            buffer = ctypes.create_unicode_buffer(32768)
            length = self.kernel.GetFinalPathNameByHandleW(self.handle, buffer, len(buffer), 0)
            if not 0 < length < len(buffer):
                raise ValueError("FINAL_HANDLE_PATH_UNAVAILABLE")
            final = buffer.value.removeprefix("\\\\?\\")
            if normalized(final) != normalized(self.path) or not Path(final).is_relative_to(self.root):
                raise ValueError("FINAL_HANDLE_PATH_OUTSIDE_SCOPE")
            descriptor = msvcrt.open_osfhandle(self.handle, os.O_RDONLY | os.O_BINARY)
            self.stream = os.fdopen(descriptor, "rb")
            before = os.fstat(self.stream.fileno())
            if self.expected_file_id is not None and [before.st_dev, before.st_ino] != self.expected_file_id:
                raise ValueError("ORIGINAL_FILE_ID_CHANGED")
            actual = hashlib.file_digest(self.stream, "sha256").hexdigest()
            after = os.fstat(self.stream.fileno())
            if (
                actual != self.expected_hash
                or after.st_size != self.expected_size
                or (before.st_ino, before.st_size, before.st_mtime_ns)
                != (after.st_ino, after.st_size, after.st_mtime_ns)
            ):
                raise ValueError("ORIGINAL_BYTES_OR_IDENTITY_CHANGED")
            info = self.StandardInfo()
            if not self.kernel.GetFileInformationByHandleEx(self.handle, 1, ctypes.byref(info), ctypes.sizeof(info)):
                raise ctypes.WinError(ctypes.get_last_error())
            allocated = info.AllocationSize
            if getattr(after, "st_file_attributes", 0) & (0x800 | 0x200):
                high = ctypes.c_uint32()
                ctypes.set_last_error(0)
                low = self.kernel.GetCompressedFileSizeW(name, ctypes.byref(high))
                if low == 0xFFFFFFFF and ctypes.get_last_error():
                    raise ctypes.WinError(ctypes.get_last_error())
                allocated = (high.value << 32) + low
            self.size, self.file_id = after.st_size, [after.st_dev, after.st_ino]
            self.allocated_bytes, self.hardlinks = allocated, info.NumberOfLinks
            return self
        except BaseException:
            if self.stream is not None:
                self.stream.close()
            else:
                self.kernel.CloseHandle(self.handle)
            raise

    def mark_delete(self, enabled=True):
        """旧空间优化删除入口已撤回；仅允许撤销尚未提交的删除标记。"""
        if enabled:
            from app.services.material_file_access import legacy_workflow_error

            legacy_workflow_error()
        value = self.DispositionInfo(False)
        if not self.kernel.SetFileInformationByHandle(self.handle, 4, ctypes.byref(value), ctypes.sizeof(value)):
            raise ctypes.WinError(ctypes.get_last_error())
        self.pending = False

    def __exit__(self, kind, value, traceback):
        try:
            if kind is not None and self.pending:
                self.mark_delete(False)
        finally:
            self.stream.close()
