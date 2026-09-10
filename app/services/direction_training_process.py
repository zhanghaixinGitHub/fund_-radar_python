"""独立训练进程的硬超时和内存额度；Windows不弹出终端。"""

import ctypes
import json
import os
import subprocess
import sys
from time import perf_counter


def windows_job(process, memory_bytes):
    from ctypes import wintypes

    class Basic(ctypes.Structure):
        _fields_ = [
            ("process_time", ctypes.c_int64),
            ("job_time", ctypes.c_int64),
            ("flags", wintypes.DWORD),
            ("min_working", ctypes.c_size_t),
            ("max_working", ctypes.c_size_t),
            ("active", wintypes.DWORD),
            ("affinity", ctypes.c_size_t),
            ("priority", wintypes.DWORD),
            ("scheduling", wintypes.DWORD),
        ]

    class IO(ctypes.Structure):
        _fields_ = [
            (name, ctypes.c_uint64) for name in ("read_ops", "write_ops", "other_ops", "read", "write", "other")
        ]

    class Limits(ctypes.Structure):
        _fields_ = [
            ("basic", Basic),
            ("io", IO),
            ("process_memory", ctypes.c_size_t),
            ("job_memory", ctypes.c_size_t),
            ("peak_process", ctypes.c_size_t),
            ("peak_job", ctypes.c_size_t),
        ]

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.CreateJobObjectW.restype = wintypes.HANDLE
    kernel.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
    kernel.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
    kernel.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    handle = kernel.CreateJobObjectW(None, None)
    if not handle:
        raise OSError("PROCESS_JOB_CREATE_FAILED")
    limits = Limits()
    limits.basic.flags = 0x100 | 0x2000  # 每进程提交内存额度；关闭Job时回收进程。
    limits.process_memory = memory_bytes
    try:
        if not kernel.SetInformationJobObject(handle, 9, ctypes.byref(limits), ctypes.sizeof(limits)):
            raise OSError("PROCESS_JOB_LIMIT_FAILED")
        if not kernel.AssignProcessToJobObject(handle, wintypes.HANDLE(int(process._handle))):
            raise OSError("PROCESS_JOB_ASSIGN_FAILED")
    except BaseException:
        kernel.CloseHandle(handle)
        raise
    return lambda: kernel.CloseHandle(handle)


def run_process(payload, *, seconds=120, memory_bytes=4 * 1024**3, command=None):
    content = json.dumps(payload, allow_nan=False).encode("utf-8")
    if len(content) > 8 * 1024**2:
        raise ValueError("JOB_INPUT_TOO_LARGE")
    environment = {
        **os.environ,
        "PYTHONIOENCODING": "utf-8",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    }
    start = perf_counter()
    process = subprocess.Popen(
        command or [sys.executable, "-m", "scripts.direction_training_worker"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=environment,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
    )

    def release():
        pass

    try:
        if os.name == "nt":
            release = windows_job(process, memory_bytes)
        else:
            import resource

            resource.prlimit(process.pid, resource.RLIMIT_AS, (memory_bytes, memory_bytes))
        # worker在收到stdin前不导入数值库；限额已生效后才交付输入。
        try:
            output, _ = process.communicate(content, timeout=seconds)
        except subprocess.TimeoutExpired:
            process.kill()
            process.communicate()
            return {"status": "FAILED", "reason": "CANDIDATE_TIMEOUT", "elapsed_seconds": perf_counter() - start}
        if process.returncode or len(output) > 8 * 1024**2:
            return {
                "status": "FAILED",
                "reason": "WORKER_EXIT_OR_OUTPUT_BUDGET",
                "exit_code": process.returncode,
                "elapsed_seconds": perf_counter() - start,
            }
        result = json.loads(output)
        result["elapsed_seconds"] = perf_counter() - start
        return result
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate()
        release()
