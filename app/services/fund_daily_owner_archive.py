"""多周期本人归档；通过既有服务令牌调用Java正式入口，不传递数据库凭据。"""

import ctypes
import json
from ctypes import wintypes
from pathlib import Path


def windows_argv(command):
    library = ctypes.windll.shell32
    library.CommandLineToArgvW.argtypes = [wintypes.LPCWSTR, ctypes.POINTER(ctypes.c_int)]
    library.CommandLineToArgvW.restype = ctypes.POINTER(wintypes.LPWSTR)
    count = ctypes.c_int()
    pointer = library.CommandLineToArgvW(command, ctypes.byref(count))
    try:
        return [pointer[i] for i in range(count.value)]
    finally:
        free = ctypes.windll.kernel32.LocalFree
        free.argtypes = [ctypes.c_void_p]
        free.restype = ctypes.c_void_p
        free(ctypes.cast(pointer, ctypes.c_void_p))


def runtime_classpath(args, core):
    """解析实际Java进程的类路径，兼容Maven的单值参数文件。

    spring-boot:run使用 `-cp @临时文件`，文件内为转义后的单个引号包围值。
    只读取已经核实归属的运行进程所引用的文件，不执行其内容、不猜依赖。
    解析失败、不是单个字符串或不含本项目路径仍拒绝归档。
    """
    index = next((i for i, arg in enumerate(args) if arg in {"-cp", "-classpath", "--class-path"}), None)
    if index is None or index + 1 >= len(args):
        raise ValueError("JAVA_RUNTIME_CLASSPATH_MISMATCH")
    cp = args[index + 1]
    if cp.startswith("@"):
        path = Path(cp[1:])
        if not path.is_file() or path.stat().st_size > 2_000_000:
            raise ValueError("JAVA_RUNTIME_CLASSPATH_ARGFILE_INVALID")
        try:
            cp = json.loads(path.read_text(encoding="utf-8"))
        except (UnicodeError, ValueError) as error:
            raise ValueError("JAVA_RUNTIME_CLASSPATH_ARGFILE_INVALID") from error
        if not isinstance(cp, str) or "\n" in cp or "\r" in cp:
            raise ValueError("JAVA_RUNTIME_CLASSPATH_ARGFILE_INVALID")
    project = core.resolve()
    # 逐个路径核对归属，避免同名前缀目录（如workSpace12-old）被当成本项目。
    if not any(Path(entry).is_absolute() and Path(entry).resolve().is_relative_to(project) for entry in cp.split(";")):
        raise ValueError("JAVA_RUNTIME_CLASSPATH_MISMATCH")
    return cp


def archive_multi(core, settings, owner, code, prediction_id, target):
    """使用已运行的正式服务归档，兼容统一入口的JAR，不读取或传输数据库密码。"""
    import httpx

    with httpx.Client(
        timeout=90,
        trust_env=False,
        headers={
            "X-Service-Token": settings.ai_service_token.get_secret_value(),
        },
    ) as client:
        response = client.post(
            settings.core_service_base_url + f"/internal/v1/predictions/sync/users/{owner}/{code}/archive",
            json={"predictionId": prediction_id, "targetDate": target},
        )
        response.raise_for_status()
        value = response.json()
    if (
        value.get("owner") != owner
        or value.get("code") != code
        or value.get("predictionId") != prediction_id
        or not value.get("verified")
    ):
        raise ValueError("OWNER_ARCHIVE_RECEIPT_MISMATCH")
    return value
