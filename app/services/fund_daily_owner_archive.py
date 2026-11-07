"""多周期本人归档的稳定本机入口；调用Java正式方法，凭据仅经子进程标准输入。"""

import ctypes
import json
import subprocess
from ctypes import wintypes
from pathlib import Path

from dotenv import dotenv_values

from app.services.fund_daily_services import owned_process


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
    process = owned_process(8080, core)
    if not process:
        raise ValueError("OFFICIAL_OWNER_SERVICE_NOT_RUNNING")
    args = windows_argv(process["CommandLine"])
    for arg in list(args):
        if arg.startswith("@"):
            args += windows_argv("java " + Path(arg[1:]).read_text(encoding="utf-8").replace("\n", " "))[1:]
    if not any("com.fundradar.core.FundCoreApplication" in arg for arg in args):
        raise ValueError("JAVA_RUNTIME_MAIN_CLASS_MISMATCH")
    cp = runtime_classpath(args, core)
    env = dotenv_values(core / ".env")
    request = {
        "owner": owner,
        "code": code,
        "predictionId": prediction_id,
        "targetDate": target,
        "jdbcUrl": "jdbc:postgresql://"
        + env.get("FUND_CORE_DB_HOST", "localhost")
        + ":"
        + env.get("FUND_CORE_DB_PORT", "54329")
        + "/"
        + env.get("FUND_CORE_DB_NAME", "fund_core"),
        "dbUser": env["FUND_CORE_DB_USERNAME"],
        "dbPassword": env["FUND_CORE_DB_PASSWORD"],
        "serviceToken": settings.ai_service_token.get_secret_value(),
        "baseUrl": "http://127.0.0.1:8000",
    }
    result = subprocess.run(
        [
            process["ExecutablePath"],
            "-Dfile.encoding=UTF-8",
            "--class-path",
            cp,
            str(core / "scripts/OwnerPredictionArchive.java"),
        ],
        input=json.dumps(request),
        cwd=core,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=90,
    )
    request = env = None
    if result.returncode:
        # 不记录完整子进程输出，避免异常携带连接串或凭据。
        raise RuntimeError("OFFICIAL_JAVA_OWNER_ARCHIVE_FAILED:EXIT_" + str(result.returncode))
    line = next(v for v in result.stdout.splitlines() if v.startswith("OWNER_ARCHIVE_RESULT:"))
    value = json.loads(line.partition(":")[2])
    if (
        value.get("owner") != owner
        or value.get("code") != code
        or value.get("predictionId") != prediction_id
        or not value.get("verified")
    ):
        raise ValueError("OWNER_ARCHIVE_RECEIPT_MISMATCH")
    return value
