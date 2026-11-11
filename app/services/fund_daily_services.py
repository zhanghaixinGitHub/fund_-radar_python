"""只启动缺失的本机项目服务；复用运行中/启动中进程，不停止或重启服务。"""

import json
import os
import subprocess
from pathlib import Path

import httpx

from app.core.config import get_settings
from app.services.fund_materials_store import versioned_save

PYTHON = Path(__file__).resolve().parents[2]
JAVA = Path(r"D:\Project-FullStack\workSpace01\02-java")


def powershell(script, *, capture=True):
    result = subprocess.run(
        ["powershell.exe", "-NoProfile", "-Command", script],
        # 隐藏启动时不建立捕获管道。Windows子进程可能继承管道句柄，
        # 即使启动命令已退出，communicate也可能等到常驻服务结束才返回。
        # 服务本身的标准输出/错误已定向到本轮日志，不依赖启动命令的输出。
        stdout=subprocess.PIPE if capture else subprocess.DEVNULL,
        stderr=subprocess.PIPE if capture else subprocess.DEVNULL,
        text=True,
        encoding="utf-8",
        timeout=20,
    )
    if result.returncode:
        raise RuntimeError("LOCAL_SERVICE_PROBE_FAILED")
    return json.loads(result.stdout.strip() or "null") if capture else None


def owned_process(port, project):
    """端口与完整命令行双重确认归属；无法证明时宁可报端口冲突，不抢占。"""
    rows = powershell(
        # Windows PowerShell 在按不存在端口查询时会返回非零退出码；服务未启动
        # 应进入现有启动分支，而不是被误判为探针故障。先成功查询监听表再筛选，
        # 同时保留真正的权限/系统查询错误，不能用忽略异常将其伪装成端口空闲。
        "Get-NetTCPConnection -State Listen -ErrorAction Stop | "
        f"Where-Object {{ $_.LocalPort -eq {port} }} | "
        "Select-Object -ExpandProperty OwningProcess -Unique | ForEach-Object { "
        'Get-CimInstance Win32_Process -Filter "ProcessId=$_" | '
        "Select-Object ProcessId,ExecutablePath,CommandLine } | ConvertTo-Json -Compress"
    )
    rows = rows if isinstance(rows, list) else [rows] if rows else []
    if len(rows) > 1:
        raise ValueError("LOCAL_SERVICE_MULTIPLE_LISTENERS")
    if rows:
        row = rows[0]
        proof = (row.get("ExecutablePath", "") + " " + row.get("CommandLine", "")).replace("\\", "/").lower()
        if str(project).replace("\\", "/").lower() not in proof:
            raise ValueError("PORT_OWNED_BY_OTHER_PROJECT:" + str(port))
        return row
    return None


def ensure_services(directory):
    """统一通过根dev.ps1恢复本项目服务；不再另起Maven或绕过D盘环境。"""
    if os.name != "nt":
        raise ValueError("LOCAL_WINDOWS_PROJECT_REQUIRED")
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    services = (("python", 8000, PYTHON), ("java", 8080, JAVA))
    before = {name: owned_process(port, project) for name, port, project in services}
    if not all(before.values()):
        result = subprocess.run(
            ["powershell.exe", "-NoProfile", "-File", str(PYTHON.parent / "dev.ps1"), "Start"],
            cwd=PYTHON.parent,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=200,
        )
        if result.returncode:
            raise RuntimeError("WORKSPACE_DEV_START_FAILED")
    cfg = get_settings()
    records = []
    with httpx.Client(
        timeout=5,
        trust_env=False,
        headers={
            "X-Service-Token": cfg.ai_service_token.get_secret_value(),
        },
    ) as client:
        for name, port, project in services:
            process = owned_process(port, project)
            url = (
                "http://127.0.0.1:8000/internal/v1/health"
                if name == "python"
                else cfg.core_service_base_url + "/actuator/health"
            )
            response = client.get(url) if process else None
            ready = (
                response is not None
                and response.status_code == 200
                and response.json().get("status", "").upper() in {"UP", "OK"}
            )
            records.append({"service": name, "port": port, "started": not bool(before[name]), "ready": ready})
            versioned_save(directory / "service-preflight.json", records)
            if not ready:
                raise RuntimeError("PROJECT_SERVICE_NOT_READY:" + name)
    return records
