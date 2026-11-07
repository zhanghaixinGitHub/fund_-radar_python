"""隔离运行混合读取相关回归：临时文件落D盘，禁止真实业务库和外网连接及C盘写入。"""

import os
import socket
import sys
import threading
from datetime import datetime

import pytest
from app.core.workspace_paths import PYTHON_ROOT, WORKSPACE_ROOT
from app.services import fund_materials_store
from sqlalchemy.engine import Engine
from sqlalchemy.exc import OperationalError


def main():
    run = WORKSPACE_ROOT / ".local-runs/mixed-reading-tests" / datetime.now().strftime("run-%Y%m%d-%H%M%S")
    run.mkdir(parents=True, exist_ok=False)
    original_connect = socket.socket.connect
    original_database_connect = Engine.connect
    original_history_root = fund_materials_store.PROJECT
    fund_materials_store.PROJECT = run / "project-history"

    def database_connect(engine, *args, **kwargs):
        # psycopg的C实现不会经过Python socket.connect，必须在SQLAlchemy连接入口再拦一次。
        if engine.dialect.name != "sqlite":
            raise OperationalError("UNIT_TEST_REAL_DATABASE_BLOCKED", {}, None)
        return original_database_connect(engine, *args, **kwargs)

    def connect(instance, address):
        # Windows asyncio的socketpair使用本机临时端口；保留内部通信但封住实际服务与业务库。
        if isinstance(address, tuple) and str(address[0]) in {"127.0.0.1", "::1", "localhost"}:
            if address[1] not in {5432, 54329, 6379, 8000, 8080, 5173}:
                return original_connect(instance, address)
        raise RuntimeError("UNIT_TEST_REAL_NETWORK_BLOCKED")

    def audit(event, args):
        if event == "open" and args and isinstance(args[0], (str, bytes, os.PathLike)):
            path = os.path.abspath(os.fsdecode(args[0])).casefold()
            mode, flags = args[1:3]
            writing = isinstance(mode, str) and any(c in mode for c in "wax+")
            writing = writing or isinstance(flags, int) and bool(
                flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND)
            )
            if writing and path.startswith("c:"):
                raise PermissionError("UNIT_TEST_C_DRIVE_WRITE_BLOCKED")

    def timeout():
        print("Mixed-reading regression exceeded 180 seconds; business database access was blocked.", flush=True)
        os._exit(124)

    socket.socket.connect = connect
    Engine.connect = database_connect
    sys.addaudithook(audit)
    timer = threading.Timer(180, timeout)
    timer.daemon = True
    timer.start()
    files = [
        "test_material_mixed_reading.py", "test_workspace_relocation.py", "test_fund_codex_analysis.py",
        "test_fund_materials_store_paths.py", "test_fund_materials_sync.py", "test_fund_rating_market_materials.py",
        "test_market_sources.py", "test_market_workflow.py", "test_market_holdings_parse.py",
        "test_market_holdings_maintenance.py", "test_fund_peer_materials.py", "test_fund_materials.py",
        "test_fund_material_nav.py", "test_fund_material_news.py",
        "test_old_project_retirement.py",
    ]
    try:
        return pytest.main([
            "-q", "-p", "no:cacheprovider", "--basetemp=" + str(run / "tmp"), "--tb=short", "--maxfail=6",
            "-o", "faulthandler_timeout=45", "--junitxml=" + str(run / "results.xml"),
            *(str(PYTHON_ROOT / "tests" / name) for name in files),
        ])
    finally:
        timer.cancel()
        socket.socket.connect = original_connect
        Engine.connect = original_database_connect
        fund_materials_store.PROJECT = original_history_root


if __name__ == "__main__":
    raise SystemExit(main())
