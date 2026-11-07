"""禁止旧工程/原件访问及外网连接，以真实只读库验收ASGI读取路由。"""

import json
import os
import sys

from app.core.config import get_settings
from app.core.workspace_paths import PYTHON_ROOT, SOURCE_ROOT, WORKSPACE_ROOT
from sqlalchemy.engine import make_url


def main():
    settings = get_settings()
    url = make_url(settings.ai_database_url).update_query_dict({"options": "-c default_transaction_read_only=on"})
    os.environ["AI_DATABASE_URL"] = url.render_as_string(hide_password=False)
    get_settings.cache_clear()
    control = PYTHON_ROOT / ".local-runs/old-project-retirement"
    blocked = [os.path.abspath(p).casefold() for p in (
        "C:/WebStormProject/workSpace05", "C:/ideaProject/workSpace12", "C:/pythonProject/workSpace06",
        "D:/fundRadarSourceFile", SOURCE_ROOT,
    )]
    file_attempts, network_attempts, write_attempts = [], [], []

    def audit(event, args):
        if event in {"open", "os.scandir"} and args and isinstance(args[0], (str, bytes, os.PathLike)):
            path = os.path.abspath(os.fsdecode(args[0])).casefold()
            if any(path == root or path.startswith(root + os.sep) for root in blocked):
                file_attempts.append({"event": event, "path": path})
                raise PermissionError("VERIFICATION_OLD_PROJECT_OR_ORIGINAL_ACCESS_BLOCKED")
            if event == "open":
                mode, flags = args[1:3]
                writing = isinstance(mode, str) and any(c in mode for c in "wax+")
                writing = writing or isinstance(flags, int) and bool(
                    flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND)
                )
                if writing and not path.startswith(str(control).casefold() + os.sep):
                    write_attempts.append(path)
                    raise PermissionError("VERIFICATION_BUSINESS_FILE_WRITE_BLOCKED")
        if event == "socket.connect":
            address = args[1]
            if isinstance(address, tuple) and str(address[0]) not in {"127.0.0.1", "::1", "localhost"}:
                network_attempts.append(str(address[0]))
                raise PermissionError("VERIFICATION_EXTERNAL_NETWORK_BLOCKED")

    sys.addaudithook(audit)
    from app.db.session import get_engine
    from app.main import create_application
    from fastapi.testclient import TestClient

    with get_engine().connect() as connection:
        readonly = connection.exec_driver_sql("SHOW transaction_read_only").scalar()
        connection.rollback()
    if readonly != "on":
        raise ValueError("DATABASE_NOT_READ_ONLY")
    headers = {"X-Service-Token": get_settings().ai_service_token.get_secret_value()}
    paths = [
        "/health", "/funds?pageSize=10", "/funds/008888", "/funds/008888/watchlist-detail",
        "/funds/008888/nav-history?startDate=2026-09-01&endDate=2026-11-06",
        "/funds/008888/share-history?startDate=2026-09-01&endDate=2026-11-06",
        "/funds/008888/same-type-comparison", "/funds/008888/share-classes", "/funds/008888/risk-summary",
        "/funds/008888/news-facts", "/funds/008888/materials",
        "/funds/008888/materials/documents?page=1&pageSize=10", "/funds/evaluation-status?fundCodes=008888",
    ]
    results = []
    with TestClient(create_application()) as client:
        for path in paths:
            response = client.get("/internal/v1" + path, headers=headers)
            result = {"path": path, "status": response.status_code}
            if path.endswith("/share-classes") and response.status_code == 200:
                body = response.json()
                result.update(business_status=body.get("status"), shares=len(body.get("shares", [])))
            results.append(result)
    report = {"workspace": str(WORKSPACE_ROOT), "validation": "ASGI routes with real read-only PostgreSQL",
              "transaction_read_only": readonly, "requests": results, "forbidden_file_accesses": file_attempts,
              "external_network_attempts": network_attempts, "business_file_write_attempts": write_attempts,
              "all_routes_200": all(item["status"] == 200 for item in results)}
    control.mkdir(parents=True, exist_ok=True)
    (control / "guarded-api-readback.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report), flush=True)
    # 拦截记录和份额业务状态均作为验收条件，避免HTTP正常掩盖旧路径回退或业务降级。
    shares_verified = any(item.get("business_status") == "VERIFIED" and item.get("shares", 0) > 0
                          for item in results)
    if not report["all_routes_200"] or file_attempts or network_attempts or write_attempts or not shares_verified:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
