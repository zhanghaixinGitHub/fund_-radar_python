"""本机三端验收夹具：Java账户/关注使用随机schema，Python只读真实资料，浏览器查看真实页面。

必须显式--execute；不创建真实账户、不改public关注、不调用第三方、不训练。
在启动输出的log_dir中放置UTF-8的command.txt，可写pause-python、resume-python或stop。
收到stop、Ctrl+C或20分钟期限到达时停止本夹具启动的进程，删除自己拥有的临时schema。
"""

import argparse
import json
import os
import re
import secrets
import shutil
import subprocess
import sys
import time
from pathlib import Path
from uuid import uuid4

import httpx
from dotenv import dotenv_values
from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL

PYTHON_PORT, JAVA_PORT, WEB_PORT = 18000, 18080, 15173
MOBILE, PASSWORD = "19900000001", "AcceptanceOnly42"  # 仅随机测试schema中的合成账号，非真实凭据。


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--core-dir", required=True)
    parser.add_argument("--web-dir", required=True)
    args = parser.parse_args()
    if not args.execute:
        print("PLAN ONLY: isolated schema and three loopback services", flush=True)
        return
    python_root = Path(__file__).resolve().parents[1]
    core, web = Path(args.core_dir).resolve(), Path(args.web_dir).resolve()
    assert (core / "pom.xml").is_file() and (web / "vite.config.ts").is_file()
    settings = {**dotenv_values(core / ".env"), **os.environ}
    host, port = settings.get("FUND_CORE_DB_HOST", "localhost"), int(settings.get("FUND_CORE_DB_PORT", "54329"))
    db = settings.get("FUND_CORE_DB_NAME", "fund_core")
    assert host in {"localhost", "127.0.0.1", "::1"} and db == "fund_core"
    schema = "prediction_e2e_" + uuid4().hex
    url = URL.create(
        "postgresql+psycopg",
        username=settings.get("FUND_CORE_DB_USERNAME"),
        password=settings.get("FUND_CORE_DB_PASSWORD"),
        host=host,
        port=port,
        database=db,
    )
    control = create_engine(
        url,
        hide_parameters=True,
        connect_args={"connect_timeout": 5, "options": "-c statement_timeout=5000 -c lock_timeout=3000"},
    )
    root = python_root / ".local-runs" / schema
    root.mkdir(parents=True, exist_ok=False)
    token = secrets.token_urlsafe(32)
    processes, streams = {}, []
    checks = {}

    def start(name, command, cwd, env):
        log = (root / (name + ".log")).open("ab")
        streams.append(log)
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        processes[name] = process
        return process

    def wait_url(url, headers=None):
        started = time.monotonic()
        while time.monotonic() - started < 55:
            if any(p.poll() is not None for p in processes.values()):
                raise RuntimeError("owned service exited; inspect its local log")
            try:
                response = httpx.get(url, headers=headers, timeout=2)
                if response.status_code < 500:
                    return
            except httpx.HTTPError:
                pass
            time.sleep(0.5)
        raise TimeoutError("owned service readiness timed out")

    def public_fingerprint():
        with control.connect() as conn:
            conn.execute(text("SET TRANSACTION READ ONLY"))
            return {
                name: conn.execute(
                    text(
                        "SELECT count(*), md5(COALESCE(string_agg(md5(to_jsonb(t)::text),'' "
                        "ORDER BY to_jsonb(t)::text),'')) FROM public." + name + " t"
                    )
                ).one()
                for name in ("user_account", "watchlist_item", "role_permission", "auth_session")
            }

    before = public_fingerprint()
    py_env = {**os.environ, "AI_SERVICE_TOKEN": token, "PYTHONIOENCODING": "utf-8"}
    java_env = {
        **os.environ,
        "SPRING_DATASOURCE_URL": f"jdbc:postgresql://{host}:{port}/{db}?currentSchema={schema}",
        "SPRING_DATASOURCE_USERNAME": settings["FUND_CORE_DB_USERNAME"],
        "SPRING_DATASOURCE_PASSWORD": settings["FUND_CORE_DB_PASSWORD"],
        "SPRING_FLYWAY_SCHEMAS": schema,
        "SPRING_FLYWAY_DEFAULT_SCHEMA": schema,
        "SERVER_ADDRESS": "127.0.0.1",
        "SERVER_PORT": str(JAVA_PORT),
        "AI_SERVICE_BASE_URL": f"http://127.0.0.1:{PYTHON_PORT}",
        "AI_SERVICE_TOKEN": token,
        "APP_WEB_ALLOWED_ORIGIN": f"http://127.0.0.1:{WEB_PORT}",
        "FUND_AUTH_INITIAL_ADMIN_PASSWORD": "",
        "FUND_AUTH_INITIAL_ADMIN_MOBILE": "",
        "FUND_READ_CACHE_ENABLED": "false",
        "ANALYSIS_DELIVERY_ENABLED": "false",
        "APP_PORTFOLIO_IMPORT_ENABLED": "false",
    }
    py_command = [sys.executable, "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", str(PYTHON_PORT)]
    java = core / ".tools/jdk17/jdk-17.0.20.1+1/bin/java.exe"
    assert java.is_file()
    try:
        # 不复用已经监听的端口，也不会为验收关闭别人的服务。
        import socket

        for target in (PYTHON_PORT, JAVA_PORT, WEB_PORT):
            with socket.socket() as sock:
                sock.bind(("127.0.0.1", target))
        with control.begin() as conn:
            conn.exec_driver_sql(f'CREATE SCHEMA "{schema}"')
        start("python", py_command, python_root, py_env)
        wait_url(f"http://127.0.0.1:{PYTHON_PORT}/internal/v1/predictions/008888", {"X-Service-Token": token})
        # 对真实冻结报告走HTTP核验；令牌只在内存中，回执不保存请求头/完整报告。
        with httpx.Client(base_url=f"http://127.0.0.1:{PYTHON_PORT}", timeout=15) as ai:
            body = {
                "fundCode": "008888",
                "researchRunId": "f70feb1a-129d-4482-b66d-f4e2e3a5425c",
                "expectedReportHash": "db527ccca8015a4f41af2ee68608dae27ec5ab86c2627ef158d39f2f6b067795",
            }
            path = "/internal/v1/predictions/generation-check"
            checks["precheck_missing_token_403"] = ai.post(path, json=body).status_code == 403
            ai.headers["X-Service-Token"] = token
            checks["precheck_force_422"] = ai.post(path, json={**body, "force": True}).status_code == 422
            checks["precheck_changed_hash_409"] = (
                ai.post(path, json={**body, "expectedReportHash": "f" * 64}).status_code == 409
            )
            for fund in ("001632", "006730", "008888"):
                response = ai.post(path, json={**body, "fundCode": fund})
                payload = response.json()
                checks[f"real_precheck_{fund}_200"] = response.status_code == 200
                checks[f"real_precheck_{fund}_blocked_without_inference"] = (
                    payload.get("status") == "GENERATION_BLOCKED"
                    and payload.get("inference_executed") is False
                    and payload.get("forecast_created") is False
                    and payload.get("database_written") is False
                    and payload.get("up_probability") is None
                    and payload.get("direction") is None
                )
                checks[f"real_precheck_{fund}_evidence_and_no_store"] = (
                    payload.get("report_hash") == body["expectedReportHash"]
                    and len(payload.get("comparisons", [])) == 16
                    and "REPEATABLE_BASELINE_GAIN_NOT_ESTABLISHED" in payload.get("blocking_codes", [])
                    and response.headers.get("cache-control") == "no-store"
                )
            assert all(checks.values()), checks
        start(
            "java",
            [str(java), "-Dfile.encoding=UTF-8", "-jar", str(core / "target/fund-core-0.1.0-SNAPSHOT.jar")],
            core,
            java_env,
        )
        wait_url(f"http://127.0.0.1:{JAVA_PORT}/actuator/health")
        node = shutil.which("node")
        assert node
        start(
            "vue",
            [
                node,
                str(web / "node_modules/vite/bin/vite.js"),
                "--host",
                "127.0.0.1",
                "--port",
                str(WEB_PORT),
                "--strictPort",
            ],
            web,
            {**os.environ, "VITE_API_BASE_URL": f"http://127.0.0.1:{JAVA_PORT}"},
        )
        wait_url(f"http://127.0.0.1:{WEB_PORT}/")
        base = f"http://127.0.0.1:{JAVA_PORT}"
        with httpx.Client(base_url=base, timeout=15) as user, httpx.Client(base_url=base, timeout=15) as other:
            path = "/api/v1/watchlist/008888/prediction"
            checks["anonymous_401"] = user.get(path).status_code == 401
            for client, mobile in ((user, MOBILE), (other, "19900000002")):
                response = client.post(
                    "/api/v1/auth/register",
                    json={"mobile": mobile, "password": PASSWORD, "displayName": "预测页面验收"},
                )
                assert response.status_code == 200, response.status_code
                client.headers["X-CSRF-Token"] = client.cookies.get("fund_radar_csrf")
            checks["not_followed_403"] = user.get(path).status_code == 403
            assert user.post("/api/v1/watchlist", json={"fundCode": "008888"}).status_code == 200
            checks["other_user_403"] = other.get(path).status_code == 403
            response = user.get(path)
            payload = response.json()["data"]
            checks["own_followed_200"] = response.status_code == 200
            checks["real_research_read"] = payload["researchRunId"] == "f70feb1a-129d-4482-b66d-f4e2e3a5425c"
            checks["probability_withheld"] = payload["upProbability"] is None and payload["direction"] is None
            checks["page_has_actual_comparison_reason"] = (
                "REPEATABLE_BASELINE_GAIN_NOT_ESTABLISHED" in payload["reasonCodes"]
            )
            checks["no_store"] = response.headers["cache-control"] == "no-store, private"
            checks["trace_present"] = bool(response.json().get("traceId"))
            checks["invalid_fund_400"] = user.get("/api/v1/watchlist/abcdef/prediction").status_code == 400
            checks["wrong_origin_403"] = (
                user.get(path, headers={"Origin": "https://invalid.example"}).status_code == 403
            )
            checks["detail_200"] = user.get("/api/v1/watchlist/008888/detail").status_code == 200
            assert all(checks.values()), checks
            user.post("/api/v1/auth/logout")
            other.post("/api/v1/auth/logout")
        (root / "http-receipt.json").write_text(
            json.dumps(
                {"checks": checks, "scope": "isolated accounts, real research, no numerical forecast"}, indent=2
            ),
            encoding="utf-8",
        )
        print(
            json.dumps(
                {
                    "ready": True,
                    "url": f"http://127.0.0.1:{WEB_PORT}/watchlist/008888",
                    "schema": schema,
                    "log_dir": str(root),
                    "checks": checks,
                    "commands": ["pause-python", "resume-python", "stop"],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        # Windows下PTY的stdin不一定转发给Python；使用本次独占目录中的控制文件。
        # 文件只接受固定三条测试命令，不执行任意脚本；最长20分钟后自动清理。
        control_file = root / "command.txt"
        session_deadline = time.monotonic() + 1200
        while time.monotonic() < session_deadline:
            if not control_file.exists():
                time.sleep(0.25)
                continue
            command = control_file.read_text(encoding="utf-8").strip()
            control_file.unlink()
            if command == "stop":
                break
            if command == "pause-python":
                processes["python"].terminate()
                processes["python"].wait(timeout=10)
                del processes["python"]
                print("python paused", flush=True)
            elif command == "resume-python" and "python" not in processes:
                start("python", py_command, python_root, py_env)
                wait_url(f"http://127.0.0.1:{PYTHON_PORT}/internal/v1/predictions/008888", {"X-Service-Token": token})
                print("python resumed", flush=True)
            elif command:
                print("unknown command", flush=True)
    finally:
        for process in reversed(list(processes.values())):
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
        for stream in streams:
            stream.close()
        assert re.fullmatch(r"prediction_e2e_[0-9a-f]{32}", schema)
        with control.begin() as conn:
            owner = conn.execute(
                text("SELECT nspowner=current_user::regrole FROM pg_namespace WHERE nspname=:name"), {"name": schema}
            ).scalar_one_or_none()
            if owner is not None:
                assert owner
                conn.exec_driver_sql(f'DROP SCHEMA "{schema}" CASCADE')
        after = public_fingerprint()
        print(
            json.dumps(
                {
                    "stopped": True,
                    "isolated_schema_removed": schema,
                    "real_accounts_watchlist_sessions_unchanged": before == after,
                }
            ),
            flush=True,
        )
        control.dispose()


if __name__ == "__main__":
    main()
