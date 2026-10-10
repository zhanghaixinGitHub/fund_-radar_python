"""行情超时诊断日志：覆盖提前告警、断开/取消、脱敏、线程上下文及既有接口边界。"""

import asyncio
import logging
import socket
import threading
import time
from datetime import date
from unittest.mock import MagicMock

import httpx
import pytest
import uvicorn
from app.core import market_observability as obs
from app.core.config import get_settings
from fastapi import FastAPI
from fastapi.testclient import TestClient

SCOPE = {"type": "http", "method": "GET", "path": "/internal/v1/simulation/funds/007045"}


async def invoke(app, scope=None):
    """用内存 ASGI 消息调用，不连接真实服务或数据库。"""
    messages = []

    async def receive():
        return {"type": "http.disconnect"}

    async def send(message):
        messages.append(message)

    await obs.SimulationMarketLoggingMiddleware(app)(scope or SCOPE, receive, send)
    assert obs._observation.get() is None
    return messages


async def respond(send, status=200):
    await send({"type": "http.response.start", "status": status, "headers": []})
    await send({"type": "http.response.body", "body": b"{}", "more_body": False})


def test_worker_stages_and_trace_survive_thread_handoff_without_sensitive_data(caplog, monkeypatch):
    caplog.set_level(logging.DEBUG, logger=obs.__name__)
    monkeypatch.setattr(obs, "get_trace_id", lambda: "market-test-trace")

    @obs.observe_market_read
    def read():
        with obs.market_stage("fund_connection"), obs.market_stage("nav_history"):
            return obs._observation.get().trace_id

    async def app(scope, receive, send):
        assert await asyncio.to_thread(read) == "market-test-trace"
        await respond(send)

    scope = {**SCOPE, "query_string": b"token=do-not-log", "headers": [(b"authorization", b"secret-header")]}
    asyncio.run(invoke(app, scope))
    assert "phase=start traceId=market-test-trace fundCode=007045" in caplog.text
    assert "outcome=response_complete" in caplog.text
    assert "service_start_delay" in caplog.text and "nav_history" in caplog.text
    assert "do-not-log" not in caplog.text and "secret-header" not in caplog.text


def test_pending_warning_identifies_active_stage_once(caplog, monkeypatch):
    caplog.set_level(logging.INFO, logger=obs.__name__)
    monkeypatch.setattr(obs, "PENDING_SECONDS", 0.01)

    async def app(scope, receive, send):
        with obs.market_stage("market_connection"):
            await asyncio.sleep(0.04)
        await respond(send)

    asyncio.run(invoke(app))
    pending = [r for r in caplog.records if "phase=pending" in r.getMessage()]
    assert len(pending) == 1
    assert pending[0].levelno == logging.WARNING
    assert "stage=market_connection" in pending[0].getMessage()


def test_cancelled_and_disconnected_request_still_has_terminal_log(caplog):
    caplog.set_level(logging.INFO, logger=obs.__name__)

    async def app(scope, receive, send):
        await receive()
        with obs.market_stage("nav_history"):
            raise asyncio.CancelledError()

    async def check():
        with pytest.raises(asyncio.CancelledError):
            await invoke(app)
        assert obs._observation.get() is None

    asyncio.run(check())
    assert "outcome=cancelled" in caplog.text
    assert "disconnectObserved=True" in caplog.text
    assert "failedStage=nav_history" in caplog.text


def test_query_failure_keeps_stack_and_stage_but_not_exception_payload(caplog):
    caplog.set_level(logging.INFO, logger=obs.__name__)
    private_message = "sql-or-token-must-not-appear"

    async def app(scope, receive, send):
        with obs.market_stage("fund_detail"), obs.market_stage("fund_latest_nav"):
            raise ValueError(private_message)

    with pytest.raises(ValueError, match="sql-or-token"):
        asyncio.run(invoke(app))
    assert "failedStage=fund_latest_nav" in caplog.text
    assert "errorType=ValueError" in caplog.text
    assert "sql-or-token-must-not-appear" not in caplog.text
    assert any(r.exc_info and r.exc_info[2] is not None for r in caplog.records)


def test_unrelated_route_has_no_market_logs_or_observation(caplog):
    caplog.set_level(logging.INFO, logger=obs.__name__)

    async def app(scope, receive, send):
        assert obs._observation.get() is None
        await respond(send)

    asyncio.run(invoke(app, {**SCOPE, "path": "/internal/v1/health"}))
    assert "SimulationMarket.http" not in caplog.text


def test_completed_request_cancels_pending_callback(caplog, monkeypatch):
    caplog.set_level(logging.INFO, logger=obs.__name__)
    monkeypatch.setattr(obs, "PENDING_SECONDS", 0.01)

    async def app(scope, receive, send):
        await respond(send)

    async def check():
        await invoke(app)
        await asyncio.sleep(0.03)

    asyncio.run(check())
    assert "phase=pending" not in caplog.text


def test_real_client_read_timeout_keeps_request_and_pending_stage_logs(caplog, monkeypatch):
    """复现原故障：HTTP 客户端先超时断开，服务端仍留下请求号、等待阶段及结束记录。"""
    from app.core.middleware import TraceIdMiddleware

    caplog.set_level(logging.INFO, logger=obs.__name__)
    monkeypatch.setattr(obs, "PENDING_SECONDS", 0.02)
    release = threading.Event()
    entered = threading.Event()
    application = FastAPI()
    application.add_middleware(obs.SimulationMarketLoggingMiddleware)
    application.add_middleware(TraceIdMiddleware)

    @application.get(SCOPE["path"])
    @obs.observe_market_read
    def market():
        with obs.market_stage("nav_history"):
            entered.set()
            release.wait(2)
        return {"fundCode": "007045"}

    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    server = uvicorn.Server(uvicorn.Config(application, log_config=None, access_log=False))
    thread = threading.Thread(target=server.run, kwargs={"sockets": [listener]}, daemon=True)
    thread.start()
    try:
        deadline = time.monotonic() + 3
        while not server.started and thread.is_alive() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert server.started
        with httpx.Client(trust_env=False, timeout=httpx.Timeout(0.15, connect=1)) as client:
            with pytest.raises(httpx.ReadTimeout):
                client.get(
                    f"http://127.0.0.1:{listener.getsockname()[1]}" + SCOPE["path"],
                    headers={"X-Trace-Id": "disconnected-client-trace"},
                )
        assert entered.is_set()
    finally:
        release.set()
        server.should_exit = True
        thread.join(timeout=3)
        listener.close()
    assert not thread.is_alive()
    assert "phase=start traceId=disconnected-client-trace" in caplog.text
    assert "phase=pending traceId=disconnected-client-trace" in caplog.text
    assert "stage=nav_history" in caplog.text
    assert "phase=end traceId=disconnected-client-trace" in caplog.text


def test_real_route_logs_database_stages_and_preserves_authentication(caplog, monkeypatch):
    """真实路由和服务函数配测试数据库替身，确保阶段不会因线程切换丢失。"""
    from app.main import create_application
    from app.services import fund_catalog_read as catalog
    from app.services import simulation_market as market

    monkeypatch.setenv("AI_SERVICE_TOKEN", "test-market-secret")
    get_settings.cache_clear()
    session = MagicMock()
    session.__enter__.return_value = session
    session.scalars.return_value.all.return_value = []
    session.execute.return_value.mappings.return_value.first.return_value = None
    for module in (catalog, market):
        monkeypatch.setattr(module, "get_engine", lambda: None)
        monkeypatch.setattr(module, "Session", lambda engine: session)
    monkeypatch.setattr(catalog, "get_fund_summary", lambda *args: object())
    monkeypatch.setattr(catalog, "list_fund_summaries_by_codes", lambda *args: [object()])
    monkeypatch.setattr(catalog, "get_fund_profile_snapshot", lambda *args: None)
    fund = MagicMock(fund_name="测试基金", unit_nav=1, data_source="TUSHARE_PRO_FUND")
    monkeypatch.setattr(catalog, "_to_detail", lambda *args: fund)
    caplog.set_level(logging.DEBUG, logger=obs.__name__)
    try:
        with TestClient(create_application()) as client:
            url = SCOPE["path"] + "?startDate=2026-09-04&endDate=" + date.today().isoformat()
            headers = {"X-Service-Token": "test-market-secret", "X-Trace-Id": "java-shared-trace"}
            response = client.get(url, headers=headers)
            assert response.status_code == 200
            assert response.headers["X-Trace-Id"] == "java-shared-trace"
            denied = client.get(url, headers={"X-Trace-Id": "denied-trace"})
            assert denied.status_code == 403
        for stage in ("fund_connection", "fund_latest_nav", "fund_summary_and_performance", "fund_profile",
                      "market_connection", "source_registry", "nav_history", "dividends", "refresh_state"):
            assert stage in caplog.text
        assert "traceId=java-shared-trace" in caplog.text and "status=403" in caplog.text
        assert "test-market-secret" not in caplog.text
    finally:
        get_settings.cache_clear()
