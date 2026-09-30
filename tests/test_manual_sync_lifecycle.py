"""服务生命周期只管理资源，不自动取数；旧配置也不能重新开启三个维护任务。"""

import threading

from app.core.config import get_settings
from app.services import fund_exposure_runtime, fund_materials_sync, tushare_fund_sync
from fastapi.testclient import TestClient


def test_startup_and_shutdown_never_start_automatic_collection(monkeypatch):
    from app import main

    calls = []
    closed = []
    monkeypatch.setenv("TUSHARE_MARKET_INCREMENTAL_ENABLED", "true")
    get_settings.cache_clear()
    monkeypatch.setattr(
        tushare_fund_sync.TushareFundSyncService, "sync_market_nav_incremental",
        lambda *args, **kwargs: calls.append("nav"),
    )
    monkeypatch.setattr(fund_exposure_runtime, "maintenance_once", lambda **kwargs: calls.append("input"))
    monkeypatch.setattr(fund_materials_sync, "schedule_if_due", lambda: calls.append("materials"))
    monkeypatch.setattr(main, "close_sync_job_manager", lambda: closed.append(True))
    original_start = threading.Thread.start

    def start(thread):
        # 旧循环首轮会等30秒，不能仅凭测试期间没有取数就宣称自动任务已移除。
        assert thread.name != "nav-repair"
        return original_start(thread)

    monkeypatch.setattr(threading.Thread, "start", start)
    try:
        with TestClient(main.create_application()) as client:
            assert client.get("/route-that-does-not-exist").status_code == 404
            assert calls == []
        assert calls == [] and closed == [True]
    finally:
        get_settings.cache_clear()
