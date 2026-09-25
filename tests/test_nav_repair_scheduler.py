"""验证后台生命周期和失败后继续检查；所有取数均使用替身。"""

from types import SimpleNamespace

from app.services import nav_repair_scheduler as scheduler


def test_scheduler_starts_stops_and_keeps_next_check_after_failure(monkeypatch):
    calls, waits = [], []

    class Stop:
        def wait(self, seconds):
            waits.append(seconds)
            return len(waits) > 2

        def set(self):
            pass

    def repair():
        calls.append("checked")
        if len(calls) == 1:
            raise RuntimeError("synthetic source failure")

    monkeypatch.setattr(scheduler, "repair_once", repair)
    exposure_calls = []

    def exposure():
        exposure_calls.append("checked")
        return {"status": "NOT_ENABLED"}

    monkeypatch.setattr(scheduler, "exposure_maintenance_once", exposure)
    materials_calls = []
    monkeypatch.setattr(scheduler, "materials_schedule_if_due", lambda: materials_calls.append("checked"))
    monkeypatch.setattr(scheduler, "get_settings", lambda: SimpleNamespace(tushare_market_incremental_enabled=True))
    worker = scheduler.NavRepairScheduler()
    worker._stop = Stop()
    worker.start()
    worker._thread.join(timeout=2)
    worker.close()
    assert calls == ["checked", "checked"] and waits == [30, 1800, 1800]
    assert exposure_calls == ["checked", "checked"]
    assert materials_calls == ["checked", "checked"]
    assert not worker._thread.is_alive()


def test_disabled_scheduler_does_not_start_source_calls(monkeypatch):
    monkeypatch.setattr(scheduler, "get_settings", lambda: SimpleNamespace(tushare_market_incremental_enabled=False))
    worker = scheduler.NavRepairScheduler()
    worker.start()
    worker.close()
    assert worker._thread is None
