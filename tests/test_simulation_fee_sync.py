"""费率任务只使用来源与 Java HTTP 替身，验证范围、保存回执、部分失败及页面状态恢复。"""

from decimal import Decimal
from threading import Event

import httpx
import pytest
from app.services import simulation_fee_sync
from app.services.eastmoney_fee import FeeBand, FeeProfile
from app.services.simulation_fee_sync import FeeSyncResult, SimulationFeeSyncService
from app.services.sync_jobs import LocalSyncJobManager, SyncJobInProgressError
from tests.test_sync_all_jobs import wait_finished


def profile(code):
    return FeeProfile(code, "测试基金", Decimal("0.001"), Decimal("0.01"), "优惠费率",
                      (FeeBand(0, 6, Decimal("0.015")), FeeBand(7, None, Decimal("0"))))


def service(handler):
    # HTTP transport 替身让真实序列化、状态码判断和并发执行全部参与验证，不触碰数据库。
    instance = object.__new__(SimulationFeeSyncService)
    instance._client = httpx.Client(base_url="http://core.test", transport=httpx.MockTransport(handler))
    return instance


def test_full_scope_deduplicates_codes_and_keeps_other_saves_after_parse_failure(monkeypatch):
    saved, progress = [], []

    def fetch(code):
        if code == "000002":
            raise ValueError("MANUAL_REQUIRED: 非自然天档位")
        return profile(code)

    def handle(request):
        if request.method == "GET":
            return httpx.Response(200, json=["000001", "000002", "000001"])
        import json
        body = json.loads(request.content)
        assert body["dataSource"] == "EASTMONEY_F10"
        assert body["purchaseRate"] == "0.001"
        assert set(body) == {"fundCode", "fundName", "purchaseRate", "purchaseOriginalRate",
                             "discountInfo", "redeemBands", "dataSource"}
        saved.append(body["fundCode"])
        return httpx.Response(204)

    monkeypatch.setattr(simulation_fee_sync, "fetch_fee_profile", fetch)
    worker = service(handle)
    try:
        result = worker.sync(progress_reporter=lambda *args: progress.append(args))
        assert (result.total, result.updated) == (2, 1)
        assert len(result.failures) == 1 and "000002" in result.failures[0]
        assert saved == ["000001"]
        assert [item[0] for item in progress] == [0, 1, 2]
    finally:
        worker.close()


def test_single_does_not_expand_scope_or_retry_an_unconfirmed_write(monkeypatch):
    calls = []

    def handle(request):
        calls.append(request.method)
        raise httpx.ReadTimeout("write result unknown", request=request)

    monkeypatch.setattr(simulation_fee_sync, "fetch_fee_profile", profile)
    worker = service(handle)
    try:
        result = worker.sync("008888", progress_reporter=lambda *args: None)
        assert (result.total, result.updated, len(result.failures)) == (1, 0, 1)
        assert calls == ["POST"]
    finally:
        worker.close()


def test_empty_scope_makes_no_fetch_or_write(monkeypatch):
    monkeypatch.setattr(simulation_fee_sync, "fetch_fee_profile", lambda code: pytest.fail("unexpected source call"))
    worker = service(lambda request: httpx.Response(200, json=[]))
    try:
        assert worker.sync(progress_reporter=lambda *args: None) == FeeSyncResult(0, 0, ())
    finally:
        worker.close()


@pytest.mark.parametrize("result,status", [
    (FeeSyncResult(2, 2, ()), "SUCCEEDED"),
    (FeeSyncResult(2, 1, ("000002：解析失败",)), "PARTIAL_SUCCESS"),
    (FeeSyncResult(1, 0, ("000002：写入未确认",)), "FAILED"),
    (FeeSyncResult(0, 0, ()), "SUCCEEDED"),
])
def test_fee_task_excludes_batch_and_preserves_real_outcome(result, status):
    entered, release = Event(), Event()
    requested = []

    class Worker:
        def sync(self, fund_code, *, progress_reporter):
            requested.append(fund_code)
            progress_reporter(0, result.total, fund_code, "正在抓取")
            entered.set()
            assert release.wait(3)
            return result

        def close(self):
            pass

    manager = LocalSyncJobManager(fee_service_factory=Worker)
    try:
        started = manager.start_simulation_fees("008888")
        assert entered.wait(3)
        assert manager.get_latest_job("SIMULATION_FEES").job_id == started.job_id
        with pytest.raises(SyncJobInProgressError):
            manager.start_all()
        with pytest.raises(SyncJobInProgressError):
            manager.start_simulation_fees()
        release.set()
        final = wait_finished(manager, started.job_id)
        assert requested == ["008888"]
        assert final.status == status
        assert final.updated_count == result.updated
        assert final.progress_current == final.progress_total == result.total
        assert final.fund_codes == ("008888",)
        assert final.finished_at is not None
    finally:
        release.set()
        manager.close()
