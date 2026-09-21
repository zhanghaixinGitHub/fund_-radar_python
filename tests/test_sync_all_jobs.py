"""一键同步的后台串行、失败继续、互斥及 HTTP 权限契约；不调用真实来源。"""

from datetime import date
from threading import Event
from time import monotonic, sleep
from uuid import uuid4

import pytest
from app.core.config import get_settings
from app.services.simulation_fee_sync import FeeSyncResult
from app.services.stock_feature_snapshot import StockFeatureBuildSummary
from app.services.sync_jobs import MARKET_ALL_JOB_TYPE, LocalSyncJobManager, SyncJobInProgressError
from app.services.tushare_free_data_completion import FreeDataCompletionResult
from app.services.tushare_fund_sync import MarketDetailSyncResult, SyncOutcome
from fastapi.testclient import TestClient

STAGES = (
    "SPX_MANUAL",
    "MARKET_DETAIL",
    "MARKET_FREE_DATA_COMPLETION",
    "MARKET_NAV_INCREMENTAL",
    "STOCK_FEATURE_SNAPSHOT",
    "SIMULATION_FEES",
)


def make_manager(calls, failures=(), stage_hook=lambda _: None, close_hook=lambda _: None):
    def run(stage, reporter):
        calls.append(stage)
        reporter(1, 2, "000001.OF", "测试来源实际进度")
        stage_hook(stage)
        if stage in failures:
            raise ValueError("test stage failure")
        return SyncOutcome(uuid4(), stage, date(2026, 9, 10), 3, 1, 1, 1)

    class FundService:
        stage = None

        def sync_market_details(self, *, progress_reporter):
            self.stage = "MARKET_DETAIL"
            return MarketDetailSyncResult(run(self.stage, progress_reporter), ())

        def sync_market_nav_incremental(self, *, progress_reporter):
            self.stage = "MARKET_NAV_INCREMENTAL"
            return run(self.stage, progress_reporter)

        def close(self):
            close_hook(self.stage)

    class FreeService:
        def sync(self, *, progress_reporter):
            return FreeDataCompletionResult(run("MARKET_FREE_DATA_COMPLETION", progress_reporter), ())

        def close(self):
            close_hook("MARKET_FREE_DATA_COMPLETION")

    class FeatureService:
        def build(self, *, progress_reporter):
            run("STOCK_FEATURE_SNAPSHOT", progress_reporter)
            return StockFeatureBuildSummary(
                status="COMPLETED",
                source_code="TUSHARE_PRO_FUND",
                source_sync_run_id=uuid4(),
                attempted_fund_count=2,
                scorable_count=1,
                data_insufficient_count=1,
                no_nav_count=0,
                created_count=1,
                updated_count=0,
                skipped_count=1,
            )

    def spx_sync():
        calls.append("SPX_MANUAL")
        stage_hook("SPX_MANUAL")
        if "SPX_MANUAL" in failures:
            return {"performedNow": False, "message": "模拟每日次数受限"}
        return {
            "performedNow": True,
            "lastAttempt": {"state": "REFERENCE_ONLY", "rowCount": 2, "message": "非交易日资料参考，不能计为早间输入"},
        }

    class FeeService:
        def sync(self, fund_code=None, *, progress_reporter):
            run("SIMULATION_FEES", progress_reporter)
            return FeeSyncResult(2, 2, ())

        def close(self):
            close_hook("SIMULATION_FEES")

    return LocalSyncJobManager(FundService, FeatureService, FreeService, spx_sync, FeeService)


def wait_finished(manager, job_id):
    deadline = monotonic() + 3
    while monotonic() < deadline:
        job = manager.get_job(job_id)
        if job.status not in {"QUEUED", "RUNNING"}:
            # 等后台函数完成 finally 的资源释放，而非仅观察状态已经变成终态。
            manager._executor.submit(lambda: None).result(timeout=3)
            return job
        sleep(0.005)
    pytest.fail("batch did not finish")


@pytest.mark.parametrize("failures", [(), *[(stage,) for stage in STAGES], STAGES])
def test_all_stages_are_attempted_once_and_result_preserves_failures(failures):
    calls = []
    manager = make_manager(calls, failures)
    try:
        started = manager.start_all()
        result = wait_finished(manager, started.job_id)
        assert calls == list(STAGES)  # 特征只在全部来源完成后生成一次。
        assert result.status == ("SUCCEEDED" if not failures else "FAILED" if len(failures) == 6 else "PARTIAL_SUCCESS")
        assert (result.progress_current, result.progress_total) == (6, 6)
        assert result.started_at and result.finished_at
        assert manager.get_latest_job(MARKET_ALL_JOB_TYPE) == result
        assert bool(result.error_message) == bool(failures)
        for stage in STAGES:
            child = manager.get_latest_job(stage)
            assert child.status == ("FAILED" if stage in failures else "SUCCEEDED")
            assert child.started_at and child.finished_at
            if stage not in failures and stage not in {"SPX_MANUAL", "SIMULATION_FEES"}:
                assert child.sync_run_id is not None
        next_batch = manager.start_all()
        assert next_batch.job_id != started.job_id
        wait_finished(manager, next_batch.job_id)
    finally:
        manager.close()


@pytest.mark.parametrize(
    ("performed", "state", "expected"),
    [
        (False, "ON_TIME", "FAILED"),  # 旧成功不能冒充本次额度受限后的新请求。
        (True, "LATE", "SUCCEEDED"),  # 晚到可算资料同步成功，不能更改原回执的时间资格。
        (True, "INCOMPLETE", "FAILED"),
    ],
)
def test_spx_batch_preserves_actual_attempt_and_timing_result(performed, state, expected):
    calls = []
    manager = make_manager(calls)
    manager._spx_synchronizer = lambda: {
        "performedNow": performed,
        "message": "本轮未请求",
        "lastAttempt": {"state": state, "message": "原时间状态保持", "rowCount": 2, "usableBeforeU08": False},
    }
    try:
        result = wait_finished(manager, manager.start_all().job_id)
        assert manager.get_latest_job("SPX_MANUAL").status == expected
        assert calls == list(STAGES[1:])
        assert result.progress_current == result.progress_total == 6
        assert result.status == ("SUCCEEDED" if expected == "SUCCEEDED" else "PARTIAL_SUCCESS")
    finally:
        manager.close()


def test_batch_holds_exclusion_across_every_stage_and_restores_live_progress():
    entered = {stage: Event() for stage in STAGES}
    release = {stage: Event() for stage in STAGES}

    def hold(stage):
        entered[stage].set()
        assert release[stage].wait(3)

    manager = make_manager([], stage_hook=hold)
    try:
        started = manager.start_all()
        for index, stage in enumerate(STAGES):
            assert entered[stage].wait(3)
            parent = manager.get_latest_job(MARKET_ALL_JOB_TYPE)
            assert parent.job_id == started.job_id
            assert parent.status == "RUNNING"
            assert parent.progress_current == index
            assert parent.current_fund_code == (None if stage == "SPX_MANUAL" else "000001.OF")
            assert ("0/1" if stage == "SPX_MANUAL" else "1/2") in parent.progress_message
            assert manager.get_latest_job(stage).status == "RUNNING"
            for next_stage in STAGES[index + 1 :]:
                assert manager.get_latest_job(next_stage).status == "QUEUED"
            for start in (
                manager.start_all,
                manager.start_market_details,
                manager.start_market_free_data_completion,
                manager.start_market_nav_incremental,
                manager.start_stock_feature_snapshots,
                manager.start_simulation_fees,
            ):
                with pytest.raises(SyncJobInProgressError):
                    start()
            release[stage].set()
        assert wait_finished(manager, started.job_id).status == "SUCCEEDED"
    finally:
        for event in release.values():
            event.set()
        manager.close()


def test_running_single_job_rejects_batch_and_cleanup_failure_does_not_drop_later_stages():
    entered, release = Event(), Event()

    def hold_close(stage):
        if stage == "MARKET_DETAIL":
            entered.set()
            assert release.wait(3)

    manager = make_manager([], close_hook=hold_close)
    try:
        single = manager.start_market_details()
        assert entered.wait(3)
        # 子任务虽已返回成功状态，连接尚未清理完成时仍然不能开始另一批次。
        with pytest.raises(SyncJobInProgressError):
            manager.start_all()
        release.set()
        wait_finished(manager, single.job_id)
    finally:
        release.set()
        manager.close()

    calls = []

    def failing_close(stage):
        if stage == "MARKET_DETAIL":
            raise RuntimeError("test close failure")

    manager = make_manager(calls, close_hook=failing_close)
    try:
        result = wait_finished(manager, manager.start_all().job_id)
        assert result.status == "PARTIAL_SUCCESS"
        assert manager.get_latest_job("MARKET_DETAIL").error_code == "SYNC_STAGE_FAILED"
        assert calls == list(STAGES)
    finally:
        manager.close()


def test_internal_batch_api_rejects_browser_and_duplicates_without_starting_more_jobs(monkeypatch):
    from app.api.routes import funds
    from app.main import create_application

    entered, release = Event(), Event()

    def hold(stage):
        if stage == STAGES[0]:
            entered.set()
            assert release.wait(3)

    manager = make_manager([], stage_hook=hold)
    monkeypatch.setenv("AI_SERVICE_TOKEN", "sync-batch-test-token")
    get_settings.cache_clear()
    monkeypatch.setattr(funds, "get_sync_job_manager", lambda: manager)
    headers = {"X-Service-Token": "sync-batch-test-token"}
    base = "/internal/v1/funds/sync-jobs"
    try:
        with TestClient(create_application()) as client:
            assert client.get(base + "/all/latest", headers=headers).json() is None
            for invalid in ({}, {"X-Service-Token": "wrong"}, {**headers, "Origin": "http://localhost:5173"}):
                assert client.post(base + "/all", headers=invalid).status_code == 403
                assert client.get(base + "/all/latest", headers=invalid).status_code == 403
                assert client.post(base + "/simulation-fees", headers=invalid).status_code == 403
                assert client.get(base + "/simulation-fees/latest", headers=invalid).status_code == 403
            assert client.post(base + "/simulation-fees?fundCode=invalid", headers=headers).status_code == 422
            assert manager.get_latest_job(MARKET_ALL_JOB_TYPE) is None
            response = client.post(base + "/all", headers=headers)
            assert response.status_code == 202
            job_id = response.json()["job_id"]
            assert entered.wait(3)
            conflict = client.post(base + "/all", headers=headers)
            assert conflict.status_code == 409
            assert conflict.json()["detail"]["code"] == "MARKET_SYNC_IN_PROGRESS"
            assert client.post(base + "/simulation-fees?fundCode=008888", headers=headers).status_code == 409
            assert client.get(base + "/all/latest", headers=headers).json()["job_id"] == job_id
            assert client.get(base + "/" + job_id, headers=headers).json()["status"] == "RUNNING"
            release.set()
    finally:
        release.set()
        manager.close()
        get_settings.cache_clear()


def test_internal_single_fee_api_returns_job_and_latest_without_expanding_scope(monkeypatch):
    from app.api.routes import funds
    from app.main import create_application

    calls = []
    manager = make_manager(calls)
    monkeypatch.setenv("AI_SERVICE_TOKEN", "sync-fee-test-token")
    get_settings.cache_clear()
    monkeypatch.setattr(funds, "get_sync_job_manager", lambda: manager)
    headers = {"X-Service-Token": "sync-fee-test-token"}
    path = "/internal/v1/funds/sync-jobs/simulation-fees"
    try:
        with TestClient(create_application()) as client:
            assert client.get(path + "/latest", headers=headers).json() is None
            response = client.post(path, params={"fundCode": "008888"}, headers=headers)
            assert response.status_code == 202
            from uuid import UUID
            result = wait_finished(manager, UUID(response.json()["job_id"]))
            latest = client.get(path + "/latest", headers=headers).json()
            assert latest["job_id"] == str(result.job_id)
            assert latest["job_type"] == "SIMULATION_FEES"
            assert latest["fund_codes"] == ["008888"]
            assert calls == ["SIMULATION_FEES"]
    finally:
        manager.close()
        get_settings.cache_clear()
