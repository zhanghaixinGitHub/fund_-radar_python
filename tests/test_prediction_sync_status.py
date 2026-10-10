"""延迟回执、断线恢复与旧批次续查；使用临时持久文件和HTTP替身，无真实分析请求。"""

from dataclasses import replace
from datetime import date
from threading import Event
from uuid import uuid4

import httpx
import pytest
from app.services import prediction_sync_status as status
from app.services.direction_1d_sync import Direction1dSyncResult, daily_item
from app.services.sync_jobs import LocalSyncJobManager
from tests.test_direction_1d_sync import worker
from tests.test_sync_all_jobs import make_manager, wait_finished


def read_job(manager, job):
    """模拟页面发起查询后收到下一次状态；只等测试替身，不引入真实网络等待。"""
    manager.get_job(job)
    manager._status_executor.submit(lambda: None).result(timeout=3)
    return manager.get_job(job)


def manager_with_pending(tmp_path, response):
    source = str(uuid4())
    calls = []

    def handle(request):
        calls.append(str(request.url))
        if request.url.path.endswith("/saved-results"):
            return httpx.Response(200, json=[{"horizonId": "T1", "targetDate": "2026-10-12", "count": 1}])
        assert request.url.path == f"/internal/v1/direction-1d/sync/002112/jobs/{source}/reconcile"
        if isinstance(response[0], Exception):
            raise response[0]
        return httpx.Response(200, json=response[0])

    pending = daily_item("002112", "2026-10-12", {"status": "RUNNING", "jobId": source})

    class Work:
        def sync(self, *, progress_reporter):
            return Direction1dSyncResult(date(2026, 10, 12), 1, 0, 0, ("尚未确认留档",), items=(pending,))

        def close(self):
            pass

    manager = LocalSyncJobManager(
        multi_prediction_service_factory=Work,
        prediction_service_factory=lambda: worker(handle),
    )
    job = manager.start_multi_predictions()
    manager._executor.submit(lambda: None).result(timeout=3)
    manager._state_path = tmp_path / "jobs.json"
    with manager._lock:
        manager._persist()
    return manager, job.job_id, calls, handle


def test_late_failure_updates_counts_and_survives_restart_without_second_generation(tmp_path):
    response = [{"status": "FAILED", "reason": "ANALYSIS_REVIEW_FAILED"}]
    manager, job, calls, handle = manager_with_pending(tmp_path, response)
    manager.close()
    restored = LocalSyncJobManager(state_path=tmp_path / "jobs.json", prediction_service_factory=lambda: worker(handle))
    try:
        actual = read_job(restored, job)
        assert actual.status == "FAILED"
        assert actual.result_summary["counts"] == {"COMPLETED": 0, "WAITING": 0, "UNSUPPORTED": 0, "ERROR": 1}
        assert actual.result_summary["pendingDailyCount"] == 0
        assert "未通过内容核验" in actual.error_message
        assert restored.get_job(job) == actual and len(calls) == 1
    finally:
        restored.close()


@pytest.mark.parametrize("reused", [True, False])
def test_late_success_counts_once_and_only_after_archive_confirmation(tmp_path, reused):
    manager, job, calls, _ = manager_with_pending(tmp_path, [{"status": "PREDICTED", "reused": reused}])
    try:
        actual = read_job(manager, job)
        assert actual.status == "SUCCEEDED"
        assert (actual.created_count, actual.updated_count, actual.skipped_count) == (int(not reused), int(reused), 0)
        assert actual.result_summary["counts"]["COMPLETED"] == 1
        assert actual.result_summary["savedResults"] == [{"horizonId": "T1", "targetDate": "2026-10-12", "count": 1}]
        assert manager.get_job(job) == actual and len(calls) == 2
    finally:
        manager.close()


def test_network_failure_keeps_original_job_and_recovers_after_throttle(tmp_path):
    response = [httpx.ReadTimeout("test")]
    manager, job, calls, _ = manager_with_pending(tmp_path, response)
    try:
        actual = read_job(manager, job)
        assert actual.result_summary["pendingDailyCount"] == 1
        assert "暂时无法确认" in actual.error_message
        manager.get_job(job)
        assert len(calls) == 1
        response[0] = {"status": "FAILED", "reason": "ANALYSIS_REVIEW_FAILED"}
        manager._refresh_after.clear()
        assert read_job(manager, job).result_summary["counts"]["ERROR"] == 1
        assert len(calls) == 2 and calls[0] == calls[1]
    finally:
        manager.close()


def test_legacy_pending_binds_only_uniquely_matched_job(tmp_path, monkeypatch):
    manager, job, calls, _ = manager_with_pending(tmp_path, [{"status": "FAILED", "reason": "ANALYSIS_REVIEW_FAILED"}])
    try:
        original = manager._jobs[job]
        old_item = {
            k: v
            for k, v in original.result_summary["items"][0].items()
            if k not in {"sourceJobId", "reasonCode", "pending"}
        }
        manager._jobs[job] = replace(original, result_summary={"items": [old_item]})
        monkeypatch.setattr(status, "legacy_job_id", lambda *args: None)
        assert read_job(manager, job).result_summary["pendingDailyCount"] == 1
        assert not calls
        monkeypatch.setattr(status, "legacy_job_id", lambda *args: original.result_summary["items"][0]["sourceJobId"])
        manager._refresh_after.clear()
        assert read_job(manager, job).result_summary["counts"]["ERROR"] == 1
        assert len(calls) == 1
    finally:
        manager.close()


def test_parent_reconciles_original_children_even_after_order_changes():
    manager = make_manager([])
    try:
        parent = wait_finished(manager, manager.start_all().job_id)
        child = manager.get_latest_job("MULTI_PREDICTIONS")
        manager._jobs[child.job_id] = replace(child, status="FAILED")
        manager._batch_child_ids[parent.job_id] = tuple(reversed(manager._batch_child_ids[parent.job_id]))
        refreshed = manager.get_job(parent.job_id)
        assert refreshed.status == "PARTIAL_SUCCESS"
        assert "全部关注多周期预测" in refreshed.error_message
        assert "基金持仓与公司资料更新" not in refreshed.error_message
        manager._jobs[child.job_id] = replace(child, status="SUCCEEDED")
        assert manager.get_job(parent.job_id).status == "SUCCEEDED"
    finally:
        manager.close()


def test_missing_data_is_not_polled():
    item = daily_item("002112", "2026-10-12", {"status": "WAITING_DATA", "reason": "NAV_GAP"})
    assert not status.pending_daily(item)
    assert status.summary([item])["pendingDailyCount"] == 0


def test_slow_check_does_not_block_status_reads_or_duplicate_requests(tmp_path):
    manager, job, _, _ = manager_with_pending(tmp_path, [{"status": "RUNNING"}])
    entered, release = Event(), Event()
    calls = []

    class Slow:
        def reconcile_item(self, item):
            calls.append(item["sourceJobId"])
            entered.set()
            assert release.wait(3)
            return daily_item(
                item["fundCode"],
                item["targetDate"],
                {"status": "FAILED", "reason": "ANALYSIS_REVIEW_FAILED"},
                item["sourceJobId"],
            )

        def close(self):
            pass

    manager._prediction_service_factory = Slow
    try:
        # 若get_job在请求线程内等网络，这一步会等到替身超时，entered已释放但release尚未设置。
        assert manager.get_job(job).result_summary["pendingDailyCount"] == 1
        assert entered.wait(1)
        assert manager.get_latest_job("MULTI_PREDICTIONS").result_summary["pendingDailyCount"] == 1
        assert len(calls) == 1
        release.set()
        manager._status_executor.submit(lambda: None).result(timeout=3)
        assert manager.get_job(job).result_summary["counts"]["ERROR"] == 1
    finally:
        release.set()
        manager.close()
