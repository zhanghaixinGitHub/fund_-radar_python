"""四周期统一入口的隔离回归；核心服务及模型任务均为替身，不写真实预测。"""

import json
from types import SimpleNamespace

import httpx
import pytest
from app.services import multi_prediction_sync as module
from app.services.sync_jobs import LocalSyncJobManager
from tests.test_sync_all_jobs import wait_finished


def setup_worker(monkeypatch, *, count=4, daily_results=None, window_failed=False, multi_failed=False):
    codes = [f"{number:06d}" for number in range(1, count + 1)]
    calls, batches, progress, finished = [], [], [], {}

    def handle(request):
        path = request.url.path
        calls.append((request.method, path))
        if path.endswith("/fund-codes"):
            after = request.url.params["after"]
            return httpx.Response(200, json=[code for code in codes if code > after][:100])
        if path.endswith("/window"):
            return httpx.Response(503 if window_failed else 200, json={"targetNavDate": "2026-09-24"})
        if path.endswith("/finalize"):
            assert json.loads(request.content)["taskId"] in finished
            return httpx.Response(200, json={"failed": 0})
        code = path.rsplit("/", 1)[-1]
        assert code in codes
        assert json.loads(request.content) == {"targetNavDate": "2026-09-24"}
        value = daily_results[codes.index(code)] if daily_results else {"status": "PREDICTED", "reused": True}
        if value == "TIMEOUT":
            raise httpx.ReadTimeout("unknown result", request=request)
        return httpx.Response(200, json=value)

    def create_task(batch):
        # 一个基金分别贡献三个长周期项，一日不进入多周期模型或其历史表。
        batches.append(batch)
        task_id = str(len(batches))
        items = [
            {"fundCode": code, "horizonId": horizon, "status": "CREATED"}
            for code in batch
            for horizon in ("T5_V1", "T20_V1", "M6_V1")
        ]
        reused = failed = 0
        if multi_failed:
            items[0].update(status="FAILED", result={"error": {"summary": "缺少净值"}})
            items[1]["status"] = "REUSED"
            reused = failed = 1
        finished[task_id] = {
            "taskId": task_id,
            "status": "PARTIAL_SUCCESS" if failed else "SUCCEEDED",
            "plannedItems": len(items),
            "pendingItems": 0,
            "createdItems": len(items) - reused - failed,
            "reusedItems": reused,
            "items": items,
        }
        return {**finished[task_id], "status": "RUNNING", "pendingItems": len(items)}

    monkeypatch.setattr(module, "create_task", create_task)
    monkeypatch.setattr(module, "task_status", lambda task_id: finished[task_id])
    monkeypatch.setattr(module.time, "sleep", lambda _: None)
    monkeypatch.setattr(module, "verify_outcomes", lambda: calls.append(("VERIFY", "outcomes")))
    service = object.__new__(module.MultiPredictionSyncService)
    service._client = httpx.Client(base_url="http://core.test", transport=httpx.MockTransport(handle))
    return SimpleNamespace(service=service, codes=codes, calls=calls, batches=batches, progress=progress)


def test_manual_job_combines_four_periods_and_keeps_window_and_data_failures(monkeypatch):
    worker = setup_worker(
        monkeypatch,
        daily_results=[
            {"status": "PREDICTED", "reused": False},
            {"status": "PREDICTED", "reused": True},
            {"status": "MISSED_DEADLINE"},
            "TIMEOUT",
        ],
        multi_failed=True,
    )
    manager = LocalSyncJobManager(multi_prediction_service_factory=lambda: worker.service)
    try:
        job = wait_finished(manager, manager.start_multi_predictions().job_id)
        assert job.status == "PARTIAL_SUCCESS"
        assert (job.fetched_count, job.created_count, job.updated_count, job.skipped_count) == (16, 11, 2, 3)
        assert (job.progress_current, job.progress_total) == (16, 16)
        assert "一日预测/000003" in job.error_message and "已收盘" in job.error_message
        assert "一日预测/000004" in job.error_message and "留档未确认" in job.error_message
        assert "000001/T5_V1：缺少净值" in job.error_message
        assert worker.batches == [worker.codes]
        assert worker.calls[-1] == ("VERIFY", "outcomes")
    finally:
        manager.close()


@pytest.mark.parametrize("count", [0, 44, 201])
def test_same_paged_scope_four_items_each_and_monotonic_progress(monkeypatch, count):
    worker = setup_worker(monkeypatch, count=count)
    try:
        result = worker.service.sync(progress_reporter=lambda *args: worker.progress.append(args))
        assert (result.total, result.created, result.existing, result.issues) == (count * 4, count * 3, count, ())
        assert [code for batch in worker.batches for code in batch] == worker.codes
        assert all(len(batch) <= 100 for batch in worker.batches)
        assert all(total == count * 4 for _, total, _, _ in worker.progress)
        values = [current for current, *_ in worker.progress]
        assert values == sorted(values) and values[-1] == count * 4
        daily_posts = [path for method, path in worker.calls if method == "POST" and "/direction-1d/" in path]
        assert len(daily_posts) == count
        if not count:
            assert not any(path.endswith("/window") for _, path in worker.calls)
    finally:
        worker.service.close()


def test_unavailable_one_day_window_does_not_block_other_periods(monkeypatch):
    worker = setup_worker(monkeypatch, window_failed=True)
    try:
        result = worker.service.sync(progress_reporter=lambda *args: worker.progress.append(args))
        assert (result.total, result.created, result.existing, len(result.issues)) == (16, 12, 0, 4)
        assert all("一日预测检查未完成" in issue for issue in result.issues)
        assert worker.batches == [worker.codes]
        assert worker.progress[-1][:2] == (16, 16)
    finally:
        worker.service.close()


def test_pending_daily_job_is_not_counted_as_generated(monkeypatch):
    worker = setup_worker(monkeypatch, count=2, daily_results=[{"status": "QUEUED"}, {"status": "RUNNING"}])
    try:
        result = worker.service.sync(progress_reporter=lambda *args: None)
        assert (result.total, result.created, result.existing, len(result.issues)) == (8, 6, 0, 2)
        assert all("尚未确认留档" in issue for issue in result.issues)
    finally:
        worker.service.close()
