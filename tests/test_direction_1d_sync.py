"""预测同步任务的 HTTP 与后台队列回归；只用替身，不生成真实预测或请求行情。"""

import json
from datetime import date

import httpx
import pytest
from app.services.direction_1d_sync import Direction1dSyncResult, Direction1dSyncService
from app.services.sync_jobs import LocalSyncJobManager
from tests.test_sync_all_jobs import make_manager, wait_finished


def worker(handler):
    result = object.__new__(Direction1dSyncService)
    result._client = httpx.Client(base_url="http://core.test", transport=httpx.MockTransport(handler))
    return result


def test_paged_funds_are_counted_once_and_failures_do_not_stop_later_funds():
    calls, progress = [], []

    def handle(request):
        path = request.url.path
        if path.endswith("/window"):
            return httpx.Response(200, json={"targetNavDate": "2026-09-23"})
        if path.endswith("/fund-codes"):
            return httpx.Response(
                200,
                json={"": ["000001", "000002"], "000002": ["000003", "000004", "000005"], "000005": []}[
                    request.url.params["after"]
                ],
            )
        code = path.rsplit("/", 1)[-1]
        calls.append(code)
        assert json.loads(request.content) == {"targetNavDate": "2026-09-23"}
        if code == "000003":
            raise httpx.ReadTimeout("result unknown", request=request)
        return httpx.Response(
            200,
            json={
                "000001": {"status": "PREDICTED", "reused": False},
                "000002": {"status": "PREDICTED", "reused": True},
                "000004": {"status": "SPECIAL_POLICY_REQUIRED"},
                "000005": {"status": "WAITING_DATA"},
            }[code],
        )

    service = worker(handle)
    try:
        result = service.sync(progress_reporter=lambda *args: progress.append(args))
        assert (result.total, result.created, result.existing, len(result.issues)) == (5, 1, 1, 3)
        assert calls == ["000001", "000002", "000003", "000004", "000005"]
        assert "该基金类型暂不支持" in result.issues[1]
        assert "净值" in result.issues[2]
        assert [item[0] for item in progress] == list(range(6))
    finally:
        service.close()


@pytest.mark.parametrize("page", [["000002", "000001"], ["000001", "000001"], ["not-a-code"], ["000000"]])
def test_invalid_or_non_advancing_page_fails_without_prediction(page):
    def handle(request):
        if request.url.path.endswith("/window"):
            return httpx.Response(200, json={"targetNavDate": "2026-09-23"})
        assert request.method == "GET"
        return httpx.Response(200, json=page)

    service = worker(handle)
    try:
        with pytest.raises(ValueError):
            service.sync(progress_reporter=lambda *args: None)
    finally:
        service.close()


@pytest.mark.parametrize(
    "result,expected",
    [
        (Direction1dSyncResult(date(2026, 9, 23), 3, 1, 1, ("000003：不适用",)), "PARTIAL_SUCCESS"),
        (Direction1dSyncResult(date(2026, 9, 23), 2, 0, 2, ()), "SUCCEEDED"),
        (Direction1dSyncResult(date(2026, 9, 23), 1, 0, 0, ("000001：不在预测时间段",)), "FAILED"),
        (Direction1dSyncResult(date(2026, 9, 23), 0, 0, 0, ()), "SUCCEEDED"),
    ],
)
def test_job_preserves_created_existing_missing_counts_and_target(result, expected):
    class Worker:
        def sync(self, *, progress_reporter):
            return result

        def close(self):
            pass

    manager = LocalSyncJobManager(prediction_service_factory=Worker)
    try:
        job = wait_finished(manager, manager.start_direction_1d_predictions().job_id)
        assert job.status == expected
        assert (job.fetched_count, job.created_count, job.updated_count, job.skipped_count) == (
            result.total,
            result.created,
            result.existing,
            len(result.issues),
        )
        assert job.requested_nav_date == date(2026, 9, 23)
        assert manager.get_latest_job("DIRECTION_1D_PREDICTIONS") == job
    finally:
        manager.close()


def test_prediction_api_requires_service_token_and_rejects_browser(monkeypatch):
    from app.api.routes import funds
    from app.core.config import get_settings
    from app.main import create_application
    from fastapi.testclient import TestClient

    manager = make_manager([])
    monkeypatch.setenv("AI_SERVICE_TOKEN", "prediction-sync-test")
    get_settings.cache_clear()
    monkeypatch.setattr(funds, "get_sync_job_manager", lambda: manager)
    path = "/internal/v1/funds/sync-jobs/direction-1d-predictions"
    headers = {"X-Service-Token": "prediction-sync-test"}
    try:
        with TestClient(create_application()) as client:
            assert client.post(path).status_code == 403
            assert client.post(path, headers={**headers, "Origin": "http://localhost"}).status_code == 403
            response = client.post(path, headers=headers)
            assert response.status_code == 202
            from uuid import UUID

            job = wait_finished(manager, UUID(response.json()["job_id"]))
            assert client.get(path + "/latest", headers=headers).json()["job_id"] == str(job.job_id)
    finally:
        manager.close()
        get_settings.cache_clear()
