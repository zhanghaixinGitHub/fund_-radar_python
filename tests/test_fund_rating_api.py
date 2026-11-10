"""内部 HTTP 契约回归；故障与业务缺资料分开，不触发真实采集或计算。"""

from types import SimpleNamespace

import pytest
from app.api import dependencies
from app.api.routes import fund_rating
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import SecretStr


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(dependencies, "get_settings", lambda: SimpleNamespace(ai_service_token=SecretStr("test-only")))
    app = FastAPI()
    app.include_router(fund_rating.router, prefix="/funds")
    return TestClient(app)


def test_auth_and_browser_origin_block_before_read(client, monkeypatch):
    monkeypatch.setattr(fund_rating, "read", lambda *_: pytest.fail("未授权不能读评级"))
    assert client.get("/funds/ratings?fundCodes=002112").status_code == 403
    assert (
        client.get(
            "/funds/ratings?fundCodes=002112", headers={"X-Service-Token": "test-only", "Origin": "http://localhost"}
        ).status_code
        == 403
    )


def test_read_failure_is_503_not_business_not_rated(client, monkeypatch):
    def broken(*_, **__):
        raise ValueError("RATING_BATCH_HASH_MISMATCH")

    monkeypatch.setattr(fund_rating, "read", broken)
    response = client.get("/funds/ratings?fundCodes=002112", headers={"X-Service-Token": "test-only"})
    assert response.status_code == 503
    assert "HASH" not in response.text


def test_incomplete_review_is_bound_to_fund_and_batch(tmp_path, monkeypatch):
    from app.services import fund_rating as service, fund_rating_completion_review as review

    monkeypatch.setattr(review, "get_settings", lambda: SimpleNamespace(fund_insight_directory=str(tmp_path)))
    rating = {"fundCode": "013180", "status": "PARTIAL", "ratingRef": "a" * 64, "asOfDate": "2026-11-06"}
    monkeypatch.setattr(service, "read", lambda codes: {"items": [rating]})
    receipt = tmp_path / "receipt.json"
    receipt.write_text("{}", encoding="utf-8")

    assert review.incomplete_reviewed_on(rating) is None
    with pytest.raises(ValueError, match="批次"):
        review.record_incomplete_review("013180", "b" * 64, receipt, "公开资料核查完毕")
    result = review.record_incomplete_review("013180", rating["ratingRef"], receipt, "公开资料核查完毕")

    assert review.incomplete_reviewed_on(rating) == result["reviewedOn"]
    assert review.incomplete_reviewed_on({**rating, "fundCode": "159755"}) is None
    assert review.incomplete_reviewed_on({**rating, "ratingRef": "b" * 64}) is None
    assert review.incomplete_reviewed_on({**rating, "status": "STALE"}) is None
    assert review.incomplete_reviewed_on({**rating, "status": "SCORED"}) is None


@pytest.mark.parametrize("body", [{}, {"fundCode": ""}, {"fundCode": "００２１１２"}])
def test_single_empty_input_never_becomes_all(client, monkeypatch, body):
    monkeypatch.setattr(fund_rating, "_start", lambda *_: pytest.fail("非法代码不能启动任务"))
    assert (
        client.post(
            "/funds/sync-jobs/fund-ratings/single", json=body, headers={"X-Service-Token": "test-only"}
        ).status_code
        == 422
    )


@pytest.mark.parametrize(
    "completed,failures,expected",
    [
        (10, [], "SUCCEEDED"),
        (10, ["failed-category"], "PARTIAL_SUCCESS"),
        (0, ["failed-category"], "FAILED"),
    ],
)
def test_rating_job_distinguishes_missing_data_from_fault(tmp_path, completed, failures, expected):
    from time import monotonic, sleep

    from app.services.sync_jobs import LocalSyncJobManager

    def updater(*_, **__):
        return {"target": 10, "rated": 0, "notRated": completed, "failures": failures}

    manager = LocalSyncJobManager(state_path=tmp_path / "jobs.json", rating_updater=updater)
    try:
        job = manager.start_fund_ratings()
        deadline = monotonic() + 2
        while manager.get_job(job.job_id).status in {"QUEUED", "RUNNING"} and monotonic() < deadline:
            sleep(0.01)
        assert manager.get_job(job.job_id).status == expected
    finally:
        manager.close()
