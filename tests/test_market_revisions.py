"""调度去重的服务契约与缺失数据边界；不调用外部行情或真实任务。"""

import hashlib
import json
from contextlib import nullcontext
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import MagicMock
from uuid import uuid4

import pytest
from app.api.routes import market_revisions as routes
from app.core.config import get_settings
from app.services import direction_1d_inference as inference
from app.services import market_revisions as service
from fastapi import FastAPI
from fastapi.testclient import TestClient

ITEM = {"key": "002112", "fund_code": "002112", "start_date": "2026-09-14", "end_date": "2026-09-15", "kind": "LABEL"}
HEADERS = {"X-Service-Token": "revision-offline-test"}


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("AI_SERVICE_TOKEN", HEADERS["X-Service-Token"])
    get_settings.cache_clear()
    app = FastAPI()  # 只挂待测路由，不执行真实应用 lifespan 或恢复后台任务。
    app.include_router(routes.router, prefix="/revisions")
    yield TestClient(app)
    get_settings.cache_clear()


@pytest.mark.parametrize("headers", [{}, {"X-Service-Token": "wrong"}, {**HEADERS, "Origin": "http://localhost"}])
def test_authentication_rejects_before_read(client, monkeypatch, headers):
    read = MagicMock()
    monkeypatch.setattr(routes, "read_revisions", read)
    assert client.post("/revisions", json={"items": [ITEM]}, headers=headers).status_code == 403
    read.assert_not_called()


@pytest.mark.parametrize(
    "items",
    [
        [],
        [ITEM, ITEM],
        [ITEM | {"key": str(i)} for i in range(51)],
        [ITEM | {"end_date": "2099-01-01"}],
        [ITEM | {"start_date": "2026-09-16"}],
        [ITEM | {"amount": 100}],
        [ITEM | {"fund_code": "invalid"}],
    ],
)
def test_scope_is_bounded_and_rejects_personal_fields(client, monkeypatch, items):
    read = MagicMock()
    monkeypatch.setattr(routes, "read_revisions", read)
    assert client.post("/revisions", json={"items": items}, headers=HEADERS).status_code == 422
    read.assert_not_called()


def test_batch_executes_one_read_and_returns_only_versions(monkeypatch):
    connection = MagicMock()
    connection.execute.return_value.mappings.return_value.all.return_value = [
        {"key": "002112", "nav_version": "nav", "dividend_version": "div", "ready": True}
    ]
    monkeypatch.setattr(service, "get_engine", lambda: SimpleNamespace(connect=lambda: nullcontext(connection)))
    result = service.read_revisions(service.RevisionRequest(items=[ITEM]))
    assert len(result["calendar_revision"]) == 64
    assert set(result["items"][0]) == {"key", "revision", "ready"}
    connection.execute.assert_called_once()
    assert json.loads(connection.execute.call_args.args[1]["queries"])[0] == ITEM


def test_version_detects_revision_and_announcement_readiness_but_not_request_identity():
    row = {"key": "first", "nav_version": "v1", "dividend_version": "d1", "ready": False}
    assert service.revision_of(row) == service.revision_of(row | {"key": "second"})
    assert service.revision_of(row) != service.revision_of(row | {"nav_version": "v2"})
    assert service.revision_of(row) != service.revision_of(row | {"dividend_version": "d2"})
    assert service.revision_of(row) != service.revision_of(row | {"ready": True})


def test_scheduled_label_read_never_submits_missing_nav_job(monkeypatch):
    from app.services import direction_1d_jobs as jobs

    body = {
        "target_nav_date": "2026-09-15",
        "base_nav_date": "2026-09-14",
        "fund_code": "002112",
        "expires_at": "2026-12-31T00:00:00+00:00",
    }
    raw = json.dumps(body)
    monkeypatch.setattr(
        inference.repo,
        "get_job",
        lambda _: {
            "state": "SUCCEEDED",
            "kind": "FORECAST",
            "result": {"payload_json": raw, "content_hash": hashlib.sha256(raw.encode()).hexdigest()},
        },
    )
    monkeypatch.setattr(inference.repo, "clock", lambda: datetime(2026, 9, 16, tzinfo=UTC))
    connection = MagicMock()
    connection.execution_options.return_value = nullcontext(connection)
    connection.begin.return_value = nullcontext()
    monkeypatch.setattr(inference, "get_engine", lambda: SimpleNamespace(connect=lambda: connection))
    monkeypatch.setattr(inference.repo, "source", lambda _: {"source_id": uuid4()})
    monkeypatch.setattr(inference.repo, "navs", lambda *args: [])
    submit = MagicMock()
    monkeypatch.setattr(jobs, "submit", submit)
    assert inference.labels(uuid4(), fetch_missing=False) == {"status": "PENDING_NAV"}
    submit.assert_not_called()
    # 显式允许补拉的既有入口保持兼容，只有定时核对关闭这个副作用。
    inference.labels(uuid4())
    assert submit.call_count == 1
