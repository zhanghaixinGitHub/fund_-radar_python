"""M3-05 受控分析与信号投递内部接口测试。"""

from datetime import UTC, datetime
from decimal import Decimal
from uuid import UUID

from app.core.config import get_settings
from app.schemas.analysis_run import InternalAnalysisRunStatus, InternalModelReleaseStatus
from app.schemas.signal import InternalSignalChange, InternalSignalChangePage
from fastapi.testclient import TestClient


def _headers() -> dict[str, str]:
    """返回测试用服务身份头；浏览器来源单独在各用例覆盖。"""
    return {"X-Service-Token": "test-service-token"}


def _analysis_run_status() -> InternalAnalysisRunStatus:
    """构造不含真实模型结果的排队状态，避免测试隐式执行回测。"""
    return InternalAnalysisRunStatus(
        analysis_run_id=UUID("00000000-0000-0000-0000-000000000301"),
        run_type="ROLLING_BACKTEST",
        status="QUEUED",
        fund_type="STOCK",
        task_id="task-301",
        backtest_run_id=None,
        model_release_id=None,
        model_release_status=None,
        failure_reason=None,
        requested_at=datetime(2026, 9, 1, tzinfo=UTC),
        started_at=None,
        finished_at=None,
    )


def test_analysis_run_endpoint_requires_service_token_and_rejects_browser_origin(monkeypatch) -> None:
    """分析运行入口只能由 Java 服务调用，浏览器和匿名请求均不得创建任务。"""
    monkeypatch.setenv("AI_SERVICE_TOKEN", "test-service-token")
    get_settings.cache_clear()

    from app.api.routes import analysis
    from app.main import create_application

    monkeypatch.setattr(analysis, "start_stock_rolling_backtest", lambda **_kwargs: _analysis_run_status())

    with TestClient(create_application()) as client:
        anonymous = client.post("/internal/v1/analysis/runs/rolling-backtest", json={})
        browser = client.post(
            "/internal/v1/analysis/runs/rolling-backtest",
            headers={**_headers(), "Origin": "http://localhost:5173"},
            json={},
        )
        authorized = client.post("/internal/v1/analysis/runs/rolling-backtest", headers=_headers(), json={})

    assert anonymous.status_code == 403
    assert browser.status_code == 403
    assert authorized.status_code == 202
    assert authorized.json()["status"] == "QUEUED"
    get_settings.cache_clear()


def test_signal_change_endpoint_requires_paired_cursor_and_returns_active_scores(monkeypatch) -> None:
    """消费游标必须成对提供；响应仍只传递评分摘要而不是运行模型。"""
    monkeypatch.setenv("AI_SERVICE_TOKEN", "test-service-token")
    get_settings.cache_clear()

    from app.api.routes import signals
    from app.main import create_application

    scored_at = datetime(2026, 9, 1, tzinfo=UTC)
    forecast_id = UUID("00000000-0000-0000-0000-000000000302")
    monkeypatch.setattr(
        signals,
        "list_active_scored_changes",
        lambda _page_size, _after_scored_at, _after_forecast_id: InternalSignalChangePage(
            items=(
                InternalSignalChange(
                    forecast_id=forecast_id,
                    fund_code="000001",
                    as_of_date=scored_at.date(),
                    model_version="baseline-v1",
                    feature_version="feature-v1",
                    model_release_id=UUID("00000000-0000-0000-0000-000000000303"),
                    direction="UP",
                    directional_probability=Decimal("0.6000"),
                    confidence=Decimal("0.5500"),
                    risk_level="MEDIUM",
                    max_drawdown_estimate=Decimal("0.120000"),
                    explanation="仅供信息参考。",
                    feature_completeness=Decimal("0.9000"),
                    scored_at=scored_at,
                ),
            ),
            has_more=False,
            next_scored_at=scored_at,
            next_forecast_id=forecast_id,
        ),
    )

    with TestClient(create_application()) as client:
        unpaired = client.get(
            "/internal/v1/signals/changes?afterScoredAt=2026-09-01T00:00:00Z",
            headers=_headers(),
        )
        response = client.get("/internal/v1/signals/changes", headers=_headers())

    assert unpaired.status_code == 400
    assert response.status_code == 200
    assert response.json()["items"][0]["forecast_id"] == str(forecast_id)
    assert response.json()["next_forecast_id"] == str(forecast_id)
    get_settings.cache_clear()


def test_model_release_transition_requires_service_token(monkeypatch) -> None:
    """发布控制只能由 Java 管理端转发，原因字段与状态响应均为内部契约。"""
    monkeypatch.setenv("AI_SERVICE_TOKEN", "test-service-token")
    get_settings.cache_clear()

    from app.api.routes import analysis
    from app.main import create_application

    release_id = UUID("00000000-0000-0000-0000-000000000304")
    monkeypatch.setattr(
        analysis,
        "activate_model_release",
        lambda _release_id, *, reason: InternalModelReleaseStatus(
            model_release_id=release_id,
            model_code="baseline",
            model_version="v1",
            feature_version="feature-v1",
            fund_type="STOCK",
            backtest_run_id=UUID("00000000-0000-0000-0000-000000000305"),
            release_status="ACTIVE",
            effective_at=datetime(2026, 9, 1, tzinfo=UTC),
            suspended_at=None,
            release_reason=reason,
        ),
    )

    with TestClient(create_application()) as client:
        anonymous = client.post(
            f"/internal/v1/analysis/model-releases/{release_id}/activate",
            json={"reason": "人工审核"},
        )
        authorized = client.post(
            f"/internal/v1/analysis/model-releases/{release_id}/activate",
            headers=_headers(),
            json={"reason": "人工审核"},
        )

    assert anonymous.status_code == 403
    assert authorized.status_code == 200
    assert authorized.json()["release_status"] == "ACTIVE"
    get_settings.cache_clear()
