"""Java 到 FastAPI 内部调用边界的集成测试。"""

from datetime import UTC, datetime
from decimal import Decimal
from types import SimpleNamespace
from uuid import UUID

from app.core.config import get_settings
from fastapi.testclient import TestClient


def test_tushare_token_is_loaded_as_a_secret(monkeypatch) -> None:
    """外部数据源凭据必须从环境变量读取，且在设置对象中保留为 SecretStr。"""
    monkeypatch.setenv("TUSHARE_TOKEN", "test-tushare-token")
    get_settings.cache_clear()

    settings = get_settings()

    assert settings.tushare_token.get_secret_value() == "test-tushare-token"
    get_settings.cache_clear()


def test_internal_health_rejects_missing_service_token(monkeypatch) -> None:
    """未携带服务令牌的浏览器式请求不得访问内部健康检查接口。"""
    monkeypatch.setenv("AI_SERVICE_TOKEN", "test-service-token")
    get_settings.cache_clear()

    from app.main import create_application

    with TestClient(create_application()) as client:
        response = client.get("/internal/v1/health")

    assert response.status_code == 403
    get_settings.cache_clear()


def test_internal_health_accepts_valid_service_token(monkeypatch) -> None:
    """Java 服务令牌可以访问最小化健康检查响应。"""
    monkeypatch.setenv("AI_SERVICE_TOKEN", "test-service-token")
    get_settings.cache_clear()

    from app.main import create_application

    with TestClient(create_application()) as client:
        response = client.get("/internal/v1/health", headers={"X-Service-Token": "test-service-token"})

    assert response.status_code == 200
    assert response.json()["service"] == "fund-ai"
    assert response.headers["X-Trace-Id"]
    get_settings.cache_clear()


def test_internal_fund_list_rejects_browser_origin(monkeypatch) -> None:
    """即使令牌有效，带浏览器来源头的请求也不能访问内部基金接口。"""
    monkeypatch.setenv("AI_SERVICE_TOKEN", "test-service-token")
    get_settings.cache_clear()

    from app.main import create_application

    with TestClient(create_application()) as client:
        response = client.get(
            "/internal/v1/funds",
            headers={"X-Service-Token": "test-service-token", "Origin": "http://localhost:5173"},
        )

    assert response.status_code == 403
    get_settings.cache_clear()


def test_internal_fund_list_and_detail_accept_service_token(monkeypatch) -> None:
    """Java 可通过受保护契约读取已持久化目录的标准字段。"""
    monkeypatch.setenv("AI_SERVICE_TOKEN", "test-service-token")
    get_settings.cache_clear()

    from app.api.routes import funds
    from app.main import create_application
    from app.schemas.fund import InternalFundDetail, InternalFundPage, InternalFundSummary

    summary = InternalFundSummary(
        fund_code="010710",
        fund_name="安信医药健康主题股票C",
        fund_type="STOCK",
        status="ACTIVE",
        as_of_date=datetime(2026, 8, 25, tzinfo=UTC).date(),
    )
    monkeypatch.setattr(
        funds,
        "list_funds",
        lambda _keyword, _page_size, _cursor: InternalFundPage(items=(summary,), next_cursor=None),
    )
    monkeypatch.setattr(
        funds,
        "get_fund",
        lambda _fund_code: InternalFundDetail(
            **summary.model_dump(),
            nav_status="SYNCED",
            data_source="TUSHARE_PRO_FUND",
            unit_nav=Decimal("1.3756"),
            accumulated_nav=Decimal("1.3756"),
        ),
    )

    with TestClient(create_application()) as client:
        headers = {"X-Service-Token": "test-service-token", "X-Trace-Id": "m0-contract-test"}
        list_response = client.get("/internal/v1/funds?pageSize=2", headers=headers)
        detail_response = client.get("/internal/v1/funds/010710", headers=headers)

    assert list_response.status_code == 200
    assert list_response.json()["items"] == [
        {
            "fund_code": "010710",
            "fund_name": "安信医药健康主题股票C",
            "fund_type": "STOCK",
            "status": "ACTIVE",
            "as_of_date": "2026-08-25",
        }
    ]
    assert list_response.headers["X-Trace-Id"] == "m0-contract-test"
    assert detail_response.status_code == 200
    assert detail_response.json()["data_source"] == "TUSHARE_PRO_FUND"
    assert detail_response.json()["unit_nav"] == "1.3756"
    assert detail_response.json()["accumulated_nav"] == "1.3756"
    get_settings.cache_clear()


def test_internal_fund_nav_history_requires_service_token_and_returns_snapshot(monkeypatch) -> None:
    """历史净值同样只能由 Java 服务身份读取，且返回明确的日期和值。"""
    monkeypatch.setenv("AI_SERVICE_TOKEN", "test-service-token")
    get_settings.cache_clear()


def test_internal_manual_focused_nav_sync_requires_service_token_and_returns_safe_counts(monkeypatch) -> None:
    """页面只能经 Java 触发手动同步，且响应只返回受控运行统计。"""
    monkeypatch.setenv("AI_SERVICE_TOKEN", "test-service-token")
    get_settings.cache_clear()

    from app.api.routes import funds
    from app.main import create_application
    from app.services.tushare_fund_sync import SyncOutcome

    received: dict[str, object] = {}

    class StubService:
        def sync_focused_nav_incremental(self, ts_codes: tuple[str, ...]) -> SyncOutcome:
            received["ts_codes"] = ts_codes
            return SyncOutcome(
                sync_run_id=UUID("00000000-0000-0000-0000-000000000301"),
                sync_type="FOCUSED_NAV_INCREMENTAL",
                requested_nav_date=datetime(2026, 8, 27, tzinfo=UTC).date(),
                fetched_count=3,
                created_count=2,
                updated_count=0,
                skipped_count=1,
            )

        def close(self) -> None:
            received["closed"] = True

    monkeypatch.setattr(funds, "TushareFundSyncService", StubService)
    monkeypatch.setattr(
        funds,
        "get_settings",
        lambda: SimpleNamespace(focused_fund_ts_codes=("002112.OF", "010710.OF")),
    )

    with TestClient(create_application()) as client:
        browser_response = client.post(
            "/internal/v1/funds/sync/focused-nav-incremental",
            headers={"X-Service-Token": "test-service-token", "Origin": "http://localhost:5173"},
        )
        response = client.post(
            "/internal/v1/funds/sync/focused-nav-incremental",
            headers={"X-Service-Token": "test-service-token", "X-Trace-Id": "manual-sync-test"},
        )

    assert browser_response.status_code == 403
    assert response.status_code == 200
    assert response.headers["X-Trace-Id"] == "manual-sync-test"
    assert response.json() == {
        "sync_run_id": "00000000-0000-0000-0000-000000000301",
        "requested_nav_date": "2026-08-27",
        "fund_codes": ["002112.OF", "010710.OF"],
        "fetched_count": 3,
        "created_count": 2,
        "updated_count": 0,
        "skipped_count": 1,
    }
    assert received == {"ts_codes": ("002112.OF", "010710.OF"), "closed": True}
    get_settings.cache_clear()


def test_internal_manual_focused_nav_sync_returns_actionable_baseline_error(monkeypatch) -> None:
    """重点基金尚无完整历史基线时，页面应获得可操作的 422 错误码。"""
    monkeypatch.setenv("AI_SERVICE_TOKEN", "test-service-token")
    get_settings.cache_clear()

    from app.api.routes import funds
    from app.main import create_application
    from app.services.tushare_fund_sync import FocusedNavIncrementalPreconditionError

    class StubService:
        def sync_focused_nav_incremental(self, _ts_codes: tuple[str, ...]) -> None:
            raise FocusedNavIncrementalPreconditionError("focused NAV baseline is missing")

        def close(self) -> None:
            return None

    monkeypatch.setattr(funds, "TushareFundSyncService", StubService)
    monkeypatch.setattr(funds, "get_settings", lambda: SimpleNamespace(focused_fund_ts_codes=("002112.OF",)))

    with TestClient(create_application()) as client:
        response = client.post(
            "/internal/v1/funds/sync/focused-nav-incremental",
            headers={"X-Service-Token": "test-service-token"},
        )

    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "FOCUSED_SYNC_BASELINE_MISSING"
    get_settings.cache_clear()

    from app.api.routes import funds
    from app.main import create_application
    from app.schemas.fund import InternalFundNavHistory, InternalFundNavPoint

    monkeypatch.setattr(
        funds,
        "get_fund_nav_history",
        lambda _fund_code, _start_date, _end_date: InternalFundNavHistory(
            fund_code="002112",
            items=(
                InternalFundNavPoint(
                    nav_date=datetime(2026, 8, 25, tzinfo=UTC).date(),
                    unit_nav=Decimal("4.8936"),
                    accumulated_nav=Decimal("5.0416"),
                ),
            ),
        ),
    )

    with TestClient(create_application()) as client:
        unauthenticated = client.get(
            "/internal/v1/funds/002112/nav-history?startDate=2026-08-01&endDate=2026-08-25"
        )
        response = client.get(
            "/internal/v1/funds/002112/nav-history?startDate=2026-08-01&endDate=2026-08-25",
            headers={"X-Service-Token": "test-service-token"},
        )

    assert unauthenticated.status_code == 403
    assert response.status_code == 200
    assert response.json() == {
        "fund_code": "002112",
        "items": [{"nav_date": "2026-08-25", "unit_nav": "4.8936", "accumulated_nav": "5.0416"}],
    }
    get_settings.cache_clear()

def test_internal_source_diagnostics_requires_service_token(monkeypatch) -> None:
    """未认证调用方不得读取数据源诊断信息。"""
    monkeypatch.setenv("AI_SERVICE_TOKEN", "test-service-token")
    get_settings.cache_clear()

    from app.main import create_application

    with TestClient(create_application()) as client:
        response = client.get("/internal/v1/sources")

    assert response.status_code == 403
    get_settings.cache_clear()


def test_internal_source_diagnostics_returns_credential_free_metadata(monkeypatch) -> None:
    """Java 可读取数据源状态，但响应不能包含外部凭据或原始内容。"""
    monkeypatch.setenv("AI_SERVICE_TOKEN", "test-service-token")
    get_settings.cache_clear()

    from app.api.routes import sources
    from app.main import create_application
    from app.schemas.source import SourceDiagnostic

    monkeypatch.setattr(
        sources,
        "list_source_diagnostics",
        lambda: (
            SourceDiagnostic(
                source_code="authorized-source",
                display_name="Authorized Source",
                source_kind="FUND_NAV",
                license_scope="metadata-only",
                rate_limit_per_minute=10,
                retention_days=30,
                enabled=False,
                last_success_at=None,
                last_error_at=None,
                last_error_summary=None,
            ),
        ),
    )

    with TestClient(create_application()) as client:
        response = client.get("/internal/v1/sources", headers={"X-Service-Token": "test-service-token"})

    assert response.status_code == 200
    assert response.json() == [
        {
            "source_code": "authorized-source",
            "display_name": "Authorized Source",
            "source_kind": "FUND_NAV",
            "license_scope": "metadata-only",
            "rate_limit_per_minute": 10,
            "retention_days": 30,
            "enabled": False,
            "last_success_at": None,
            "last_error_at": None,
            "last_error_summary": None,
        }
    ]
    get_settings.cache_clear()


def test_internal_events_require_service_identity_and_only_return_reviewed_contract(monkeypatch) -> None:
    """M2 事件读取必须保持内部访问，并且只暴露获许可的展示元数据。"""
    monkeypatch.setenv("AI_SERVICE_TOKEN", "test-service-token")
    get_settings.cache_clear()

    from app.api.routes import events
    from app.main import create_application
    from app.schemas.event import InternalEventPage, InternalEventSummary

    monkeypatch.setattr(
        events,
        "list_reviewed_events",
        lambda _fund_code, _page_size, _cursor: InternalEventPage(
            items=(
                InternalEventSummary(
                    event_id=UUID("00000000-0000-0000-0000-000000000101"),
                    fund_code="000001",
                    event_type="FUND_ANNOUNCEMENT",
                    summary="A permitted event summary.",
                    source_name="Authorized source",
                    source_url="https://example.test/event/1",
                    published_at=datetime(2026, 8, 25, tzinfo=UTC),
                    confidence=Decimal("0.9000"),
                    relevance_score=Decimal("0.7500"),
                    relation_reason="Fund code was present in the approved reference.",
                ),
            )
        ),
    )

    with TestClient(create_application()) as client:
        unauthenticated = client.get("/internal/v1/events?fundCode=000001")
        invalid_fund_code = client.get(
            "/internal/v1/events?fundCode=bad", headers={"X-Service-Token": "test-service-token"}
        )
        response = client.get("/internal/v1/events?fundCode=000001", headers={"X-Service-Token": "test-service-token"})

    assert unauthenticated.status_code == 403
    assert invalid_fund_code.status_code == 422
    assert response.status_code == 200
    assert response.json()["items"][0]["event_type"] == "FUND_ANNOUNCEMENT"
    assert "content" not in response.json()["items"][0]
    get_settings.cache_clear()


def test_internal_signals_do_not_invent_direction_when_data_is_insufficient(monkeypatch) -> None:
    """M3 结果可报告数据不足，但不得将其伪造成方向预测。"""
    monkeypatch.setenv("AI_SERVICE_TOKEN", "test-service-token")
    get_settings.cache_clear()

    from app.api.routes import signals
    from app.main import create_application
    from app.schemas.signal import InternalSignalPage, InternalSignalSummary

    monkeypatch.setattr(
        signals,
        "list_completed_signals",
        lambda _fund_code, _page_size, _cursor: InternalSignalPage(
            items=(
                InternalSignalSummary(
                    forecast_id=UUID("00000000-0000-0000-0000-000000000201"),
                    fund_code="000001",
                    as_of_date=datetime(2026, 8, 25, tzinfo=UTC).date(),
                    score_status="DATA_INSUFFICIENT",
                    direction=None,
                    directional_probability=None,
                    confidence=None,
                    risk_level=None,
                    max_drawdown_estimate=None,
                    explanation="Insufficient licensed NAV history.",
                    model_version="baseline-v1",
                    feature_version="feature-v1",
                    feature_completeness=Decimal("0.2000"),
                    scored_at=datetime(2026, 8, 25, tzinfo=UTC),
                ),
            )
        ),
    )

    with TestClient(create_application()) as client:
        browser_response = client.get(
            "/internal/v1/signals?fundCode=000001",
            headers={"X-Service-Token": "test-service-token", "Origin": "http://localhost:5173"},
        )
        response = client.get("/internal/v1/signals?fundCode=000001", headers={"X-Service-Token": "test-service-token"})

    assert browser_response.status_code == 403
    assert response.status_code == 200
    item = response.json()["items"][0]
    assert item["score_status"] == "DATA_INSUFFICIENT"
    assert item["direction"] is None
    assert item["directional_probability"] is None
    assert item["confidence"] is None
    get_settings.cache_clear()
