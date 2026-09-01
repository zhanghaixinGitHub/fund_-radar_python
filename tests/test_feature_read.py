"""M3-G1 特征状态内部读取的边界验证。"""

from datetime import UTC, date, datetime
from decimal import Decimal

from app.schemas.feature import InternalFeatureSnapshot, InternalFeatureStatus, InternalStockFeatureMetrics
from fastapi.testclient import TestClient


def _available_status() -> InternalFeatureStatus:
    return InternalFeatureStatus(
        status="AVAILABLE",
        snapshot=InternalFeatureSnapshot(
            fund_code="000001",
            as_of_date=date(2026, 8, 31),
            fund_type="STOCK",
            feature_version="M3_STOCK_FEATURE_V1",
            completeness=Decimal("1"),
            eligibility_status="SCORABLE",
            unavailable_reason=None,
            source_code="TUSHARE_PRO_FUND",
            source_sync_finished_at=datetime(2026, 9, 1, 6, 0, tzinfo=UTC),
            nav_value_basis="ACCUMULATED_NAV",
            metrics=InternalStockFeatureMetrics(
                return_5d=Decimal("0.01"),
                return_20d=Decimal("0.02"),
                return_60d=Decimal("0.03"),
                volatility_20d=Decimal("0.04"),
                max_drawdown_60d=Decimal("-0.05"),
            ),
            computed_at=datetime(2026, 9, 1, 6, 1, tzinfo=UTC),
        ),
    )


def test_internal_feature_status_requires_service_identity_and_rejects_browser(monkeypatch) -> None:
    """特征统计只能由 Java 服务读取，浏览器不得跨过授权边界。"""
    monkeypatch.setenv("AI_SERVICE_TOKEN", "test-service-token")
    from app.api.routes import features
    from app.core.config import get_settings
    from app.main import create_application

    get_settings.cache_clear()
    monkeypatch.setattr(features, "get_latest_stock_feature_status", lambda _fund_code: _available_status())
    with TestClient(create_application()) as client:
        unauthenticated = client.get("/internal/v1/features/latest?fundCode=000001")
        browser = client.get(
            "/internal/v1/features/latest?fundCode=000001",
            headers={"X-Service-Token": "test-service-token", "Origin": "http://localhost:5173"},
        )
        response = client.get(
            "/internal/v1/features/latest?fundCode=000001",
            headers={"X-Service-Token": "test-service-token"},
        )

    assert unauthenticated.status_code == 403
    assert browser.status_code == 403
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "AVAILABLE"
    assert payload["snapshot"]["metrics"]["return_20d"] == "0.02"
    assert "direction" not in payload["snapshot"]
    get_settings.cache_clear()


def test_internal_feature_status_can_report_missing_snapshot_without_inventing_metrics(monkeypatch) -> None:
    """无快照是正常状态，不能用默认指标或方向占位。"""
    monkeypatch.setenv("AI_SERVICE_TOKEN", "test-service-token")
    from app.api.routes import features
    from app.core.config import get_settings
    from app.main import create_application

    get_settings.cache_clear()
    monkeypatch.setattr(
        features,
        "get_latest_stock_feature_status",
        lambda _fund_code: InternalFeatureStatus(status="NOT_AVAILABLE", snapshot=None),
    )
    with TestClient(create_application()) as client:
        response = client.get(
            "/internal/v1/features/latest?fundCode=000001",
            headers={"X-Service-Token": "test-service-token"},
        )

    assert response.status_code == 200
    assert response.json() == {"status": "NOT_AVAILABLE", "snapshot": None}
    get_settings.cache_clear()
