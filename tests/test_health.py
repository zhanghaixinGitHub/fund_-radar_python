"""Tests for the Java-to-FastAPI internal health boundary."""

from app.core.config import get_settings
from fastapi.testclient import TestClient


def test_internal_health_rejects_missing_service_token(monkeypatch) -> None:
    """A browser-like request must not access the internal health endpoint."""
    monkeypatch.setenv("AI_SERVICE_TOKEN", "test-service-token")
    get_settings.cache_clear()

    from app.main import create_application

    with TestClient(create_application()) as client:
        response = client.get("/internal/v1/health")

    assert response.status_code == 403
    get_settings.cache_clear()


def test_internal_health_accepts_valid_service_token(monkeypatch) -> None:
    """The Java service token can access the minimal health payload."""
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
    """A valid service token cannot make the internal API browser-accessible."""
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
    """Java can obtain the M0 mock list and detail through the protected contract."""
    monkeypatch.setenv("AI_SERVICE_TOKEN", "test-service-token")
    get_settings.cache_clear()

    from app.main import create_application

    with TestClient(create_application()) as client:
        headers = {"X-Service-Token": "test-service-token", "X-Trace-Id": "m0-contract-test"}
        list_response = client.get("/internal/v1/funds?pageSize=2", headers=headers)
        detail_response = client.get("/internal/v1/funds/000001", headers=headers)

    assert list_response.status_code == 200
    assert len(list_response.json()["items"]) == 2
    assert list_response.headers["X-Trace-Id"] == "m0-contract-test"
    assert detail_response.status_code == 200
    assert detail_response.json()["data_source"] == "M0_MOCK"
    get_settings.cache_clear()
