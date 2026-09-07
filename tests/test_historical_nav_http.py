"""检查 HTTP 接口的请求、返回值、认证和错误处理。

TestClient 在测试进程内模拟 HTTP 调用，不需要启动 8000/8001 服务。
monkeypatch 临时替换函数或配置，测试结束后恢复；GET 测试不连接真实数据库。
"""

import json
from pathlib import Path

import pytest
from app.core.config import get_settings
from fastapi.testclient import TestClient
from tests.test_historical_nav_batch import reader as reader  # noqa: F401
from tests.test_historical_nav_repository import session as session  # noqa: F401

PREVIEW_PATH = "/internal/v1/features/historical-nav-samples/preview"
# 下面的示例文件仅供 POST 手工传数据的测试；平时用 GET 查库无需导入它。
EXAMPLE_PATH = Path(__file__).resolve().parents[1] / "docs_zhx/examples/historical-nav-preview.request.json"
# 这是测试专用假 Token，不是本机或线上服务的真实凭据。
HEADERS = {"X-Service-Token": "test-service-token"}


@pytest.fixture
def payload() -> dict:
    """每个用例获取一份新的人工示例数据，避免上个用例的改动影响下个用例。"""
    return json.loads(EXAMPLE_PATH.read_text(encoding="utf-8"))


@pytest.fixture
def client(monkeypatch):
    """创建测试客户端；清理配置缓存，让临时 Token 配置生效且不残留。"""
    monkeypatch.setenv("AI_SERVICE_TOKEN", "test-service-token")
    get_settings.cache_clear()
    from app.main import create_application

    try:
        with TestClient(create_application()) as test_client:
            yield test_client
    finally:
        get_settings.cache_clear()


def test_preview_returns_verifiable_example_without_database(client, payload, monkeypatch) -> None:
    """POST 使用请求中的净值计算，并返回可手算核对的样本，而不是查询数据库。"""
    import app.db.session as database

    def forbidden_database():
        pytest.fail("preview must not open a database connection")

    monkeypatch.setattr(database, "get_engine", forbidden_database)
    response = client.post(PREVIEW_PATH, json=payload, headers={**HEADERS, "X-Trace-Id": "nav-preview-test"})

    assert response.status_code == 200
    assert response.headers["X-Trace-Id"] == "nav-preview-test"
    body = response.json()
    assert body["mode"] == "PREVIEW_ONLY"
    assert body["input_nav_count"] == 81
    assert body["sample_count"] == body["scorable_count"] == 1
    sample = body["items"][0]
    assert sample["as_of_date"] == "2025-03-26"
    assert sample["available_at"] == "2025-03-27"
    assert sample["feature_payload"]["metrics"]["return_20d"] == "0.14285714"
    assert sample["offline_label"]["label_end_date"] == "2025-04-23"
    assert sample["offline_label"]["future_return_20d"] == "0.125"
    assert sample["offline_label"]["label_up_20d"] == 1
    assert "offline_label" not in sample["feature_payload"]


def test_preview_all_dates_returns_counts_that_match_items(client, payload) -> None:
    """不指定日期时返回全部 81 份样本：60 份历史不足、1 份完整、20 份答案未齐。"""
    del payload["as_of_date"]
    body = client.post(PREVIEW_PATH, json=payload, headers=HEADERS).json()
    assert body["sample_count"] == len(body["items"]) == 81
    assert body["scorable_count"] == 1
    assert body["data_insufficient_count"] == 60
    assert body["label_not_matured_count"] == 20


@pytest.mark.parametrize("headers", [{}, {"X-Service-Token": "wrong"}, {**HEADERS, "Origin": "http://localhost"}])
def test_preview_requires_internal_auth_before_calculation(client, payload, headers, monkeypatch) -> None:
    """缺少凭据、凭据错误或带浏览器 Origin 的请求，都应在计算前被拒绝。"""
    from app.api.routes import features

    def forbidden_calculation(_input):
        pytest.fail("unauthenticated requests must not calculate")

    monkeypatch.setattr(features, "build_historical_nav_samples", forbidden_calculation)
    assert client.post(PREVIEW_PATH, json=payload, headers=headers).status_code == 403


def test_preview_missing_configured_token_is_unavailable(client, payload, monkeypatch) -> None:
    """服务自身未配置 Token 时返回 503，不得因此放开认证。"""
    monkeypatch.setenv("AI_SERVICE_TOKEN", "")
    get_settings.cache_clear()
    assert client.post(PREVIEW_PATH, json=payload, headers=HEADERS).status_code == 503


@pytest.mark.parametrize("value", ["0", "-1", "NaN", "Infinity", "1e10000", "0.123456789"])
def test_preview_rejects_invalid_nav(client, payload, value) -> None:
    """非法数字、非正数或超出精度限制的净值，应作为参数错误返回 422。"""
    payload["nav_points"][60]["unit_nav"] = value
    assert client.post(PREVIEW_PATH, json=payload, headers=HEADERS).status_code == 422


@pytest.mark.parametrize("case", ["empty", "oversized", "duplicate", "unordered", "type", "extra", "target"])
def test_preview_rejects_invalid_request_shape(client, payload, case) -> None:
    """逐个检查空数据、超量、重复、乱序、非股票型、多余字段和无效目标日期。"""
    if case == "empty":
        payload["nav_points"] = []
    elif case == "oversized":
        payload["nav_points"] = [payload["nav_points"][0]] * 513
    elif case == "duplicate":
        payload["nav_points"][1] = payload["nav_points"][0]
    elif case == "unordered":
        payload["nav_points"].reverse()
    elif case == "type":
        payload["fund_type"] = "BOND"
    elif case == "extra":
        payload["user_id"] = "not-accepted"
    else:
        payload["as_of_date"] = "2024-01-01"
    assert client.post(PREVIEW_PATH, json=payload, headers=HEADERS).status_code == 422


def test_preview_reports_missing_announcement_as_normal_data_status(client, payload) -> None:
    """HTTP 200 只表示请求处理成功；公告日期缺失仍需在样本内标明数据不足。"""
    payload["nav_points"][60]["ann_date"] = None
    response = client.post(PREVIEW_PATH, json=payload, headers=HEADERS)
    assert response.status_code == 200
    sample = response.json()["items"][0]
    assert sample["eligibility_status"] == "DATA_INSUFFICIENT"
    assert sample["unavailable_reason"] == "MISSING_NAV_ANNOUNCEMENT_DATE"


def test_post_preview_rejects_stale_anchor_without_reading_database(client, payload) -> None:
    """自备数据使用同一条旧起点规则；只变公告日，不改净值和请求业务日。"""
    payload["nav_points"][60]["ann_date"] = payload["nav_points"][61]["ann_date"]
    response = client.post(PREVIEW_PATH, json=payload, headers=HEADERS)
    assert response.status_code == 200
    sample = response.json()["items"][0]
    assert sample["as_of_date"] == payload["as_of_date"]
    assert sample["unavailable_reason"] == "STALE_NAV_AT_CUTOFF"
    assert sample["feature_payload"]["metrics"] is None
    assert sample["offline_label"] is None
    assert sample["sample_rule_version"] == "HISTORICAL_NAV_SAMPLE_RULE_V2"


def test_get_and_batch_report_same_stale_sample_through_real_test_reader(client, reader) -> None:
    """接通HTTP、服务、仓储和构建器，仅数据库换为内存库；完整JSON应一致。"""
    single = client.get(PREVIEW_PATH, params={"fundCode": "008888", "asOfDate": "2025-03-21"}, headers=HEADERS)
    batch_response = client.get(BATCH_PATH, params={
        "fundCode": "008888", "startDate": "2025-03-20", "endDate": "2025-03-22", "pageSize": 1,
    }, headers=HEADERS)
    assert single.status_code == batch_response.status_code == 200
    assert single.json()["unavailable_reason"] == "STALE_NAV_AT_CUTOFF"
    assert single.json() == batch_response.json()["items"][1]
    assert batch_response.json()["unavailable_reasons"] == {"STALE_NAV_AT_CUTOFF": 1}


def test_preview_future_missing_accumulated_nav_keeps_past_features(client, payload) -> None:
    """通过 HTTP 再验证一次：未来累计净值缺失，不能反过来改变过去特征。"""
    original = client.post(PREVIEW_PATH, json=payload, headers=HEADERS).json()["items"][0]
    payload["nav_points"][-1]["accumulated_nav"] = None
    changed = client.post(PREVIEW_PATH, json=payload, headers=HEADERS).json()["items"][0]
    assert changed["nav_value_basis"] == "ACCUMULATED_NAV"
    assert changed["feature_hash"] == original["feature_hash"]
    assert changed["feature_payload"] == original["feature_payload"]
    assert changed["offline_label"] is None
    assert changed["eligibility_status"] == "DATA_INSUFFICIENT"


def test_postman_request_matches_verified_example(payload) -> None:
    """核对可选的 Postman 示例与测试数据一致，并且没有预填真实 Token。"""
    collection = json.loads(EXAMPLE_PATH.with_name("historical-nav-preview.postman_collection.json").read_text("utf-8"))
    assert json.loads(collection["item"][0]["request"]["body"]["raw"]) == payload
    assert next(v["value"] for v in collection["variable"] if v["key"] == "service_token") == ""


def test_get_preview_needs_only_code_and_date(client, monkeypatch) -> None:
    """GET 只需基金代码和日期，无需请求体；此处用替身服务验证接口传参和返回。"""
    from datetime import date

    from app.api.routes import features
    from app.services.historical_nav_samples import build_historical_nav_samples
    from tests.test_historical_nav_samples import _input_with_points, _stage_one_points

    def read_sample(*, fund_code, as_of_date):
        assert fund_code == "008888"
        assert as_of_date == date(2025, 8, 7)
        return build_historical_nav_samples(_input_with_points(_stage_one_points()))[60]

    monkeypatch.setattr(features, "preview_stored_historical_nav_sample", read_sample)
    response = client.get(PREVIEW_PATH + "?fundCode=008888&asOfDate=2025-08-07", headers=HEADERS)
    assert response.status_code == 200
    assert response.json()["fund_code"] == "008888"
    assert response.json()["feature_payload"]["metrics"]["return_20d"] == "0.06821425"
    assert response.json()["offline_label"]["label_up_20d"] == 1


@pytest.mark.parametrize("code,expected", [("FUND_NOT_FOUND", 404), ("NAV_NOT_FOUND", 404),
                                          ("SOURCE_NOT_READY", 409), ("NOT_APPLICABLE", 409)])
def test_get_preview_reports_missing_or_inapplicable_data(client, monkeypatch, code, expected) -> None:
    """数据不存在返回 404，来源未就绪或基金不适用返回 409，并保留业务错误码。"""
    from app.api.routes import features
    from app.repositories.historical_nav import HistoricalNavPreviewReadError

    def fail_read(**kwargs):
        raise HistoricalNavPreviewReadError(code, "该记录不可用。")

    monkeypatch.setattr(features, "preview_stored_historical_nav_sample", fail_read)
    response = client.get(PREVIEW_PATH + "?fundCode=008888&asOfDate=2025-08-07", headers=HEADERS)
    assert response.status_code == expected
    assert response.json()["detail"]["code"] == code


def test_get_preview_database_error_is_controlled(client, monkeypatch) -> None:
    """模拟数据库异常时返回 503，响应里不能泄露内部连接细节。"""
    from app.api.routes import features
    from sqlalchemy.exc import SQLAlchemyError

    def fail_read(**kwargs):
        raise SQLAlchemyError("internal-connection-details")

    monkeypatch.setattr(features, "preview_stored_historical_nav_sample", fail_read)
    response = client.get(PREVIEW_PATH + "?fundCode=008888&asOfDate=2025-08-07", headers=HEADERS)
    assert response.status_code == 503
    assert "internal-connection-details" not in response.text


@pytest.mark.parametrize("headers", [{}, {**HEADERS, "Origin": "http://localhost"}])
def test_get_preview_auth_rejects_before_database_read(client, monkeypatch, headers) -> None:
    """GET 也必须先认证再读库；未经授权时，替身查库函数绝不能被调用。"""
    from app.api.routes import features

    def forbidden_read(**kwargs):
        pytest.fail("unauthorized request must not access database")

    monkeypatch.setattr(features, "preview_stored_historical_nav_sample", forbidden_read)
    assert client.get(PREVIEW_PATH + "?fundCode=008888&asOfDate=2025-08-07", headers=headers).status_code == 403


BATCH_PATH = "/internal/v1/features/historical-nav-samples/dry-run"
BATCH_PARAMS = {"fundCode": "008888", "startDate": "2025-08-01", "endDate": "2025-08-31"}


def test_batch_get_parses_query_and_returns_all_items(client, monkeypatch):
    """批量入口不需要 Body；pageSize 是读取批次，响应明确标记只读试跑。"""
    from app.api.routes import features
    from app.schemas.historical_nav import HistoricalNavBatchPreviewResponse

    def fake_batch(request):
        assert request.fund_code == "008888"
        assert request.page_size == 10
        return HistoricalNavBatchPreviewResponse(
            fund_code=request.fund_code, start_date=request.start_date, end_date=request.end_date,
            page_size=request.page_size, page_count=0, sample_count=0, scorable_count=0,
            data_insufficient_count=0, label_not_matured_count=0, unavailable_reasons={}, items=(),
        )

    monkeypatch.setattr(features, "preview_stored_historical_nav_batch", fake_batch)
    response = client.get(BATCH_PATH, params=BATCH_PARAMS, headers=HEADERS)
    assert response.status_code == 200
    assert response.json()["mode"] == "DRY_RUN"
    assert response.json()["items"] == []


@pytest.mark.parametrize("change", [
    {"fundCode": "8888"}, {"startDate": "bad-date"}, {"startDate": "2025-09-01"},
    {"endDate": "2025-09-01"}, {"pageSize": "0"}, {"pageSize": "31"},
    {"pageSize": "1.5"}, {"unknown": "field"},
])
def test_batch_invalid_query_never_reads_database(client, monkeypatch, change):
    """反向范围、超过31天、非法页大小等，在读库之前返回422。"""
    from app.api.routes import features

    monkeypatch.setattr(features, "preview_stored_historical_nav_batch", lambda *_: pytest.fail("must not read"))
    response = client.get(BATCH_PATH, params={**BATCH_PARAMS, **change}, headers=HEADERS)
    assert response.status_code == 422


@pytest.mark.parametrize("headers", [{}, {"X-Service-Token": "wrong"}, {**HEADERS, "Origin": "http://localhost"}])
def test_batch_auth_precedes_reading(client, monkeypatch, headers):
    """批量入口不能因多了一个路径就绕过原有内部认证。"""
    from app.api.routes import features

    monkeypatch.setattr(features, "preview_stored_historical_nav_batch", lambda *_: pytest.fail("must not read"))
    assert client.get(BATCH_PATH, params=BATCH_PARAMS, headers=headers).status_code == 403


@pytest.mark.parametrize("code,expected", [("FUND_NOT_FOUND", 404), ("NOT_APPLICABLE", 409), ("SOURCE_NOT_READY", 409)])
def test_batch_business_errors_have_clear_status(client, monkeypatch, code, expected):
    """基金不存在与品类/来源不适用分别使用404和409。"""
    from app.api.routes import features
    from app.repositories.historical_nav import HistoricalNavPreviewReadError

    def fail(_request):
        raise HistoricalNavPreviewReadError(code, "当前基金不可用。")

    monkeypatch.setattr(features, "preview_stored_historical_nav_batch", fail)
    response = client.get(BATCH_PATH, params=BATCH_PARAMS, headers=HEADERS)
    assert response.status_code == expected
    assert response.json()["detail"]["code"] == code


@pytest.mark.parametrize("timeout", [False, True])
def test_batch_database_and_timeout_errors_do_not_expose_partial_items(client, monkeypatch, timeout):
    """数据库故障和超时均受控失败，不返回部分题目或数据库内部细节。"""
    from app.api.routes import features
    from app.services.historical_nav_preview import HistoricalNavBatchTimeoutError
    from sqlalchemy.exc import SQLAlchemyError

    def fail(_request):
        error_type = HistoricalNavBatchTimeoutError if timeout else SQLAlchemyError
        raise error_type("internal-connection-details")

    monkeypatch.setattr(features, "preview_stored_historical_nav_batch", fail)
    response = client.get(BATCH_PATH, params=BATCH_PARAMS, headers=HEADERS)
    assert response.status_code == 503
    assert "items" not in response.json()
    assert "internal-connection-details" not in response.text
