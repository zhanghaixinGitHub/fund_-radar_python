"""Java 到 FastAPI 内部调用边界的集成测试。"""

from datetime import UTC, date, datetime
from decimal import Decimal
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
        lambda _keyword, _fund_type, _page_size, _cursor, _page: InternalFundPage(
            items=(summary,), next_cursor=None, page=None, page_size=2, total_count=1, total_pages=1
        ),
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
            "day_change_rate": None,
            "week_change_rate": None,
            "month_change_rate": None,
        }
    ]
    assert list_response.json()["page"] is None
    assert list_response.json()["page_size"] == 2
    assert list_response.json()["total_count"] == 1
    assert list_response.json()["total_pages"] == 1
    assert list_response.headers["X-Trace-Id"] == "m0-contract-test"
    assert detail_response.status_code == 200
    assert detail_response.json()["data_source"] == "TUSHARE_PRO_FUND"
    assert detail_response.json()["unit_nav"] == "1.3756"
    assert detail_response.json()["accumulated_nav"] == "1.3756"
    get_settings.cache_clear()


def test_internal_watchlist_detail_requires_service_token_and_has_no_user_state(monkeypatch) -> None:
    """完整详情只向 Java 服务身份开放，Python 响应不得包含关注关系或用户标识。"""
    monkeypatch.setenv("AI_SERVICE_TOKEN", "test-service-token")
    get_settings.cache_clear()

    from app.api.routes import funds
    from app.main import create_application
    from app.schemas.fund import InternalFundDetail, InternalFundWatchlistDetail

    basic = InternalFundDetail(
        fund_code="010710",
        fund_name="安信医药健康主题股票C",
        fund_type="STOCK",
        status="ACTIVE",
        as_of_date=date(2026, 8, 25),
        nav_status="SYNCED",
        data_source="TUSHARE_PRO_FUND",
    )
    monkeypatch.setattr(
        funds,
        "get_fund_watchlist_detail",
        lambda fund_code: InternalFundWatchlistDetail(
            basic=basic,
            managers_status="SYNCED",
            managers=(),
            latest_share_status="SYNCED",
            latest_share=None,
            dividends_status="SYNCED",
            dividends=(),
        )
        if fund_code == "010710"
        else None,
    )

    with TestClient(create_application()) as client:
        unauthorized_response = client.get("/internal/v1/funds/010710/watchlist-detail")
        detail_response = client.get(
            "/internal/v1/funds/010710/watchlist-detail",
            headers={"X-Service-Token": "test-service-token"},
        )
        missing_response = client.get(
            "/internal/v1/funds/999999/watchlist-detail",
            headers={"X-Service-Token": "test-service-token"},
        )

    assert unauthorized_response.status_code == 403
    assert detail_response.status_code == 200
    assert detail_response.json()["basic"]["fund_code"] == "010710"
    assert "is_watched" not in detail_response.json()
    assert missing_response.status_code == 404
    assert missing_response.json()["detail"]["code"] == "FUND_NOT_FOUND"
    get_settings.cache_clear()


def test_internal_fund_list_uses_page_mode_and_rejects_cursor_conflict(monkeypatch) -> None:
    """页码模式将总数传给 Java，且不允许与旧游标模式混用。"""
    monkeypatch.setenv("AI_SERVICE_TOKEN", "test-service-token")
    get_settings.cache_clear()

    from app.api.routes import funds
    from app.main import create_application
    from app.schemas.fund import InternalFundPage

    received: list[dict[str, object]] = []
    monkeypatch.setattr(
        funds,
        "list_funds",
        lambda keyword, fund_type, page_size, cursor, page: received.append(
            {
                "keyword": keyword,
                "fund_type": fund_type,
                "page_size": page_size,
                "cursor": cursor,
                "page": page,
            }
        )
        or InternalFundPage(items=(), page=3, page_size=10, total_count=43, total_pages=5),
    )

    with TestClient(create_application()) as client:
        headers = {"X-Service-Token": "test-service-token"}
        default_response = client.get("/internal/v1/funds", headers=headers)
        page_response = client.get("/internal/v1/funds?fundType=BOND&pageSize=10&page=3", headers=headers)
        conflict_response = client.get("/internal/v1/funds?page=2&cursor=010710", headers=headers)

    assert default_response.status_code == 200
    assert page_response.status_code == 200
    assert page_response.json()["page"] == 3
    assert page_response.json()["page_size"] == 10
    assert page_response.json()["total_count"] == 43
    assert page_response.json()["total_pages"] == 5
    assert received == [
        {"keyword": None, "fund_type": None, "page_size": 10, "cursor": None, "page": None},
        {"keyword": None, "fund_type": "BOND", "page_size": 10, "cursor": None, "page": 3},
    ]
    assert conflict_response.status_code == 422
    assert conflict_response.json()["detail"]["code"] == "PAGINATION_MODE_CONFLICT"
    get_settings.cache_clear()


def test_internal_fund_batch_returns_summaries_and_rejects_invalid_codes(monkeypatch) -> None:
    """关注页仅能通过服务身份批量读取基金摘要，不能携带重复或非法代码。"""
    monkeypatch.setenv("AI_SERVICE_TOKEN", "test-service-token")
    get_settings.cache_clear()

    from app.api.routes import funds
    from app.main import create_application
    from app.schemas.fund import InternalFundSummary

    received: list[tuple[str, ...]] = []
    summary = InternalFundSummary(
        fund_code="002112",
        fund_name="德邦鑫星价值灵活配置混合-C",
        fund_type="MIXED",
        status="ACTIVE",
        as_of_date=datetime(2026, 8, 26, tzinfo=UTC).date(),
        day_change_rate=Decimal("0.0123"),
        week_change_rate=Decimal("-0.0456"),
        month_change_rate=Decimal("0.0789"),
    )
    monkeypatch.setattr(
        funds,
        "get_funds_by_codes",
        lambda fund_codes: received.append(fund_codes) or (summary,),
    )

    with TestClient(create_application()) as client:
        headers = {"X-Service-Token": "test-service-token"}
        response = client.get("/internal/v1/funds/batch?fundCode=002112", headers=headers)
        invalid_response = client.get(
            "/internal/v1/funds/batch?fundCode=002112&fundCode=002112", headers=headers
        )

    assert response.status_code == 200
    assert response.json()[0]["fund_code"] == "002112"
    assert response.json()[0]["day_change_rate"] == "0.0123"
    assert received == [("002112",)]
    assert invalid_response.status_code == 422
    assert invalid_response.json()["detail"]["code"] == "INVALID_FUND_CODE_BATCH"
    get_settings.cache_clear()


def test_internal_fund_nav_history_requires_service_token_and_returns_snapshot(monkeypatch) -> None:
    """历史净值同样只能由 Java 服务身份读取，且返回明确的日期和值。"""
    monkeypatch.setenv("AI_SERVICE_TOKEN", "test-service-token")
    get_settings.cache_clear()


def test_internal_market_nav_sync_job_requires_service_token_and_returns_progress(monkeypatch) -> None:
    """页面只能经 Java 创建任务，并能轮询安全的进度摘要。"""
    monkeypatch.setenv("AI_SERVICE_TOKEN", "test-service-token")
    get_settings.cache_clear()

    from app.api.routes import funds
    from app.main import create_application
    from app.services.sync_jobs import MARKET_NAV_INCREMENTAL_JOB_TYPE, SyncJobSnapshot

    received: dict[str, object] = {}
    job_id = UUID("00000000-0000-0000-0000-000000000301")
    snapshot = SyncJobSnapshot(
        job_id=job_id,
        job_type=MARKET_NAV_INCREMENTAL_JOB_TYPE,
        status="RUNNING",
        requested_nav_date=datetime(2026, 8, 27, tzinfo=UTC).date(),
        fund_codes=("002112.OF", "010710.OF"),
        progress_current=1,
        progress_total=3,
        current_fund_code="002112.OF",
        progress_message="已读取 002112.OF 的待补齐净值",
        sync_run_id=None,
        fetched_count=0,
        created_count=0,
        updated_count=0,
        skipped_count=0,
        error_code=None,
        error_message=None,
        started_at=datetime(2026, 8, 27, tzinfo=UTC),
        finished_at=None,
    )

    class StubManager:
        def start_market_nav_incremental(self) -> SyncJobSnapshot:
            received["started"] = True
            return snapshot

        def get_job(self, received_job_id: UUID) -> SyncJobSnapshot | None:
            return snapshot if received_job_id == job_id else None

        def get_latest_job(self) -> SyncJobSnapshot:
            return snapshot

    monkeypatch.setattr(funds, "get_sync_job_manager", lambda: StubManager())
    with TestClient(create_application()) as client:
        browser_response = client.post(
            "/internal/v1/funds/sync-jobs/market-nav-incremental",
            headers={"X-Service-Token": "test-service-token", "Origin": "http://localhost:5173"},
        )
        response = client.post(
            "/internal/v1/funds/sync-jobs/market-nav-incremental",
            headers={"X-Service-Token": "test-service-token", "X-Trace-Id": "manual-sync-test"},
        )
        status_response = client.get(
            f"/internal/v1/funds/sync-jobs/{job_id}", headers={"X-Service-Token": "test-service-token"}
        )
        latest_response = client.get(
            "/internal/v1/funds/sync-jobs/market-nav-incremental/latest",
            headers={"X-Service-Token": "test-service-token"},
        )

    assert browser_response.status_code == 403
    assert response.status_code == 202
    assert response.headers["X-Trace-Id"] == "manual-sync-test"
    assert response.json() == {
        "job_id": "00000000-0000-0000-0000-000000000301",
        "job_type": "MARKET_NAV_INCREMENTAL",
        "status": "RUNNING",
        "requested_nav_date": "2026-08-27",
        "fund_codes": ["002112.OF", "010710.OF"],
        "progress_current": 1,
        "progress_total": 3,
        "current_fund_code": "002112.OF",
        "progress_message": "已读取 002112.OF 的待补齐净值",
        "sync_run_id": None,
        "fetched_count": 0,
        "created_count": 0,
        "updated_count": 0,
        "skipped_count": 0,
        "error_code": None,
        "error_message": None,
        "started_at": "2026-08-27T00:00:00Z",
        "finished_at": None,
    }
    assert status_response.status_code == 200
    assert latest_response.status_code == 200
    assert status_response.json()["progress_current"] == 1
    assert latest_response.json()["job_id"] == str(job_id)
    assert received == {"started": True}
    get_settings.cache_clear()


def test_internal_market_nav_sync_job_returns_latest_failed_state(monkeypatch) -> None:
    """后台发现历史基线缺失时，页面从任务状态获得可操作错误。"""
    monkeypatch.setenv("AI_SERVICE_TOKEN", "test-service-token")
    get_settings.cache_clear()

    from app.api.routes import funds
    from app.main import create_application
    from app.services.sync_jobs import MARKET_NAV_INCREMENTAL_JOB_TYPE, SyncJobSnapshot

    snapshot = SyncJobSnapshot(
        job_id=UUID("00000000-0000-0000-0000-000000000302"),
        job_type=MARKET_NAV_INCREMENTAL_JOB_TYPE,
        status="FAILED",
        requested_nav_date=datetime(2026, 8, 27, tzinfo=UTC).date(),
        fund_codes=("002112.OF",),
        progress_current=0,
        progress_total=2,
        current_fund_code=None,
        progress_message="同步未完成",
        sync_run_id=None,
        fetched_count=0,
        created_count=0,
        updated_count=0,
        skipped_count=0,
        error_code="MARKET_SYNC_BASELINE_MISSING",
        error_message="请先完成基金市场历史净值回填或来源代码校验。",
        started_at=datetime(2026, 8, 27, tzinfo=UTC),
        finished_at=datetime(2026, 8, 27, tzinfo=UTC),
    )

    class StubManager:
        def get_latest_job(self) -> SyncJobSnapshot:
            return snapshot

    monkeypatch.setattr(funds, "get_sync_job_manager", lambda: StubManager())

    with TestClient(create_application()) as client:
        response = client.get(
            "/internal/v1/funds/sync-jobs/market-nav-incremental/latest",
            headers={"X-Service-Token": "test-service-token"},
        )

    assert response.status_code == 200
    assert response.json()["status"] == "FAILED"
    assert response.json()["error_code"] == "MARKET_SYNC_BASELINE_MISSING"
    get_settings.cache_clear()


def test_internal_market_detail_sync_job_returns_progress_and_last_success(monkeypatch) -> None:
    """完整资料任务只接受 Java 服务令牌，并返回阶段进度和持久化成功时间。"""
    monkeypatch.setenv("AI_SERVICE_TOKEN", "test-service-token")
    get_settings.cache_clear()

    from app.api.routes import funds
    from app.main import create_application
    from app.services.sync_jobs import MARKET_DETAIL_JOB_TYPE, SyncJobSnapshot

    job_id = UUID("00000000-0000-0000-0000-000000000305")
    snapshot = SyncJobSnapshot(
        job_id=job_id,
        job_type=MARKET_DETAIL_JOB_TYPE,
        status="RUNNING",
        requested_nav_date=datetime(2026, 8, 28, tzinfo=UTC).date(),
        fund_codes=(),
        progress_current=5,
        progress_total=12,
        current_fund_code="010710.OF",
        progress_message="已读取 010710.OF 的基金经理资料",
        sync_run_id=None,
        fetched_count=0,
        created_count=0,
        updated_count=0,
        skipped_count=0,
        error_code=None,
        error_message=None,
        started_at=datetime(2026, 8, 28, tzinfo=UTC),
        finished_at=None,
    )

    class StubManager:
        def start_market_details(self) -> SyncJobSnapshot:
            return snapshot

        def get_latest_job(self, job_type: str) -> SyncJobSnapshot | None:
            return snapshot if job_type == MARKET_DETAIL_JOB_TYPE else None

        def get_last_successful_time(self, job_type: str) -> datetime | None:
            return datetime(2026, 8, 27, 12, 0, tzinfo=UTC) if job_type == MARKET_DETAIL_JOB_TYPE else None

    monkeypatch.setattr(funds, "get_sync_job_manager", lambda: StubManager())
    with TestClient(create_application()) as client:
        browser_response = client.post(
            "/internal/v1/funds/sync-jobs/market-details",
            headers={"X-Service-Token": "test-service-token", "Origin": "http://localhost:5173"},
        )
        response = client.post(
            "/internal/v1/funds/sync-jobs/market-details",
            headers={"X-Service-Token": "test-service-token"},
        )
        latest_response = client.get(
            "/internal/v1/funds/sync-jobs/market-details/latest",
            headers={"X-Service-Token": "test-service-token"},
        )
        success_response = client.get(
            "/internal/v1/funds/sync-jobs/last-success",
            headers={"X-Service-Token": "test-service-token"},
        )

    assert browser_response.status_code == 403
    assert response.status_code == 202
    assert response.json()["job_type"] == MARKET_DETAIL_JOB_TYPE
    assert latest_response.status_code == 200
    assert latest_response.json()["progress_current"] == 5
    assert success_response.status_code == 200
    assert success_response.json() == [
        {"job_type": "MARKET_NAV_INCREMENTAL", "last_successful_at": None},
        {"job_type": "MARKET_DETAIL", "last_successful_at": "2026-08-27T12:00:00Z"},
    ]
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
