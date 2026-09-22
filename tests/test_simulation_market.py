"""模拟行情内部契约、精度和基金支持范围；不调用真实外部行情。"""

from datetime import date
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from app.core.config import get_settings
from app.schemas.simulation_market import SimulationDividend, SimulationMarket, SimulationNav
from app.services.simulation_market import unsupported_reason
from fastapi.testclient import TestClient

BASE = "/internal/v1/simulation"
HEADERS = {"X-Service-Token": "simulation-test-only-token"}


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("AI_SERVICE_TOKEN", HEADERS["X-Service-Token"])
    get_settings.cache_clear()
    from app.main import create_application

    try:
        with TestClient(create_application()) as value:
            yield value
    finally:
        get_settings.cache_clear()


@pytest.mark.parametrize("headers", [{}, {"X-Service-Token": "wrong"}, {**HEADERS, "Origin": "http://localhost:5173"}])
def test_browser_and_unauthenticated_requests_cannot_read_or_refresh(client, headers):
    assert client.get(BASE + "/calendar", headers=headers).status_code == 403
    assert client.post(BASE + "/refresh", headers=headers, json={"fundCodes": ["000001"]}).status_code == 403


def test_calendar_is_versioned_and_holidays_are_not_sessions(client):
    response = client.get(BASE + "/calendar", headers=HEADERS)
    assert response.status_code == 200
    body = response.json()
    assert body["version"] == "CN_A_SHARE_2026_V1"
    assert "2026-10-01" not in body["sessions"]
    assert "2026-10-08" in body["sessions"]
    assert body["coverageEnd"] == "2026-12-31"


def test_nav_and_dividend_keep_decimal_source_units_and_no_personal_fields(client, monkeypatch):
    from app.api.routes import simulation_market as routes

    monkeypatch.setattr(
        routes,
        "read_market",
        lambda *args: SimulationMarket(
            fund_code="000001",
            fund_name="示例混合基金",
            supported=True,
            reason=None,
            dividends_verified_at=None,
            refresh_status="NOT_SYNCED",
            refresh_message="测试来源",
            navs=[
                SimulationNav(
                    nav_date=date(2026, 9, 7),
                    unit_nav=Decimal("1.12345678"),
                    accumulated_nav=Decimal("1.22345678"),
                    announced_on=date(2026, 9, 8),
                    revision="nav-source-revision",
                )
            ],
            dividends=[
                SimulationDividend(
                    event_key="cash",
                    record_date=date(2026, 9, 7),
                    ex_date=date(2026, 9, 8),
                    pay_date=date(2026, 9, 10),
                    cash_per_unit=Decimal("0.1"),
                    implemented=True,
                    revision="dividend-source-revision",
                )
            ],
        ),
    )
    response = client.get(BASE + "/funds/000001?startDate=2026-09-01&endDate=2026-09-09", headers=HEADERS)
    assert response.status_code == 200
    body = response.json()
    assert body["navs"][0]["unitNav"] == "1.12345678"
    assert body["dividends"][0]["cashPerUnit"] == "0.1"
    assert not {"userId", "amount", "shares", "orders"}.intersection(body)


@pytest.mark.parametrize(
    "query", ["startDate=2026-09-09&endDate=2026-09-01", "startDate=2020-01-01&endDate=2099-01-01"]
)
def test_invalid_date_ranges_are_rejected_before_database(client, query):
    assert client.get(BASE + "/funds/000001?" + query, headers=HEADERS).status_code == 422


def test_refresh_rejects_personal_payload_and_only_queues_registered_codes(client, monkeypatch):
    from app.api.routes import simulation_market as routes

    seen = []
    monkeypatch.setattr(routes, "validate_registered_codes", lambda codes: tuple(sorted(set(codes))))
    monkeypatch.setattr(routes, "refresh_market", lambda codes: seen.append(codes))
    assert (
        client.post(
            BASE + "/refresh", headers=HEADERS, json={"fundCodes": ["000001"], "userId": "must-not-cross-boundary"}
        ).status_code
        == 422
    )
    response = client.post(BASE + "/refresh", headers=HEADERS, json={"fundCodes": ["000001", "000001"]})
    assert response.status_code == 202
    assert seen == [("000001",)]


@pytest.mark.parametrize(
    "changes",
    [
        {"market": "E"},
        {"profile_status": "PENDING"},
        {"fund_name": "示例QDII"},
        {"fund_type": "QDII", "fund_name": "博时标普石油天然气勘探及生产精选行业指数(QDII)-C-CNY"},
        {"fund_type": "FOF", "fund_name": "示例养老FOF"},
        {"fund_type": "OTHER", "fund_name": "示例REIT"},
        {"fund_type": "UNKNOWN", "profile_status": "PENDING"},
        {"fund_name": "示例美元份额"},
        {"fund_name": "示例港元份额"},
        {"fund_name": "示例滚动持有"},
        {"fund_name": "示例封闭基金"},
        {"fund_name": "示例一年持有"},
        {"fund_name": "示例定期开放"},
        {"fund_type": "MONEY"},
        {"status": "INACTIVE"},
    ],
)
def test_all_registered_product_types_use_the_same_nav_simulation(changes):
    values = dict(
        status="ACTIVE",
        market="O",
        profile_status="SYNCED",
        fund_type="HYBRID",
        fund_name="示例混合基金",
        source_fund_type="混合型",
        invest_type="成长型",
        data_source="TUSHARE_PRO_FUND",
        unit_nav=Decimal("1"),
    )
    assert unsupported_reason(SimpleNamespace(**values)) is None
    values.update(changes)
    assert unsupported_reason(SimpleNamespace(**values)) is None


@pytest.mark.parametrize("nav", [None, Decimal("0"), Decimal("-1")])
def test_open_product_scope_still_rejects_missing_or_invalid_nav(nav):
    """取消类型限制不等于允许用空净值、零净值或负净值计算份额。"""
    fund = SimpleNamespace(data_source="TUSHARE_PRO_FUND", unit_nav=nav)
    assert unsupported_reason(fund) == "尚无可核验的同源单位净值。"


def test_open_product_scope_does_not_admit_demo_prices():
    fund = SimpleNamespace(data_source="DEMO", unit_nav=Decimal("1"))
    assert unsupported_reason(fund) == "尚无可核验的同源单位净值。"


@pytest.mark.parametrize("ts_code", ["018853.OF", "510050.SH", "159915.SZ"])
def test_market_refresh_checks_nav_and_dividends_for_every_registered_market(monkeypatch, ts_code):
    """放开场内外类型后，两种资料都必须进入刷新流程；测试不请求真实供应商。"""
    from app.services import simulation_market as service

    session = MagicMock()
    session.__enter__.return_value = session
    session.execute.return_value.scalar.return_value = None
    session.scalar.side_effect = [ts_code, date(2026, 9, 17)]
    monkeypatch.setattr(service, "Session", lambda _engine: session)
    monkeypatch.setattr(service, "get_engine", lambda: None)
    sync = MagicMock()
    monkeypatch.setattr(service, "TushareFundSyncService", lambda: sync)

    service._refresh_one(ts_code.split(".")[0])

    sync.sync_market_nav_history.assert_called_once()
    assert sync.sync_market_nav_history.call_args.args == ((ts_code,),)
    sync.sync_market_dividends.assert_called_once_with((ts_code,))
    sync.close.assert_called_once()
    assert any("SUCCEEDED" in str(call.args[0]) for call in session.execute.call_args_list)
