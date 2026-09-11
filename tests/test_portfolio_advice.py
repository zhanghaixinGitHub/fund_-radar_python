"""核验真实时间边界、固定交易日窗口、现金再投和资料缺口，不接触个人账本或外部源。"""

from datetime import UTC, date, datetime
from decimal import Decimal

import pytest
from app.core.config import get_settings
from app.repositories.cash_reinvestment_samples import CashDividend, CashNavPoint
from app.services import portfolio_advice as service
from app.services.trading_calendar import load_current_calendar


@pytest.fixture
def window():
    calendar = load_current_calendar()
    start = date(2026, 7, 1)
    dates = (start, *calendar.future_sessions(start, 20))
    now = datetime.combine(calendar.future_sessions(dates[-1], 1)[0], datetime.min.time(), tzinfo=UTC)
    return dates, now


def inputs(dates, values, events=()):
    return (
        tuple(CashNavPoint(d, d, Decimal(v)) for d, v in zip(dates, values, strict=True)),
        events,
        {"source": "test"},
        None,
    )


def test_not_due_never_reads_future_prices(monkeypatch, window):
    dates, _ = window
    monkeypatch.setattr(service, "read_outcome_inputs", lambda *args: pytest.fail("must not read prices"))
    result = service.get_advice_outcome("006730", dates[0], dates[-1], now=datetime(2026, 7, 2, tzinfo=UTC))
    assert result.status == "WAITING" and result.total_return is None


def test_cash_dividend_reinvestment_is_not_mistaken_for_nav_loss(monkeypatch, window):
    dates, now = window
    event = CashDividend("cash-event", dates[0], dates[0], dates[5], dates[5], Decimal("0.1"), "实施")
    # 第5日净值除息从1变成0.9，终点回到1；再投资后收益为1/9，不是0。
    values = ["1"] * 5 + ["0.9"] * 15 + ["1"]
    monkeypatch.setattr(service, "read_outcome_inputs", lambda *args: inputs(dates, values, (event,)))
    result = service.get_advice_outcome("006730", dates[0], dates[-1], now=now)
    assert result.status == "ASSESSED"
    assert result.total_return == Decimal("0.111111111111")
    assert len(result.evidence["values_used"]) == 21
    assert len(result.evidence_hash) == 64


@pytest.mark.parametrize("last,expected", [("1", "0"), ("0.9", "-0.1"), ("1.1", "0.1")])
def test_positive_negative_and_flat_are_preserved(monkeypatch, window, last, expected):
    dates, now = window
    monkeypatch.setattr(service, "read_outcome_inputs", lambda *args: inputs(dates, ["1"] * 20 + [last]))
    result = service.get_advice_outcome("006730", dates[0], dates[-1], now=now)
    assert result.total_return == Decimal(expected)


def test_missing_nav_is_not_filled_or_scored(monkeypatch, window):
    dates, now = window
    nav, events, audit, _ = inputs(dates, ["1"] * 21)
    monkeypatch.setattr(service, "read_outcome_inputs", lambda *args: (nav[:-1], events, audit, None))
    result = service.get_advice_outcome("006730", dates[0], dates[-1], now=now)
    assert result.status == "DATA_INSUFFICIENT" and result.total_return is None


def test_unverified_dividends_are_not_a_success(monkeypatch, window):
    dates, now = window
    monkeypatch.setattr(service, "read_outcome_inputs", lambda *args: ((), (), None, "分红尚未核验"))
    assert service.get_advice_outcome("006730", dates[0], dates[-1], now=now).status == "DATA_INSUFFICIENT"


def test_non_session_or_wrong_horizon_is_rejected(window):
    dates, now = window
    with pytest.raises(ValueError):
        service.get_advice_outcome("006730", dates[0], dates[-2], now=now)
    with pytest.raises(ValueError):
        service.get_advice_outcome("006730", date(2026, 7, 4), dates[-1], now=now)


def test_internal_endpoint_rejects_browser_and_missing_token(monkeypatch):
    from fastapi.testclient import TestClient

    monkeypatch.setenv("AI_SERVICE_TOKEN", "advice-test-only-token")
    get_settings.cache_clear()
    from app.main import create_application

    try:
        with TestClient(create_application()) as client:
            url = "/internal/v1/portfolio-advice/006730/outcome?startDate=2026-07-01&endDate=2026-07-29"
            assert client.get(url).status_code == 403
            assert (
                client.get(
                    url, headers={"X-Service-Token": "advice-test-only-token", "Origin": "http://localhost:5173"}
                ).status_code
                == 403
            )
            assert (
                client.get(
                    url.replace("2026-07-29", "2026-07-30"), headers={"X-Service-Token": "advice-test-only-token"}
                ).status_code
                == 422
            )
    finally:
        get_settings.cache_clear()
