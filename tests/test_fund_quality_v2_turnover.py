"""成交负荷候选的量纲和拒收边界；虚构测试数据不进入真实研究。"""

from datetime import date, timedelta
from decimal import Decimal

import pytest
from app.services.fund_quality_v2_turnover import stock_turnover_load


def inputs():
    days = [str(date(2026, 1, 1) + timedelta(days=i)) for i in range(20)]
    return {
        "positions": [{"stock_code": "600001.SH", "market_value_cny": "1000", "quantity_shares": "100"}],
        "stock_amount": "1000",
        "restricted_rows": [{"code": "600001", "amount": "200", "quantity": "20"}],
        "restricted_amount": "200",
        "days": days,
        "report_date": days[-1],
        "stress_fraction": "0.05",
        "participation_fraction": "0.1",
        "quotes": {d: {"600001.SH": {"amount_cny": "400", "source_hash": "TEST_ONLY"}} for d in days},
    }


def test_free_stock_amount_and_units_and_linear_sensitivity():
    kwargs = inputs()
    result = stock_turnover_load(**kwargs)
    assert result["free_stock_amount_cny"] == "800"
    assert Decimal(result["weighted_mean_load_days"]) == 1
    kwargs["stress_fraction"] = "0.1"
    assert Decimal(stock_turnover_load(**kwargs)["weighted_mean_load_days"]) == 2
    kwargs["participation_fraction"] = "0.2"
    assert Decimal(stock_turnover_load(**kwargs)["weighted_mean_load_days"]) == 1


@pytest.mark.parametrize("value", [None, "0"])
def test_missing_or_observed_zero_amount_never_becomes_ready(value):
    kwargs = inputs()
    for day in kwargs["days"]:
        kwargs["quotes"][day]["600001.SH"]["amount_cny"] = value
    result = stock_turnover_load(**kwargs)
    assert result["state"] == "NOT_READY" and result["weighted_mean_load_days"] is None
    assert result["gaps"][0]["reason"] == (
        "MISSING_DAILY_AMOUNT_NOT_ZERO" if value is None else "OBSERVED_ZERO_TURNOVER_NO_FINITE_LOAD"
    )


def test_one_missing_day_prevents_shorter_window_average():
    kwargs = inputs()
    del kwargs["quotes"][kwargs["days"][0]]
    result = stock_turnover_load(**kwargs)
    assert result["gaps"][0]["dates"] == [kwargs["days"][0]]
    assert result["weighted_mean_load_days"] is None


def test_fully_locked_amount_is_proven_by_table_not_missing_quote_zero():
    kwargs = inputs()
    kwargs["positions"].append({"stock_code": "600002.SH", "market_value_cny": "100", "quantity_shares": "10"})
    kwargs["restricted_rows"].append({"code": "600002", "amount": "100", "quantity": "10"})
    kwargs.update(stock_amount="1100", restricted_amount="300")
    result = stock_turnover_load(**kwargs)
    assert result["state"] == "UNSCORED_STOCK_BOOK_DIAGNOSTIC"
    assert result["securities"][1]["state"] == "FULLY_RESTRICTED_EXCLUDED_FROM_FREE_STOCK_BOOK"


def test_missing_full_stock_table_and_excess_locked_quantity_are_rejected():
    kwargs = inputs()
    kwargs["stock_amount"] = "1001"
    with pytest.raises(ValueError, match="FULL_STOCK_TABLE"):
        stock_turnover_load(**kwargs)
    kwargs["stock_amount"] = "1000"
    kwargs["restricted_rows"][0]["quantity"] = "101"
    with pytest.raises(ValueError, match="EXCEEDS_HOLDING"):
        stock_turnover_load(**kwargs)
