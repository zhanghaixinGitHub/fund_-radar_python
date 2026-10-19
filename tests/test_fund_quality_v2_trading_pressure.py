"""交易承压研究的数学反例和缺失边界；测试数据不进入真实校准。"""

from decimal import Decimal

import pytest
from app.services.fund_quality_v2_trading_pressure import stock_activity_pressure


def actual_shape_input(first="500", second="500", amount_a="10000", amount_b="10000"):
    days = [f"2025-12-{day:02}" for day in (4, 5, 8, 9, 10, 11, 12, 15, 16, 17,
                                            18, 19, 22, 23, 24, 25, 26, 29, 30, 31)]
    codes = ["600000.SH", "000001.SZ"]
    return {
        "positions": [{"stock_code": code, "market_value_cny": value, "quantity_shares": "100"}
                      for code, value in zip(codes, (first, second), strict=True)],
        "stock_amount": str(Decimal(first) + Decimal(second)), "restricted_rows": [], "restricted_amount": "0",
        "days": days, "report_date": days[-1],
        "quotes": {day: {code: {"amount_cny": amount, "source_hash": "a" * 64}
                         for code, amount in zip(codes, (amount_a, amount_b), strict=True)} for day in days},
    }


def test_equal_market_turnover_does_not_mechanically_reward_or_penalize_stock_hhi():
    even = stock_activity_pressure(**actual_shape_input())
    concentrated = stock_activity_pressure(**actual_shape_input("900", "100"))
    assert even["activity_pressure"] == concentrated["activity_pressure"] == "0.1"
    assert even["free_stock_hhi"] != concentrated["free_stock_hhi"]
    assert not even["quality_metric_admitted"]


def test_size_and_market_activity_have_transparent_scaling_without_capacity_threshold():
    base = stock_activity_pressure(**actual_shape_input())
    doubled_size = stock_activity_pressure(**actual_shape_input("1000", "1000"))
    doubled_turnover = stock_activity_pressure(**actual_shape_input(amount_a="20000", amount_b="20000"))
    assert Decimal(doubled_size["activity_pressure"]) == 2 * Decimal(base["activity_pressure"])
    assert 2 * Decimal(doubled_turnover["activity_pressure"]) == Decimal(base["activity_pressure"])


def test_different_security_turnover_retains_real_market_information_at_same_hhi():
    liquid = stock_activity_pressure(**actual_shape_input(amount_a="20000", amount_b="20000"))
    mixed = stock_activity_pressure(**actual_shape_input(amount_a="20000", amount_b="100"))
    assert liquid["free_stock_hhi"] == mixed["free_stock_hhi"]
    assert Decimal(mixed["activity_pressure"]) > Decimal(liquid["activity_pressure"])


@pytest.mark.parametrize("mutation", ["missing_day", "no_hash", "all_zero_turnover"])
def test_one_positive_position_without_usable_turnover_prevents_aggregate(mutation):
    values = actual_shape_input()
    if mutation == "missing_day":
        del values["quotes"][values["days"][0]]["600000.SH"]
    elif mutation == "no_hash":
        values["quotes"][values["days"][0]]["600000.SH"]["source_hash"] = ""
    else:
        for rows in values["quotes"].values():
            rows["600000.SH"]["amount_cny"] = "0"
    result = stock_activity_pressure(**values)
    assert result["state"] == "NOT_READY" and result["activity_pressure"] is None and result["gaps"]


def test_exchange_suffix_cannot_prove_b_share_amount_currency():
    values = actual_shape_input()
    values["positions"][0]["stock_code"] = "900901.SH"
    with pytest.raises(ValueError, match="B_SHARE_AMOUNT_CURRENCY_UNVERIFIED"):
        stock_activity_pressure(**values)


def test_arithmetic_market_activity_can_mask_a_small_position_with_a_tight_security_constraint():
    """合成反例只验证两种指标含义不同，不能当作真实股票或校准产品。"""
    value = stock_activity_pressure(**actual_shape_input("99000000", "1000000", "1000000000000", "10000"))
    constraints = value["security_volume_constraints"]
    assert Decimal(value["activity_pressure"]) < 1
    assert Decimal(constraints["parallel_bottleneck"]) == 100
    assert constraints["bottleneck_codes"] == ["000001.SZ"]
    assert constraints["strategy_capacity_upper_bound"] is None and constraints["actual_liquidation_days"] is None


def test_the_same_arithmetic_activity_does_not_identify_the_parallel_security_bottleneck():
    even = stock_activity_pressure(**actual_shape_input())
    concentrated = stock_activity_pressure(**actual_shape_input("900", "100"))
    assert even["activity_pressure"] == concentrated["activity_pressure"]
    assert Decimal(even["security_volume_constraints"]["parallel_bottleneck"]) == Decimal(".05")
    assert Decimal(concentrated["security_volume_constraints"]["parallel_bottleneck"]) == Decimal(".09")
