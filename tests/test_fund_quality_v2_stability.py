"""真实端点、合同基准合成和重叠年度的计算边界。"""

from decimal import Decimal, getcontext

import pytest
from app.services.fund_quality_v2_stability import blended_benchmark, score_stability


def month_dates():
    return [f"{2023 + index // 12}-{index % 12 + 1:02d}-28" for index in range(37)]


def test_daily_rebalance_differs_from_fixed_period_allocation():
    days = ["2026-01-05", "2026-01-06", "2026-01-07"]
    result = blended_benchmark(days, {"stock": dict(zip(days, ["100", "200", "100"], strict=True)),
                                       "bond": dict.fromkeys(days, "100")}, {"stock": ".3", "bond": ".7"})
    # 先涨100%再跌50%的股票，与平价债券每日重平衡后为1.105，而期初买入持有为1。
    assert Decimal(result[days[-1]]) == Decimal("1.105")


def test_original_monthly_annual_weights_and_exact_ties():
    days = month_dates()
    benchmark = dict.fromkeys(days, "1")
    total = {day: str(Decimal(index + 100) / 100) for index, day in enumerate(days)}
    output = score_stability(total, benchmark, days)
    assert len(output["monthly"]) == 36 and len(output["rolling_annual"]) == 25
    assert Decimal(output["score"]) == 100
    ties = score_stability(benchmark, benchmark, days)
    assert Decimal(ties["score"]) == 50 and all(row["credit"] == "0.5" for row in ties["monthly"])
    decreasing = {day: str(Decimal(200 - index) / 100) for index, day in enumerate(days)}
    assert Decimal(score_stability(decreasing, benchmark, days)["score"]) == 0
    last_drop = dict(total)
    last_drop[days[-1]] = "0.8"
    output = score_stability(last_drop, benchmark, days)
    assert abs(Decimal(output["score"]) - Decimal("96.36666666666666666666666667")) < Decimal("1e-24")


def test_missing_date_or_month_cannot_be_silently_removed():
    days = month_dates()
    levels = dict.fromkeys(days, "1")
    with pytest.raises(ValueError, match="REQUIRES_37"):
        score_stability(levels, levels, days[:-1])
    missing = dict(levels)
    del missing[days[13]]
    with pytest.raises(ValueError, match="MISSING_DATE"):
        score_stability(missing, levels, days)
    wrong_month = days[:]
    wrong_month[13] = "2024-03-01"
    with pytest.raises(ValueError, match="MONTH_GAP"):
        score_stability(levels, levels, wrong_month)


def test_weights_positive_levels_and_decimal_context():
    days = ["2026-01-05", "2026-01-06"]
    levels = dict.fromkeys(days, "1")
    before = getcontext().prec
    assert blended_benchmark(days, {"stock": levels}, {"stock": "1"})[days[-1]] == "1"
    assert getcontext().prec == before
    with pytest.raises(ValueError, match="WEIGHTS"):
        blended_benchmark(days, {"stock": levels}, {"stock": ".3"})
    with pytest.raises(ValueError, match="LEVEL"):
        blended_benchmark(days, {"stock": {days[0]: "0", days[1]: "1"}}, {"stock": "1"})
