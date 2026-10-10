"""隔离算例验证，所有数字都是测试数据，不代表任何真实基金评级。"""

from copy import deepcopy
from datetime import date, timedelta
from decimal import Decimal

import pytest
from app.services.fund_rating_metrics import history_metrics, holdings_metrics, standard_cost, total_return
from app.services.fund_rating_rules import DIMENSIONS, calculate, candidate, grade, percentile, stored_score


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (0, "WEAK"),
        ("49.999999", "WEAK"),
        (50, "AVERAGE"),
        ("69.999999", "AVERAGE"),
        (70, "GOOD"),
        ("89.999999", "GOOD"),
        (90, "EXCELLENT"),
        (100, "EXCELLENT"),
    ],
)
def test_exact_grade_boundaries(value, expected):
    assert grade(value) == expected


@pytest.mark.parametrize("value", [None, -1, "100.000000000001", "NaN", "Infinity", "-Infinity", True])
def test_invalid_scores_never_map(value):
    with pytest.raises(ValueError):
        grade(value)


def test_persisted_precision_is_only_mapping_basis():
    assert stored_score("49.9999999999999") == Decimal(50)
    assert grade("49.9999999999999") == "AVERAGE"
    assert percentile(1, [1, 1]) == 50
    assert percentile(2, [1, 2, 3]) == 50


def make_members(n=30):
    metrics = {m: "1" for terms in candidate("ACTIVE_EQUITY")["formula"].values() for m, _, _ in terms}
    return [
        {"fund_code": f"{i:06}", "product_id": str(i), "representative": True, "metrics": deepcopy(metrics)}
        for i in range(n)
    ]


def test_equal_peers_and_duplicate_shares_do_not_inflate_products():
    config = candidate("ACTIVE_EQUITY")
    members = make_members()
    result = calculate(members, config)
    assert all(Decimal(r["score"]) == 50 and r["grade"] == "AVERAGE" for r in result)
    c_share = deepcopy(members[0])
    c_share.update(fund_code="999999", representative=False)
    assert calculate([*members, c_share], config)[-1]["score"] == result[0]["score"]
    with pytest.raises(ValueError, match="INSUFFICIENT"):
        calculate([*members[:29], c_share], config)


@pytest.mark.parametrize("dimension", list(DIMENSIONS))
def test_each_dimension_missing_blocks_full_batch(dimension):
    members = make_members()
    key = candidate("ACTIVE_EQUITY")["formula"][dimension][0][0]
    del members[-1]["metrics"][key]
    with pytest.raises(KeyError):
        calculate(members, candidate("ACTIVE_EQUITY"))


def test_candidate_weights_have_all_eight_and_are_not_active():
    from app.services.fund_rating_rules import CATEGORY_LABELS

    for family in set(CATEGORY_LABELS) - {"OTHER"}:
        config = candidate(family)
        assert set(config["formula"]) == set(DIMENSIONS)
        assert sum(config["weights"].values()) == 100
        assert config["status"] == "CANDIDATE"
    assert "sharpe" not in str(candidate("MONEY")["formula"])
    assert candidate("INDEX")["formula"]["management"] != candidate("ACTIVE_EQUITY")["formula"]["management"]


def test_dividend_split_and_real_drop_are_distinct():
    assert total_return([1, ".9"], [0, ".1"], [1, 1]) == [1, 1]
    assert total_return([1, ".5"], [0, 0], [1, 2]) == [1, 1]
    assert total_return([1, ".5"], [0, 0], [1, 1]) == [1, Decimal(".5")]
    with pytest.raises(ValueError):
        total_return([1, None], [0, 0], [1, 1])


def test_history_windows_and_zero_variance():
    import calendar

    months = [date(2023 + i // 12, i % 12 + 1, calendar.monthrange(2023 + i // 12, i % 12 + 1)[1]) for i in range(37)]
    days = [months[0] + timedelta(days=i) for i in range((months[-1] - months[0]).days + 1)]
    index = [Decimal(100) + Decimal(i % 29) / 10 + Decimal(i) / 100 for i in range(len(days))]
    result = history_metrics(days, index, [0] * (len(days) - 1), months, 252)
    assert len(result["rolling"]) == 25
    assert abs(result["return_1y"] - (index[-1] / index[days.index(months[-13])] - 1)) < Decimal("1e-26")
    assert result["drawdown"] > 0
    with pytest.raises(ValueError, match="ZERO_VARIANCE"):
        history_metrics(days, [1] * len(days), [0] * (len(days) - 1), months, 252)
    with pytest.raises(ValueError):
        history_metrics(days, index, [], months, 252)


def test_complete_holdings_reconcile_and_known_suspension():
    holdings = [{"issuer": "A", "industry": "I", "amount": "80", "saleable_amount": "0", "turnover_20d": [100] * 20}]
    result = holdings_metrics(holdings, 100, 20, 0, ".1")
    assert result["issuer_hhi"] == 1 and result["liquid_5d"] == Decimal(".2")
    with pytest.raises(ValueError, match="RECONCILED"):
        holdings_metrics(holdings, 200, 20, 0, ".1")
    with pytest.raises(ValueError):
        holdings_metrics(holdings, 100, 20, 0, None)
    holdings[0]["turnover_20d"] = [100]
    with pytest.raises(ValueError, match="TURNOVER"):
        holdings_metrics(holdings, 100, 20, 0, ".1")


def test_standard_fees_do_not_double_subtract_history():
    assert standard_cost(".01", ".001", 0, 0, 0, mode="EXTERNAL") == Decimal(".011")
    assert standard_cost(0, 0, 0, ".01", 0, mode="EXTERNAL") == Decimal(".01") / Decimal("1.01")
    with pytest.raises(ValueError):
        standard_cost(0, None, 0, 0, 0, mode="EXTERNAL")
