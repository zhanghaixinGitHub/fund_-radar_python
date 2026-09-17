"""用人工报价检验集中度、严格前20日基线和缺失边界，不读取真实基金成绩。"""

import json
import math
from copy import deepcopy
from datetime import date, timedelta

import pytest
from app.services import direction_1d_sprint_stock_activity_features as s


def raw_quotes():
    return {
        "code": 0,
        "data": {
            "fields": s.old.source.FIELDS,
            "items": [[f"{600000 + i:06d}.SH", "20260915", 100, 100, 0, 1, i + 1] for i in range(25)]
            + [["920001.BJ", "20260915", 100, 100, 0, 1, 1000000]],
        },
    }


def parsed(v):
    return s.parse(json.dumps(v).encode(), "2026-09-15", minimum_included=25)


def test_top20_sorted_by_amount_with_bj_outside_fixed_scope():
    q = raw_quotes()
    v = parsed(q)
    assert v["total_amount"] == 325 and v["top20_amount"] == 310
    assert v["top20_share"] == pytest.approx(310 / 325) and v["included_rows"] == 25
    assert v["amount_unit"] == "THOUSAND_CNY"
    q["data"]["items"].reverse()
    assert parsed(q) == v


def test_all_zero_amount_is_explicit_missing():
    q = raw_quotes()
    for row in q["data"]["items"]:
        row[-1] = 0
    v = parsed(q)
    assert not v["available"] and v["top20_share"] is None
    assert v["unavailable_reason"] == "ZERO_TOTAL_RETURNED_AMOUNT"


@pytest.mark.parametrize("bad", ["negative", "boolean", "wrong_day", "wrong_pct", "duplicate"])
def test_invalid_raw_source_rejected(bad):
    q = raw_quotes()
    row = q["data"]["items"][0]
    if bad == "negative":
        row[-1] = -1
    elif bad == "boolean":
        row[-1] = True
    elif bad == "wrong_day":
        row[1] = "20260916"
    elif bad == "wrong_pct":
        row[4] = 9
    else:
        q["data"]["items"].append(row)
    with pytest.raises(ValueError):
        parsed(q)


@pytest.fixture
def market_days():
    days = [str(date(2024, 1, 1) + timedelta(days=i)) for i in range(30)]
    points = {
        d: {
            "date": d,
            "scope": s.SCOPE,
            "available": True,
            "total_amount": (i + 1) * 10.0,
            "top20_share": (i + 1) / 100.0,
        }
        for i, d in enumerate(days)
    }
    return days, points, days[25], days[26]


def test_exact_preceding_twenty_days_excludes_t_and_all_future(market_days):
    days, points, t, u = market_days
    x = s.rolling(t, u, points, days)
    assert x["baseline_dates"] == days[5:25]
    assert x["amount_baseline_median"] == 155
    assert x["concentration_baseline_median"] == pytest.approx(0.155)
    assert x["raw_log_activity"] == pytest.approx(math.log(260 / 155))
    assert x["concentration_change"] == pytest.approx(0.105)
    points[u] = {"invalid_future": True}
    assert s.rolling(t, u, points, days) == x
    points[t]["total_amount"] = 520
    assert s.rolling(t, u, points, days)["amount_baseline_median"] == 155


def test_amount_unit_rescaling_preserves_relative_activity(market_days):
    days, points, t, u = market_days
    before = s.rolling(t, u, points, days)
    for row in points.values():
        row["total_amount"] *= 1000
    after = s.rolling(t, u, points, days)
    assert all(after[k] == pytest.approx(before[k]) for k in s.KEYS)


@pytest.mark.parametrize("kind", ["missing_t", "missing_history", "zero_t", "zero_history", "short_calendar"])
def test_incomplete_window_is_not_filled_or_treated_as_neutral(market_days, kind):
    days, points, t, u = market_days
    target = t if kind in ["missing_t", "zero_t"] else days[10]
    if kind.startswith("missing"):
        del points[target]
    elif kind.startswith("zero"):
        points[target].update(
            available=False, total_amount=0, top20_share=None, unavailable_reason="ZERO_TOTAL_RETURNED_AMOUNT"
        )
    else:
        t, u = days[3], days[4]
    x = s.rolling(t, u, points, days)
    assert not x["available"] and all(x[k] is None for k in s.KEYS)


@pytest.mark.parametrize("amount", [1e-200, 1e200])
def test_activity_is_clipped_after_computing_log_ratio(market_days, amount):
    days, points, t, u = market_days
    points[t]["total_amount"] = amount
    x = s.rolling(t, u, points, days)
    assert x["log_activity"] == (-3 if amount < 1 else 3)


@pytest.mark.parametrize(
    "field,value",
    [("date", "wrong"), ("available", 1), ("total_amount", True), ("top20_share", float("nan")), ("top20_share", 1.1)],
)
def test_malformed_baseline_day_rejected(market_days, field, value):
    days, points, t, u = market_days
    points[days[10]][field] = value
    with pytest.raises(ValueError):
        s.rolling(t, u, points, days)


def test_nonadjacent_target_rejected(market_days):
    days, points, t, _ = market_days
    with pytest.raises(ValueError, match="ADJACENCY_INVALID"):
        s.rolling(t, days[27], points, days)


def test_feature_computation_does_not_mutate_source(market_days):
    days, points, t, u = market_days
    frozen = deepcopy(points)
    s.rolling(t, u, points, days)
    assert points == frozen
