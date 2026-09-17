"""用人工行情验证金额权重、平盘分母、零额缺失及来源校验；不跑基金成绩。"""

import json

import pytest
from app.services import direction_1d_sprint_stock_turnover_features as s


def payload():
    return {
        "code": 0,
        "data": {
            "fields": s.source.FIELDS,
            "items": [
                ["600001.SH", "20260915", 98, 100, -2, 1, 10],
                ["000001.SZ", "20260915", 100, 100, 0, 1, 20],
                ["300001.SZ", "20260915", 101, 100, 1, 1, 70],
                ["920001.BJ", "20260915", 105, 100, 5, 1, 9000],
            ],
        },
    }


def parsed(v):
    return s.parse(json.dumps(v).encode(), "2026-09-15", minimum_included=3)


def test_flat_amount_in_denominator_and_bj_excluded():
    x = parsed(payload())
    assert x["total_amount"] == 100 and x["flat_amount"] == 20 and x["included_rows"] == 3
    assert x["turnover_breadth"] == pytest.approx(0.6)
    assert x["turnover_return_pct"] == pytest.approx(0.5)
    assert x["all_listed_stocks_independently_verified"] is False


def test_equal_amount_recovers_equal_count_breadth():
    v = payload()
    for row in v["data"]["items"]:
        row[-1] = 10
    assert parsed(v)["turnover_breadth"] == 0


def test_amount_unit_rescaling_does_not_change_features():
    v = payload()
    old = parsed(v)
    for row in v["data"]["items"]:
        row[-1] *= 1000
    new = parsed(v)
    assert new["total_amount"] == old["total_amount"] * 1000
    assert all(new[k] == pytest.approx(old[k]) for k in s.KEYS)


def test_zero_amount_is_retained_without_weight_and_all_zero_is_missing():
    v = payload()
    v["data"]["items"][0][-1] = 0
    x = parsed(v)
    assert x["zero_amount_rows"] == 1 and x["included_rows"] == 3 and x["turnover_breadth"] == pytest.approx(7 / 9)
    for row in v["data"]["items"]:
        row[-1] = 0
    x = parsed(v)
    assert not x["available"] and all(x[k] is None for k in s.KEYS)


def test_order_does_not_change_derived_identity():
    v = payload()
    before = parsed(v)
    v["data"]["items"].reverse()
    assert parsed(v) == before


def test_extreme_return_is_kept_then_aggregate_clipped():
    v = payload()
    v["data"]["items"][2][2] = 300
    v["data"]["items"][2][4] = 200
    x = parsed(v)
    assert x["raw_turnover_return_pct"] == pytest.approx(139.8)
    assert x["turnover_return_pct"] == 20 and x["included_rows"] == 3


@pytest.mark.parametrize(
    "bad", ["negative_amount", "nan_amount", "duplicate", "wrong_day", "wrong_pct", "missing_sh_value"]
)
def test_invalid_source_is_never_silently_filtered(bad):
    v = payload()
    rows = v["data"]["items"]
    if bad == "negative_amount":
        rows[0][-1] = -1
    elif bad == "nan_amount":
        rows[0][-1] = float("nan")
    elif bad == "duplicate":
        rows.append(rows[0])
    elif bad == "wrong_day":
        rows[0][1] = "20260916"
    elif bad == "wrong_pct":
        rows[0][4] = 2
    else:
        rows[0][-1] = None
    with pytest.raises(ValueError):
        parsed(v)


def test_excluded_bj_missing_values_stay_outside_fixed_scope():
    v = payload()
    before = parsed(v)
    v["data"]["items"][-1][-1] = None
    after = parsed(v)
    assert all(after[k] == before[k] for k in s.KEYS)
    assert after["source_distribution_hash"] != before["source_distribution_hash"]
