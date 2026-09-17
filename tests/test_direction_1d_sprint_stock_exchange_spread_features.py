"""合成数据核验交易所分母、成交金额权重、缺失及T/U隔离。"""

import copy
import json

import pytest
from app.services import direction_1d_sprint_stock_exchange_spread_features as s


def payload():
    rows = []
    for code, change, amount in [
        ("600001.SH", 30, 2),
        ("600002.SH", -10, 1),
        ("000001.SZ", -5, 3),
        ("000002.SZ", 0, 0),
    ]:
        rows.append([code, "20260915", 100 + change, 100, change, 1, amount])
    rows.append(["920001.BJ", "20260915", 150, 100, 50, 1, 1000])
    return {"code": 0, "data": {"fields": s.source.FIELDS, "items": rows}}


def parse(value):
    return s.parse(json.dumps(value).encode(), "2026-09-15", minimum_included=4)


def test_exchange_denominators_and_clipped_amount_weights():
    value = parse(payload())
    assert value["breadth_spread"] == 0.5  # 沪0减深-0.5，深市零成交平盘仍计入数量分母。
    assert value["return_spread"] == 15  # 沪(20*2-10)/3=10个百分点，深-5。
    s.validate_day(value, "2026-09-15")


def test_raw_order_and_common_amount_units_do_not_change_model_features():
    p = payload()
    before = parse(p)
    p["data"]["items"].reverse()
    assert parse(p) == before
    for row in p["data"]["items"]:
        row[-1] *= 1000
    after = parse(p)
    assert all(after[k] == before[k] for k in s.KEYS)


def test_missing_exchange_is_unavailable_even_if_total_stock_count_sufficient():
    p = payload()
    for i, row in enumerate(p["data"]["items"][:4]):
        row[0] = f"{600001 + i}.SH"
    value = parse(p)
    assert not value["available"] and all(value[k] is None for k in s.KEYS)
    s.validate_day(value, "2026-09-15")


def test_zero_amount_in_one_exchange_cannot_be_neutral_input():
    p = payload()
    for row in p["data"]["items"]:
        if row[0].endswith(".SZ"):
            row[-1] = 0
    value = parse(p)
    assert not value["available"] and value["return_spread"] is None
    s.validate_day(value, "2026-09-15")


@pytest.mark.parametrize("bad", ["duplicate", "date", "formula", "nan", "negative_amount"])
def test_source_errors_are_rejected_not_filtered(bad):
    p = payload()
    rows = p["data"]["items"]
    if bad == "duplicate":
        rows.append(rows[0])
    elif bad == "date":
        rows[0][1] = "20260916"
    elif bad == "formula":
        rows[0][4] = 31
    elif bad == "nan":
        rows[0][4] = float("nan")
    else:
        rows[0][-1] = -1
    with pytest.raises(ValueError):
        parse(p)


@pytest.mark.parametrize(
    "field,value",
    [("breadth_spread", 0), ("return_spread", float("nan")), ("date", "2026-09-16"), ("included_rows", 3)],
)
def test_changed_derived_identity_rejected(field, value):
    point = parse(payload())
    point[field] = value
    with pytest.raises(ValueError):
        s.validate_day(point, "2026-09-15")


def test_group_counts_and_amounts_rechecked():
    point = parse(payload())
    point["groups"]["SH"]["up"] = 3
    with pytest.raises(ValueError, match="COUNTS_INVALID"):
        s.validate_day(point, "2026-09-15")
    point = parse(payload())
    point["groups"]["SH"]["total_amount"] = -1
    with pytest.raises(ValueError, match="AMOUNT_INVALID"):
        s.validate_day(point, "2026-09-15")


def test_exact_t_only_without_future_fill_or_input_mutation():
    point = parse(payload())
    days = ["2026-09-14", "2026-09-15", "2026-09-16"]
    points = {"2026-09-15": point, "2026-09-16": {"bad_future": True}}
    original = copy.deepcopy(points)
    assert s.aligned("2026-09-15", "2026-09-16", points, days) == point and points == original
    del points["2026-09-15"]
    assert s.aligned("2026-09-15", "2026-09-16", points, days) == {"date": "2026-09-15", "available": False}
    with pytest.raises(ValueError, match="ADJACENCY"):
        s.aligned("2026-09-14", "2026-09-16", points, days)
