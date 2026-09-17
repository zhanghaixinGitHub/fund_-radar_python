"""用边界行情验证尾部比例与时点隔离，避免变成涨跌停口径或悄悄删除难样本。"""

import copy
import json

import pytest
from app.services import direction_1d_sprint_stock_tail_features as s


def payload():
    changes = [5.0, -5.0, 4.9999, -4.9999, 0, 100]
    rows = [[f"{600001 + i:06d}.SH", "20260915", 100 + x, 100, x, 1, 10] for i, x in enumerate(changes)]
    rows.append(["920001.BJ", "20260915", 150, 100, 50, 1, 1000])
    return {"code": 0, "data": {"fields": s.source.FIELDS, "items": rows}}


def parse(value):
    return s.parse(json.dumps(value).encode(), "2026-09-15", minimum_included=6)


def test_inclusive_boundaries_extreme_and_flat_denominator():
    got = parse(payload())
    assert got["tail_up"] == 2 and got["tail_down"] == 1 and got["included_rows"] == 6
    assert got["tail_balance"] == 1 / 6 and got["tail_mass"] == 0.5
    assert not got["all_listed_stocks_independently_verified"]


def test_zero_amount_and_row_order_do_not_change_count_features():
    value = payload()
    before = parse(value)
    for row in value["data"]["items"]:
        row[-1] = 0
    value["data"]["items"].reverse()
    assert parse(value) == before


def test_no_extreme_is_valid_zero_not_missing():
    value = payload()
    for row in value["data"]["items"]:
        row[2], row[4] = 100, 0
    got = parse(value)
    assert got["available"] and got["tail_balance"] == got["tail_mass"] == 0


@pytest.mark.parametrize("bad", ["duplicate", "date", "formula", "nan", "negative_amount"])
def test_invalid_source_is_rejected_without_filtering(bad):
    value = payload()
    rows = value["data"]["items"]
    if bad == "duplicate":
        rows.append(rows[0])
    elif bad == "date":
        rows[0][1] = "20260916"
    elif bad == "formula":
        rows[0][4] = 6
    elif bad == "nan":
        rows[0][4] = float("nan")
    else:
        rows[0][-1] = -1
    with pytest.raises(ValueError):
        parse(value)


def test_future_points_and_missing_t_do_not_affect_features():
    value = parse(payload())
    days = ["2026-09-14", "2026-09-15", "2026-09-16"]
    points = {"2026-09-15": value, "2026-09-16": {"invalid": True}}
    before = copy.deepcopy(points)
    assert s.aligned("2026-09-15", "2026-09-16", points, days) == value
    assert points == before
    del points["2026-09-15"]
    assert s.aligned("2026-09-15", "2026-09-16", points, days) == {"date": "2026-09-15", "available": False}


def test_nonadjacent_target_rejected():
    with pytest.raises(ValueError, match="ADJACENCY"):
        s.aligned("2026-09-14", "2026-09-16", {}, ["2026-09-14", "2026-09-15", "2026-09-16"])


@pytest.mark.parametrize(
    "field,value",
    [
        ("tail_mass", 0.7),
        ("tail_balance", float("nan")),
        ("tail_up", -1),
        ("tail_down", 7),
        ("included_rows", True),
        ("threshold_pct", 4),
        ("date", "2026-09-16"),
    ],
)
def test_modified_derived_identity_rejected(field, value):
    got = parse(payload())
    got[field] = value
    with pytest.raises(ValueError):
        s.validate_day(got, "2026-09-15")
