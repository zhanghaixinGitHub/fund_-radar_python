"""订单分类输入的真实数值约束与时间隔离，不读取基金标签或实际未来数据。"""

import json
from copy import deepcopy

import pytest
from app.integrations import tushare_sprint_stock_moneyflow as parser
from app.services import direction_1d_sprint_stock_moneyflow_data_v2 as s


@pytest.fixture
def point():
    row = {k: 10 for k in parser.FIELDS}
    row.update(ts_code="600001.SH", trade_date="20260915", buy_lg_amount=30, net_mf_amount=-7)
    raw = json.dumps(
        {"code": 0, "data": {"fields": parser.FIELDS, "items": [[row[k] for k in parser.FIELDS]]}}
    ).encode()
    return parser.parse(raw, "2026-09-15", minimum_included=1)


def test_actual_parser_output_passes_derived_identity(point):
    s.validate_feature(point, "2026-09-15")


@pytest.mark.parametrize(
    "field,value",
    [
        ("date", "2026-09-16"),
        ("amount_unit", "CNY"),
        ("large_imbalance", 2),
        ("net_fraction", float("nan")),
        ("net_fraction", True),
        ("classified_gross_amount", 0),
        ("raw_net_fraction", 0.1),
        ("net_fraction_clipped", True),
    ],
)
def test_changed_units_dates_or_derived_values_rejected(point, field, value):
    point[field] = value
    with pytest.raises(ValueError):
        s.validate_feature(point, "2026-09-15")


def test_gross_amount_is_checked_even_if_net_ratio_adjusted(point):
    point["classified_gross_amount"] *= 2
    point["raw_net_fraction"] /= 2
    point["net_fraction"] /= 2
    with pytest.raises(ValueError, match="FEATURE_FORMULA"):
        s.validate_feature(point, "2026-09-15")


def test_exact_t_only_missing_t_stays_missing_and_inputs_unchanged(monkeypatch, point):
    monkeypatch.setattr(s.base, "calendar", lambda: (["2026-09-14", "2026-09-15", "2026-09-16"], "cal"))
    points = {"2026-09-15": point, "2026-09-16": {"invalid_future": True}}
    before = deepcopy(points)
    market = {"available": True, "baseline": 1}
    got = s.extend(market, "2026-09-15", "2026-09-16", points)
    assert got["stock_moneyflow"] == point and points == before and "stock_moneyflow" not in market
    del points["2026-09-15"]
    assert s.extend(market, "2026-09-15", "2026-09-16", points)["stock_moneyflow"] == {
        "date": "2026-09-15",
        "available": False,
    }


def test_nonadjacent_target_rejected(monkeypatch):
    monkeypatch.setattr(s.base, "calendar", lambda: (["2026-09-14", "2026-09-15", "2026-09-16"], "cal"))
    with pytest.raises(ValueError, match="ADJACENCY"):
        s.extend({}, "2026-09-14", "2026-09-16", {})


def test_explicit_zero_denominator_is_missing_not_neutral(point):
    point.update(
        available=False, unavailable_reason="ZERO_CLASSIFIED_DENOMINATOR", large_imbalance=None, net_fraction=None
    )
    s.validate_feature(point, "2026-09-15")
    point["net_fraction"] = 0
    with pytest.raises(ValueError, match="NOT_NULL"):
        s.validate_feature(point, "2026-09-15")
