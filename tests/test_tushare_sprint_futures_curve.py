"""合约期限、源字段和缺失边界测试，防止连续主力价格冒充具体合约价差。"""

import json

import pytest
from app.integrations import tushare_sprint_futures_curve as s


def raw(rows):
    return json.dumps({"code": 0, "data": {"fields": s.FIELDS, "items": rows}}).encode()


def row(code="IF2501.CFX", day="20250102"):
    return [code, day, 3926.0, 3930.2, 3786.4, 3819.0, 77698.0, 90877.0]


def test_exact_contract_calendar_and_prices():
    assert s.parse(raw([row()]), "IF2501.CFX", ["2025-01-02"])["2025-01-02"]["close"] == 3819.0


@pytest.mark.parametrize("items", [[row("IF.CFX")], [row(day="20250103")], [row(), row()], []])
def test_wrong_contract_date_duplicate_or_empty_rejected(items):
    with pytest.raises(ValueError):
        s.parse(raw(items), "IF2501.CFX", ["2025-01-02"])


@pytest.mark.parametrize("field,value", [(5, float("nan")), (5, True), (5, -1), (6, 0), (7, None), (4, 3900)])
def test_invalid_prices_and_liquidity_rejected(field, value):
    item = row()
    item[field] = value
    with pytest.raises(ValueError):
        s.parse(raw([item]), "IF2501.CFX", ["2025-01-02"])


def test_missing_day_not_interpolated():
    with pytest.raises(ValueError, match="MISSING_DATES"):
        s.parse(raw([row()]), "IF2501.CFX", ["2025-01-02", "2025-01-03"])


@pytest.fixture
def pair():
    near = {
        "ts_code": "IF2501.CFX",
        "exchange": "CFFEX",
        "fut_code": "IF",
        "multiplier": 300,
        "quote_unit": "指数点",
        "list_date": "20241118",
        "delist_date": "20250117",
    }
    far = near | {"ts_code": "IF2502.CFX", "delist_date": "20250221"}
    a = dict(zip(s.FIELDS, row(), strict=True))
    b = a | {"ts_code": far["ts_code"], "close": 3821.2}
    return ["2025-01-02", near, far, a, b]


def test_actual_expiry_gap_and_fixed_clipping(pair):
    result = s.point(*pair)
    assert result["expiry_gap_days"] == 35 and result["annualized_spread_pct"] == pytest.approx(0.600755620394119)
    pair[-1]["close"] = 8000
    result = s.point(*pair)
    assert result["raw_annualized_spread_pct"] > 50 and result["annualized_spread_pct"] == 50


@pytest.mark.parametrize(
    "change", ["wrong_unit", "expiry_before_day", "reversed_expiry", "future_listing", "wrong_price_day"]
)
def test_contract_metadata_or_price_time_must_match(pair, change):
    if change == "wrong_unit":
        pair[1]["multiplier"] = 100
    elif change == "expiry_before_day":
        pair[1]["delist_date"] = "20250101"
    elif change == "reversed_expiry":
        pair[2]["delist_date"] = "20250110"
    elif change == "future_listing":
        pair[1]["list_date"] = "20250103"
    else:
        pair[-1]["trade_date"] = "20250103"
    with pytest.raises(ValueError):
        s.point(*pair)
