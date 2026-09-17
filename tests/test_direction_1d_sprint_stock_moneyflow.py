"""订单分类来源边界：金额量纲、独立净额定义、零分母及不擅删坏记录。"""

import json

import pytest
from app.integrations import tushare_sprint_stock_moneyflow as s


def payload():
    rows = []
    for code in ["600001.SH", "000001.SZ", "920001.BJ"]:
        row = {k: 10 for k in s.FIELDS}
        row.update(ts_code=code, trade_date="20260915", buy_lg_amount=30, net_mf_amount=-7)
        rows.append([row[k] for k in s.FIELDS])
    return {"code": 0, "data": {"fields": s.FIELDS, "items": rows}}


def parse(value):
    return s.parse(json.dumps(value).encode(), "2026-09-15", minimum_included=2)


def test_net_value_is_not_reconstructed_from_classified_amounts():
    got = parse(payload())
    assert got["included_rows"] == 2 and got["excluded_bj_rows"] == 1
    assert got["large_imbalance"] == pytest.approx(1 / 3)
    assert got["classified_gross_amount"] == 100
    assert got["raw_net_fraction"] == -0.14
    assert got["totals"]["net_mf_amount"] == -14


def test_order_and_uniform_amount_unit_rescaling():
    value = payload()
    got = parse(value)
    value["data"]["items"].reverse()
    assert parse(value) == got
    for row in value["data"]["items"]:
        for i, k in enumerate(s.FIELDS):
            if k.endswith("amount"):
                row[i] *= 10
    after = parse(value)
    assert after["large_imbalance"] == got["large_imbalance"]
    assert after["raw_net_fraction"] == got["raw_net_fraction"]


def test_zero_amount_rows_kept_and_all_zero_denominator_is_missing():
    value = payload()
    for row in value["data"]["items"]:
        for i in range(2, len(s.FIELDS)):
            row[i] = 0
    got = parse(value)
    assert got["included_rows"] == 2 and not got["available"]
    assert got["large_imbalance"] is None and got["net_fraction"] is None


@pytest.mark.parametrize("bad", ["date", "duplicate", "negative", "nan", "missing", "bool", "schema", "truncated"])
def test_invalid_input_rejected(bad):
    value = payload()
    rows = value["data"]["items"]
    if bad == "date":
        rows[0][1] = "20260916"
    elif bad == "duplicate":
        rows.append(rows[0])
    elif bad == "schema":
        value["data"]["fields"] = list(reversed(s.FIELDS))
    elif bad == "truncated":
        value["data"]["items"] = rows * 2000
    else:
        rows[0][2] = {"negative": -1, "nan": float("nan"), "missing": None, "bool": True}[bad]
    with pytest.raises(ValueError):
        parse(value)


def test_extreme_net_fraction_is_preserved_and_clipping_explicit():
    value = payload()
    for row in value["data"]["items"]:
        row[-1] = -200
    got = parse(value)
    assert got["raw_net_fraction"] == -4 and got["net_fraction"] == -1 and got["net_fraction_clipped"]
