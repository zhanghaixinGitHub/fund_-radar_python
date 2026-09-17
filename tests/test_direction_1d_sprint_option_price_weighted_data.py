"""验证价格权重独立于手数、真实零值与来源绑定；全部使用固定样本，不访问供应商。"""

import json
import math
from datetime import date

import pytest
from app.services import direction_1d_sprint_option_price_weighted_data as s


def raw(call_price=2, put_price=8):
    rows = [
        [f"IO2609-{side}-4000.CFX", "20260915", "CFFEX", price, 3, 5]
        for side, price in [("C", call_price), ("P", put_price)]
    ]
    return json.dumps({"code": 0, "data": {"fields": s.old.parser.FIELDS, "items": rows}}).encode()


def test_equal_contract_quantities_have_different_price_weights():
    body = raw()
    assert s.old.parser.parse(body, ["2026-09-15"])["rows"]["2026-09-15"]["log_put_call_volume"] == 0
    v = s.parse(body, ["2026-09-15"])["2026-09-15"]
    assert v[s.KEYS[0]] == pytest.approx(math.log(25 / 7))
    assert v[s.KEYS[1]] == pytest.approx(math.log(41 / 11))
    assert v["actual_turnover"] is False and v["io_contracts"] == 2


def test_explicit_zero_price_retained_and_counted():
    v = s.parse(raw(0, 0), ["2026-09-15"])["2026-09-15"]
    assert v[s.KEYS[0]] == v[s.KEYS[1]] == 0
    assert v["zero_close_with_volume"] == v["zero_close_contracts"] == 2
    assert v["available"] is True


@pytest.mark.parametrize("mutation", ["unpaired", "duplicate", "date", "negative", "overflow"])
def test_invalid_or_unbounded_raw_rejected(mutation):
    v = json.loads(raw())
    rows = v["data"]["items"]
    if mutation == "unpaired":
        rows[1][0] = "IO2609-P-4100.CFX"
    elif mutation == "duplicate":
        rows.append(rows[0])
    elif mutation == "date":
        rows[1][1] = "20260916"
    elif mutation == "negative":
        rows[1][3] = -1
    else:
        rows[1][3] = rows[1][4] = 1e308
    with pytest.raises(ValueError):
        s.parse(json.dumps(v).encode(), ["2026-09-15"])


def test_missing_date_does_not_become_zero_or_use_later_data(monkeypatch):
    monkeypatch.setattr(s.base, "calendar", lambda: ([date(2026, 9, d) for d in (15, 16, 17)], "fixture"))
    points = s.parse(raw(), ["2026-09-15"])
    assert s.extend({}, "2026-09-15", "2026-09-16", points)["option_price_weighted"]["available"]
    assert s.extend({}, "2026-09-16", "2026-09-17", points)["option_price_weighted"] == {
        "date": "2026-09-16",
        "available": False,
    }
    with pytest.raises(ValueError):
        s.extend({}, "2026-09-15", "2026-09-17", points)


def test_unavailable_actual_parent_keeps_actual_timestamp_and_hash(monkeypatch):
    parent = {"at": "2026-09-17T08:14:32+08:00", "available": False, "errors": ["MISSING_ACTUAL_RESPONSE"]}
    monkeypatch.setattr(s.old, "load_live", lambda *args: parent)
    v = s.live("2026-09-16", "2026-09-17", "frozen-parent")
    assert v["at"] == parent["at"] and v["parent_source_hash"] == s.base.digest(parent)
    assert v["available"] is False and v["new_source_requests"] == 0 and v["snapshot"]["points"] == {}
