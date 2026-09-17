"""原始期货源解析边界；换月价格不被强行接成连续收益。"""

import json

import pytest
from app.integrations.tushare_sprint_futures import FIELDS, parse


def body(api="fut_daily", changes=None):
    row = {
        "ts_code": "IF.CFX",
        "trade_date": "20250102",
        "pre_close": 3921.0,
        "open": 3921.0,
        "high": 3931.0,
        "low": 3784.4,
        "close": 3817.8,
        "settle": 3812.8,
        "vol": 82958.0,
        "oi": 151506.0,
        "mapping_ts_code": "IF2503.CFX",
    }
    row.update(changes or {})
    return json.dumps({"code": 0, "data": {"fields": FIELDS[api], "items": [[row[k] for k in FIELDS[api]]]}}).encode()


def test_daily_raw_points_preserved():
    row = parse(body(), "fut_daily", "20250101", "20251231")["2025-01-02"]
    assert row["close"] == 3817.8 and "oi_chg" not in row


def test_mapping_and_spot():
    assert (
        parse(body("fut_mapping"), "fut_mapping", "20250101", "20251231")["2025-01-02"]["mapping_ts_code"]
        == "IF2503.CFX"
    )
    assert parse(body("index_daily", {"ts_code": "000300.SH"}), "index_daily", "20250101", "20251231")


@pytest.mark.parametrize(
    "changes",
    [
        {"close": float("nan")},
        {"close": True},
        {"high": 3800.0},
        {"oi": 0.0},
        {"ts_code": "IF2503.CFX"},
        {"trade_date": "20260102"},
        {"trade_date": "20250104"},
    ],
)
def test_bad_quote_rejected(changes):
    with pytest.raises(ValueError):
        parse(body(changes=changes), "fut_daily", "20250101", "20251231")


def test_continuous_alias_not_a_monthly_mapping():
    with pytest.raises(ValueError, match="CONTRACT_MAPPING_INVALID"):
        parse(body("fut_mapping", {"mapping_ts_code": "IFL.CFX"}), "fut_mapping", "20250101", "20251231")


def test_duplicate_day_rejected():
    value = json.loads(body())
    value["data"]["items"] *= 2
    with pytest.raises(ValueError, match="DUPLICATE"):
        parse(json.dumps(value).encode(), "fut_daily", "20250101", "20251231")


def test_permission_failure_not_treated_as_empty_market():
    with pytest.raises(ValueError, match="PROVIDER_REJECTED"):
        parse(b'{"code":40203}', "fut_daily", "20250101", "20251231")
