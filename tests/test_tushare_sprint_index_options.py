"""有界股指期权聚合的范围、完整性和单位检查，无供应商请求。"""

import json
import math

import pytest
from app.integrations import tushare_sprint_index_options as s


def raw(rows, code=0):
    return json.dumps({"code": code, "data": {"fields": s.FIELDS, "items": rows}}).encode()


@pytest.fixture
def rows():
    return [
        ["IO2609-C-4400.CFX", "20260915", "CFFEX", 10, 100, 200],
        ["IO2609-P-4400.CFX", "20260915", "CFFEX", 11, 50, 300],
    ]


def test_io_pair_only_and_fixed_one_contract_smoothing(rows):
    rows.append(["MO2609-C-7000.CFX", "20260915", "CFFEX", 50, 9999, 9999])
    v = s.parse(raw(rows), ["2026-09-15"])
    assert v["response_rows"] == 3 and v["io_rows"] == 2
    d = v["rows"]["2026-09-15"]
    assert d["log_put_call_volume"] == math.log(51 / 101) and d["log_put_call_oi"] == math.log(301 / 201)
    assert d["call_contracts"] == d["put_contracts"] == 1


def test_real_zero_activity_is_not_missing(rows):
    for r in rows:
        r[4:] = [0, 0]
    d = s.parse(raw(rows), ["2026-09-15"])["rows"]["2026-09-15"]
    assert d["log_put_call_volume"] == d["log_put_call_oi"] == 0


@pytest.mark.parametrize(
    "kind",
    [
        "duplicate",
        "missing_put",
        "mismatched_strike",
        "missing_day",
        "extra_day",
        "negative",
        "null",
        "boolean",
        "exchange",
        "expired_month",
    ],
)
def test_incomplete_or_invalid_response_rejected(rows, kind):
    expected = ["2026-09-15"]
    if kind == "duplicate":
        rows.append(rows[0])
    elif kind == "missing_put":
        rows.pop()
    elif kind == "mismatched_strike":
        rows[1][0] = "IO2609-P-4500.CFX"
    elif kind == "missing_day":
        expected.append("2026-09-16")
    elif kind == "extra_day":
        rows[0][1] = "20260916"
    elif kind == "negative":
        rows[0][4] = -1
    elif kind == "null":
        rows[0][5] = None
    elif kind == "boolean":
        rows[0][5] = True
    elif kind == "exchange":
        rows[0][2] = "SSE"
    else:
        rows[0][0] = "IO2608-C-4400.CFX"
    with pytest.raises(ValueError):
        s.parse(raw(rows), expected)


def test_permission_denial_not_empty_success(rows):
    with pytest.raises(ValueError, match="IO_PROVIDER_REJECTED"):
        s.parse(raw(rows, 40203), ["2026-09-15"])
