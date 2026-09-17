"""验证股票广度分母、同日身份、除权参考口径与无法证明完整性的边界。"""

import json

import pytest
from app.integrations import tushare_sprint_stock_breadth as s


def body():
    rows = [
        [code, "20260915", 100 + pct, 100, pct, 1, 10]
        for code, pct in [("600001.SH", -2), ("000001.SZ", 0), ("300001.SZ", 1), ("920001.BJ", 5)]
    ]
    return {"code": 0, "data": {"fields": s.FIELDS, "items": rows}}


def test_fixed_sh_sz_scope_counts_flat_and_retains_excluded_bj_count():
    x = s.parse(json.dumps(body()).encode(), "2026-09-15", minimum_included=3)
    assert x["included_rows"] == 3 and x["excluded_bj_rows"] == 1 and x["returned_rows"] == 4
    assert x["up"] == x["down"] == x["flat"] == 1
    assert x["breadth"] == x["median_pct"] == 0 and x["iqr_pct"] == 1.5
    assert x["all_listed_stocks_independently_verified"] is False


@pytest.mark.parametrize("bad", ["duplicate", "date", "pct", "negative", "nan", "bool", "denied", "truncated"])
def test_invalid_response_rejected_as_a_whole(bad):
    v = body()
    rows = v["data"]["items"]
    if bad == "duplicate":
        rows.append(rows[0])
    elif bad == "date":
        rows[0][1] = "20260916"
    elif bad == "pct":
        rows[0][4] = 2
    elif bad == "negative":
        rows[0][5] = -1
    elif bad == "nan":
        rows[0][4] = float("nan")
    elif bad == "bool":
        rows[0][5] = True
    elif bad == "denied":
        v["code"] = 40203
    else:
        v["data"]["items"] = rows * 1500
    with pytest.raises(ValueError):
        s.parse(json.dumps(v).encode(), "2026-09-15", minimum_included=3)


def test_shuffling_does_not_change_distributions_or_set_hash():
    v = body()
    expected = s.parse(json.dumps(v).encode(), "2026-09-15", minimum_included=3)
    v["data"]["items"].reverse()
    assert s.parse(json.dumps(v).encode(), "2026-09-15", minimum_included=3) == expected


def test_stock_quote_count_floor_does_not_silently_drop_missing_quotes():
    with pytest.raises(ValueError, match="INSUFFICIENT_RETURNED_UNIVERSE"):
        s.parse(json.dumps(body()).encode(), "2026-09-15")
