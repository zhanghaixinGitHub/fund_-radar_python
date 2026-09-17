"""验证排名独立名单、空值和跨合约/日期边界，防止把不完整数据当有效特征。"""

import json

import pytest
from app.integrations import tushare_sprint_futures_holdings as source


def fixture_rows():
    return [["20260915", "IF2609", f"member{i:02d}", 100, 0, 100, 2, 90, -1, "CFFEX"] for i in range(20)]


def raw(rows):
    return json.dumps({"code": 0, "data": {"fields": list(source.FIELDS), "items": rows}}).encode()


def run(rows):
    return source.parse(raw(rows), "IF2609", ["2026-09-15"])


def test_independent_lists_and_units():
    rows = fixture_rows()
    # 多方榜外行不作零仓位；空方另一个会员入榜，仍恰好各20名。
    rows[0][7:9] = [None, None]
    rows.append(["20260915", "IF2609", "different", None, None, None, None, 90, -1, "CFFEX"])
    point = run(rows)["2026-09-15"]
    assert point["level_imbalance"] == pytest.approx(200 / 3800)
    assert point["change_imbalance"] == pytest.approx(60 / 3800)
    assert point["ranked_long_contracts"] == 2000 and point["ranked_short_contracts"] == 1800
    assert point["rows"][0]["long_hld"] is None


@pytest.mark.parametrize(
    "column,value",
    [
        (0, "20260914"),
        (1, "IF2612"),
        (2, "member01"),
        (2, "合计"),
        (5, -1),
        (5, True),
        (5, 1.5),
        (5, float("nan")),
        (5, None),
        (6, None),
        (8, None),
        (9, "SHFE"),
    ],
)
def test_invalid_rows_fail_closed(column, value):
    rows = fixture_rows()
    rows[0][column] = value
    with pytest.raises(ValueError):
        run(rows)


def test_missing_day_and_truncation_rejected():
    with pytest.raises(ValueError, match="RANKING_COVERAGE"):
        source.parse(raw(fixture_rows()), "IF2609", ["2026-09-15", "2026-09-16"])
    with pytest.raises(ValueError, match="TRUNCATED"):
        run(fixture_rows() * 100)


def test_large_change_is_preserved_and_clipped():
    rows = fixture_rows()
    rows[0][6] = 10000
    point = run(rows)["2026-09-15"]
    assert point["change_clipped"] and point["change_imbalance"] == 1.0
    assert point["raw_change_imbalance"] > 1


def test_provider_rejection_and_empty():
    with pytest.raises(ValueError, match="PROVIDER_REJECTED"):
        source.parse(b'{"code":40203}', "IF2609", ["2026-09-15"])
    with pytest.raises(ValueError, match="EMPTY"):
        run([])
