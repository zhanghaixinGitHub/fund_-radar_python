"""来自实际本期原文的列错位回归；不把原解析器不支持当来源不存在。"""

from pathlib import Path

import pytest
from app.services.market_holdings_parse import gold_quarter_rows, partial_overseas_stock_rows

ROOT = Path(__file__).parent / "fixtures/quarter-layout-20261103"


def test_us_market_code_and_cents_stay_in_their_original_row():
    rows = partial_overseas_stock_rows((ROOT / "159632.txt").read_text(encoding="utf-8"))
    assert [r["stockCode"] for r in rows] == [
        "NVDA.US",
        "AAPL.US",
        "MSFT.US",
        "MU.US",
        "AMD.US",
        "AMZN.US",
        "META.US",
        "GOOGL.US",
        "GOOG.US",
        "TSLA.US",
    ]
    values = {r["stockCode"]: r["marketValue"] for r in rows}
    assert values["NVDA.US"] == 911042667.20
    assert values["MU.US"] == 549305785.05
    assert values["GOOG.US"] == 311629206.61
    assert sum(r["weightPct"] for r in rows) == pytest.approx(43.25)


@pytest.mark.parametrize(
    "before,after",
    [("GOOGL", "GOOG"), ("    2.61", "    --"), (" 911,042,667.", " 911,042,66."), ("美国    美国", "日本    美国")],
)
def test_missing_tail_duplicate_symbol_or_foreign_market_is_rejected(before, after):
    body = (ROOT / "159632.txt").read_text(encoding="utf-8")
    assert before in body
    with pytest.raises(ValueError):
        partial_overseas_stock_rows(body.replace(before, after, 1))


def test_gold_only_original_has_no_stock_and_keeps_two_denominators_separate():
    assets, rows = gold_quarter_rows((ROOT / "159812.txt").read_text(encoding="utf-8"))
    assert assets[0]["weightPct"] == 99.37
    assert rows[0] == {"stockCode": "Au99.99", "stockName": "Au99.99", "marketValue": 1436169608.40, "weightPct": 99.47}
    assert assets[-1]["value"] == 1445317047.84


@pytest.mark.parametrize(
    "before,after",
    [
        ("1,445,317,047.84", "1,445,317,048.84"),
        ("1,582,870", "--"),
        ("99.47", "199.47"),
        ("仅持有以上贵金属", "还持有其他贵金属"),
    ],
)
def test_gold_incomplete_or_unreconciled_original_is_not_published(before, after):
    body = (ROOT / "159812.txt").read_text(encoding="utf-8")
    assert before in body
    with pytest.raises(ValueError):
        gold_quarter_rows(body.replace(before, after, 1))
