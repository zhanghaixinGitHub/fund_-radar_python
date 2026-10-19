"""季度披露只使用原表可证明的证券范围，不把后发的中报全仓塞回早发季报。"""

import pytest
from app.services.market_holdings_parse import disclosed_single_fund, disclosed_stocks


def test_half_year_public_table_cannot_expand_quarterly_top_ten():
    rows = [{"stockCode": code} for code in ("000001.SZ", "600001.SH", "000003.SZ")]
    original = "5.4 前十名股票投资明细\n1 000001 公司甲\n2 600001 公司乙\n5.5 债券投资\n000003"
    assert disclosed_stocks(rows, original) == rows[:2]
    with pytest.raises(ValueError, match="IDENTITY"):
        disclosed_stocks(rows, "另一份报告没有披露这些股票代码")


def test_single_target_fund_keeps_wrapped_amount_and_never_uses_type_as_name():
    original = """5.9 前十名基金投资明细
  序号    基金名称  基金类型  运作方式    管理人    公允价值      占净值比
    1      南方标普  股票型    交易型开  南方基金  13,629,644      92.39
            中国 A 股              放式      管理股份      ,235.91
            大盘红利                        有限公司
            低波
            50ETF
5.10 股指期货投资说明
"""
    title = "南方标普中国A股大盘红利低波50交易型开放式指数证券投资基金联接基金2026年第2季度报告"
    row = disclosed_single_fund(original, title)[0]
    assert row["marketValue"] == 13_629_644_235.91
    assert row["weightPct"] == 92.39
    assert row["stockName"] == title.split("联接")[0]
    with pytest.raises(ValueError, match="NAME"):
        disclosed_single_fund(original, "另一个完全不同名称的目标基金联接基金2026年第2季度报告")


def test_multiple_funds_and_ambiguous_amount_stay_pending():
    original = """5.2 期末投资目标基金明细
    1    目标基金ETF      ETF                         123,456.78      92.39
    2    另一基金ETF      ETF                         100,000.00       2.39
5.3 股票投资
"""
    with pytest.raises(ValueError, match="MULTIPLE"):
        disclosed_single_fund(original, "目标基金ETF联接基金2026年第2季度报告")
    single = original.replace("    2    另一基金ETF      ETF                         100,000.00       2.39\n", "")
    with pytest.raises(ValueError, match="AMOUNT"):
        disclosed_single_fund(single.replace("123,456.78", "123,456.78 999"), "目标基金ETF联接基金2026年第2季度报告")
