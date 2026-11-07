"""季度披露只使用原表可证明的证券范围，不把后发的中报全仓塞回早发季报。"""

import pytest
from app.services.market_holdings_parse import disclosed_single_fund, disclosed_stocks, partial_stock_sections


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


def test_index_and_active_quarter_tables_keep_independent_rank_and_partial_scope():
    text = """5.3 期末股票投资明细
5.3.1 指数投资前十名股票投资明细
序号 股票代码 股票名称 数量 公允价值 占基金资产净值比例
1 600001 公司甲 100 123,456.78 9.12
2 000001 公司乙 200 100,000.00 8.00
5.3.2 积极投资前五名股票投资明细
序号 股票代码 股票名称 数量 公允价值 占基金资产净值比例
1 300001 公司丙 100 12,345.67 0.00
5.4 报告期末按债券品种分类
"""
    rows, groups = partial_stock_sections(text)
    assert len(rows) == 3 and len(groups) == 2
    assert rows[-1]["weightPct"] == 0 and rows[-1]["marketValue"] == 12345.67
    with pytest.raises(ValueError, match="IDENTITY"):
        partial_stock_sections(text.replace("300001", "600001"))
    with pytest.raises(ValueError, match="RANK"):
        partial_stock_sections(text.replace("2 000001", "3 000001"))
    with pytest.raises(ValueError, match="INCOMPLETE"):
        partial_stock_sections(text.replace("12,345.67 0.00", "12,345.67"))


def test_quarter_header_wrap_and_same_issuer_a_h_share_rank_are_preserved():
    text = """5.3期末按公允价值占基金资产净值比例大小排序的股票投资明细
5.3.1 期末前十名股票投资明细
占基金资产净
序号 股票代码 股票名称 数量 公允价值
值比例（％）
1 603259 药明康德 100 12,345.67 9.12
1 02359 药明康德 200 11,000.00 8.00
2 000001 公司乙 100 10,000.00 7.00
5.4报告期末债券投资组合
"""
    rows, _ = partial_stock_sections(text)
    assert [r["stockCode"] for r in rows] == ["603259.SH", "02359.HK", "000001.SZ"]
    with pytest.raises(ValueError, match="RANK"):
        partial_stock_sections(text.replace("02359 药明康德", "02359 另一公司"))
    with pytest.raises(ValueError, match="RANK"):
        partial_stock_sections(text.replace("02359 药明康德", "600259 药明康德"))
    with pytest.raises(ValueError, match="RANK"):
        partial_stock_sections(text.replace("2 000001", "1 000001"))


def test_interleaved_serial_header_character_does_not_change_numeric_rows():
    text = """5.3 期末股票投资明细
序 股票代码 股票名称 数量 公允价值 占基金资产净
号 值比例（％）
1 600001 公司甲 100 123,456.78 9.12
5.4 报告期末债券投资组合
"""
    assert partial_stock_sections(text)[0][0]["marketValue"] == 123456.78
    with pytest.raises(ValueError, match="INCOMPLETE"):
        partial_stock_sections(text.replace("123,456.78 9.12", "123,456.78"))


@pytest.mark.parametrize(
    "code,count,first_value", [("007356", 11, 1073987462.90), ("015596", 15, 476595910.80), ("160323", 10, 11411296.55)]
)
def test_real_quarter_row_wraps_preserve_source_digits_and_partial_scope(code, count, first_value):
    from pathlib import Path

    original = (Path(__file__).parent / "fixtures/quarter-layout-20261028" / (code + ".txt")).read_text("utf-8")
    rows, groups = partial_stock_sections(original)
    assert len(rows) == count and rows[0]["marketValue"] == first_value
    assert all("部分" in g["notice"] for g in groups)
    if code == "007356":
        assert rows[2]["stockCode"] == "300308.SZ" and rows[3]["stockCode"] == "03308.HK"
        assert rows[3]["marketValue"] == 1002806.24
        assert next(r["stockName"] for r in rows if r["stockCode"] == "688012.SH") == "中微半导体设备"
    if code == "015596":
        assert rows[-1]["weightPct"] == 0 and rows[-1]["marketValue"] == 109096.02


@pytest.mark.parametrize("problem", ["missing_amount_digit", "missing_quantity_digit", "wrong_duplicate_rank"])
def test_real_wrapped_rows_still_reject_missing_digits_and_unsupported_same_rank(problem):
    from pathlib import Path

    original = (Path(__file__).parent / "fixtures/quarter-layout-20261028/015596.txt").read_text("utf-8")
    if problem == "missing_amount_digit":
        original = original.replace("20              0", "20", 1)
    elif problem == "missing_quantity_digit":
        original = original.replace("20              0", "2              0", 1)
    else:
        original = original.replace("2    603993", "1    603993", 1)
    with pytest.raises(ValueError):
        partial_stock_sections(original)


@pytest.mark.parametrize(
    "code,first,last",
    [("017145", "MRVL.US", "AVGO.US"), ("018853", "PBF.US", "DK.US"), ("019449", "6098.JP", "8766.JP")],
)
def test_foreign_quarter_preserves_reported_market_codes_and_cny(code, first, last):
    from pathlib import Path

    from app.services.market_holdings_parse import partial_overseas_stock_rows

    original = (Path(__file__).parent / "fixtures/quarter-layout-20261028" / (code + ".txt")).read_text("utf-8")
    rows = partial_overseas_stock_rows(original)
    assert len(rows) == 10 and rows[0]["stockCode"] == first and rows[-1]["stockCode"] == last
    if code == "017145":
        assert rows[6]["stockCode"] == "AMD.US" and rows[6]["marketValue"] == 21371813.39
    with pytest.raises(ValueError):
        partial_overseas_stock_rows(original.replace("人民币", "美元").replace("人民", "美元"))


def test_us_quarter_keeps_exact_amount_fragments_and_reported_market():
    from pathlib import Path

    from app.services.market_holdings_parse import partial_overseas_stock_rows

    original = (Path(__file__).parent / "fixtures/quarter-layout-20261028/018156.txt").read_text("utf-8")
    rows = partial_overseas_stock_rows(original)
    assert len(rows) == 10
    assert rows[0]["stockCode"] == "PTGX.US" and rows[0]["marketValue"] == 12531282.20
    assert rows[6]["stockCode"] == "DFTX.US" and rows[6]["marketValue"] == 11278867.86
    assert rows[-1]["stockCode"] == "URGN.US" and rows[-1]["marketValue"] == 10945884.52
    for broken in [original.replace("12,531,", "12,53,", 1), original.replace("US", "HK", 1),
                   original.replace("40,000", "-", 1), original.replace("民币", "美元")]:
        with pytest.raises(ValueError):
            partial_overseas_stock_rows(broken)


def test_no_stock_scope_requires_literal_own_report_or_two_matching_sections():
    from app.services.market_holdings_parse import _reported_no_stocks

    assert _reported_no_stocks('注：本基金在本报告期末未持有股票。')
    assert _reported_no_stocks('报告期末本基金未持有股票。')
    original = ('5.1 报告期末基金资产组合情况\n其中：股票 - -\n5.2 目标基金明细\n'
                '5.4 报告期末按比例排序的前十名股票投资明细\n无。\n5.5 债券投资组合\n')
    assert _reported_no_stocks(original)
    assert not _reported_no_stocks(original.replace('股票 - -','股票 1.00 0.01'))
    assert not _reported_no_stocks(original.replace('其中：股票 - -','其中：债券 - -'))
    assert not _reported_no_stocks(original.replace('无。','待核验'))


def test_vietnam_quarter_fragments_are_joined_within_same_source_columns():
    from pathlib import Path

    from app.services.market_holdings_parse import partial_vietnam_stock_rows

    original = (Path(__file__).parent / "fixtures/quarter-layout-20261028/008764.txt").read_text("utf-8")
    rows = partial_vietnam_stock_rows(original)
    assert len(rows) == 10 and rows[0]["stockCode"] == "VN000000VIC9"
    assert rows[0]["marketValue"] == 307333352.48
    assert rows[6]["stockCode"] == "VN000000MBB5" and rows[6]["marketValue"] == 197371152.79
    assert rows[-1]["marketValue"] == 158387944.80
    with pytest.raises(ValueError):
        partial_vietnam_stock_rows(original.replace("VIC9", "VIC", 1))
    with pytest.raises(ValueError):
        partial_vietnam_stock_rows(original.replace("9,200    48", "9,200    4", 1))


def test_multiple_fof_products_keep_their_own_name_amount_and_weight():
    from pathlib import Path

    from app.services.market_holdings_parse import disclosed_multiple_funds

    original = (Path(__file__).parent / "fixtures/quarter-layout-20261028/006476.txt").read_text("utf-8")
    rows = disclosed_multiple_funds(original)
    assert len(rows) == 7 and rows[0]["marketValue"] == 320131062.48
    assert rows[5]["stockName"] == "Simplex WTI ETF" and rows[5]["weightPct"] == 5.26
    assert rows[-1]["stockName"] == "NEXT FUNDS NOMURA Crude Oil Long Index Linked Exchange Traded"
    assert rows[-1]["marketValue"] == 84166697.02 and all(r["stockCode"] == "" for r in rows)
    with pytest.raises(ValueError):
        disclosed_multiple_funds(original.replace("2.48      19.12", "2.4      19.12", 1))
    with pytest.raises(ValueError):
        disclosed_multiple_funds(original.replace("    2      Brent Oil", "    1      Brent Oil", 1))


def test_etc_and_etf_table_retains_names_that_start_above_rank():
    from pathlib import Path

    from app.services.market_holdings_parse import disclosed_multiple_funds

    text = (Path(__file__).parent / "fixtures/quarter-layout-20261028/003321-multi.txt").read_text("utf8")
    rows = disclosed_multiple_funds(text)
    assert len(rows) == 6
    assert rows[0]["stockName"] == "WisdomTree WTI Crude Oil"
    assert rows[-1]["stockName"] == "Samsung S&P GSCI Crude Oil ER Futures ETF"
    assert rows[0]["marketValue"] == 171172800.95
    assert round(sum(r["marketValue"] for r in rows), 2) == 786986932.47
    with pytest.raises(ValueError, match="COLUMNS"):
        disclosed_multiple_funds(text.replace("ETC", "未知"))
    with pytest.raises(ValueError):
        disclosed_multiple_funds(text.replace("171,172,800.95", "171,172,800.9"))


def test_local_us_codes_keep_both_share_classes_under_one_issuer_rank():
    from pathlib import Path

    from app.services.market_holdings_parse import _local_us_code_rows, partial_overseas_stock_rows

    text = (Path(__file__).parent / "fixtures/quarter-layout-20261028/000041-verified.txt").read_text("utf8")
    rows = partial_overseas_stock_rows(text)
    assert len(rows) == 11 and rows[0]["stockCode"] == "NVDA.US"
    assert [row["stockCode"] for row in rows[1:3]] == ["GOOGL.US", "GOOG.US"]
    assert rows[1]["marketValue"] == 88061701.90 and rows[-1]["stockCode"] == "DELL.US"
    with pytest.raises(ValueError):
        _local_us_code_rows(text.replace("纳斯达", "不明市场"))
    with pytest.raises(ValueError):
        _local_us_code_rows(text.replace("GOOGL", "GOOG"))


def test_money_assets_and_cd_names_come_from_reported_cells():
    from pathlib import Path

    from app.services.market_holdings_parse import _money_quarter_assets, _money_quarter_bonds

    text = (Path(__file__).parent / "fixtures/quarter-layout-20261028/000009-verified.txt").read_text("utf8")
    assets = _money_quarter_assets(text)
    assert len(assets) == 4 and assets[0]["value"] == 34278514824.16
    bonds = _money_quarter_bonds(text)
    assert len(bonds) == 10 and bonds[0]["stockName"] == "26宁波银行CD247"
    assert bonds[-1]["marketValue"] == 497079420.56
    with pytest.raises(ValueError, match="RECONCILIATION"):
        _money_quarter_assets(text.replace("73,742,970,893.28", "73,742,970,893.29" + "0"))
    with pytest.raises(ValueError):
        _money_quarter_bonds(text.replace("CD247", "不明尾码"))


def test_domestic_fof_names_do_not_include_operation_type_or_next_fund():
    from pathlib import Path

    from app.services.fund_holdings_publication import _domestic_fund_rows

    text = (Path(__file__).parent / "fixtures/quarter-layout-20261028/005156-verified.txt").read_text("utf8")
    rows = _domestic_fund_rows(text)
    assert len(rows) == 10 and rows[0]["stockName"] == "鹏扬利沣短债A"
    assert rows[3]["stockName"] == "国泰海通君得利短债A"
    assert rows[8]["stockName"] == "工银瑞信信用添利债券A"
    assert rows[-1]["stockName"] == "鑫元泽利债券A"
    assert rows[-1]["marketValue"] == 10209813.12
    with pytest.raises(ValueError, match="VERTICAL"):
        _domestic_fund_rows(text + "\n契\n约\n")
    with pytest.raises(ValueError, match="RANK"):
        _domestic_fund_rows(text.replace("2      008864", "3      008864"))


def test_decimal_tail_cannot_be_misread_as_weight_or_quantity_inside_name():
    from pathlib import Path

    original = (Path(__file__).parent / "fixtures/quarter-layout-20261028/014156.txt").read_text("utf-8")
    rows, groups = partial_stock_sections(original)
    assert len(rows) == 15 and len(groups) == 2
    assert rows[0]["stockName"] == "源杰科技" and rows[0]["marketValue"] == 52504763.00 and rows[0]["weightPct"] == 1.44
    assert rows[1]["marketValue"] == 50800261.92 and rows[1]["weightPct"] == 1.39
    with pytest.raises(ValueError):
        partial_stock_sections(original.replace("52,504,763.0", "52,504,76.0", 1))


def test_link_fund_stock_table_in_section_54_keeps_market_code_and_numeric_columns():
    original = """5.4 期末按公允价值占基金资产净值比例大小排序的股票投资明细
5.4.1 报告期末按公允价值占基金资产净值比例大小排序的前十名股票投资明细
序号 股票代码 股票名称 数量（股） 公允价值（元） 占基金资产净值比例（%）
1 688256 寒武纪 32,516 32,808,969.16 0.23
2 002371 北方华创 52,300 32,215,231.00 0.23
5.5 报告期末按债券品种分类的债券投资组合
"""
    rows, _ = partial_stock_sections(original)
    assert [r["stockCode"] for r in rows] == ["688256.SH", "002371.SZ"]
    assert rows[0]["marketValue"] == 32808969.16 and rows[0]["weightPct"] == .23


def test_strict_quarter_rows_override_successful_but_misaligned_generic_parser(monkeypatch, tmp_path):
    from pathlib import Path
    from types import SimpleNamespace

    from app.services import market_holdings_parse as module

    original = (Path(__file__).parent / "fixtures/quarter-layout-20261028/018125.txt").read_text("utf-8")
    monkeypatch.setattr(module, "read_source", lambda *a: b"original")
    monkeypatch.setattr(module, "public_document", lambda *a: ([original], None))
    monkeypatch.setattr(module, "_fund_name_matches", lambda *a: True)
    monkeypatch.setattr(module, "publication", lambda *a: "2026-10-27")
    bad = {
        "report_end": "2026-09-30",
        "holdings": [
            {
                "stock_code": "301550.SZ",
                "stock_name": "斯菱智驱10,329,188",
                "nav_weight_pct": 6,
                "market_value_cny": 9.65,
            }
        ],
        "stock_nav_pct": 90,
        "assets": {},
        "reported_industries": [],
    }
    monkeypatch.setattr(module, "parse_text", lambda *a, **k: bad)
    monkeypatch.setattr(module, "_public_table", lambda *a: [])
    title = "永赢先进制造智选混合型发起式证券投资基金2026年第3季度报告"
    fund = {"fund_code": "018125", "fund_name": "永赢先进制造智选混合-C"}
    entry = {
        "kind": "QUARTER",
        "title": title,
        "id": "AN202610271830993090",
        "end": "2026-09-30",
        "published": "2026-10-27",
    }
    report, _ = module.parse_report(
        SimpleNamespace(table=lambda *a: {}),
        tmp_path,
        fund,
        entry,
        {"sha256": "a" * 64, "url": "https://example.com/report"},
    )
    assert report["holdings"][0]["stockName"] == "斯菱智驱"
    assert report["holdings"][0]["weightPct"] == 9.65 and report["holdings"][0]["marketValue"] == 1035191221.36
    assert report["fullDisclosure"] is False
