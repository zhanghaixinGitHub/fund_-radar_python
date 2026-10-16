"""完整金额、跨分表重复和缺失边界；不能用凑行或拆桶绕过持仓核验。"""

import io
import zipfile

import pytest
from app.services.fund_rating_public_market import public_document, publication
from app.services.fund_rating_public_stock_tables import cells, parse


def test_issuer_docx_keeps_table_money_and_rejects_entity_declaration():
    def document(xml):
        stream = io.BytesIO()
        with zipfile.ZipFile(stream, "w") as archive:
            archive.writestr("word/document.xml", xml)
        return stream.getvalue()

    xml = (
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        "<w:body><w:tbl><w:tr><w:tc><w:p><w:r><w:t>货币资金</w:t></w:r></w:p></w:tc>"
        "<w:tc><w:p><w:r><w:t>1,234.56</w:t></w:r></w:p></w:tc></w:tr></w:tbl></w:body></w:document>"
    )
    assert public_document(document(xml), {}, "018853", "中期报告")[0] == ["货币资金 1,234.56"]
    with pytest.raises(ValueError, match="XML_ENTITY"):
        public_document(document('<!DOCTYPE x [<!ENTITY x "unsafe">]>' + xml), {}, "018853", "中期报告")


def test_wrapped_money_keeps_every_printed_digit():
    name, values = cells(["中控技术 12,760,839 1,499,398,58", "2.50 9.05"])
    assert name == "中控技术"
    assert values == ["12,760,839", "1,499,398,582.50", "9.05"]


def test_truncated_amount_is_not_filled_from_weight():
    with pytest.raises(ValueError, match="INCOMPLETE_OR_AMBIGUOUS"):
        cells(["中控技术 12,760,839 1,499,398,58 9.05"])


def test_index_and_active_sleeves_merge_same_security_with_amount_conservation():
    pages = [
        """7.2 报告期末按行业分类的股票投资组合
J 金融业 1,500.00 15.00
合计 1,500.00 15.00
7.3 期末所有股票投资明细
7.3.1 期末指数投资全部股票
1 600000 浦发银行 100 1,000.00 10.00
7.3.2 期末积极投资全部股票
1 600000 浦发银行 50 500.00 5.00
7.4 报告期股票变动
"""
    ]
    result = parse(pages, "1500.00", "10000")
    assert len(result["holdings"]) == 1
    assert result["holdings"][0]["quantity_shares"] == "150"
    assert result["holdings"][0]["market_value_cny"] == "1500.00"
    assert len(result["holdings"][0]["reported_rows"]) == 2
    with pytest.raises(ValueError, match="FULL_AMOUNT_MISMATCH"):
        parse(pages, "1600.00", "10000")


def test_mixed_industry_taxonomies_cannot_be_admitted_as_diversification():
    pages = [
        """7.2 报告期末按行业分类的股票投资组合
C 制造业 1000.00 10.00
合计 1000.00 10.00
7.2.2 港股通股票投资组合
45 信息技术 500.00 5.00
合计 500.00 5.00
7.3 期末所有股票投资明细
1 688981 中芯国际 100 1000.00 10.00
2 00981 中芯国际 100 500.00 5.00
7.4 报告期股票变动
"""
    ]
    with pytest.raises(ValueError, match="MIXED_INDUSTRY_TAXONOMY"):
        parse(pages, "1500.00", "10000")
    partial = parse(pages, "1500.00", "10000", require_industry=False)
    assert len(partial["holdings"]) == 2
    assert partial["holdings"][1]["stock_code"] == "00981.HK"


def test_report_publication_accepts_iso_but_not_compile_date():
    assert publication(["编制日期2026-04-30 送出日期：2026-05-30"]) == "2026-05-30"
    assert publication(["送出日期：二0二六年八月三十一日"]) == "2026-08-31"
    with pytest.raises(ValueError, match="PUBLICATION_DATE_MISSING"):
        publication(["编制日期2026-04-30 信息截至2026-04-29"])


def test_article_metadata_without_full_text_is_not_a_report():
    import json

    raw = json.dumps(
        {"success": 1, "data": {"art_code": "AN1", "security": [{"stock": "004237"}], "notice_title": "2026年中期报告"}}
    ).encode()
    with pytest.raises(ValueError, match="IDENTITY_OR_COMPLETENESS"):
        public_document(raw, {"article_id": "AN1"}, "004237", "中期报告")
