"""历史时点与报告类型的反例；测试原文不能作为真实基金校准资料。"""

from datetime import date

import pytest
from app.services import fund_quality_v2_disclosures as module
from app.services.fund_quality_v2_stock_book import periodic_financial_view


def document_refs(tmp_path, monkeypatch, definitions):
    """只替换文档解码；实际选择、日期和正文期末核对均走生产代码。"""
    refs = []
    for label, observed, published, sent, year in definitions:
        text = (
            f"测试股票型证券投资基金123456 {year}年"
            + ("年度报告" if "annual" in label else "中期报告")
            + f"\n报告送出日期{sent}\n本报告期自{year}年1月1日起至"
            + (f"{year}年12月31日止。" if "annual" in label else f"{year}年6月30日止。")
        )
        name = str(len(refs))
        (tmp_path / name).write_bytes(text.encode("utf-8"))
        refs.append({
            "label": label, "report_end": observed, "published_on": published,
            "file": name, "sha256": "TEST_ONLY_" + name,
            "url": "https://www.cmfchina.com/" + name,
        })
    monkeypatch.setattr(module, "public_document", lambda raw, *args: ([raw.decode("utf-8")], None))
    return refs


def test_annual_primary_uses_available_publication_month_and_preserves_actual_type(tmp_path, monkeypatch):
    refs = document_refs(tmp_path, monkeypatch, [
        ("historical-half", "2025-06-30", "2025-08-30", "2025年8月30日", 2025),
        ("historical-annual", "2025-12-31", "2026-03-31", "2026年3月31日", 2025),
    ])
    current = module.source_documents("123456", list(reversed(refs)), tmp_path, date(2026, 4, 30))
    assert current["report"]["report_kind"] == "annual" and "half" not in current
    assert current["report"]["economic_period"] == ["2025-01-01", "2025-12-31"]
    assert current["report"]["original_sent_on"] == "2026-03-31"
    before = module.source_documents("123456", refs, tmp_path, date(2025, 12, 31))
    assert before["report"]["report_kind"] == "half"
    assert before["report"]["economic_period"][1] == "2025-06-30"


@pytest.mark.parametrize(
    ("observed", "published", "sent", "error"),
    [
        ("2025-12-31", "2026-03-31", "2026年5月1日", "ORIGINAL_SENT_AFTER_AS_OF"),
        ("2025-06-30", "2026-03-31", "2026年3月31日", "REGISTERED_PERIOD_CONFLICT"),
        ("2025-12-31", None, "2026年3月31日", "PUBLICATION_RECEIPT_REQUIRED"),
    ],
)
def test_metadata_never_overrules_actual_report_date(tmp_path, monkeypatch, observed, published, sent, error):
    refs = document_refs(tmp_path, monkeypatch, [
        ("historical-annual", observed, published, sent, 2025),
    ])
    result = module.source_documents("123456", refs, tmp_path, date(2026, 4, 30))
    assert error in result["report"]["error"]
    assert "text" not in result["report"]


def test_newer_half_replaces_annual_by_economic_period_and_quarter_does_not(tmp_path, monkeypatch):
    refs = document_refs(tmp_path, monkeypatch, [
        ("historical-half", "2026-06-30", "2026-08-31", "2026年8月31日", 2026),
        ("historical-annual", "2025-12-31", "2026-03-31", "2026年3月31日", 2025),
    ])
    refs += [{"label": "historical-quarter", "report_end": "2026-09-30", "published_on": "2026-10-15"}]
    result = module.source_documents("123456", refs, tmp_path, date(2026, 10, 17))
    assert result["report"]["report_kind"] == "half"
    assert result["report"]["economic_period"][1] == "2026-06-30"


def test_annual_view_preserves_source_pages_and_financial_digits():
    pages = ["7.1 资产负债表\n股票投资 7.4.7.1 7.40 100.00\n7.2 利润表\n"
             "8.3 期末按公允价值占基金资产净值比例大小排序的所有股票投资明细\n001248 10 100.00"]
    original = list(pages)
    view, evidence = periodic_financial_view(pages)
    assert pages == original
    assert "股票投资 6.4.7.1 7.40 100.00" in view[0]
    assert "001248 10 100.00" in view[0]
    assert evidence["chapter_edits"] and evidence["original_text_sha256"] != evidence["parser_text_sha256"]
