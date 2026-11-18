from app.integrations import dbfund_supplement


def test_existing_material_page_supports_range_without_native_bounded_api(monkeypatch):
    class Page:
        def get_textpage(self):
            return self

        def get_text_range(self):
            return "证券代码300308，实际保存的完整正文内容。"

        def close(self):
            pass

    class Document:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def __len__(self):
            return 1

        def __iter__(self):
            return iter([Page()])

    monkeypatch.setattr(dbfund_supplement, "document_pdfium", lambda raw: Document())
    pages, status = dbfund_supplement.article_text(b"%PDF-test-original", True)
    assert status == "TEXT_EXTRACTED"
    assert pages == ["证券代码300308，实际保存的完整正文内容。"]
