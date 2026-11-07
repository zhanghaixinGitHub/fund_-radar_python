"""图像xref目录异常不能漏OCR页，也不能把OCR候选冒充已核验的数值。"""

import json

import pytest
from app.services import material_archive_ocr as ocr
from app.services import material_content as store
from tests.test_material_source_archive import setup  # noqa: F401


def test_bad_image_directory_falls_back_to_memory_without_skipping_page(setup, monkeypatch):  # noqa: F811
    pymupdf = pytest.importorskip("pymupdf")
    path, disk, repo = setup
    document = pymupdf.open()
    document.new_page(width=100, height=120)
    raw = document.tobytes()
    document.close()
    path = path.with_suffix(".pdf")
    store.write_source(path, raw)
    target = next((disk/"originals").rglob("*.pdf"))

    def broken(*args, **kwargs):
        raise ValueError("bad page number(s)")

    monkeypatch.setattr(pymupdf.Page, "get_image_info", broken)

    def recognizer(pixels, **kwargs):
        assert pixels.shape[:2] == (240, 200) and kwargs["text_score"] == 0.0
        return [([[1, 1], [30, 1], [30, 8], [1, 8]], "-1,234.5000 CNY", 0.1)], []

    result = ocr.capture_pdf(target, recognizer=recognizer, repo=repo)
    assert result["OCR_pages_saved"] == 1
    saved = json.loads(repo.page(store.checksum(raw), ocr.ENGINE, 0))
    assert saved["text"] == "-1,234.5000 CNY" and saved["can_publish_without_review"] is False
    assert saved["in_memory_render_fallback"] == "ValueError:bad page number(s)"
    assert saved["binary_payload_saved"] is False
