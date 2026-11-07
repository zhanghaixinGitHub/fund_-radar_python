"""纯PDF文字缓存不得跳过每次原件/日期核验，也不缓存原件对象或评分。"""

import hashlib
import io
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path

import pytest
from app.services import fund_rating_huaan_qdii_relative as module
from pypdf import PdfReader, PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject


def pdf_bytes(words):
    writer = PdfWriter()
    for word in words:
        page = writer.add_blank_page(width=120, height=120)
        font = DictionaryObject(
            {
                NameObject("/Type"): NameObject("/Font"),
                NameObject("/Subtype"): NameObject("/Type1"),
                NameObject("/BaseFont"): NameObject("/Helvetica"),
            }
        )
        page[NameObject("/Resources")] = DictionaryObject(
            {NameObject("/Font"): DictionaryObject({NameObject("/F1"): writer._add_object(font)})}
        )
        stream = DecodedStreamObject()
        stream.set_data(b"BT /F1 12 Tf 10 50 Td (" + word.encode("ascii") + b") Tj ET")
        page[NameObject("/Contents")] = writer._add_object(stream)
    output = io.BytesIO()
    writer.write(output)
    return output.getvalue()


@pytest.fixture(autouse=True)
def empty_cache():
    with module._PDF_TEXT_LOCK:
        module._PDF_TEXT_CACHE.clear()
        module._PDF_TEXT_BYTES = 0
    yield
    with module._PDF_TEXT_LOCK:
        module._PDF_TEXT_CACHE.clear()
        module._PDF_TEXT_BYTES = 0


@pytest.fixture
def original(tmp_path):
    raw = pdf_bytes(["alpha", "beta"])
    digest = hashlib.sha256(raw).hexdigest()
    path = tmp_path / "sources" / (digest + ".pdf")
    path.parent.mkdir()
    path.write_bytes(raw)
    inputs = {
        "as_of_date": "2026-09-30",
        "research_on": "2026-10-29T13:00:00+00:00",
        "source_registry": {
            digest: {
                "file": str(path.relative_to(tmp_path)),
                "sha256": digest,
                "received_at": "2026-10-29T12:00:00+00:00",
                "available_on": "2026-09-29",
                "scope": "FUNDAMENTAL",
                "format": "PDF",
                "kind": "ORIGINAL",
            }
        },
    }
    return tmp_path, path, inputs, digest


def test_real_pdf_output_and_explicit_page_ranges(original):
    root, _, inputs, digest = original
    source = module.Originals(inputs, root)
    assert source.text(digest, head_pages=1) == "alpha"
    assert source.text(digest) == "alphabeta"
    assert len(module._PDF_TEXT_CACHE) == 2


def test_eight_requests_recheck_bytes_but_extract_once(original, monkeypatch):
    root, path, inputs, digest = original
    reads, readers = [], []
    actual_read = Path.read_bytes

    def read(current):
        if current == path:
            reads.append(True)
        return actual_read(current)

    def reader(stream):
        readers.append(True)
        return PdfReader(stream)

    monkeypatch.setattr(Path, "read_bytes", read)
    monkeypatch.setattr(module, "PdfReader", reader)
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda _: module.Originals(deepcopy(inputs), root).text(digest), range(8)))
    assert results == ["alphabeta"] * 8
    assert len(reads) == 8 and len(readers) == 1
    assert all(isinstance(value[0], str) and isinstance(value[1], int) for value in module._PDF_TEXT_CACHE.values())


def test_changed_bytes_rejected_after_cache_hit(original):
    root, path, inputs, digest = original
    module.Originals(inputs, root).text(digest)
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="QDII_ORIGINAL_HASH"):
        module.Originals(inputs, root)


@pytest.mark.parametrize(
    "field,value,error",
    [
        ("received_at", "2026-10-30T00:00:00+00:00", "SOURCE_NOT_AVAILABLE"),
        ("available_on", "2026-10-01", "FUTURE_FUNDAMENTAL"),
    ],
)
def test_source_dates_rechecked_after_cache_hit(original, field, value, error):
    root, _, inputs, digest = original
    module.Originals(inputs, root).text(digest)
    changed = deepcopy(inputs)
    changed["source_registry"][digest][field] = value
    with pytest.raises(ValueError, match=error):
        module.Originals(changed, root)


def test_root_scope_rechecked_after_cache_hit(original):
    root, path, inputs, digest = original
    module.Originals(inputs, root).text(digest)
    outside = root / "outside.pdf"
    outside.write_bytes(path.read_bytes())
    changed = deepcopy(inputs)
    changed["source_registry"][digest]["file"] = "outside.pdf"
    with pytest.raises(ValueError, match="ORIGINAL_ROOT_SCOPE"):
        module.Originals(changed, root)


def test_cache_is_bounded_and_large_text_is_not_retained(original, monkeypatch):
    root, _, inputs, digest = original
    monkeypatch.setattr(module, "_PDF_TEXT_MAX_ENTRIES", 1)
    module.Originals(inputs, root).text(digest, head_pages=1)
    module.Originals(inputs, root).text(digest)
    assert len(module._PDF_TEXT_CACHE) == 1
    assert module._PDF_TEXT_BYTES <= module._PDF_TEXT_LIMIT_BYTES
    module._PDF_TEXT_CACHE.clear()
    module._PDF_TEXT_BYTES = 0
    monkeypatch.setattr(module, "_PDF_TEXT_LIMIT_BYTES", 1)
    assert module.Originals(inputs, root).text(digest) == "alphabeta"
    assert not module._PDF_TEXT_CACHE and module._PDF_TEXT_BYTES == 0


def test_direct_cache_helper_rejects_wrong_digest(original):
    _, path, _, digest = original
    module._cached_pdf_text(digest, path.read_bytes(), None)
    with pytest.raises(ValueError, match="QDII_ORIGINAL_HASH"):
        module._cached_pdf_text(digest, path.read_bytes() + b"wrong", None)
