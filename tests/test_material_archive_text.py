"""全基金通用文本政策：精确数字、外链缺口和普通业务入口的实际读库。"""

import pytest
from app.models.material_content import MaterialDocument, MaterialPageRepresentation
from app.services import material_content as store
from app.services import material_important_text as important
from app.services.material_archive_text import capture
from app.services.material_document import read_document
from sqlalchemy import select
from tests.test_material_source_archive import setup  # noqa: F401


def test_json_full_lexemes_publish_to_common_policy_and_read_without_original(setup, monkeypatch):  # noqa: F811
    path, disk, repo = setup
    MaterialDocument.__table__.create(repo.engine)
    raw = b'{"amount":-123456789012345.000010,"conditions":["all","rows"],"note":"not a summary"}'
    store.write_source(path, raw)
    target = next((disk / "originals").rglob("*.json"))
    result = capture(target, repo=repo)
    assert result["status"] == "IMPORTANT_TEXT_VERIFIED"
    monkeypatch.setattr(type(target), "read_bytes", lambda self: pytest.fail("opened original"))
    with store.database_only():
        document = read_document(store.source_bytes(path).source_hash)
        assert document["schema"] == important.COMMON_POLICY
        assert document["content"]["text"] == raw.decode()


def test_html_visuals_remain_pending_instead_of_being_marked_complete(setup):  # noqa: F811
    path, disk, repo = setup
    MaterialDocument.__table__.create(repo.engine)
    path = path.with_suffix(".html")
    raw = b'<h1>Fund</h1><table><tr><td>-1.2300%</td></tr></table><img src="chart.png" alt="important">'
    store.write_source(path, raw)
    result = capture(next((disk / "originals").rglob("*.html")), repo=repo)
    assert result["status"] == "NATIVE_TEXT_SAVED_REVIEW_PENDING"
    assert result["tables"] == 1 and result["pending"]
    assert important.release(store.checksum(raw), repo) is None
    with repo.engine.connect() as c:
        assert not c.execute(select(MaterialPageRepresentation).where(
            MaterialPageRepresentation.engine == "GENERAL_ASSET_V1")).first()


def test_legacy_word_gets_business_table_representation_without_replacing_base(setup, monkeypatch):  # noqa: F811
    from tests.test_material_document import word_bytes

    path, disk, repo = setup
    MaterialDocument.__table__.create(repo.engine)
    raw = word_bytes()
    original = store.extract(raw, ".docx")
    old_data = {"text": original.extracted["text"], "paragraphs": original.extracted["paragraphs"]}
    legacy = store.SourceContent(original.source_hash, len(raw), "DOCX", old_data,
                                 store.checksum(store.encoded(old_data)))
    target = disk / "originals" / original.source_hash[:2] / (original.source_hash + ".docx")
    target.parent.mkdir(parents=True)
    target.write_bytes(raw)
    repo.put(target, legacy)
    capture(target, repo=repo)
    monkeypatch.setattr(type(target), "read_bytes", lambda self: pytest.fail("business reopened Word original"))
    with store.using_repository(repo), store.database_only():
        actual = repo.get(target, fresh=True)
        assert actual.content_hash == legacy.content_hash
        structure = store.docx_structure(actual)
        assert structure["tables"] == [[["金额（人民币元）", "123.45"]]]
        assert structure == original.extracted


def test_large_html_splits_losslessly_and_reads_all_values_from_database(setup, monkeypatch):  # noqa: F811
    from pathlib import Path

    from app.services.material_archive_text import HTML_ENGINE, read_html_text, save_html_text

    path, disk, repo = setup
    raw = b'<p>native source identity</p>'
    value = store.extract(raw, ".html")
    repo.put(path, value)
    payload = {"source_hash": value.source_hash, "plain": "财务正文-1.2300\n"*1_000_000,
               "tables": [[["期末", "-0.0100 CNY"]]], "links": [], "pending": []}
    save_html_text(repo, value.source_hash, payload)
    assert 'UTF8_JSON_TEXT_PARTS' in repo.page(value.source_hash, HTML_ENGINE, 0)
    monkeypatch.setattr(Path, "open", lambda *a, **k: pytest.fail("read an original during database readback"))
    with store.database_only():
        assert read_html_text(value.source_hash, repo=repo) == payload


def test_json_unicode_escaped_visual_payload_stays_only_in_external_original(setup):  # noqa: F811
    import zlib

    from app.models.material_content import MaterialContent

    path, disk, repo = setup
    raw = b'{"cover":"\\u0064ata:image/png;base64,not-a-business-text-payload"}'
    store.write_source(path, raw)
    assert not path.exists()
    assert next((disk/"originals").rglob("*.bin")).read_bytes() == raw
    metadata = repo.metadata(path=path)
    assert metadata.kind == "UNSUPPORTED" and not metadata.complete
    with repo.engine.connect() as c:
        payload = c.scalar(select(MaterialContent.extracted_data).where(
            MaterialContent.source_hash == store.checksum(raw)))
    body = zlib.decompress(payload)
    assert b'data:image/' not in body and b'not-a-business-text-payload' not in body
    with pytest.raises(ValueError, match="TEXT_EXTRACTION_REQUIRED"):
        store.source_bytes(path)
