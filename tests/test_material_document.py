"""通用内容与独立验收：改用新问题仍可读库，业务成败不能替代内容验收。"""

import io
import zipfile
import zlib

import pytest
from app.models.material_content import MaterialContent, MaterialDocument, MaterialLocation, MaterialPageRepresentation
from app.services import material_content as content
from app.services.material_document import finish_source, get_review, read_document, review_source
from sqlalchemy import create_engine, update
from tests.test_material_content import pdf_bytes


@pytest.fixture
def setup(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from app.services import material_source_archive as archive

    disk = tmp_path.parent / (tmp_path.name + "-external")
    disk.mkdir()
    monkeypatch.setattr(archive, "get_settings", lambda: SimpleNamespace(fund_source_archive_root=str(disk)))
    monkeypatch.setattr(content, "PROJECT", tmp_path)
    engine = create_engine(f"sqlite:///{tmp_path / 'materials.db'}")
    for table in (MaterialContent, MaterialLocation, MaterialPageRepresentation, MaterialDocument):
        table.__table__.create(engine)
    repo = content.MaterialRepository(engine)
    (tmp_path / "data").mkdir()
    (tmp_path / "data/material-original-cleanup-hold.json").write_text('{"hold":false}', encoding="utf-8")
    return tmp_path, repo


def source(setup, name, raw):
    root, repo = setup
    path = root / "data/sources" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    repo.ingest(path)
    return path


def test_plain_pdf_full_content_survives_original_cleanup(setup):
    path = source(
        setup, "notice.pdf", pdf_bytes("Effective 2026-07-13; ETF excluded; resume only after active trading")
    )
    root, repo = setup
    with content.using_repository(repo):
        review = review_source(path)
        assert review["ready"] and review["content"]["page_count"] == 1
        result = finish_source(path)
        assert result["deleted_bytes"] > 0 and not path.exists()
        with content.database_only():
            doc = read_document(review["source_hash"])
            page = doc["content"]["pages"][0]
            assert "ETF excluded" in page["text"]
            assert "active trading" in page["layout_text"]
            assert page["text_runs"]
            assert content.document_reader(content.source_bytes(path)).pages[0].extract_text()


def test_old_batch_hold_is_independent_of_quality_pass(setup):
    path = source(setup, "old.json", b'{"meaning":"keep original until user review"}')
    root, repo = setup
    with repo.engine.begin() as c:
        c.execute(update(MaterialLocation).values(retention_hold=True))
    with content.using_repository(repo):
        result = finish_source(path)
        assert result["ready"] and result["retention_hold"] and path.exists()
        doc = get_review(result["source_hash"])
        repo.certify([path], doc["document_hash"])
        with pytest.raises(ValueError, match="EXISTING_BATCH_HOLD"):
            content.delete_verified(path, root=root, verification=doc["document_hash"], repo=repo)


def test_business_error_does_not_block_independent_valid_content_cleanup(setup):
    root, repo = setup
    from app.services.fund_exposure_common import _publish

    path = root / "data/sources/new.json"
    path.parent.mkdir(parents=True)

    @content.material_operation
    def failed_prediction():
        _publish(path, b'{"amount":123.12345678901234567890,"condition":"ETF excluded"}')
        return {"status": "FAILED", "errors": ["calendar evidence missing"]}

    with content.using_repository(repo):
        assert failed_prediction()["status"] == "FAILED"
        assert not path.exists()
        stored = content.source_bytes(path)
        assert "123.12345678901234567890" in stored.decode()


def test_visual_pdf_retains_native_text_and_gap_without_storing_image(setup):
    path = source(setup, "scanned.pdf", pdf_bytes("Heading only", image=True))
    root, repo = setup
    with content.using_repository(repo):
        result = finish_source(path)
        assert not result["ready"] and path.exists()
        doc = read_document(result["source_hash"], include_assets=True)
        assert "Heading only" in doc["content"]["pages"][0]["text"]
        assert "visual_content" not in doc["content"]["pages"][0]
        with repo.engine.connect() as connection:
            from sqlalchemy import select

            assert not connection.execute(select(MaterialPageRepresentation).where(
                MaterialPageRepresentation.engine == "GENERAL_ASSET_V1")).first()
        assert any(i["code"] == "PDF_VISUAL_CONTENT_REVIEW_REQUIRED" for i in result["issues"])


def word_bytes():
    buffer = io.BytesIO()
    start = '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body>'
    body = '<w:p><w:r><w:t>正文：估值调整</w:t><w:footnoteReference w:id="1"/></w:r></w:p>'
    body += (
        "<w:tbl><w:tr><w:tc><w:p><w:r><w:t>金额（人民币元）</w:t></w:r></w:p></w:tc>"
        "<w:tc><w:p><w:r><w:t>123.45</w:t></w:r></w:p></w:tc></w:tr></w:tbl>"
    )
    ns = 'xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"'
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("word/document.xml", start + body + "</w:body></w:document>")
        archive.writestr(
            "word/footnotes.xml",
            f'<w:footnotes {ns}><w:footnote w:id="1">'
            "<w:p><w:r><w:t>脚注：ETF除外</w:t></w:r></w:p></w:footnote></w:footnotes>",
        )
        archive.writestr(
            "word/endnotes.xml",
            f'<w:endnotes {ns}><w:endnote w:id="2">'
            "<w:p><w:r><w:t>尾注：复牌且交易活跃</w:t></w:r></w:p></w:endnote></w:endnotes>",
        )
        archive.writestr("word/header1.xml", f"<w:hdr {ns}><w:p><w:r><w:t>适用基金范围</w:t></w:r></w:p></w:hdr>")
    return buffer.getvalue()


def test_word_tables_and_notes_survive_removal_and_new_questions(setup):
    path = source(setup, "notice.docx", word_bytes())
    root, repo = setup
    with content.using_repository(repo):
        result = finish_source(path)
        assert result["ready"] and not path.exists()
        doc = read_document(result["source_hash"])
        parts = {p["name"]: p for p in doc["content"]["parts"]}
        assert parts["word/footnotes.xml"]["paragraphs"][0]["text"] == "脚注：ETF除外"
        assert "复牌且交易活跃" in parts["word/endnotes.xml"]["paragraphs"][0]["text"]
        assert parts["word/document.xml"]["tables"][0][0][1]["paragraphs"][0]["text"] == "123.45"


def test_review_is_independent_of_scores_and_rejects_corrupt_parts(setup, monkeypatch):
    import app.services.fund_rating as rating

    monkeypatch.setattr(
        rating, "read", lambda *a, **kw: (_ for _ in ()).throw(AssertionError("rating must not be consulted"))
    )
    path = source(setup, "notice.pdf", pdf_bytes())
    root, repo = setup
    with content.using_repository(repo):
        doc = review_source(path)
        assert doc["ready"]
        with repo.engine.begin() as c:
            c.execute(
                update(MaterialPageRepresentation)
                .where(MaterialPageRepresentation.engine == "GENERAL_PDF_PAGE_V1")
                .values(extracted_text=zlib.compress(b"corrupt"))
            )
        with pytest.raises(ValueError, match="CONTENT_MISMATCH|PART_MISSING|CONTENT_HASH"):
            finish_source(path)
    assert path.exists()


def test_old_business_proof_never_authorizes_production_cleanup(setup):
    path = source(setup, "notice.json", b'{"amount":123}')
    root, repo = setup
    repo.certify([path], "a" * 64)
    with content.using_repository(repo), pytest.raises(ValueError, match="CONTENT_REVIEW_REQUIRED"):
        content.delete_verified(path, root=root, verification="a" * 64, repo=repo)
    assert path.exists()


def test_changed_source_is_not_reviewed_as_old_version(setup):
    path = source(setup, "notice.json", b'{"amount":123}')
    root, repo = setup
    path.write_bytes(b'{"amount":999}')
    with pytest.raises(ValueError, match="SOURCE_CHANGED"):
        review_source(path, repo=repo)
    assert path.exists()


def test_source_publish_without_rating_operation_still_runs_content_lifecycle(setup):
    from app.services.fund_exposure_common import _publish

    root, repo = setup
    path = root / "data/sources/independent.json"
    path.parent.mkdir(parents=True)
    raw = '{"未来新问题":"ETF是否除外","原文":"ETF除外，复牌且交易活跃后恢复估值"}'.encode()
    with content.using_repository(repo):
        _publish(path, raw)
        assert not path.exists()
        value = content.source_bytes(path)
        assert content.source_text(path) == raw.decode("utf-8")
        from app.services.material_source_archive import archive_route

        assert archive_route(value.source_hash, repo)


def test_missing_word_footnote_prevents_deletion(setup):
    buffer = io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(word_bytes())) as old, zipfile.ZipFile(buffer, "w") as new:
        for name in old.namelist():
            if name != "word/footnotes.xml":
                new.writestr(name, old.read(name))
    path = source(setup, "missing-note.docx", buffer.getvalue())
    root, repo = setup
    with content.using_repository(repo):
        result = finish_source(path)
        assert not result["ready"] and path.exists()
        assert any(i["code"] == "WORD_NOTE_REFERENCE_NOT_CLOSED" for i in result["issues"])


def test_archive_members_are_independently_reviewed_and_read_after_cleanup(setup):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("notice.pdf", pdf_bytes("ETF excluded; source includes all conditions"))
        archive.writestr("facts.json", b'{"amount":123.45678901234567890}')
    path = source(setup, "bundle.zip", buffer.getvalue())
    root, repo = setup
    with content.using_repository(repo):
        result = finish_source(path)
        assert result["ready"] and not path.exists()
        document = read_document(result["source_hash"])
        assert "ETF excluded" in document["content"]["members"]["notice.pdf"]["content"]["pages"][0]["text"]
        assert "123.45678901234567890" in document["content"]["members"]["facts.json"]["content"]["text"]


def test_cleanup_database_failure_preserves_business_result(setup, monkeypatch):
    root, repo = setup
    from app.services.fund_exposure_common import _publish

    path = root / "data/sources/retained.json"
    path.parent.mkdir(parents=True)

    @content.material_operation
    def business():
        _publish(path, b'{"amount":123}')
        monkeypatch.setattr(
            repo.engine, "connect", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("database lost"))
        )
        return {"status": "SUCCEEDED"}

    with content.using_repository(repo):
        assert business() == {"status": "SUCCEEDED"}
    assert not path.exists()


def _add_policy_document(repo, value, document, policy):
    """构造追加政策底稿，不覆盖通用底稿或源内容。"""
    data = content.encoded(document)
    with repo.engine.begin() as connection:
        connection.execute(
            MaterialDocument.__table__.insert().values(
                source_hash=value.source_hash,
                policy=policy,
                base_content_hash=value.content_hash,
                document_hash=content.checksum(data),
                document_data=zlib.compress(data),
                ready=document["ready"],
            )
        )


def test_explicit_content_policy_is_not_silently_replaced_by_default(setup):
    path = source(setup, "history.json", b'{"all_fields":{"future_use":12.1234567890123456789}}')
    _, repo = setup
    policy = "HISTORICAL_FULL_SOURCE_V1"
    with content.using_repository(repo):
        initial = review_source(path)
        value = repo.get(path)
        document = {k: v for k, v in initial.items() if k != "document_hash"}
        document["schema"] = policy
        _add_policy_document(repo, value, document, policy)
        assert read_document(value.source_hash)["schema"] == "GENERAL_CONTENT_V1"
        assert read_document(value.source_hash, policy=policy)["schema"] == policy
        with pytest.raises(ValueError, match="NOT_REVIEWED"):
            read_document(value.source_hash, policy="UNKNOWN_POLICY")
        assert path.exists()


def test_historical_pdf_layout_and_visual_assets_read_without_original(setup):
    path = source(setup, "historical.pdf", pdf_bytes("All conditions and footnotes retained"))
    _, repo = setup
    policy = "HISTORICAL_FULL_SOURCE_V1"
    with content.using_repository(repo):
        value = repo.get(path)
        visual = {"text": '<svg xmlns="http://www.w3.org/2000/svg"><text>Full page</text></svg>'}
        visual_text = content.encoded(visual).decode()
        repo.page(value.source_hash, "HISTORY_SVG_V1", 0, visual_text)
        visual_ref = {"engine": "HISTORY_SVG_V1", "index": 0, "hash": content.checksum(visual_text.encode())}
        page = {
            "page": 1,
            "native_page_index": 0,
            "native_text": value.extracted["pdfium"][0],
            "layout": {"blocks": [{"text": "All conditions and footnotes retained"}]},
            "visual": visual_ref,
        }
        page_text = content.encoded(page).decode()
        repo.page(value.source_hash, "HISTORY_PDF_PAGE_V1", 0, page_text)
        document = {
            "schema": policy,
            "source_hash": value.source_hash,
            "base_content_hash": value.content_hash,
            "format": "PDF",
            "ready": True,
            "content": {"pages": [{"engine": "HISTORY_PDF_PAGE_V1", "index": 0,
                                    "hash": content.checksum(page_text.encode())}]},
        }
        _add_policy_document(repo, value, document, policy)
        # 只让本隔离测试的原件不可用，验证显式政策读回不依赖原PDF。
        path.rename(path.with_suffix(".temporarily-unavailable"))
        with content.database_only():
            page = read_document(value.source_hash, policy=policy)["content"]["pages"][0]
            assert "conditions" in page["text"] and page["layout"]["blocks"]
            assert page["visual"] == {"asset_ref": visual_ref}
            expanded = read_document(value.source_hash, policy=policy, include_assets=True)
            assert "Full page" in expanded["content"]["pages"][0]["visual"]["text"]
        with repo.engine.begin() as connection:
            connection.execute(update(MaterialPageRepresentation)
                               .where(MaterialPageRepresentation.engine == "HISTORY_SVG_V1")
                               .values(extracted_text=zlib.compress(b'{}')))
        with pytest.raises(ValueError, match="CONTENT_HASH|CONTENT_MISMATCH|PART_MISSING"):
            read_document(value.source_hash, policy=policy)


@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("catalog", [False, True])
def test_shared_glyph_document_reads_without_pdf_and_detects_asset_corruption(setup, nested, catalog):
    from app.services.material_svg import ASSET_FORMAT, glyph_payload, pack_svg
    from tests.test_material_svg import SOURCE

    path = source(setup, "shared.pdf", pdf_bytes("Complete source conditions"))
    root, repo = setup
    policy = "HISTORICAL_FULL_SOURCE_V3" if catalog else "HISTORICAL_FULL_SOURCE_V2"
    with content.using_repository(repo):
        value = repo.get(path)
        identities = []
        glyph_catalog = []

        def store(node):
            reference, payload = glyph_payload(node)
            asset = content.SourceContent(reference["source_hash"], len(payload), ASSET_FORMAT,
                                          {"glyph_node": node}, reference["content_hash"])
            repo.put(root / "data/derived-glyphs" / reference["source_hash"], asset)
            identities.append(reference["source_hash"])
            glyph_catalog.append(reference)
            return {"glyph_id": len(glyph_catalog) - 1} if catalog else reference

        packed = pack_svg(SOURCE, store)
        body = packed.pop("svg")
        payload = content.encoded({"text": body}).decode()
        repo.page(value.source_hash, "HISTORY_SVG_COMPACT_V2", 0, payload)
        packed.update(parts=[{"engine": "HISTORY_SVG_COMPACT_V2", "index": 0,
                              "hash": content.checksum(payload.encode())}],
                      characters=len(body), sha256=content.checksum(body.encode()))
        page = {"page": 1, "native_page_index": 0, "native_text": value.extracted["pdfium"][0], "visual": packed}
        page_text = content.encoded(page).decode()
        repo.page(value.source_hash, "HISTORY_PDF_PAGE_V2", 0, page_text)
        if catalog:
            # 新压缩封装和原始zlib片段可共存，内容摘要与库内还原规则保持一致。
            with repo.engine.begin() as connection:
                connection.execute(update(MaterialPageRepresentation)
                                   .where(MaterialPageRepresentation.engine == "HISTORY_PDF_PAGE_V2")
                                   .values(extracted_text=content.compress_page(page_text.encode())))
        document = {"schema": policy, "source_hash": value.source_hash, "base_content_hash": value.content_hash,
                    "format": "PDF", "ready": True,
                    "content": {"pages": [{"engine": "HISTORY_PDF_PAGE_V2", "index": 0,
                                            "hash": content.checksum(page_text.encode())}]}}
        if catalog:
            document["shared_glyph_catalog"] = glyph_catalog
        if nested:
            document["format"] = "OPENXML_OR_ARCHIVE"
            document["content"] = {"members": [{"name": "embedded.pdf", "parsed_content": document["content"]}]}
        _add_policy_document(repo, value, document, policy)
        path.rename(path.with_suffix(".temporarily-unavailable"))
        def pages(result):
            body = result["content"]["members"][0]["parsed_content"] if nested else result["content"]
            return body["pages"]
        with content.database_only():
            visible = read_document(value.source_hash, policy=policy)
            assert "svg" not in pages(visible)[0]["visual"]
            restored = read_document(value.source_hash, policy=policy, include_assets=True)
            assert "chart" in pages(restored)[0]["visual"]["svg"]
        with repo.engine.begin() as connection:
            connection.execute(update(MaterialContent).where(MaterialContent.source_hash == identities[0])
                               .values(extracted_data=zlib.compress(b'{"glyph_node":{}}')))
        with pytest.raises(ValueError, match="SHARED_ASSET_CONTENT_MISMATCH"):
            read_document(value.source_hash, policy=policy)
