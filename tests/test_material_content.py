"""真实提取、事务读回、无原件复算和清理门槛的隔离回归。"""

import hashlib
import json
import lzma
import zlib

import pytest
from app.models.material_content import MaterialContent, MaterialLocation, MaterialPageRepresentation
from app.services.material_content import (
    MaterialRepository,
    content_digest,
    delete_verified,
    document_pdfium,
    document_reader,
    extract,
    source_bytes,
    source_exists,
    using_repository,
)
from sqlalchemy import create_engine, update


def test_page_codecs_preserve_full_text_and_reject_truncated_or_appended_streams():
    from app.services.material_content import compress_page, decompress_page

    raw = ('完整正文：金额123.456789012345、单位、条件、脚注\n' * 2000).encode()
    assert decompress_page(zlib.compress(raw)) == raw
    compressed = compress_page(raw)
    assert decompress_page(compressed) == raw
    for broken in [compressed[:-4], compressed + b'not another page']:
        with pytest.raises((ValueError, lzma.LZMAError)):
            decompress_page(broken)


def pdf_bytes(text="Numeric fact 123.45 percent 8.9", *, image=False):
    """实际 PDF 字节，包含 xref；测试进入真实安装的两个解析器。"""
    stream = f"BT /F1 12 Tf 10 80 Td ({text}) Tj ET".encode()
    if image:
        stream = b"q 300 0 0 80 0 0 cm /Im1 Do Q " + stream
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 500 100] "
        b"/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream",
    ]
    if image:
        objects[2] = objects[2].replace(b"/Font << /F1 4 0 R >>", b"/Font << /F1 4 0 R >> /XObject << /Im1 6 0 R >>")
        objects.append(
            b"<< /Type /XObject /Subtype /Image /Width 1 /Height 1 /ColorSpace /DeviceRGB "
            b"/BitsPerComponent 8 /Length 3 >>\nstream\n\x00\x00\x00\nendstream"
        )
    content, offsets = b"%PDF-1.4\n", []
    for i, obj in enumerate(objects, 1):
        offsets.append(len(content))
        content += f"{i} 0 obj\n".encode() + obj + b"\nendobj\n"
    start = len(content)
    content += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    content += b"".join(f"{p:010d} 00000 n \n".encode() for p in offsets)
    return content + f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{start}\n%%EOF".encode()


@pytest.fixture
def repo(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from app.services import material_source_archive as archive

    disk = tmp_path.parent / (tmp_path.name + "-external")
    disk.mkdir()
    monkeypatch.setattr(archive, "get_settings", lambda: SimpleNamespace(fund_source_archive_root=str(disk)))
    engine = create_engine(f"sqlite:///{tmp_path / 'material.db'}")
    MaterialContent.__table__.create(engine)
    MaterialLocation.__table__.create(engine)
    MaterialPageRepresentation.__table__.create(engine)
    return MaterialRepository(engine)


def test_indexed_run_file_refreshes_database_and_cached_read_after_save(tmp_path, repo, monkeypatch):
    """迁入库的可变续办文件保存后必须读到新版本，旧正文仍可按摘要查回。"""
    from app.services import material_content as module
    from app.services.fund_exposure_common import read, save

    path = tmp_path / "run.json"
    monkeypatch.setattr(module, "persistent_location", lambda path: False)
    with using_repository(repo):
        save(path, {"stage": "BEFORE_SUPPLEMENT"})
        previous = repo.ingest(path)

        @module.material_read_cache
        def continue_run():
            assert read(path) == {"stage": "BEFORE_SUPPLEMENT"}
            save(path, {"stage": "ACTUAL_NEW_ORIGINAL"}, replace=True)
            assert read(path) == {"stage": "ACTUAL_NEW_ORIGINAL"}

        continue_run()
        assert b"BEFORE_SUPPLEMENT" in repo.get(source_hash=previous.source_hash)
        assert b"ACTUAL_NEW_ORIGINAL" in source_bytes(path)


def test_unindexed_temporary_run_file_is_not_automatically_imported(tmp_path, repo, monkeypatch):
    """不扩大临时文件入库范围，只修复原来已有索引的续办衔接。"""
    from app.services import material_content as module
    from app.services.fund_exposure_common import save

    path = tmp_path / "temporary.json"
    monkeypatch.setattr(module, "persistent_location", lambda path: False)
    with using_repository(repo):
        save(path, {"temporary": True})
        assert repo.metadata(path=path) is None


def test_pdf_database_pages_match_both_original_parsers_and_survive_removal(tmp_path, repo):
    raw = pdf_bytes()
    path = tmp_path / "original.pdf"
    path.write_bytes(raw)
    from app.services.fund_rating_pdf import text_pages

    before_native = text_pages(path)
    before_pdf = [p.extract_text() for p in document_reader(raw).pages]
    stored = repo.ingest(path)
    assert stored.complete and stored.kind == "PDF"
    assert bytes(stored) == b""  # 数据库绝不保存或重新生成原 PDF。
    assert len(stored) == len(raw)
    with using_repository(repo):
        # 先验证当前计算实际使用的页面，并把它需要的排版留在库内，随后才能删除。
        assert [p.extract_text() for p in document_reader(source_bytes(path)).pages] == before_pdf
    verification = hashlib.sha256(b"verified parsers and consumer").hexdigest()
    repo.certify([path], verification)
    delete_verified(path, root=tmp_path, verification=verification, repo=repo)
    assert not path.exists()
    with using_repository(repo):
        assert source_exists(path)
        assert text_pages(path) == before_native
        assert [p.extract_text() for p in document_reader(source_bytes(path)).pages] == before_pdf
        assert content_digest(source_bytes(path)) == hashlib.sha256(raw).hexdigest()
        with document_pdfium(path) as doc:
            assert "123.45" in doc[0].get_textpage().get_text_range()


def test_exact_numeric_tokens_and_source_versions_are_preserved(tmp_path, repo):
    path = tmp_path / "statement.json"
    raw = b'{ "amount": 123.12345678901234567890, "note": "space in string" }'
    path.write_bytes(raw)
    first = repo.ingest(path)
    from decimal import Decimal

    assert json.loads(first, parse_float=Decimal)["amount"] == Decimal("123.12345678901234567890")
    path.write_text('{"amount":2}', encoding="utf-8")
    second = repo.ingest(path)
    assert second.source_hash != first.source_hash
    assert repo.get(source_hash=first.source_hash).content_hash == first.content_hash


def test_deletion_requires_business_verification_and_unchanged_source(tmp_path, repo):
    path = tmp_path / "notice.json"
    path.write_text('{"value":1}', encoding="utf-8")
    repo.ingest(path)
    verification = "a" * 64
    with pytest.raises(ValueError, match="NOT_BUSINESS_VERIFIED"):
        delete_verified(path, root=tmp_path, verification=verification, repo=repo)
    repo.certify([path], verification)
    with pytest.raises(ValueError, match="SCOPE"):
        delete_verified(path, root=tmp_path / "elsewhere", verification=verification, repo=repo)
    path.write_text('{"value":2}', encoding="utf-8")
    with pytest.raises(ValueError, match="SOURCE_CHANGED"):
        delete_verified(path, root=tmp_path, verification=verification, repo=repo)
    assert path.exists()


def test_database_corruption_is_detected_before_use_and_deletion(tmp_path, repo):
    path = tmp_path / "notice.json"
    path.write_text('{"value":1}', encoding="utf-8")
    record = repo.ingest(path)
    with repo.engine.begin() as connection:
        connection.execute(
            update(MaterialContent)
            .where(MaterialContent.source_hash == record.source_hash)
            .values(content_hash="0" * 64)
        )
    with pytest.raises(ValueError, match="CONTENT_MISMATCH"):
        repo.get(path)
    assert path.exists()


def test_empty_pdf_is_not_eligible_for_deletion(tmp_path, repo):
    path = tmp_path / "scan.pdf"
    path.write_bytes(pdf_bytes(""))
    record = repo.ingest(path)
    assert not record.complete
    verification = "b" * 64
    repo.certify([path], verification)
    with pytest.raises(ValueError, match="INCOMPLETE"):
        delete_verified(path, root=tmp_path, verification=verification, repo=repo)
    assert path.exists()


def test_distinct_pdfs_never_share_cached_page_values():
    first, second = extract(pdf_bytes("first 10"), ".pdf"), extract(pdf_bytes("second 20"), ".pdf")
    from app.services.fund_rating_public_market import _pdf_pages

    assert first != second and len({first, second}) == 2
    assert "first 10" in "".join(_pdf_pages(first))
    assert "second 20" in "".join(_pdf_pages(second))


def test_new_material_is_persisted_and_original_kept_on_external_disk(tmp_path, repo, monkeypatch):
    import app.services.material_content as materials
    from app.services.fund_exposure_common import _publish

    monkeypatch.setattr(materials, "PROJECT", tmp_path)
    path = tmp_path / "data" / "sources" / "new.pdf"
    path.parent.mkdir(parents=True)

    @materials.material_operation
    def successful():
        _publish(path, pdf_bytes("New fact 123.45"))
        assert repo.get(path) is not None  # 发布文件后信息已在独立事务中可读。
        body = document_reader(source_bytes(path)).pages[0].extract_text()
        assert "123.45" in body
        return {"status": "SUCCEEDED"}

    with using_repository(repo):
        successful()
        assert not path.exists()
        with materials.database_only():
            assert "123.45" in document_reader(source_bytes(path)).pages[0].extract_text()

    failed_path = tmp_path / "data" / "sources" / "failed.pdf"

    @materials.material_operation
    def failed():
        _publish(failed_path, pdf_bytes("Retain 99"))
        source_bytes(failed_path)
        return {"status": "FAILED", "errors": ["required field missing"]}

    with using_repository(repo):
        failed()
    assert not failed_path.exists()
    from app.services.material_source_archive import archive_route

    assert archive_route(repo.get(failed_path).source_hash, repo)


def test_republishing_current_snapshot_switches_database_version(tmp_path, repo, monkeypatch):
    import app.services.material_content as materials
    from app.services.fund_exposure_common import read, save

    monkeypatch.setattr(materials, "PROJECT", tmp_path)
    path = tmp_path / "data" / "fund-materials" / "000001.json"
    with using_repository(repo):
        save(path, {"amount": 1})
        old = repo.get(path)
        save(path, {"amount": 2}, replace=True)
        assert read(path) == {"amount": 2}
        assert repo.get(source_hash=old.source_hash).source_hash == old.source_hash


def test_deleted_material_is_discoverable_from_database(tmp_path, repo, monkeypatch):
    import app.services.material_content as materials

    monkeypatch.setattr(materials, "PROJECT", tmp_path)
    path = tmp_path / "data" / "sources" / "notice.json"
    path.parent.mkdir(parents=True)
    path.write_text('{"amount":123}', encoding="utf-8")
    repo.ingest(path)
    path.unlink()  # 仅删除本测试创建的临时文件，模拟历史原件不可用。
    with using_repository(repo):
        assert materials.location_exists(path)
        assert list(materials.source_glob(path.parent, "*.json")) == [path]


def test_parse_response_before_archive_retains_needed_page_representation(tmp_path, repo, monkeypatch):
    import app.services.material_content as materials
    from app.services.fund_exposure_common import _publish

    monkeypatch.setattr(materials, "PROJECT", tmp_path)
    path = tmp_path / "data/sources/response.pdf"
    path.parent.mkdir(parents=True)
    raw = pdf_bytes("Response before archive 456.78")

    @materials.material_operation
    def collect():
        # 真实采集器先读HTTP响应再公开原件名称，不能因此漏掉旧计算需要的排版。
        expected = document_reader(raw).pages[0].extract_text()
        _publish(path, raw)
        return {"status": "SUCCEEDED", "body": expected}

    with using_repository(repo):
        result = collect()
        assert not path.exists()
        with materials.database_only():
            assert document_reader(source_bytes(path)).pages[0].extract_text() == result["body"]


def test_archive_retains_member_information_without_original_zip(tmp_path, repo):
    import zipfile
    from io import BytesIO

    import app.services.material_content as materials

    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("report.pdf", pdf_bytes("Archive fact 12.34"))
        archive.writestr("amounts.json", '{"amount":123.45}')
    path = tmp_path / "source.zip"
    path.write_bytes(buffer.getvalue())
    value = repo.ingest(path)
    assert value.kind == "ARCHIVE" and value.complete
    verification = "d" * 64
    with using_repository(repo):
        parent = source_bytes(path)
        member = materials.archive_member(parent, "report.pdf")
        before = document_reader(member).pages[0].extract_text()
        repo.certify([path], verification)
        delete_verified(path, root=tmp_path, verification=verification, repo=repo)
        assert not path.exists()
        with materials.database_only():
            parent = source_bytes(path)
            member = materials.archive_member(parent, "report.pdf")
            assert document_reader(member).pages[0].extract_text() == before
            amounts = materials.archive_member(parent, "amounts.json")
            assert json.loads(amounts)["amount"] == 123.45


def test_unreadable_binary_is_indexed_but_never_deleted(tmp_path, repo):
    raw = b"\x89PNG\r\n\x1a\n\xff\xf0\xf1\xf2"
    path = tmp_path / "scan.png"
    path.write_bytes(raw)
    value = repo.ingest(path)
    assert not value.complete
    assert value.kind == "UNSUPPORTED"
    assert content_digest(value) == hashlib.sha256(raw).hexdigest()
    verification = "e" * 64
    repo.certify([path], verification)
    with pytest.raises(ValueError, match="INCOMPLETE"):
        delete_verified(path, root=tmp_path, verification=verification, repo=repo)
    assert path.exists()


def test_archive_reuses_zero_byte_member_across_suffixes_without_overwriting(tmp_path, repo):
    """空签名成员与已存空文件同源；不可因长度为0而误用新扩展名重建提取。"""
    import zipfile
    from io import BytesIO

    from app.services.material_content import archive_member, opaque_content

    existing = repo.put(tmp_path / "empty.bin", opaque_content(b"", ".bin", "ValueError"))
    assert not existing and existing is not None
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("_xmlsignatures/origin.sigs", b"")
        archive.writestr("amounts.json", '{"amount":-123.45000,"unit":"CNY"}')
    parent = repo.put(tmp_path / "source.xlsx", extract(buffer.getvalue(), ".xlsx"))
    assert parent.kind == "ARCHIVE" and not parent.complete
    assert parent.issue == "ARCHIVE_MEMBER_INCOMPLETE"
    with using_repository(repo):
        zero = archive_member(parent, "_xmlsignatures/origin.sigs")
        assert zero is not None and zero.source_size == 0
        assert zero.content_hash == existing.content_hash and zero.extracted == existing.extracted
        assert archive_member(parent, "amounts.json").decode() == '{"amount":-123.45000,"unit":"CNY"}'


def test_snapshot_reads_database_when_json_cache_is_missing(tmp_path, repo, monkeypatch):
    from types import SimpleNamespace

    from app.services import fund_materials
    from app.services import material_content as materials
    from app.services.fund_exposure_common import save

    monkeypatch.setattr(materials, "PROJECT", tmp_path)
    directory = tmp_path / "data/fund-materials"
    path = directory / "000001.json"
    monkeypatch.setattr(fund_materials, "get_settings", lambda: SimpleNamespace(fund_material_directory=str(directory)))
    payload = {"fundCode": "000001", "reports": [], "documents": []}
    with using_repository(repo):
        save(path, payload)
        path.unlink()
        fund_materials._snapshot.cache_clear()
        assert fund_materials.snapshot("000001") == payload


def test_original_mode_uses_actual_file_instead_of_database(tmp_path, repo):
    import app.services.material_content as materials

    path = tmp_path / "notice.json"
    path.write_text('{"amount":10}', encoding="utf-8")
    repo.ingest(path)
    path.write_text('{"amount":20}', encoding="utf-8")
    with using_repository(repo):
        assert json.loads(source_bytes(path))["amount"] == 10
        with materials.original_materials():
            assert json.loads(source_bytes(path))["amount"] == 20


def test_pdf_with_text_heading_and_large_image_is_not_complete():
    value = extract(pdf_bytes("Income table heading", image=True), ".pdf")
    assert "Income" in value.extracted["pdfium"][0]
    assert not value.complete
    assert value.issue == "LARGE_IMAGE_CONTENT_REQUIRES_REVIEW"


def test_old_pdf_extraction_is_rechecked_for_unread_image_before_deletion(tmp_path, repo):
    import app.services.material_content as materials

    raw = pdf_bytes("Income table heading", image=True)
    path = tmp_path / "old-report.pdf"
    path.write_bytes(raw)
    native = extract(raw, ".pdf")
    old_data = {"pdfium": native.extracted["pdfium"]}
    old = materials.SourceContent(
        native.source_hash, len(raw), "PDF", old_data, materials.checksum(materials.encoded(old_data)), True
    )
    repo.put(path, old)
    verification = "f" * 64
    repo.certify([path], verification)
    with using_repository(repo):
        with pytest.raises(ValueError, match="UNEXTRACTED_IMAGE"):
            delete_verified(path, root=tmp_path, verification=verification, repo=repo)
    assert path.exists()


def test_cleanup_never_removes_new_file_published_at_original_name(tmp_path, repo, monkeypatch):
    path = tmp_path / "notice.json"
    old, new = b'{"amount":1}', b'{"amount":200}'
    path.write_bytes(old)
    repo.ingest(path)
    verification = "1" * 64
    repo.certify([path], verification)
    rename = type(path).rename

    def replace_after_move(self, target):
        result = rename(self, target)
        if self == path:
            path.write_bytes(new)
        return result

    monkeypatch.setattr(type(path), "rename", replace_after_move)
    assert delete_verified(path, root=tmp_path, verification=verification, repo=repo) == len(old)
    assert path.read_bytes() == new
    assert not list(tmp_path.glob(".material-delete-*.tmp"))


def test_interrupted_cleanup_can_resume_from_verified_staging_file(tmp_path, repo):
    path = tmp_path / "notice.json"
    raw = b'{"amount":123}'
    path.write_bytes(raw)
    record = repo.ingest(path)
    verification = "2" * 64
    repo.certify([path], verification)
    staged = tmp_path / (".material-delete-" + record.source_hash + "-" + "0" * 32 + ".tmp")
    path.rename(staged)
    assert delete_verified(path, root=tmp_path, verification=verification, repo=repo) == len(raw)
    assert not staged.exists()
    assert not path.exists()


def test_project_document_backup_is_outside_cleanup_scope(tmp_path, repo, monkeypatch):
    import app.services.material_content as materials

    monkeypatch.setattr(materials, "PROJECT", tmp_path)
    path = tmp_path / "before-files" / "document.json"
    path.parent.mkdir()
    path.write_text('{"note":"project documentation"}', encoding="utf-8")
    repo.ingest(path)
    verification = "3" * 64
    repo.certify([path], verification)
    with pytest.raises(ValueError, match="NOT_COLLECTION_DIRECTORY"):
        delete_verified(path, root=tmp_path, verification=verification, repo=repo)
    assert path.exists()


def test_business_success_does_not_certify_originals_for_deletion(tmp_path, repo, monkeypatch):
    import app.services.material_content as materials
    from app.services.fund_exposure_common import _publish

    monkeypatch.setattr(materials, "PROJECT", tmp_path)
    path = tmp_path / "data/sources/notice.json"
    path.parent.mkdir(parents=True)

    calls = []

    def unavailable(*args, **kwargs):
        calls.append(True)
        raise AssertionError("business result must not certify extraction quality")

    @materials.material_operation
    def business():
        _publish(path, b'{"amount":123}')
        source_bytes(path)
        return {"status": "SUCCEEDED", "processed": 1}

    monkeypatch.setattr(repo, "certify", unavailable)
    with using_repository(repo):
        assert business() == {"status": "SUCCEEDED", "processed": 1}
    assert not path.exists()
    assert repo.get(path) is not None
    assert not calls


def test_project_cleanup_defaults_to_hold_even_with_old_business_proof(tmp_path, repo, monkeypatch):
    import app.services.material_content as materials

    monkeypatch.setattr(materials, "PROJECT", tmp_path)
    path = tmp_path / "data/sources/notice.json"
    path.parent.mkdir(parents=True)
    path.write_text('{"amount":123}', encoding="utf-8")
    repo.ingest(path)
    verification = "4" * 64
    repo.certify([path], verification)
    with pytest.raises(ValueError, match="ON_USER_HOLD"):
        delete_verified(path, root=tmp_path, verification=verification, repo=repo)
    assert path.exists()


def test_unparsed_markdown_placeholder_is_completed_without_losing_initial_attempt(tmp_path, repo):
    from app.services.material_content import opaque_content

    raw = "# 时间口径\n名义截止日是月末；净值用各自有效日历对应的实际估值日。\n".encode()
    path = tmp_path / "time-contract.md"
    path.write_bytes(raw)
    failed = repo.put(path, opaque_content(raw, ".md", "ValueError"))
    assert not failed.complete
    fixed = repo.ingest(path, metadata_only=True)
    assert fixed.source_hash == failed.source_hash
    assert fixed.decode() == raw.decode("utf-8")
    assert fixed.complete and fixed.kind == "TEXT"
    assert fixed.extracted["initial_extraction"]["content_hash"] == failed.content_hash
    assert repo.ingest(path).content_hash == fixed.content_hash
    assert path.exists()


def test_extracted_docx_retains_short_zip_signature_for_existing_parser():
    from app.services.material_content import SourceContent, checksum, encoded

    data = {"text": "原文表格123.45", "paragraphs": ["原文表格123.45"]}
    value = SourceContent("a" * 64, 1234, "DOCX", data, checksum(encoded(data)))
    assert value.startswith(b"PK")
    assert value.startswith(b"PK\x03\x04")
    assert value.startswith((b"%PDF", b"PK"))
    assert not value.startswith(b"%PDF")


def test_html_character_offsets_remain_exact_when_original_is_unavailable(tmp_path, repo):
    from app.services.material_content import SourceContent, checksum, encoded, exact_source_text

    raw = '<html>\r\n  <p data-date="2026-09-30" >公开日期：2026年9月30日</p>\r\n</html>'
    path = tmp_path / "publication.html"
    path.write_bytes(raw.encode("utf-8"))
    old = {"text": "<html><p>公开日期：2026年9月30日</p></html>"}
    value = repo.put(
        path, SourceContent(checksum(raw.encode()), len(raw.encode()), "HTML", old, checksum(encoded(old)))
    )
    value.source_path = str(path)
    with using_repository(repo):
        assert exact_source_text(value) == raw
        path.unlink()
        import app.services.material_content as materials

        with materials.database_only():
            after = exact_source_text(value)
            assert after == raw
            start = raw.index("公开日期")
            assert after[start : start + len("公开日期：2026年9月30日")] == "公开日期：2026年9月30日"


def test_known_legacy_word_uses_stored_body_without_reopening_binary():
    from app.services.fund_rating_yy_manufacturing_legacy_word import KNOWN_ORIGINALS, extract_body
    from app.services.material_content import SourceContent, checksum, encoded

    data = {"text": "已核旧Word正文及表格\t123.45\n单位：元"}
    value = SourceContent(next(iter(KNOWN_ORIGINALS)), 100000, "LEGACY_DOC", data, checksum(encoded(data)), False)
    assert value.startswith(bytes.fromhex("d0cf11e0a1b11ae1"))
    assert extract_body(value) == data["text"]
    assert not value.complete


def test_original_and_database_verification_do_not_reuse_calculated_results(monkeypatch):
    import sys
    from functools import lru_cache
    from types import ModuleType

    import app.services.material_content as materials

    module = ModuleType("app.services.material_verification_cache_fixture")

    @lru_cache
    def calculate():
        return (materials._original_only.get(), materials._database_only.get())

    calculate.__module__ = module.__name__
    module.calculate = calculate
    monkeypatch.setitem(sys.modules, module.__name__, module)
    assert calculate() == (False, False)
    with materials.original_materials():
        assert calculate() == (True, False)
    assert calculate() == (False, False)
    with materials.database_only():
        assert calculate() == (False, True)
    assert calculate() == (False, False)


def test_scoring_regression_cannot_authorize_cleanup_even_if_hold_flag_is_cleared(tmp_path, monkeypatch):
    import app.services.material_content as materials
    import app.services.material_migration_validation as validation
    from app.commands.material_content import delete

    monkeypatch.setattr(materials, "cleanup_on_hold", lambda: False)
    monkeypatch.setattr(
        validation,
        "validated_proof",
        lambda directory: {
            "evidence": {"passed": True, "purpose": "STORAGE_REGRESSION_ONLY", "authorizes_deletion": False},
            "verification": "5" * 64,
        },
    )
    with pytest.raises(ValueError, match="INDEPENDENT_EXTRACTION_REVIEW_REQUIRED"):
        delete(tmp_path)


def test_prediction_body_references_share_one_version_and_restore_exact_hash(repo):
    from app.services.direction_1d_protocol import digest
    from app.services.material_packet_storage import pack, unpack
    from sqlalchemy import func, select

    body = "公告原值123.456789、2026-10-30、人民币元。\n" * 300
    first = {"data": {"as_of": "2026-11-02T00:00:00+08:00", "documents": [{"body": body}]}, "horizon": "T1"}
    second = {"data": {"as_of": "2026-11-02T00:01:00+08:00", "documents": [{"body": body}]}, "horizon": "T5"}
    with using_repository(repo):
        one, two = pack(first), pack(second)
        assert digest(unpack(one)) == digest(first)
        assert unpack(two) == second
        assert one["references"][0]["source_hash"] == two["references"][0]["source_hash"]
        assert len(json.dumps(one)) < len(json.dumps(first)) / 10
        with repo.engine.connect() as connection:
            assert connection.scalar(select(func.count()).select_from(MaterialContent)) == 1
        two["references"][0]["content_hash"] = "0" * 64
        with pytest.raises(ValueError, match="REFERENCE_INVALID"):
            unpack(two)


def test_prediction_snapshot_reference_checks_original_logical_hash(repo):
    from app.services.direction_1d_protocol import digest
    from app.services.fund_prediction_snapshot_compat import checked_snapshot
    from app.services.material_packet_storage import pack

    source = {"data": {"fund_code": "016008", "documents": [{"body": "正文\n" * 3000}]}, "review": {"horizon": "T1"}}
    with using_repository(repo):
        stored = pack(source)
        assert checked_snapshot(stored, digest(source)) == source
        stored["value"]["review"]["horizon"] = "T5"
        with pytest.raises((ValueError, KeyError)):
            checked_snapshot(stored, digest(source))


def test_prediction_json_body_keeps_exact_character_positions(tmp_path, repo, monkeypatch):
    import app.services.material_content as materials
    from app.services.fund_exposure_common import read, save
    from app.services.material_packet_storage import pack, unpack

    monkeypatch.setattr(materials, "PROJECT", tmp_path)
    body = '{ "amount" : 123.456, "comment" : "' + "原文\n".replace("\n", "\\n") * 1800 + '" }'
    original = tmp_path / "data/sources/statement.json"
    original.parent.mkdir(parents=True)
    original.write_text(body, encoding="utf-8")
    with using_repository(repo):
        repo.ingest(original)
        logical = {"format": "CODEX_FUND_INPUT_V1", "input": {"documents": [{"body": body}]}}
        assert unpack(pack(logical)) == logical
        path = tmp_path / ".local-runs/frozen/input.json"
        save(path, logical)
        assert path.stat().st_size < len(body) / 5
        path.unlink()
        assert read(path) == logical
