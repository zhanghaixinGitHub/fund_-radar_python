"""混合读取的业务边界：缓存复用、原件版本变化、D盘归档路由和无自动清理。"""

import hashlib
import io
import json
import os
from types import SimpleNamespace

import pytest
from app.core import workspace_paths
from app.models.material_content import MaterialContent, MaterialLocation, MaterialPageRepresentation
from app.services import material_content as store
from app.services import material_file_access as access
from app.services import material_source_archive as archive
from sqlalchemy import create_engine
from sqlalchemy.exc import OperationalError


@pytest.fixture
def mixed(tmp_path, monkeypatch):
    project, disk = tmp_path / "03-python", tmp_path / "04-SourceFile"
    project.mkdir()
    disk.mkdir()
    settings = SimpleNamespace(fund_source_archive_root=str(disk))
    monkeypatch.setattr(store, "PROJECT", project)
    monkeypatch.setattr(archive, "get_settings", lambda: settings)
    monkeypatch.setattr("app.core.config.get_settings", lambda: settings)
    engine = create_engine("sqlite://")
    for table in (MaterialContent, MaterialLocation, MaterialPageRepresentation):
        table.__table__.create(engine)
    repo = store.MaterialRepository(engine)
    access._file_hash.cache_clear()
    with store.using_repository(repo):
        yield project, disk, repo
    engine.dispose()


def source(project, relative, raw):
    path = project / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return path


def forbid_repository(monkeypatch):
    monkeypatch.setattr(store, "repository", lambda: pytest.fail("unexpected material database access"))


def test_new_source_reads_original_when_material_database_is_unavailable(mixed, monkeypatch):
    project, _, _ = mixed
    raw = b'{"amount":1.2300,"condition":"original"}'
    path = source(project, "data/fund-materials/sources/new.json", raw)
    monkeypatch.setattr(store, "repository", lambda: (_ for _ in ()).throw(OperationalError("offline", {}, None)))
    assert store.source_bytes(path) == raw


def test_same_version_reuses_extraction_without_reopening_original_each_time(mixed, monkeypatch):
    project, _, repo = mixed
    raw = b'{"amount":1.2300, "condition":"original"}'
    path = source(project, "data/fund-materials/sources/current.json", raw)
    repo.ingest(path)
    value = store.source_bytes(path)
    assert isinstance(value, store.SourceContent)
    assert store.content_digest(value) == hashlib.sha256(raw).hexdigest()
    monkeypatch.setattr(type(access.physical_path(path)), "open", lambda *a, **k: pytest.fail("reopened original"))
    assert isinstance(store.source_bytes(path), store.SourceContent)


def test_changed_same_length_source_never_uses_old_cached_version(mixed):
    project, _, repo = mixed
    path = source(project, "data/fund-materials/sources/current.json", b'{"value":1}')
    repo.ingest(path)
    assert isinstance(store.source_bytes(path), store.SourceContent)
    path.write_bytes(b'{"value":2}')
    assert type(store.source_bytes(path)) is bytes
    assert store.source_bytes(path) == b'{"value":2}'


def test_mutable_run_file_reads_current_disk_without_material_database(mixed, monkeypatch):
    project, _, repo = mixed
    path = source(project, ".local-runs/research/run.json", b'{"state":"before"}')
    repo.ingest(path)
    path.write_bytes(b'{"state":"after"}')
    forbid_repository(monkeypatch)
    assert store.source_bytes(path) == b'{"state":"after"}'
    assert store.source_size(path) == len(b'{"state":"after"}')
    assert store.source_exists(path)
    assert store.location_exists(path)


def archived(mixed, *, unsupported=False):
    project, disk, repo = mixed
    raw = b'{"exact":1.2300,"source":"retained original"}'
    path = project / "data/fund-materials/sources/migrated.json"
    value = store.opaque_content(raw, ".json", "PENDING") if unsupported else store.extract(raw, ".json")
    repo.put(path, value)
    target = disk / "originals" / value.source_hash[:2] / (value.source_hash + ".json")
    target.parent.mkdir(parents=True)
    target.write_bytes(raw)
    route = {"source_hash": value.source_hash, "source_size": len(raw), "format": value.kind,
             "archive_path": str(target)}
    # 未适配格式的旧路由使用.bin，本例仍按真实旧路由规则构造。
    if unsupported:
        target = target.rename(target.with_suffix(".bin"))
        route["archive_path"] = str(target)
    repo.page(value.source_hash, archive.ROUTE_ENGINE, 0, json.dumps(route))
    return path, target, raw, value


def test_migrated_original_is_read_on_demand_and_checked(mixed):
    path, _, raw, value = archived(mixed)
    assert isinstance(store.source_bytes(path), store.SourceContent)
    assert access.original_bytes(path, expected_hash=value.source_hash) == raw


def test_uncaptured_source_is_usable_from_archive_without_text_gate(mixed):
    path, _, raw, _ = archived(mixed, unsupported=True)
    assert store.source_bytes(path) == raw


@pytest.mark.parametrize("damage", ["same_size", "truncated", "missing"])
def test_bad_archive_cannot_be_hidden_when_original_is_requested(mixed, damage):
    path, target, raw, value = archived(mixed)
    if damage == "missing":
        target.unlink()
    else:
        target.write_bytes(b"0" * len(raw) if damage == "same_size" else b"short")
    with pytest.raises(ValueError, match="MATERIAL_ARCHIVE_"):
        access.original_bytes(path, expected_hash=value.source_hash)


def test_deleted_historical_original_can_still_read_existing_representation(mixed):
    project, _, repo = mixed
    path = project / "data/fund-materials/sources/deleted.json"
    value = store.extract(b'{"historical":1.2300}', ".json")
    repo.put(path, value)
    assert store.content_digest(store.source_bytes(path)) == value.source_hash
    with pytest.raises(FileNotFoundError, match="MATERIAL_ORIGINAL_NOT_RETAINED"):
        access.original_bytes(path)


def test_old_c_alias_only_opens_current_d_equivalent(mixed, monkeypatch):
    project, _, _ = mixed
    relative = ".local-runs/research/operational.json"
    new = source(project, relative, b"current D file")
    forbid_repository(monkeypatch)
    assert store.source_bytes(workspace_paths.LEGACY_PYTHON_ROOT / relative) == new.read_bytes()


def test_save_and_operation_do_not_pack_ingest_archive_or_finalize(mixed, monkeypatch):
    from app.services.fund_exposure_common import read, save

    project, _, _ = mixed
    forbid_repository(monkeypatch)
    monkeypatch.setattr(store, "finalize_originals", lambda *a: pytest.fail("automatic cleanup"))
    path = project / ".local-runs/input.json"
    payload = {"format": "CODEX_FUND_INPUT_V1", "body": "完整输入" * 2000}

    @store.material_operation
    def execute():
        save(path, payload)
        assert read(path) == payload

    execute()
    assert json.loads(path.read_text(encoding="utf-8"))["payload"] == payload


def test_old_cleanup_entry_is_disabled_even_with_old_receipt(mixed):
    project, _, _ = mixed
    path = source(project, "data/fund-materials/sources/retained.json", b'{"keep":true}')
    with pytest.raises(RuntimeError, match="MATERIAL_SPACE_OPTIMIZATION_RETIRED"):
        store.delete_verified(path, root=project, verification="old-approved-receipt")
    assert path.read_bytes() == b'{"keep":true}'


def test_explicit_database_validation_does_not_read_changed_file(mixed):
    project, _, repo = mixed
    path = source(project, "data/fund-materials/sources/current.json", b'{"version":1}')
    value = repo.ingest(path)
    path.write_bytes(b'{"version":2}')
    with store.database_only():
        assert store.content_digest(store.source_bytes(path)) == value.source_hash


def test_existing_files_can_be_discovered_when_material_index_is_offline(mixed, monkeypatch):
    project, _, _ = mixed
    path = source(project, "data/fund-materials/sources/present.json", b'{"present":true}')
    monkeypatch.setattr(store, "repository", lambda: (_ for _ in ()).throw(OperationalError("offline", {}, None)))
    assert list(store.source_glob(path.parent, "*.json")) == [path]


def test_pdf_missing_representation_can_parse_original_once_then_reuse_cache(mixed, monkeypatch):
    from pypdf import PdfWriter

    project, _, repo = mixed
    writer = PdfWriter()
    writer.add_blank_page(width=100, height=100)
    writer.add_metadata({"/Title": "Original report"})
    stream = io.BytesIO()
    writer.write(stream)
    path = source(project, "data/fund-materials/sources/report.pdf", stream.getvalue())
    repo.ingest(path)
    with store.document_reader(store.source_bytes(path)) as reader:
        assert reader.metadata["/Title"] == "Original report"
        assert reader.pages[0].extract_text(extraction_mode="layout") == ""
    path.unlink()
    monkeypatch.setattr(access, "original_bytes", lambda *a, **k: pytest.fail("reparsed original"))
    with store.document_reader(store.source_bytes(path)) as reader:
        assert reader.metadata["/Title"] == "Original report"
        assert reader.pages[0].extract_text(extraction_mode="layout") == ""


@pytest.mark.skipif(os.name != "nt", reason="Windows盘符边界")
def test_unrecognized_c_path_is_never_read_or_written(mixed):
    with pytest.raises(ValueError, match="MUST_BE_ON_D_DRIVE"):
        access.physical_path("C:/some-other-project/source.pdf")


def test_migrated_html_exact_representation_is_extracted_only_when_requested(mixed):
    project, disk, repo = mixed
    raw = b'<html><body>amount: 1.2300; condition: original</body></html>'
    value = store.extract(raw, ".html")
    extracted = {**value.extracted, "exact_html_text": False}
    value = store.SourceContent(value.source_hash, len(raw), "HTML", extracted,
                                store.checksum(store.encoded(extracted)), value.complete, value.issue)
    path = project / "data/fund-materials/sources/migrated.html"
    repo.put(path, value)
    target = disk / "originals" / value.source_hash[:2] / (value.source_hash + ".html")
    target.parent.mkdir(parents=True)
    target.write_bytes(raw)
    route = {"source_hash": value.source_hash, "source_size": len(raw), "format": "HTML", "archive_path": str(target)}
    repo.page(value.source_hash, archive.ROUTE_ENGINE, 0, json.dumps(route))
    cached = store.source_bytes(path)
    assert repo.page(value.source_hash, "EXACT_HTML_TEXT_V1", 0) is None
    assert store.exact_source_text(cached) == raw.decode()
    assert repo.page(value.source_hash, "EXACT_HTML_TEXT_V1", 0) == raw.decode()


def test_deleted_immutable_source_cannot_be_replaced_by_a_different_version(mixed):
    from app.services.fund_exposure_common import _publish

    project, _, repo = mixed
    path = project / "data/fund-materials/sources/old.json"
    path.parent.mkdir(parents=True)
    previous = repo.put(path, store.extract(b'{"old":true}', ".json"))
    with pytest.raises(FileExistsError, match="IMMUTABLE_LOCATION_CONFLICT"):
        _publish(path, b'{"different":true}')
    assert not path.exists() and repo.get(path).source_hash == previous.source_hash


def test_new_source_is_saved_on_disk_without_automatic_ingestion(mixed):
    project, _, repo = mixed
    path = project / "data/fund-materials/sources/new-source.json"
    raw = b'{"exact":1.2300}'
    store.write_source(path, raw)
    assert path.read_bytes() == raw
    assert repo.metadata(path=path) is None
