"""外接盘归档、失败保留及普通业务禁止原件回退；只操作测试自己的临时目录。"""

from pathlib import Path
from types import SimpleNamespace

import pytest
from app.models.material_content import MaterialContent, MaterialLocation, MaterialPageRepresentation
from app.services import material_content as store
from app.services import material_source_archive as archive
from sqlalchemy import create_engine


@pytest.fixture
def setup(tmp_path, monkeypatch):
    project, disk = tmp_path / "project", tmp_path / "external-disk"
    project.mkdir()
    disk.mkdir()
    settings = SimpleNamespace(fund_source_archive_root=str(disk), fund_source_archive_funds="008888")
    monkeypatch.setattr(archive, "get_settings", lambda: settings)
    monkeypatch.setattr(store, "PROJECT", project)
    engine = create_engine(f"sqlite:///{tmp_path / 'database.sqlite3'}")
    for table in (MaterialContent, MaterialLocation, MaterialPageRepresentation):
        table.__table__.create(engine)
    repo = store.MaterialRepository(engine)
    path = project / ".local-runs/task/008888/sources/source.json"
    with store.using_repository(repo):
        yield path, disk, repo
    engine.dispose()


def original(path, repo, raw=b'{"value":1.2300,"note":"full"}'):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    value = repo.put(path, store.extract(raw, path.suffix))
    return raw, value


def test_fund_identity_is_explicit_and_not_limited_to_pilot(setup):
    path, _, _ = setup
    assert archive.fund_for(path) == "008888"
    assert archive.fund_for(path.with_name("hash008888abc.json").parents[2] / "sources/hash008888abc.json") is None
    assert archive.fund_for(path.parents[2] / "010737/sources/item.json") == "010737"
    assert archive.fund_for(store.PROJECT / "data/company-sources/600089/source.pdf") is None
    assert archive.fund_for(store.PROJECT / ".local-runs/run/funds/008888/company-sources/600089/source.pdf") == "008888"


def test_new_original_is_written_to_archive_and_business_reads_only_database(setup, monkeypatch):
    path, disk, _ = setup
    raw = b'{"exact":1.2300,"all":[1,2,3]}'
    store.write_source(path, raw)
    assert not path.exists()
    target = next((disk / "originals").rglob("*.json"))
    assert target.read_bytes() == raw
    assert store.source_bytes(path).decode() == raw.decode()
    # 模拟外接盘不可读，不删除真实归档；普通资料入口必须仍可取得库内正文。
    monkeypatch.setattr(Path, "read_bytes", lambda self: pytest.fail("business opened an original"))
    assert store.source_bytes(path).decode() == raw.decode()
    assert store.source_bytes(target).decode() == raw.decode()


def test_existing_source_relocated_after_archive_and_database_readback(setup):
    path, _, repo = setup
    raw, value = original(path, repo)
    result = archive.relocate(path, source_hash=value.source_hash, delete_local=True, repo=repo)
    assert not path.exists() and result["removed_C_copy"]
    assert Path(result["archive_path"]).read_bytes() == raw
    assert store.source_bytes(path).source_hash == value.source_hash
    assert store.source_bytes(result["archive_path"]).decode() == raw.decode()


def test_deduplicates_content_but_preserves_both_source_aliases(setup):
    path, disk, repo = setup
    raw, value = original(path, repo)
    other = path.with_name("another.json")
    other.write_bytes(raw)
    repo.link(other, value.source_hash)
    for p in (path, other):
        archive.relocate(p, source_hash=value.source_hash, delete_local=True, repo=repo)
    assert len(list((disk / "originals").rglob("*.json"))) == 1
    assert len(list((disk / "manifests/008888").rglob("*.json"))) == 2
    assert repo.metadata(path=path).source_hash == repo.metadata(path=other).source_hash


def test_archive_offline_does_not_fall_back_to_local_source_write(setup):
    path, disk, _ = setup
    disk.rmdir()
    with pytest.raises(ValueError, match="DRIVE_UNAVAILABLE"):
        store.write_source(path, b'{"do_not_write_C":true}')
    assert not path.exists()


def test_corrupt_archive_prevents_local_copy_deletion(setup):
    path, disk, repo = setup
    raw, value = original(path, repo)
    folder = disk / "originals" / value.source_hash[:2]
    folder.mkdir(parents=True)
    (folder / (value.source_hash + ".json")).write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="COPY_MISMATCH"):
        archive.relocate(path, source_hash=value.source_hash, delete_local=True, repo=repo)
    assert path.read_bytes() == raw


def test_database_link_failure_preserves_local_copy(setup, monkeypatch):
    path, _, repo = setup
    raw, value = original(path, repo)
    monkeypatch.setattr(repo, "link", lambda *a: (_ for _ in ()).throw(RuntimeError("database failure")))
    with pytest.raises(RuntimeError, match="database failure"):
        archive.relocate(path, source_hash=value.source_hash, delete_local=True, repo=repo)
    assert path.read_bytes() == raw


def test_unindexed_pilot_source_is_not_implicitly_parsed(setup):
    path, _, _ = setup
    path.parent.mkdir(parents=True)
    path.write_bytes(b'{"not_yet_ingested":true}')
    with pytest.raises(ValueError, match="DATABASE_MIGRATION_MISSING"):
        store.source_bytes(path)


def test_missing_pdf_layout_does_not_reopen_archived_source(setup, monkeypatch):
    _, _, repo = setup
    from tests.test_material_content import pdf_bytes

    raw = pdf_bytes()
    value = store.extract(raw, ".pdf")
    value.database_only_source = True
    value.source_path = str(Path(archive.get_settings().fund_source_archive_root) / "original.pdf")
    repo.put(value.source_path, value)
    monkeypatch.setattr(Path, "read_bytes", lambda self: pytest.fail("opened original PDF"))
    with store.StoredDocument(value) as document:
        with pytest.raises(ValueError, match="LAYOUT_REPRESENTATION_NOT_CAPTURED"):
            document.page_text(0, layout=True)


def test_interrupted_move_resumes_only_with_committed_database_marker(setup):
    from app.commands.material_source_archive import resumed_item

    path, _, repo = setup
    _, value = original(path, repo)
    row = {
        "location": str(path),
        "source_hash": value.source_hash,
        "format": value.kind,
        "source_size": value.source_size,
    }
    archive.relocate(path, source_hash=value.source_hash, delete_local=True, repo=repo)
    result = resumed_item(row, repo, {})
    assert result["allocation_unknown_after_interrupted_progress_write"]
    assert result["reclaimed_C_allocated_bytes"] == 0


def test_html_full_table_saved_but_external_image_remains_unverified(setup):
    from app.commands.material_source_archive import capture_text

    path, _, repo = setup
    path = path.with_suffix(".html")
    raw = (
        '<html><body><table><tr><td>精确值</td><td>-1.2300%</td></tr></table><img src="chart.png"></body></html>'
    ).encode()
    _, value = original(path, repo, raw)
    item = archive.relocate(path, source_hash=value.source_hash, repo=repo)
    result = capture_text(item["archive_path"], repo)
    assert result["status"] == "HTML_TEXT_SAVED_EXTERNAL_REGIONS_UNVERIFIED"
    saved = repo.page(value.source_hash, "ARCHIVE_HTML_TEXT_V1", 0)
    assert "-1.2300%" in saved and "chart.png" in saved
    assert store.source_bytes(path).database_only_source


def test_legacy_html_added_whitespace_is_recorded_without_losing_source_text(setup):
    from app.commands.material_source_archive import capture_text

    path, _, repo = setup
    path = path.with_suffix(".html")
    body = "<html><body>完整正文及-1.2300%</body></html>"
    raw = body.encode()
    path.parent.mkdir(parents=True)
    path.write_bytes(raw)
    data = {"text": "\n" + body, "plain": "完整正文及-1.2300%", "tables": []}
    value = store.SourceContent(store.checksum(raw), len(raw), "HTML", data, store.checksum(store.encoded(data)))
    repo.put(path, value)
    item = archive.relocate(path, source_hash=value.source_hash, repo=repo)
    assert capture_text(item["archive_path"], repo)["status"] == "COMPLETE_SOURCE_TEXT"
    assert capture_text(item["archive_path"], repo)["legacy_whitespace_difference"]
    assert repo.page(value.source_hash, "EXACT_HTML_TEXT_V1", 0) == body


def test_explicit_shared_copy_removal_keeps_database_alias_and_external_original(setup):
    path, _, repo = setup
    raw, value = original(path, repo)
    moved = archive.relocate(path, source_hash=value.source_hash, delete_local=True, repo=repo)
    shared = path.parents[3] / "shared/sources/previous-run.json"
    shared.parent.mkdir(parents=True)
    shared.write_bytes(raw)
    repo.link(shared, value.source_hash)
    assert archive.fund_for(shared) is None
    with pytest.raises(ValueError, match="EXPLICIT_SCOPE"):
        archive.remove_archived_copy(shared, source_hash=value.source_hash, approved_paths=[], repo=repo)
    assert shared.exists()
    archive.remove_archived_copy(shared, source_hash=value.source_hash, approved_paths=[shared], repo=repo)
    assert not shared.exists()
    assert Path(moved["archive_path"]).read_bytes() == raw
    assert store.source_bytes(shared).decode() == raw.decode()


def test_explicit_shared_copy_removal_requires_identical_external_original(setup):
    path, _, repo = setup
    raw, value = original(path, repo)
    moved = archive.relocate(path, source_hash=value.source_hash, repo=repo)
    Path(moved["archive_path"]).write_bytes(b"changed")
    with pytest.raises(ValueError):
        archive.remove_archived_copy(path, source_hash=value.source_hash, approved_paths=[path], repo=repo)
    assert path.read_bytes() == raw


@pytest.mark.parametrize("relative", [
    "data/010737/sources/new.json", "data/shared/sources/new.json", ".local-runs/job/raw.bin",
])
def test_all_new_sources_go_directly_to_external_disk_without_fund_gate(setup, relative, monkeypatch):
    _, disk, repo = setup
    path = store.PROJECT / relative
    raw = b'{"exact":-1.2300,"all_rows":[1,2,3]}'
    store.write_source(path, bytearray(raw))
    assert not path.exists()
    target = next((disk / "originals").rglob("*.json"))
    assert target.read_bytes() == raw
    assert archive.archive_route(store.checksum(raw), repo)
    monkeypatch.setattr(Path, "read_bytes", lambda self: pytest.fail("business opened a source file"))
    assert store.source_bytes(path).decode() == raw.decode()
    assert store.source_bytes(path).database_only_source


def test_inline_images_are_pending_text_extraction_without_binary_database_payload(setup):
    path, disk, _ = setup
    path = path.with_suffix(".html")
    raw = (
        b'<p>Exact -1.2300%</p><svg><image src="data:image/png;base64,AAAA"/></svg>'
        b'<img alt="chart" src="data:image/png;base64,BBBB">'
    )
    store.write_source(path, raw)
    value = store.source_bytes(path)
    assert not value.complete and "-1.2300%" in value.decode()
    assert "base64" not in value.decode() and "<svg" not in value.decode()
    assert value.extracted["visuals_requiring_text"]
    assert next((disk / "originals").rglob("*.html")).read_bytes() == raw


def test_image_source_is_archived_but_pixels_are_not_saved_as_parsed_content(setup):
    path, disk, repo = setup
    path = path.with_suffix(".png")
    raw = b'\x89PNG\r\n\x1a\nRAW_PIXEL_PLACEHOLDER'
    store.write_source(path, raw)
    value = repo.get(path)
    assert not value.complete and value.issue == "IMAGE_REQUIRES_VERIFIED_TEXT_EXTRACTION"
    assert "RAW_PIXEL_PLACEHOLDER" not in store.encoded(value.extracted).decode()
    assert next((disk / "originals").rglob("*.bin")).read_bytes() == raw
    with pytest.raises(ValueError, match="TEXT_EXTRACTION_REQUIRED"):
        store.source_bytes(path)
