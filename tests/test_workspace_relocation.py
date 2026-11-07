"""迁移路径兼容只读历史资料，且不能跨目录、覆盖旧来源或重新读取原件。"""

import json
from types import SimpleNamespace

import pytest
from app.core import workspace_paths as paths
from app.models.material_content import MaterialContent, MaterialLocation, MaterialPageRepresentation
from app.services import material_content as store
from app.services import material_source_archive as archive
from sqlalchemy import create_engine


@pytest.fixture
def relocated(tmp_path, monkeypatch):
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
    # 本组验证“显式只读库”迁移契约；默认混合读取另由test_material_mixed_reading覆盖。
    with store.using_repository(repo), store.database_only():
        yield project, disk, repo
    engine.dispose()


def test_old_and_new_paths_read_database_without_opening_original(relocated, monkeypatch):
    project, _, repo = relocated
    relative = "data/fund-insights/ratings/sources/reviewed/item.json"
    old, new = paths.LEGACY_PYTHON_ROOT / relative, project / relative
    value = store.extract(b'{"exact":1.2300,"condition":"full"}', ".json")
    repo.put(old, value)
    monkeypatch.setattr(type(new), "read_bytes", lambda self: pytest.fail("opened an original"))
    assert repo.metadata(new).source_hash == value.source_hash
    assert store.source_bytes(new).decode() == store.source_bytes(old).decode()
    # 兼容查询不建立新别名，不把已删除原件的旧凭据改成新路径。
    with repo.engine.connect() as connection:
        assert connection.exec_driver_sql("SELECT count(*) FROM material_source_location").scalar() == 1


def test_current_version_precedes_fallback_and_old_explicit_version_remains(relocated):
    project, _, repo = relocated
    old = paths.LEGACY_PYTHON_ROOT / "data/fund-materials/sources/item.json"
    new = project / "data/fund-materials/sources/item.json"
    before, after = store.extract(b'{"value":1}', ".json"), store.extract(b'{"value":2}', ".json")
    repo.put(old, before)
    repo.put(new, after)
    assert repo.get(new).source_hash == after.source_hash
    assert repo.get(old).source_hash == before.source_hash


def test_sibling_prefix_and_parent_escape_do_not_become_material_aliases(relocated):
    project, disk, _ = relocated
    sibling = project.with_name("03-python-other") / "sources/item.json"
    assert paths.material_location_aliases(sibling, project_root=project, archive_root=disk) == (str(sibling),)
    escaped = project / "../outside/sources/item.json"
    assert len(paths.material_location_aliases(escaped, project_root=project, archive_root=disk)) == 1
    legacy_escape = paths.LEGACY_PYTHON_ROOT / "../outside/item.json"
    assert not paths.runtime_material_path(legacy_escape, project_root=project).is_relative_to(project)


def test_old_archive_route_resolves_new_entity_without_overwriting_receipt(relocated):
    _, disk, repo = relocated
    value = store.extract(b'{"keep":1.2300}', ".json")
    old = paths.LEGACY_SOURCE_ROOT / "originals" / value.source_hash[:2] / (value.source_hash + ".json")
    target = disk / "originals" / value.source_hash[:2] / old.name
    receipt = {"source_hash": value.source_hash, "source_size": value.source_size,
               "format": value.kind, "archive_path": str(old)}
    repo.put(old, value)
    body = json.dumps(receipt)
    repo.page(value.source_hash, archive.ROUTE_ENGINE, 0, body)
    assert archive.archive_path(old)
    assert archive.archive_route(value.source_hash, repo)["archive_path"] == str(target)
    assert archive._register_verified_archive(target, value, repo)["archive_path"] == str(target)
    assert repo.page(value.source_hash, archive.ROUTE_ENGINE, 0) == body


@pytest.mark.parametrize("change", ["hash", "bucket", "extension", "outside"])
def test_archive_identity_validation_still_rejects_changed_route(relocated, change):
    _, _, repo = relocated
    value = store.extract(b'{"keep":1}', ".json")
    target = paths.LEGACY_SOURCE_ROOT / "originals" / value.source_hash[:2] / (value.source_hash + ".json")
    if change == "bucket":
        target = target.parent.parent / "wrong" / target.name
    elif change == "extension":
        target = target.with_suffix(".pdf")
    elif change == "outside":
        target = target.parent.parent.parent.with_name("unapproved") / target.name
    receipt = {"source_hash": "0" * 64 if change == "hash" else value.source_hash,
               "source_size": value.source_size, "format": value.kind, "archive_path": str(target)}
    repo.put(target, value)
    repo.page(value.source_hash, archive.ROUTE_ENGINE, 0, json.dumps(receipt))
    with pytest.raises(ValueError, match="MATERIAL_ARCHIVE_ROUTE_IDENTITY_CHANGED"):
        archive.archive_route(value.source_hash, repo)


def test_market_source_reference_reads_migrated_database_without_physical_stat(relocated, monkeypatch):
    from app.integrations.market_fund_reports import read_source

    project, _, repo = relocated
    relative = "data/fund-insights/share-classes/sources/item.json"
    old, new = paths.LEGACY_PYTHON_ROOT / relative, project / relative
    raw = b'{"value":1.2300,"condition":"unchanged"}'
    value = store.extract(raw, ".json")
    repo.put(old, value)
    monkeypatch.setattr(type(new), "stat", lambda *args, **kwargs: pytest.fail("stat original"))
    monkeypatch.setattr(type(new), "read_bytes", lambda *args, **kwargs: pytest.fail("opened original"))
    result = read_source(new.parent.parent, {"file": "sources/item.json", "sha256": value.source_hash})
    assert result.source_hash == value.source_hash
    assert result.decode() == raw.decode()


def test_market_source_migrated_metadata_still_enforces_size_and_hash(relocated):
    from app.integrations.market_fund_reports import read_source

    project, _, repo = relocated
    root = project / "data/fund-insights/share-classes"
    old = paths.LEGACY_PYTHON_ROOT / "data/fund-insights/share-classes/sources/large.json"
    value = store.extract(b'{"value":1}', ".json")
    value.source_size = 20_000_001
    repo.put(old, value)
    with pytest.raises(ValueError, match="MARKET_REPORT_SOURCE_PATH"):
        read_source(root, {"file": "sources/large.json", "sha256": value.source_hash})
    with pytest.raises(ValueError, match="MARKET_REPORT_SOURCE_PATH"):
        read_source(root, {"file": "../outside.json", "sha256": value.source_hash})


def test_directory_discovery_uses_old_database_prefix_and_returns_new_virtual_paths(relocated, monkeypatch):
    project, _, repo = relocated
    relative = "data/fund-materials/sources"
    first, second = "first.json", "nested/second.json"
    value = store.extract(b'{"value":1}', ".json")
    for name in (first, second):
        repo.put(paths.LEGACY_PYTHON_ROOT / relative / name, value)
    root = project / relative
    with monkeypatch.context() as patcher, store.database_only():
        patcher.setattr(type(root), "glob", lambda *args, **kwargs: pytest.fail("scanned physical source directory"))
        patcher.setattr(type(root), "rglob", lambda *args, **kwargs: pytest.fail("scanned physical source directory"))
        assert list(store.source_glob(root, "*.json")) == [root / first]
        assert set(store.source_glob(root, "*.json", recursive=True)) == {root / first, root / second}
        old = paths.LEGACY_PYTHON_ROOT / relative
        assert list(store.source_glob(old, "*.json")) == [old / first]


def test_legacy_operational_file_checks_use_current_physical_directory(relocated):
    project, _, _ = relocated
    relative = ".local-runs/workspace-test/current.json"
    current = project / relative
    current.parent.mkdir(parents=True)
    current.write_bytes(b'{"value":1}')
    old = paths.LEGACY_PYTHON_ROOT / relative
    assert store.source_exists(old)
    assert store.location_exists(old.parent)
    assert store.source_size(old) == current.stat().st_size


def test_explicit_original_validation_takes_precedence_over_database_only(relocated):
    project, _, _ = relocated
    root = project / ".local-runs/explicit-original/sources"
    root.mkdir(parents=True)
    path = root / "item.json"
    path.write_bytes(b'{"value":1}')
    with store.database_only(), store.original_materials():
        assert list(store.source_glob(root, "*.json")) == [path]
