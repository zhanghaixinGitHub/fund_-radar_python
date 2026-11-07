"""全基金分组迁移保护测试；只操作隔离项目和临时外接盘。"""

from pathlib import Path

import pytest
from app.models.material_content import MaterialLocation
from app.services import material_archive_batch as batch
from app.services import material_archive_scope as scope
from app.services import material_content as store
from app.services import material_source_archive as archive
from sqlalchemy import event, select
from tests.test_material_source_archive import original, setup  # noqa: F401


def rows(paths, value):
    return [{"path": str(p), "size": p.stat().st_size, "source_hash": value.source_hash} for p in paths]


def test_group_deduplicates_originals_and_keeps_database_aliases(setup):  # noqa: F811
    path, disk, repo = setup
    raw, value = original(path, repo)
    other = path.parents[2] / "013472/sources/another.json"
    other.parent.mkdir(parents=True)
    other.write_bytes(raw)
    repo.link(other, value.source_hash)
    result = batch.move_group(rows([path, other], value), approved_paths=[path, other], repo=repo)
    assert not path.exists() and not other.exists()
    assert len(list((disk / "originals").rglob("*.json"))) == 1
    assert Path(result["archive_path"]).read_bytes() == raw
    assert store.source_bytes(path).decode() == store.source_bytes(other).decode() == raw.decode()
    assert result["parse_status"] == "TEXT_REVIEW_PENDING"
    assert sum(x["reclaimed_allocated_bytes"] for x in result["removed"]) > 0


def test_group_requires_explicit_scope_and_rejects_frozen_inputs(setup):  # noqa: F811
    path, _, repo = setup
    _, value = original(path, repo)
    with pytest.raises(ValueError, match="NOT_APPROVED"):
        batch.move_group(rows([path], value), approved_paths=[], repo=repo)
    snapshot = path.parent / "snapshot.json"
    _, snapshot_value = original(snapshot, repo, b'{"hash":"identity","payload":{"score":99}}')
    with pytest.raises(ValueError, match="PROTECTED_FILE"):
        batch.move_group(rows([snapshot], snapshot_value), approved_paths=[snapshot], repo=repo)
    assert path.exists() and snapshot.exists()
    assert scope.classify(path.parents[2] / "013472/frozen/sources/a.pdf", store.PROJECT)[0] is False


def test_group_refuses_changed_original_or_corrupt_destination(setup):  # noqa: F811
    path, disk, repo = setup
    raw, value = original(path, repo)
    directory = disk / "originals" / value.source_hash[:2]
    directory.mkdir(parents=True)
    (directory / (value.source_hash + ".json")).write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="COPY_MISMATCH"):
        batch.move_group(rows([path], value), approved_paths=[path], repo=repo)
    assert path.read_bytes() == raw


def test_separate_pdf_approval_is_exact_and_restored_after_failure(setup):  # noqa: F811
    from tests.test_material_content import pdf_bytes

    path, _, repo = setup
    selected = store.PROJECT / ".local-runs/legacy-eval/holdout/approved.pdf"
    other = selected.with_name("not-approved.pdf")
    raw = pdf_bytes("Source fact 123.45000 unchanged")
    _, value = original(selected, repo, raw)
    original(other, repo, raw)
    approved = [{"path": str(selected), "source_hash": value.source_hash, "kind": "PDF", "size": len(raw)}]
    assert not scope.classify(selected, store.PROJECT, indexed=True)[0]
    with pytest.raises(RuntimeError, match="intentional failure"):
        with scope.approved_pdf_input_scope(approved):
            assert scope.classify(selected, store.PROJECT)[0]
            assert not scope.classify(other, store.PROJECT)[0]
            raise RuntimeError("intentional failure")
    assert not scope.classify(selected, store.PROJECT)[0]
    with scope.approved_pdf_input_scope(approved):
        moved = batch.move_group(approved, approved_paths=[selected], repo=repo, external_manifest=False)
    assert not selected.exists() and other.read_bytes() == raw
    assert Path(moved["archive_path"]).read_bytes() == raw
    assert not scope.classify(selected, store.PROJECT)[0]
    with pytest.raises(ValueError, match="APPROVAL_INVALID"):
        with scope.approved_pdf_input_scope([{**approved[0], "path": str(path.with_suffix('.json'))}]):
            pytest.fail("non-PDF entered exceptional source scope")


def test_database_failure_cancels_all_pending_deletions(setup):  # noqa: F811
    path, _, repo = setup
    raw, value = original(path, repo)
    other = path.with_name("another.json")
    other.write_bytes(raw)
    repo.link(other, value.source_hash)

    def fail_update(connection, cursor, statement, parameters, context, many):
        if statement.lstrip().upper().startswith("UPDATE MATERIAL_SOURCE_LOCATION"):
            raise RuntimeError("simulated transaction failure")

    event.listen(repo.engine, "before_cursor_execute", fail_update)
    try:
        with pytest.raises(RuntimeError, match="transaction failure"):
            batch.move_group(rows([path, other], value), approved_paths=[path, other], repo=repo)
    finally:
        event.remove(repo.engine, "before_cursor_execute", fail_update)
    assert path.read_bytes() == other.read_bytes() == raw
    with repo.engine.connect() as c:
        assert all(r.deleted_at is None for r in c.execute(select(MaterialLocation)))
    target = archive.root_path() / "originals" / value.source_hash[:2] / (value.source_hash + ".json")
    assert target.read_bytes() == raw


def test_unknown_response_is_parsed_from_external_disk_and_keeps_one_original(setup):  # noqa: F811
    path, disk, repo = setup
    path = path.with_suffix(".raw")
    path.parent.mkdir(parents=True)
    raw = b'{"exact":123.450000,"conditions":["all","rows"]}'
    path.write_bytes(raw)
    item = {"path": str(path), "size": len(raw)}
    identified = batch.capture_unindexed(item, repo=repo)
    assert path.exists()
    assert identified["kind"] == "JSON"
    originals = [p for p in (disk / "originals").rglob("*") if p.is_file()]
    assert len(originals) == 1 and originals[0].suffix == ".json" and originals[0].read_bytes() == raw
    batch.move_group([identified], approved_paths=[path], repo=repo)
    assert not path.exists() and store.source_bytes(path).decode() == raw.decode()


def test_long_windows_source_path_keeps_normal_database_alias(setup):  # noqa: F811
    from app.services.material_file_cleanup import physical_path

    path, _, repo = setup
    path = path.parent / ("long-folder-" * 12) / ("long-file-" * 12 + ".json")
    physical = physical_path(path)
    physical.parent.mkdir(parents=True)
    raw = b'{"exact":-12.3400}'
    physical.write_bytes(raw)
    value = repo.put(path, store.extract(raw, ".json"))
    result = batch.move_group([{"path": str(path), "size": len(raw), "source_hash": value.source_hash}],
                              approved_paths=[path], repo=repo)
    assert not physical.exists() and result["removed"]
    assert store.source_bytes(path).decode() == raw.decode()


def test_existing_earnings_consumer_reads_pdf_text_and_metadata_from_database(setup, monkeypatch):  # noqa: F811
    from app.services.fund_earnings_batch_v3 import extract_pdf
    from tests.test_material_content import pdf_bytes

    path, disk, repo = setup
    path = path.with_suffix(".pdf")
    raw = pdf_bytes("Revenue 123.4500; negative cash flow; original date unchanged")
    store.write_source(path, raw)
    assert not path.exists()
    monkeypatch.setattr(Path, "read_bytes", lambda self: pytest.fail("business opened source"))
    with store.database_only():
        pages, metadata = extract_pdf(store.source_bytes(path), 10)
    assert "123.4500" in pages[0] and "CreationDate" in metadata
