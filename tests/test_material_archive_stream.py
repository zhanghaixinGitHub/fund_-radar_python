"""顺序清单与内存索引：保护历史结果、清单篡改阻断、迁移中断日志恢复。"""


import pytest
from app.commands import material_archive_stream as command
from app.services import material_content as store
from tests.test_material_source_archive import original, setup  # noqa: F401


def test_verified_legacy_responses_do_not_expand_to_results_or_frozen_inputs(tmp_path):
    from app.services import material_archive_scope as scope

    run = tmp_path / ".local-runs/direction-1d-sprint-20260914"
    for folder, name in scope.VERIFIED_LEGACY_RESPONSES:
        assert scope.classify(run / folder / name, tmp_path) == (True, "VERIFIED_LEGACY_RESPONSE")
        assert not scope.classify(run / folder / "frozen" / name, tmp_path)[0]
        assert not scope.classify(run / folder / "result.json", tmp_path)[0]
    assert not scope.classify(run / "unconfirmed/response.xls", tmp_path)[0]
    assert not scope.classify(tmp_path / ".local-runs/other/fxi-data-feasibility-v1/response.xls", tmp_path)[0]


def test_stream_plan_protects_receipts_snapshots_and_results(setup):  # noqa: F811
    path, disk, repo = setup
    original(path, repo)
    receipt = path.with_name("source.receipt.json")
    receipt.write_bytes(b'{"sha256":"known","received_at":"2026-11-05"}')
    frozen = path.parent.parent / "frozen/raw/snapshot.json"
    frozen.parent.mkdir(parents=True)
    frozen.write_bytes(b'{"amount":-123.4500}')
    result = path.parent.parent / "experiment-result.json"
    result.write_bytes(b'{"score":99}')
    run = disk / "runs/all-funds/test"
    planned = command.plan(run)
    assert planned["totals"]["SOURCE"][0] == 1
    assert planned["totals"]["PROTECTED"][0] == 3
    assert path.exists() and receipt.exists() and frozen.exists() and result.exists()
    memory = command.memory_index(run)
    assert memory.execute("PRAGMA database_list").fetchone()[2] == ""
    assert not list(run.glob("*.sqlite3"))
    memory.close()


def test_modified_manifest_blocks_any_source_deletion(setup):  # noqa: F811
    path, disk, repo = setup
    original(path, repo)
    run = disk / "runs/all-funds/test"
    command.plan(run)
    with (run/"source-plan-v3.jsonl").open("ab") as changed:
        changed.write(b"\n")
    with pytest.raises(ValueError, match="PLAN_CHANGED"):
        command.migrate(run)
    assert path.exists()


def test_migration_reuses_original_and_recovers_only_incomplete_journal_tail(setup):  # noqa: F811
    path, disk, repo = setup
    raw, value = original(path, repo)
    other = path.with_name("same-source.json")
    other.write_bytes(raw)
    repo.link(other, value.source_hash)
    run = disk / "runs/all-funds/test"
    command.plan(run)
    first = command.migrate(run)
    assert first["moved_this_process"] == 2 and not path.exists() and not other.exists()
    with (run/"stream-migration-events.jsonl").open("ab") as journal:
        journal.write(b'{"event":"MOVE')
    second = command.migrate(run)
    assert second["moved_this_process"] == 0
    assert second["C_reclaimed_allocated_bytes"] == first["C_reclaimed_allocated_bytes"]
    assert len([p for p in (disk/"originals").rglob("*") if p.is_file()]) == 1
    assert store.source_bytes(other).decode() == raw.decode()


def test_parallel_migration_keeps_groups_isolated_and_readable(setup):  # noqa: F811
    path, disk, repo = setup
    raw, value = original(path, repo)
    other = path.with_name("different.json")
    other.write_bytes(b'{"amount":-987.6500,"unit":"CNY"}')
    second = store.extract(other.read_bytes(), ".json")
    repo.put(other, second)
    run = disk/"runs/all-funds/test"
    command.plan(run)
    moved = command.migrate(run, workers=2)
    assert moved["moved_this_process"] == 2 and moved["errors"] == []
    assert not path.exists() and not other.exists()
    assert store.source_bytes(path).source_hash == value.source_hash
    assert store.source_bytes(other).source_hash == second.source_hash


def test_hardlink_group_keeps_destination_locked_until_last_alias_removed(setup):  # noqa: F811
    import os

    path, disk, repo = setup
    raw, value = original(path, repo)
    other = path.with_name("hardlink.json")
    os.link(path, other)
    repo.link(other, value.source_hash)
    run = disk/"runs/all-funds/test"
    command.plan(run)
    result = command.migrate(run, workers=2)
    assert result["errors"] == [] and result["moved_this_process"] == 2
    assert not path.exists() and not other.exists()
    assert len(list((disk/"originals").rglob("*.json"))) == 1
    assert store.source_bytes(other).source_hash == value.source_hash


def test_parse_publishes_exact_text_and_resumes_without_source_open(setup, monkeypatch):  # noqa: F811
    from app.models.material_content import MaterialDocument

    path, disk, repo = setup
    MaterialDocument.__table__.create(repo.engine)
    raw, value = original(path, repo)
    run = disk/"runs/all-funds/test"
    command.plan(run)
    command.migrate(run)
    first = command.parse_archived(run)
    assert first["status_counts"] == {"IMPORTANT_TEXT_VERIFIED": 1}
    assert first["not_yet_captured"] == 0
    from app.services import material_archive_text
    monkeypatch.setattr(material_archive_text, "capture", lambda *a, **k: pytest.fail("reparsed completed source"))
    second = command.parse_archived(run)
    assert second["attempted_this_process"] == 0


def test_write_guard_rejects_business_results_and_all_deletes(setup):  # noqa: F811
    from app.services.material_archive_guard import material_writes_only
    from sqlalchemy import text

    path, disk, repo = setup
    with material_writes_only(repo.engine) as writes:
        with repo.engine.begin() as c:
            assert c.scalar(text("SELECT 1")) == 1
            for sql in ("UPDATE fund_rating_result SET score=1", "DELETE FROM material_source_location"):
                with pytest.raises(RuntimeError, match="BUSINESS_WRITE_FORBIDDEN"):
                    c.execute(text(sql))
        assert writes == {}


def test_named_originals_are_eligible_but_reports_backups_and_eval_inputs_are_protected(setup):  # noqa: F811
    from app.services import material_archive_scope as scope

    path, disk, repo = setup
    root = store.PROJECT
    from app.services.material_source_archive import fund_for

    assert scope.classify(root/'.local-runs/fund-rating-market-test/006730/official-half-report.pdf', root,
                          indexed=True)[0]
    assert fund_for(root/'.local-runs/fund-rating-market-test/006730/official-half-report.pdf') == '006730'
    assert scope.classify(root/'.local-runs/test/companies/primary-ir/discovered/issuer.pdf', root)[0]
    for relative in [
        '.local-runs/fund-rating-market-test/006730/generated-result.pdf',
        '.local-runs/test/before-files/sources/original.pdf',
        '.local-runs/test/office-output/generated.pdf',
        '.local-runs/local-qwen35-test/new-pdfs/new-01.pdf',
        '.local-runs/direction-training-test/evidence/input.pdf',
    ]:
        assert not scope.classify(root/relative, root, indexed=True)[0]


def test_supplement_freezes_named_original_without_overwriting_old_plans(setup, monkeypatch):  # noqa: F811
    import hashlib

    from app.services import material_archive_scope as scope

    path, disk, repo = setup
    original(path, repo)
    extra = store.PROJECT/'.local-runs/fund-rating-market-test/006730/official-notice.pdf'
    extra.parent.mkdir(parents=True)
    raw = b'full original notice, 2026-11-05, amount -1.2300'
    extra.write_bytes(raw)
    repo.put(extra, store.extract(raw, ".txt"))
    identify = scope.named_collection_document
    monkeypatch.setattr(scope, "named_collection_document", lambda *a, **k: False)
    run = disk/'runs/all-funds/test'
    command.plan(run)
    before = hashlib.sha256((run/'source-plan-v3.jsonl').read_bytes()).hexdigest()
    monkeypatch.setattr(scope, "named_collection_document", identify)
    additional = command.supplement(run)
    assert additional['files'] == 1 and additional['old_manifests_unchanged']
    assert hashlib.sha256((run/'source-plan-v3.jsonl').read_bytes()).hexdigest() == before
    assert command.supplement(run)['additional_files'] == 0
    moved = command.migrate(run, workers=2)
    assert moved['moved_this_process'] == 2 and moved['errors'] == []
    assert not extra.exists() and store.source_bytes(extra).decode() == raw.decode()


def test_external_disk_fault_blocks_migration_and_handle_delete_but_allows_undo(setup):  # noqa: F811
    from app.services.material_archive_io import record_fault
    from app.services.material_file_cleanup import LockedOriginal

    path, disk, repo = setup
    raw, value = original(path, repo)
    run = disk/'runs/all-funds/test'
    command.plan(run)
    with LockedOriginal(path, root=store.PROJECT, expected_hash=value.source_hash, expected_size=len(raw)) as locked:
        locked.mark_delete()
        record_fault(run, OSError(9, 'external volume unavailable'))
        locked.mark_delete(False)
        with pytest.raises(RuntimeError, match="IO_FAULT_C_DELETION_BLOCKED"):
            locked.mark_delete()
    with pytest.raises(RuntimeError, match="IO_FAULT_REQUIRES_FULL_VERIFICATION"):
        command.migrate(run)
    with pytest.raises(RuntimeError, match="IO_FAULT_REQUIRES_FULL_VERIFICATION"):
        command.migrate(run, control_dir=store.PROJECT / ".local-runs/material-migration-control/test")
    assert path.read_bytes() == raw


def test_storage_failure_stops_remaining_aliases_in_same_group(setup, monkeypatch):  # noqa: F811
    """模拟首个硬链接迁移即失联；同来源的下一路径不能再被尝试。"""
    import os

    path, disk, repo = setup
    raw, value = original(path, repo)
    other = path.with_name("same-physical-original.json")
    os.link(path, other)
    repo.link(other, value.source_hash)
    items = [{"path": str(p), "source_hash": value.source_hash, "size": len(raw)} for p in (path, other)]
    attempts = []

    def failed_move(chunk, **kwargs):
        attempts.extend(row["path"] for row in chunk)
        raise OSError(9, "external volume disappeared")

    monkeypatch.setattr(command, "move_group", failed_move)
    events = command.move_source_group(items, repo)
    assert attempts == [str(path)]
    assert len(events) == 1 and events[0]["fatal_storage_failure"]
    assert path.read_bytes() == other.read_bytes() == raw


@pytest.mark.parametrize("migration_failed", [True, False])
def test_unavailable_D_audit_preserves_first_error_and_C_fault_gate(setup, monkeypatch, migration_failed):  # noqa: F811
    """收尾报告丢失也应阻止下次删除，已有设备异常不能被报告路径错误覆盖。"""
    import json
    import sys
    from pathlib import Path

    from app.services.material_archive_io import control_path

    path, disk, repo = setup
    raw, value = original(path, repo)
    run = disk / "runs/all-funds/test"
    run.mkdir(parents=True)
    actual_save = command.save

    def unavailable_D_save(target, record):
        if Path(target).is_relative_to(disk):
            raise FileNotFoundError("D audit path unavailable")
        return actual_save(target, record)

    def migrate(*args, **kwargs):
        if migration_failed:
            raise OSError(9, "original device disappeared")
        return {"moved_this_process": 0}

    monkeypatch.setattr(command, "save", unavailable_D_save)
    monkeypatch.setattr(command, "migrate", migrate)
    monkeypatch.setattr(sys, "argv", ["material_archive_stream", "migrate", "--run", str(run)])
    expected = "original device disappeared" if migration_failed else "D audit path unavailable"
    with pytest.raises(OSError, match=expected):
        command.main()
    assert control_path().is_file()
    fallback = json.loads(control_path().with_name("migrate-write-scope-D-unavailable.json").read_text())
    assert fallback["business_table_writes"] == 0
    assert fallback["source_file_bytes_saved_on_C"] is False
    assert bool(fallback["primary_error"]) == migration_failed
    assert path.read_bytes() == raw


def test_local_control_resumes_legacy_events_without_new_external_metadata_writes(setup, monkeypatch):  # noqa: F811
    """前批D盘流水保持原字节；新批只写原件，路径关系读库，工程流水不含正文。"""
    import json
    import sys
    from pathlib import Path

    from app.models.material_content import MaterialDocument
    from app.services import material_source_archive as archive

    path, disk, repo = setup
    MaterialDocument.__table__.create(repo.engine)
    raw, value = original(path, repo, ('{"legacy_note":"' + 'x' * 512 + '"}').encode())
    other = path.with_name("new-source.json")
    second_raw, second = original(other, repo, b'{"precision":-123.000010,"note":"unique body not in engineering log"}')
    run = disk / "runs/all-funds/test"
    command.plan(run)
    command.migrate(run, limit=1)
    with (run / "stream-migration-events.jsonl").open("ab") as interrupted:
        interrupted.write(b'{"event":"unfinished legacy tail')
    old_journal = (run / "stream-migration-events.jsonl").read_bytes()
    before = {str(p): p.read_bytes() for p in run.iterdir() if p.is_file()}
    control = store.PROJECT / ".local-runs/material-migration-control/test"
    directory = archive._directory

    def no_external_manifest(root, relative):
        if str(relative).startswith("manifests"):
            pytest.fail("new group wrote an external alias manifest")
        return directory(root, relative)

    actual_open = Path.open

    def readonly_legacy_metadata(target, mode="r", *args, **kwargs):
        if target.is_relative_to(run) and any(char in mode for char in "wax+"):
            pytest.fail("mutated legacy external engineering metadata")
        return actual_open(target, mode, *args, **kwargs)

    monkeypatch.setattr(archive, "_directory", no_external_manifest)
    monkeypatch.setattr(Path, "open", readonly_legacy_metadata)
    monkeypatch.setattr(sys, "argv", ["material_archive_stream", "migrate", "--run", str(run),
                                     "--control-dir", str(control)])
    command.main()
    moved = json.loads((control / "stream-migration-result.json").read_text())
    assert moved["moved_this_process"] == 1 and not moved["errors"]
    assert not other.exists() and store.source_bytes(other).decode() == second_raw.decode()
    assert (run / "stream-migration-events.jsonl").read_bytes() == old_journal
    assert before == {str(p): p.read_bytes() for p in run.iterdir() if p.is_file()}
    entries = [json.loads(line) for line in (control / "stream-migration-events.jsonl").read_text().splitlines()]
    assert len(entries) == 1 and entries[0]["source_hash"] == second.source_hash
    assert b"unique body not in engineering log" not in (control / "stream-migration-events.jsonl").read_bytes()
    with (control / "stream-migration-events.jsonl").open("ab") as interrupted:
        interrupted.write(b'{"event":"unfinished new tail')
    assert command.migrate(run, control_dir=control)["moved_this_process"] == 0
    parsed = command.parse_archived(run, control_dir=control)
    assert parsed["status_counts"] == {"IMPORTANT_TEXT_VERIFIED": 2}
    assert command.parse_archived(run, control_dir=control)["attempted_this_process"] == 0


def test_local_control_rejects_unrelated_run_and_budget_overflow_before_deletion(setup, monkeypatch):  # noqa: F811
    path, disk, repo = setup
    raw, value = original(path, repo)
    run = disk / "runs/all-funds/test"
    command.plan(run)
    monkeypatch.setattr(command, "CONTROL_BUDGET_BYTES", 32 * 1024)
    control = store.PROJECT / ".local-runs/material-migration-control/test"
    command.state_root(run, control)
    with pytest.raises(ValueError, match="ENGINEERING_SCOPE"):
        command.migrate(run, control_dir=store.PROJECT / "data/unapproved")
    with pytest.raises(ValueError, match="SINGLE_WORKER"):
        command.migrate(run, control_dir=control, workers=2)
    with (control / "engineering-budget-placeholder").open("wb") as placeholder:
        placeholder.truncate(32 * 1024)
    with pytest.raises(RuntimeError, match="METADATA_BUDGET_EXCEEDED"):
        command.migrate(run, control_dir=control)
    assert path.read_bytes() == raw
    other_run = disk / "runs/all-funds/other"
    other_run.mkdir()
    (other_run / "stream-plan.json").write_text('{"source_manifest_sha256":"different"}')
    with pytest.raises(ValueError, match="RUN_IDENTITY_CHANGED"):
        command.state_root(other_run, control)


def test_control_budget_survives_atomic_report_temp_removal(tmp_path, monkeypatch):
    """监控写者替换临时报告不应中止迁移，但稳定文件达到预算仍须阻断。"""
    from pathlib import Path

    stable = tmp_path / "events.jsonl"
    stable.write_bytes(b"x" * 32)
    transient = tmp_path / "report.tmp"
    transient.write_bytes(b"temporary")
    original_stat = Path.stat

    def stat_during_replace(path, *args, **kwargs):
        if path == transient:
            transient.unlink(missing_ok=True)
            raise FileNotFoundError("report was atomically replaced")
        return original_stat(path, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", stat_during_replace)
    assert command.control_metadata_bytes(tmp_path) == 32
    monkeypatch.setattr(command, "CONTROL_BUDGET_BYTES", 32)
    with pytest.raises(RuntimeError, match="METADATA_BUDGET_EXCEEDED"):
        command.check_control_budget(tmp_path, tmp_path / "external-run")


@pytest.mark.parametrize("error", [PermissionError("access denied"), OSError(5, "device I/O failure")])
def test_control_budget_keeps_other_filesystem_failures(tmp_path, monkeypatch, error):
    from pathlib import Path

    metadata = tmp_path / "events.jsonl"
    metadata.write_bytes(b"existing receipt")
    original_stat = Path.stat

    def failed_stat(path, *args, **kwargs):
        if path == metadata:
            raise error
        return original_stat(path, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", failed_stat)
    with pytest.raises(type(error), match=str(error).split("] ")[-1]):
        command.check_control_budget(tmp_path, tmp_path / "external-run")


@pytest.mark.parametrize("storage_failure", [True, False])
def test_local_control_stops_identification_on_first_error(setup, monkeypatch, storage_failure):  # noqa: F811
    path, disk, repo = setup
    path.parent.mkdir(parents=True)
    raw = b'{"exact":1.2300,"rows":[1,2,3]}'
    path.write_bytes(raw)
    other = path.with_name("another.json")
    other.write_bytes(raw + b" ")
    run = disk / "runs/all-funds/test"
    assert command.plan(run)["unindexed_sources"] == 2
    attempts = []

    def failed_identification(item, **kwargs):
        attempts.append(item["path"])
        if storage_failure:
            raise OSError(9, "device unavailable")
        raise ValueError("source scope needs review")

    monkeypatch.setattr(command, "capture_unindexed", failed_identification)
    control = store.PROJECT / ".local-runs/material-migration-control/test"
    with pytest.raises(RuntimeError, match="FATAL_STORAGE_FAILURE|IDENTIFICATION_REQUIRES_REVIEW"):
        command.migrate(run, control_dir=control)
    assert len(attempts) == 1
    assert path.read_bytes() == raw and other.read_bytes() == raw + b" "


def test_recovered_only_archive_is_included_in_text_capture(setup):  # noqa: F811
    import json

    from app.models.material_content import MaterialDocument

    path, disk, repo = setup
    MaterialDocument.__table__.create(repo.engine)
    raw, value = original(path, repo)
    run = disk / "runs/all-funds/test"
    command.plan(run)
    command.migrate(run)
    event = {"event": "RECOVERED", "path": str(path), "source_hash": value.source_hash,
             "allocation_unknown": True}
    (run / "stream-migration-events.jsonl").write_text(json.dumps(event) + "\n")
    result = command.parse_archived(run)
    assert result["archived_sources"] == 1 and result["status_counts"] == {"IMPORTANT_TEXT_VERIFIED": 1}
    assert not path.exists() and store.source_bytes(path).decode() == raw.decode()


def test_engineering_lock_is_shared_across_path_case_and_released_after_exit(setup):  # noqa: F811
    from pathlib import Path

    path, disk, repo = setup
    run = disk / "runs/all-funds/test"
    run.mkdir(parents=True)
    with command.run_lock(run, "migrate"):
        with pytest.raises(RuntimeError, match="ALREADY_RUNNING"):
            with command.run_lock(Path(str(run).upper()), "migrate"):
                pytest.fail("second writer obtained the same batch")
    with command.run_lock(run, "migrate"):
        pass


def test_local_control_text_io_failure_stops_remaining_originals(setup, monkeypatch):  # noqa: F811
    from app.models.material_content import MaterialDocument
    from app.services import material_archive_text
    from app.services.material_archive_io import control_path

    path, disk, repo = setup
    MaterialDocument.__table__.create(repo.engine)
    raw, value = original(path, repo)
    other = path.with_name("second.json")
    original(other, repo, b'{"another":-1.2300}')
    run = disk / "runs/all-funds/test"
    command.plan(run)
    control = store.PROJECT / ".local-runs/material-migration-control/test"
    command.migrate(run, control_dir=control)
    attempts = []

    def unavailable_original(target, **kwargs):
        attempts.append(target)
        raise OSError(9, "external drive unavailable during extraction")

    monkeypatch.setattr(material_archive_text, "capture", unavailable_original)
    with pytest.raises(RuntimeError, match="TEXT_IO_FAILURE"):
        command.parse_archived(run, control_dir=control)
    assert len(attempts) == 1 and control_path().is_file()
    assert store.source_bytes(path).decode() == raw.decode()
