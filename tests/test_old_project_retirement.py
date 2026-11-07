"""删除前核验的关键保护：C盘只读、D盘既有内容不覆盖、变化不能误报成功。"""

import sys

import pytest
from scripts import verify_old_project_retirement as audit


@pytest.fixture
def setup(tmp_path, monkeypatch):
    source, workspace = tmp_path / "old", tmp_path / "workspace"
    source.mkdir()
    workspace.mkdir()
    monkeypatch.setattr(audit, "ROOTS", {"03-python": source})
    monkeypatch.setattr(audit, "WORKSPACE", workspace)
    monkeypatch.setattr(audit, "CONTROL", workspace / "audit-control")
    return source, workspace


def row(source, relative, body=b"retained historical data"):
    path = source / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(body)
    info = path.stat()
    return "03-python", relative, info.st_size, info.st_mtime_ns


def test_equal_data_is_verified_without_changing_source_or_destination(setup):
    source, workspace = setup
    item = row(source, "data/snapshot.json")
    target = workspace / "03-python/data/snapshot.json"
    target.parent.mkdir(parents=True)
    target.write_bytes((source / item[1]).read_bytes())
    before = (source / item[1]).stat()
    result = audit.compare(item)
    assert result[2] == "equal"
    assert (source / item[1]).stat().st_mtime_ns == before.st_mtime_ns


def test_missing_history_is_restored_and_source_remains(setup):
    source, workspace = setup
    item = row(source, ".local-runs/frozen-input.json")
    assert audit.compare(item)[2] == "restored"
    assert (source / item[1]).read_bytes() == (workspace / "03-python" / item[1]).read_bytes()


def test_newer_destination_is_preserved_and_old_version_gets_independent_backup(setup):
    source, workspace = setup
    item = row(source, "data/snapshot.json", b"old history")
    target = workspace / "03-python/data/snapshot.json"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"new history")
    result = audit.compare(item)
    assert result[2] == "different-retained"
    assert target.read_bytes() == b"new history"
    assert audit.io_path(result[4]).read_bytes() == b"old history"
    assert (source / item[1]).read_bytes() == b"old history"


def test_changed_source_cannot_get_a_success_receipt(setup):
    source, _ = setup
    item = row(source, "data/snapshot.json", b"before")
    (source / item[1]).write_bytes(b"after, changed length")
    with pytest.raises(ValueError, match="SOURCE_CHANGED_AFTER_FREEZE"):
        audit.compare(item)


def test_missing_git_reference_is_backed_up_without_writing_into_current_git(setup):
    source, workspace = setup
    item = row(source, ".git/refs/codex/checkpoint")
    result = audit.compare(item)
    assert result[2] == "git-retained"
    assert not (workspace / "03-python" / item[1]).exists()
    assert audit.io_path(result[4]).read_bytes() == (source / item[1]).read_bytes()


def test_disconnected_workspace_stops_before_creating_destination(setup, monkeypatch):
    source, workspace = setup
    item = row(source, "data/snapshot.json")
    monkeypatch.setattr(audit, "WORKSPACE", workspace / "disconnected")
    with pytest.raises(audit.WorkspaceStorageUnavailable, match="WORKSPACE_DRIVE_UNAVAILABLE"):
        audit.compare(item)
    assert not (workspace / "disconnected").exists()


def test_device_io_failure_requires_a_storage_stop(setup):
    failure = OSError("device removed")
    failure.winerror = 433
    assert audit.storage_failure(failure)


def test_storage_failure_exits_unsuccessfully_and_keeps_sources(setup, tmp_path, monkeypatch):
    source, _ = setup
    row(source, "data/snapshot.json")
    monkeypatch.setattr(audit, "PAIRS", (("03-python", source),))
    monkeypatch.setattr(sys, "argv",
                        ["verify_old_project_retirement.py", "--control-dir", str(tmp_path / "control")])

    def unavailable(_row):
        failure = OSError("device removed")
        failure.winerror = 433
        raise failure

    monkeypatch.setattr(audit, "compare", unavailable)
    with pytest.raises(SystemExit) as result:
        audit.main()
    assert result.value.code == 2
    assert (source / "data/snapshot.json").exists()
