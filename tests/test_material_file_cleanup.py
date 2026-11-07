"""高影响清理的实体保护测试，仅操作pytest自行创建的临时文件。"""

import hashlib

import pytest
from app.services.material_file_cleanup import LockedOriginal
from app.services.material_pilot_cleanup import _references


def locked(path, root, data, **kwargs):
    return LockedOriginal(
        path, root=root, expected_hash=hashlib.sha256(data).hexdigest(), expected_size=len(data), **kwargs
    )


def test_path_outside_authorized_root_is_rejected(tmp_path):
    root = tmp_path / "approved"
    root.mkdir()
    source = tmp_path / "outside.pdf"
    source.write_bytes(b"keep")
    with pytest.raises(ValueError, match="OUTSIDE|ESCAPES"):
        locked(source, root, b"keep")
    assert source.read_bytes() == b"keep"


def test_source_hash_change_preserves_file(tmp_path):
    source = tmp_path / "source.pdf"
    source.write_bytes(b"new version")
    with pytest.raises(ValueError, match="ORIGINAL_BYTES_OR_IDENTITY_CHANGED"):
        with locked(source, tmp_path, b"old version"):
            pytest.fail("changed source must not be opened for deletion")
    assert source.read_bytes() == b"new version"


def test_file_entity_change_preserves_file(tmp_path):
    source = tmp_path / "source.pdf"
    source.write_bytes(b"version")
    with pytest.raises(ValueError, match="ORIGINAL_FILE_ID_CHANGED"):
        with locked(source, tmp_path, b"version", expected_file_id=[0, 0]):
            pytest.fail("replacement entity must be rejected")
    assert source.read_bytes() == b"version"


def test_transaction_failure_cancels_pending_delete(tmp_path):
    source = tmp_path / "source.pdf"
    source.write_bytes(b"sole source")
    with pytest.raises(RuntimeError, match="commit failed"):
        with locked(source, tmp_path, b"sole source") as handle:
            handle.mark_delete()
            raise RuntimeError("commit failed")
    assert source.read_bytes() == b"sole source"


def test_verified_temp_entity_is_deleted_on_handle_close(tmp_path):
    source = tmp_path / "source.pdf"
    source.write_bytes(b"test only")
    with locked(source, tmp_path, b"test only") as handle:
        assert handle.allocated_bytes > 0 and handle.hardlinks == 1
        handle.mark_delete()
    assert not source.exists()


def test_reference_audit_detects_nested_historical_policy_dependencies():
    target = ("source", "OLD_POLICY")
    reference = {"document_source_hash": "source", "policy": "OLD_POLICY", "document_hash": "old"}
    assert _references({"nested": [reference]}, {target}) == [reference]
    assert _references({"text": str(reference)}, {target}) == []
